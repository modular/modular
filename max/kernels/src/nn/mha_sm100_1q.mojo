# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from math import ceildiv, exp2, recip, align_up
from math.constants import log2e

from sys import align_of, simd_width_of, size_of

from gpu import warp_id
import gpu.primitives.warp as warp
from algorithm.functional import unswitch
from collections import OptionalReg
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    lane_id,
    thread_idx,
    block_idx,
)
from gpu.primitives.cluster import elect_one_sync
from gpu.host import DeviceContext, FuncAttribute, DeviceBuffer
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import external_memory
from gpu.compute.mma import MMAOperandDescriptor
from gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptor,
    UMMAInsDescriptor,
    UMMAKind,
    mma,
    mma_arrive,
)
from gpu.sync import named_barrier
from gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)
from layout.int_tuple import IntTuple, UNKNOWN_VALUE
from layout.layout import Layout
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    RaggedTMA3DTile,
)
from logger import Logger
from memory import bitcast, stack_allocation
from nn.mha_fa3_utils import (
    _apply_mask,
    _get_position,
    get_q_head_idx,
    KVTMATile,
    MHAPosition,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    Pack,
    produce,
    q_tma,
    QTMATile,
)
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_tile_scheduler import (
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    SeqInfo,
    TransientScheduler,
)
from nn.mha_utils import (
    FlashAttentionAlgorithm,
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
)
from nn.softmax import (
    _online_softmax_correction,
    _rowmax_online_softmax,
    _rowsum,
)
from tensor import ManagedTensorSlice

from utils.index import Index
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple

comptime logger = Logger()


struct RegisterAccumulatorDescription:
    var num_mmas: Int
    var frag_size: Int

    @always_inline
    fn __init__(out self, num_mmas: Int, frag_size: Int):
        self.num_mmas = num_mmas
        self.frag_size = frag_size


# consumer_group_size equals
# sm90: 128 (warp group size)
# sm100: num_softmax_threads
struct RegisterAccumulatorLayout[
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    consumer_group_size: Int,
    *,
    frag_simdwidth: Int = 2,
](TrivialRegisterPassable):
    comptime frag_size: Int = Self.MMA_M * Self.MMA_N // Self.consumer_group_size
    comptime num_row_blocks_per_mma = 2
    comptime element_layout: Layout = Layout.row_major(1, Self.frag_simdwidth)
    comptime rows_of_frags_layout: Layout = Layout.row_major(
        Self.num_m_mmas * Self.num_n_mmas, Self.frag_size
    )
    comptime vec_output_layout: Layout = Layout(
        IntTuple(
            IntTuple(Self.num_row_blocks_per_mma, Self.num_m_mmas),
            IntTuple(
                Self.frag_size
                // (Self.num_row_blocks_per_mma * Self.frag_simdwidth),
                Self.num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(Self.frag_simdwidth, Self.frag_size),
            IntTuple(
                Self.num_row_blocks_per_mma * Self.frag_simdwidth,
                Self.num_m_mmas * Self.frag_size,
            ),
        ),
    )

    @staticmethod
    @always_inline
    fn description() -> RegisterAccumulatorDescription:
        comptime assert Self.vec_output_layout.size() > 0, "layout: " + String(
            Self.vec_output_layout
        )

        return RegisterAccumulatorDescription(
            Self.num_m_mmas * Self.num_n_mmas, Self.frag_size
        )


struct MMAOperandOffsetFn[
    dtype: DType,
    BMN: Int,
    BK: Int,
    swizzle: TensorMapSwizzle,
    is_k_major: Bool,
    WMMA_MN: Int,
    WMMA_K: Int,
](TrivialRegisterPassable):
    comptime layout = tile_layout_k_major[
        Self.dtype, Self.BMN, Self.BK, Self.swizzle
    ]() if Self.is_k_major else tile_layout_mn_major[
        Self.dtype, Self.BMN, Self.BK, Self.swizzle
    ]()
    comptime layout_size: Int = Self.layout.size()

    comptime canonical_K = Self.swizzle.bytes() // size_of[
        Self.dtype
    ]() if Self.swizzle != TensorMapSwizzle.SWIZZLE_NONE else Self.BK
    comptime canonical_layout_flat = tile_layout_k_major[
        Self.dtype, Self.BMN, Self.canonical_K, Self.swizzle
    ]() if Self.is_k_major else Self.layout
    comptime canonical_layout = tile_to_descriptor[
        Self.dtype, Self.canonical_layout_flat, Self.is_k_major
    ]()
    comptime canonical_layout_size = Self.canonical_layout.size()

    @always_inline
    fn __init__(out self):
        pass


trait DescriptorPair(TrivialRegisterPassable):
    comptime a_t: MMAOperandDescriptor
    comptime b_t: MMAOperandDescriptor

    @always_inline
    fn get_a(self) -> Self.a_t:
        ...

    @always_inline
    fn get_b(self) -> Self.b_t:
        ...


trait WriteableMMAOperandDescriptor(TrivialRegisterPassable):
    @always_inline
    fn copy_from[
        src_type: DType, src_layout: Layout, src_element_layout: Layout, //
    ](
        self,
        src: LayoutTensor[
            src_type,
            src_layout,
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout=src_element_layout,
        ],
    ):
        ...


trait DescriptorPairTS(TrivialRegisterPassable):
    comptime a_t: WriteableMMAOperandDescriptor
    comptime b_t: MMAOperandDescriptor

    @always_inline
    fn get_a(self) -> Self.a_t:
        ...

    @always_inline
    fn get_b(self) -> Self.b_t:
        ...


fn local_tensor_type[
    dtype: DType, layout: Layout, element_layout: Layout
](
    out dummy_arg: LayoutTensor[
        dtype,
        layout,
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=element_layout,
    ]
):
    dummy_arg = {
        UnsafePointer[
            Scalar[dtype], MutAnyOrigin, address_space = AddressSpace.LOCAL
        ]()
    }


trait AccumulatorTile(TrivialRegisterPassable):
    comptime dtype: DType
    comptime element_layout: Layout
    comptime vec_output_layout: Layout
    comptime rows_of_frags_layout: Layout

    @staticmethod
    @always_inline
    fn _empty_tensor() -> (
        type_of(
            local_tensor_type[
                Self.dtype, Self.vec_output_layout, Self.element_layout
            ]()
        )
    ):
        ...

    @staticmethod
    @always_inline
    fn rows_of_frags(
        src: type_of(Self._empty_tensor()),
        out res: LayoutTensor[
            Self.dtype,
            Self.rows_of_frags_layout,
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ],
    ):
        ...

    @staticmethod
    @always_inline
    fn allocate_register_tile(
        out res: type_of(Self._empty_tensor()),
    ):
        ...

    @always_inline
    fn copy_from(
        self,
        src: type_of(Self._empty_tensor()),
    ):
        ...

    @always_inline
    fn copy_to(
        self,
        dst: type_of(Self._empty_tensor()),
    ):
        ...


struct UMMADescriptorSS[operand_type: DType](
    DescriptorPair, TrivialRegisterPassable
):
    comptime operand_t = Self.operand_type
    comptime a_t = MMASmemDescriptor
    comptime b_t = MMASmemDescriptor

    var a: Self.a_t
    var b: Self.b_t

    @always_inline
    fn __init__(out self, a: Self.a_t, b: Self.b_t):
        self.a = a
        self.b = b

    @always_inline
    fn get_a(self) -> Self.a_t:
        return self.a

    @always_inline
    fn get_b(self) -> Self.b_t:
        return self.b


@always_inline
fn _tmem_offset(dtype_size: Int, *, MMA_N: Int, m_mma: Int, n_mma: Int) -> Int:
    row = 16 * m_mma
    col = (MMA_N * n_mma * dtype_size) // 4
    return (row << 16) + col


@always_inline
fn _tmem_offset[dtype: DType, *, MMA_N: Int, m_mma: Int, n_mma: Int]() -> Int:
    comptime linear = _tmem_offset(
        size_of[dtype](), MMA_N=MMA_N, m_mma=m_mma, n_mma=n_mma
    )
    return linear


struct TMemAccumulator[
    dtype_: DType,
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    num_softmax_threads: Int,
](AccumulatorTile, TrivialRegisterPassable):
    comptime dtype: DType = Self.dtype_
    comptime layout_t = RegisterAccumulatorLayout[
        Self.MMA_M,
        Self.MMA_N,
        Self.num_m_mmas,
        Self.num_n_mmas,
        Self.num_softmax_threads,
    ]
    comptime vec_output_layout = Self.layout_t.vec_output_layout
    comptime element_layout = Self.layout_t.element_layout
    comptime rows_of_frags_layout = Self.layout_t.rows_of_frags_layout
    comptime frag_size = Self.layout_t.frag_size

    var tmem_addr: UInt32

    @always_inline
    fn __init__(out self, tmem_addr: UInt32):
        Self.check_constraints()
        self.tmem_addr = tmem_addr

    @staticmethod
    @always_inline
    fn _empty_tensor() -> (
        type_of(
            local_tensor_type[
                Self.dtype, Self.vec_output_layout, Self.layout_t.element_layout
            ]()
        )
    ):
        Self.check_constraints()
        return local_tensor_type[
            Self.dtype, Self.vec_output_layout, Self.layout_t.element_layout
        ]()

    @always_inline
    fn __getitem__(self, i: UInt32) -> Self:
        return {self.tmem_addr + i * UInt32(Self.MMA_N)}

    @always_inline
    @staticmethod
    fn check_constraints():
        comptime assert Self.vec_output_layout[0].size() > 0, (
            "layout: "
            + String(Self.vec_output_layout)
            + "\nnum_m_mmas = "
            + String(Self.num_m_mmas)
        )
        comptime assert (
            Self.vec_output_layout[1].size() > 0
        ), "layout: " + String(Self.vec_output_layout)
        comptime assert Self.MMA_M > 0, (
            "MMA_M = "
            + String(Self.MMA_M)
            + "\nMMA_N = "
            + String(Self.MMA_N)
            + "\nnum_m_mmas = "
            + String(Self.num_m_mmas)
            + "\nnum_n_mmas = "
            + String(Self.num_n_mmas)
            + "\n"
        )
        comptime assert Self.MMA_N > 0, (
            "MMA_M = "
            + String(Self.MMA_M)
            + "\nMMA_N = "
            + String(Self.MMA_N)
            + "\nnum_m_mmas = "
            + String(Self.num_m_mmas)
            + "\nnum_n_mmas = "
            + String(Self.num_n_mmas)
            + "\n"
        )
        comptime assert Self.num_m_mmas > 0, (
            "MMA_M = "
            + String(Self.MMA_M)
            + "\nMMA_N = "
            + String(Self.MMA_N)
            + "\nnum_m_mmas = "
            + String(Self.num_m_mmas)
            + "\nnum_n_mmas = "
            + String(Self.num_n_mmas)
            + "\n"
        )
        comptime assert Self.num_n_mmas > 0, (
            "MMA_M = "
            + String(Self.MMA_M)
            + "\nMMA_N = "
            + String(Self.MMA_N)
            + "\nm_mma = "
            + String(Self.num_m_mmas)
            + "\nnum_n_mmas = "
            + String(Self.num_n_mmas)
            + "\n"
        )

    @always_inline
    fn offset[m_mma: Int, n_mma: Int](self) -> UInt32:
        Self.check_constraints()

        @parameter
        if m_mma == 0 and n_mma == 0:
            return self.tmem_addr
        else:
            comptime linear = _tmem_offset[
                Self.dtype, MMA_N = Self.MMA_N, m_mma=m_mma, n_mma=n_mma
            ]()

            return self.tmem_addr + UInt32(linear)

    @staticmethod
    @always_inline
    fn rows_of_frags(
        src: type_of(Self._empty_tensor()),
        out res: LayoutTensor[
            Self.dtype,
            Self.rows_of_frags_layout,
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ],
    ):
        Self.check_constraints()
        res = {src.ptr}

    @staticmethod
    @always_inline
    fn allocate_register_tile(
        out res: type_of(Self._empty_tensor()),
    ):
        res = type_of(res).stack_allocation()

    @always_inline
    fn copy_from(
        self,
        src: type_of(Self._empty_tensor()),
    ):
        frags = Self.rows_of_frags(src).vectorize[1, Self.frag_size]()
        comptime dtype_size = size_of[Self.dtype]()
        comptime assert dtype_size == 4
        comptime frag_size_b32 = Self.frag_size * dtype_size // 4
        # 16 x 256b results in repeated 8x4<1x2> pattern
        # each repetition thus fills 8 columns
        # and writes 4 values per thread.
        comptime repeat = frag_size_b32 // 4

        @parameter
        for m_mma in range(Self.num_m_mmas):

            @parameter
            for n_mma in range(Self.num_n_mmas):
                comptime mma_id = n_mma * Self.num_m_mmas + m_mma
                comptime tmem_offset = _tmem_offset(
                    dtype_size,
                    MMA_N=Self.MMA_N,
                    m_mma=m_mma,
                    n_mma=n_mma,
                )
                tmem = self.tmem_addr + UInt32(tmem_offset)
                frag = bitcast[DType.uint32, frag_size_b32](frags[mma_id, 0])
                # 16 x 256b results in repeated 8x4 matrix of <1,2> vector pattern
                tcgen05_st[
                    datapaths=16,  # first dimension of the shape
                    bits=256,  # second dimension of the shape
                    repeat=repeat,
                    pack=False,
                ](tmem, frag)
        tcgen05_store_wait()
        named_barrier[Int32(Self.num_softmax_threads)]()

    @always_inline
    fn copy_to(
        self,
        dst: type_of(Self._empty_tensor()),
    ):
        frags = Self.rows_of_frags(dst).vectorize[1, Self.frag_size]()
        comptime dtype_size = size_of[Self.dtype]()
        comptime assert dtype_size == 4
        comptime frag_size_b32 = (Self.frag_size * dtype_size) // 4
        # 16 x 256b results in repeated 8x4<1x2> pattern
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        comptime repeat = frag_size_b32 // 4
        comptime assert (
            Self.num_m_mmas * Self.num_n_mmas == type_of(frags).layout.size()
        )

        @parameter
        for m_mma in range(Self.num_m_mmas):

            @parameter
            for n_mma in range(Self.num_n_mmas):
                comptime mma_id = n_mma * Self.num_m_mmas + m_mma
                comptime tmem_offset = _tmem_offset(
                    dtype_size,
                    MMA_N=Self.MMA_N,
                    m_mma=m_mma,
                    n_mma=n_mma,
                )
                tmem = self.tmem_addr + UInt32(tmem_offset)
                frags[mma_id, 0] = bitcast[
                    Self.dtype, frags.element_layout.size()
                ](
                    tcgen05_ld[
                        datapaths=16,  # first dimension of the shape
                        bits=256,  # second dimension of the shape
                        repeat=repeat,
                        dtype = DType.uint32,
                        pack=False,
                        width=frag_size_b32,
                    ](tmem)
                )

        tcgen05_load_wait()


struct TMemOperand[
    dtype: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    num_softmax_threads: Int,
](TrivialRegisterPassable, WriteableMMAOperandDescriptor):
    var tmem_addr: UInt32

    comptime reg_layout = RegisterAccumulatorLayout[
        Self.MMA_M,
        Self.MMA_N,
        Self.num_m_mmas,
        Self.num_n_mmas,
        Self.num_softmax_threads,
    ]
    comptime frag_size = Self.reg_layout.frag_size
    comptime vec_output_layout = Self.reg_layout.vec_output_layout
    comptime reg_tile_t = type_of(
        local_tensor_type[
            Self.dtype, Self.vec_output_layout, Self.reg_layout.element_layout
        ]()
    )

    @always_inline
    fn __init__(out self, tmem_addr: UInt32):
        self.tmem_addr = tmem_addr

    @always_inline
    fn offset[m_mma: Int, k_mma: Int](self) -> UInt32:
        comptime assert Self.MMA_M > 0, "MMA_M = " + String(Self.MMA_M) + "\n"
        comptime assert Self.MMA_K > 0, "MMA_K = " + String(Self.MMA_K) + "\n"

        @parameter
        if m_mma == 0 and k_mma == 0:
            return self.tmem_addr
        else:
            comptime linear = _tmem_offset[
                DType.bfloat16, MMA_N = Self.MMA_K, m_mma=m_mma, n_mma=k_mma
            ]()
            return self.tmem_addr + UInt32(linear)

    @always_inline
    fn copy_from[
        src_type: DType,
        src_layout: Layout,
        src_element_layout: Layout,
        //,
    ](
        self,
        src: LayoutTensor[
            src_type,
            src_layout,
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout=src_element_layout,
        ],
    ):
        # src has row of frags layout
        comptime num_frags = src_layout[0].size()
        comptime assert num_frags == Self.num_m_mmas * Self.num_n_mmas
        comptime assert Self.num_n_mmas == 1
        comptime assert Self.frag_size == src_layout[1].size(), (
            "Self.frag_size = "
            + String(Self.frag_size)
            + "\nsrc_layout = "
            + String(src_layout)
        )
        comptime assert src_element_layout.size() == 1
        comptime src_size = size_of[src_type]()
        comptime dst_size = size_of[Self.dtype]()
        comptime frag_size_b32 = (Self.frag_size * dst_size) // 4
        # 16 x 256b results in repeated 8x4<1xN> pattern, where
        comptime N = 32 // (4 * src_size)
        comptime bytes = 4 * dst_size * N
        comptime bits = 8 * bytes
        # e.g., N = 2 for fp32
        #
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        # width == (repeat * bits * datapaths) // (32 * 32)
        comptime repeat = 64 * frag_size_b32 // bits
        # We need to reshape into a row of frags
        comptime assert (
            Self.num_m_mmas * Self.num_n_mmas * Self.frag_size
            == src_layout.size() * src_element_layout.size()
        )
        frags = LayoutTensor[
            src_type,
            Layout(
                IntTuple(Self.num_m_mmas * Self.num_n_mmas),
                IntTuple(Self.frag_size),
            ),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout = Layout.row_major(Self.frag_size),
        ](src.ptr)
        # frags = src.vectorize[1, Self.frag_size]()
        # assume src loaded with 256 bits
        comptime assert src_size >= dst_size
        comptime assert Self.num_m_mmas == 1
        comptime assert Self.num_n_mmas == 1

        @parameter
        for m_mma in range(Self.num_m_mmas):
            tmem = self.offset[m_mma, 0]()
            frag = bitcast[DType.uint32, frag_size_b32](
                frags[m_mma].cast[Self.dtype]()
            )
            # 16 x 256b results in repeated 8x4<1x64b> pattern
            # 256b means 256 // 4 = 64b per thread
            tcgen05_st[
                datapaths=16,  # first dimension of the shape
                bits=bits,  # second dimension of the shape
                repeat=repeat,
                pack=False,
            ](tmem, frag)
        tcgen05_store_wait()
        named_barrier[Int32(Self.num_softmax_threads)]()

    @always_inline
    fn copy_to[
        dst_type: DType,
        dst_layout: Layout,
        dst_element_layout: Layout,
        //,
    ](
        self,
        dst: LayoutTensor[
            dst_type,
            dst_layout,
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout=dst_element_layout,
        ],
    ):
        # src has row of frags layout
        comptime num_frags = dst_layout[0].size()
        comptime assert num_frags == Self.num_m_mmas * Self.num_n_mmas
        comptime assert Self.frag_size == dst_layout[1].size()
        comptime assert dst_element_layout.size() == 1
        comptime assert size_of[dst_type]() == 4
        # 16 x 256b results in repeated 8x4<1x2> pattern
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        comptime src_size = size_of[Self.dtype]()
        comptime dst_size = size_of[dst_type]()
        comptime frag_size_b32 = (Self.frag_size * src_size) // 4
        # 16 x 256b results in repeated 8x4<1xN> pattern, where
        comptime N = 32 // (4 * dst_size)
        comptime bytes = 4 * src_size * N
        comptime bits = 8 * bytes
        # e.g., N = 2 for fp32
        #
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        # width == (repeat * bits * datapaths) // (32 * 32)
        comptime repeat = 64 * frag_size_b32 // bits
        #
        frags = dst.vectorize[1, Self.frag_size]()
        # assume src loaded with 256 bits
        comptime assert src_size <= dst_size
        comptime assert Self.num_n_mmas == 1

        @parameter
        for m_mma in range(Self.num_m_mmas):
            tmem = self.offset[m_mma, 0]()
            # 16 x 256b results in repeated 8x4<1x2> pattern
            frags[m_mma, 0] = rebind[
                SIMD[dst_type, type_of(frags).element_size]
            ](
                bitcast[Self.dtype, Self.frag_size](
                    tcgen05_ld[
                        datapaths=16,  # first dimension of the shape
                        bits=bits,  # second dimension of the shape
                        repeat=repeat,
                        dtype = DType.uint32,
                        pack=False,
                        width=frag_size_b32,
                    ](tmem)
                ).cast[dst_type]()
            )
        tcgen05_load_wait()


struct UMMADescriptorTS[
    operand_type: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    *,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    consumer_group_size: Int,
](DescriptorPairTS, TrivialRegisterPassable):
    comptime operand_t = Self.operand_type
    comptime a_t = TMemOperand[
        Self.operand_type,
        Self.num_m_mmas,
        Self.num_n_mmas,
        Self.MMA_M,
        Self.MMA_N,
        Self.MMA_K,
        Self.consumer_group_size,
    ]
    comptime b_t = MMASmemDescriptor

    var a: Self.a_t
    var b: Self.b_t

    @always_inline
    fn __init__(out self, a: Self.a_t, b: Self.b_t):
        self.a = a
        self.b = b

    @always_inline
    fn get_a(self) -> Self.a_t:
        return self.a

    @always_inline
    fn get_b(self) -> Self.b_t:
        return self.b


struct SM100TensorAccumulatorSS[
    operand_type: DType,
    accum_type: DType,
    MMA_M: Int,
    MMA_N: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    compute_BK: Int,
    num_softmax_threads: Int,
    swizzle_a: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    *,
    transpose_b: Bool = True,
    cta_group: Int = 1,
    pipeline_stages: Int = 1,
](TrivialRegisterPassable):
    comptime operand_t: DType = Self.operand_type
    comptime accum_t: DType = Self.accum_type

    comptime MMA_K = 16

    comptime num_m_mmas = Self.BM // Self.MMA_M
    comptime num_n_mmas = Self.BN // Self.MMA_N
    comptime num_k_mmas = Self.compute_BK // Self.MMA_K

    comptime num_m_blocks_per_warp = 2 * Self.BM // Self.num_softmax_threads

    comptime smem_ptr_t = UnsafePointer[
        Scalar[Self.operand_t],
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ]

    comptime a_offset = MMAOperandOffsetFn[
        Self.operand_t,
        Self.BM,
        Self.BK,
        Self.swizzle_a,
        True,
        Self.MMA_M,
        Self.MMA_K,
    ]()
    comptime b_offset = MMAOperandOffsetFn[
        Self.operand_t,
        Self.BN,
        Self.BK,
        Self.swizzle_b,
        Self.transpose_b,
        Self.MMA_N,
        Self.MMA_K,
    ]()

    comptime idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype = DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b = Self.transpose_b,
    ]()

    comptime ab_t: DescriptorPair = UMMADescriptorSS[Self.operand_t]
    comptime a_t: MMAOperandDescriptor = Self.ab_t.a_t
    comptime b_t: MMAOperandDescriptor = Self.ab_t.b_t
    comptime c_t: AccumulatorTile = TMemAccumulator[
        Self.accum_t,
        Self.BM // Self.num_m_blocks_per_warp,
        Self.MMA_N,
        Self.num_m_blocks_per_warp,
        Self.num_n_mmas,
        Self.num_softmax_threads,
    ]

    var mbar: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space = AddressSpace.SHARED
    ]
    var pipeline: PipelineState[Self.pipeline_stages]

    @always_inline
    @staticmethod
    fn check_constraints():
        comptime assert (Self.BM % Self.MMA_M) == 0, (
            "BM, MMA_M = " + String(Self.BM) + ", " + String(Self.MMA_M)
        )
        comptime assert ((Self.BN % Self.MMA_N) == 0) and (
            Self.num_n_mmas > 0
        ), ("BN, MMA_N = " + String(Self.BN) + ", " + String(Self.MMA_N))
        comptime assert ((Self.compute_BK % Self.MMA_K) == 0) and (
            Self.num_k_mmas > 0
        ), (
            "compute_BK, MMA_K = "
            + String(Self.compute_BK)
            + ", "
            + String(Self.MMA_K)
        )

    @always_inline
    fn __init__(
        out self,
        smem: UnsafePointer[
            SharedMemBarrier, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
    ):
        Self.check_constraints()
        self.mbar = smem
        self.pipeline = {}

    @always_inline
    fn init(self):
        @parameter
        for i in range(Self.pipeline_stages):
            self.mbar[i].init()
            self.mbar[i + Self.pipeline_stages].init(
                Int32(Self.num_softmax_threads)
            )

    @staticmethod
    @always_inline
    fn mma_descriptors[
        dtype_a: DType, dtype_b: DType
    ](
        p_a: UnsafePointer[
            Scalar[dtype_a], MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
        p_b: UnsafePointer[
            Scalar[dtype_b], MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
    ) -> Self.ab_t:
        Self.check_constraints()
        comptime a_canonical_layout = Self.a_offset.canonical_layout
        comptime a_type = Self.operand_t
        comptime aSBO = a_canonical_layout[0].stride[1].value() * size_of[
            a_type
        ]()
        comptime aLBO = a_canonical_layout[1].stride[1].value() * size_of[
            a_type
        ]()
        adesc_base = MMASmemDescriptor.create[aSBO, aLBO, Self.swizzle_a](p_a)

        comptime b_canonical_layout = Self.b_offset.canonical_layout
        comptime b_type = Self.operand_t
        comptime b_stride01 = b_canonical_layout[0].stride[1].value()
        comptime b_stride11 = b_canonical_layout[1].stride[1].value()
        comptime bSBO = (
            b_stride01 if Self.transpose_b else b_stride11
        ) * size_of[b_type]()
        comptime bLBO = (
            b_stride11 if Self.transpose_b else b_stride01
        ) * size_of[b_type]()
        bdesc_base = MMASmemDescriptor.create[bSBO, bLBO, Self.swizzle_b](p_b)

        return Self.ab_t(adesc_base, bdesc_base)

    @always_inline
    fn mma(
        mut self,
        a: Self.a_t,
        b: Self.b_t,
        c_base: Self.c_t,
        scale_c: UInt32,
    ):
        c = c_base[self.pipeline.index()]

        @parameter
        for k_mma in range(Self.num_k_mmas):

            @parameter
            for m_mma in range(Self.num_m_mmas):
                comptime a_offset = Self.a_offset.layout(
                    IntTuple(Self.MMA_M * m_mma, Self.MMA_K * k_mma)
                )
                comptime a_offset_bytes = a_offset * size_of[Self.operand_t]()
                a_desc = a + a_offset_bytes

                @parameter
                for n_mma in range(Self.num_n_mmas):
                    c_tmem = c.offset[m_mma, n_mma]()

                    comptime b_offset = Self.b_offset.layout(
                        IntTuple(Self.MMA_N * n_mma, Self.MMA_K * k_mma)
                    ) * size_of[Self.operand_t]()
                    b_desc = b + b_offset

                    @parameter
                    if k_mma == 0:
                        mma[Self.cta_group](
                            a_desc,
                            b_desc,
                            c_tmem,
                            Self.idesc,
                            scale_c,
                        )
                    else:
                        mma[Self.cta_group, c_scale=1](
                            a_desc, b_desc, c_tmem, Self.idesc
                        )

        mma_arrive(self.mbar + self.pipeline.index())
        self.pipeline.step()

    # the mma thread
    # loop:
    #   wait_for_tmem() # self.mbar[Stages + index()].wait(phase())
    #   mma()           # self.mbar[index()].arrive(), step()
    #
    # the softmax thread
    #
    # tmem_arrive_init() # for i in range(Stages): self.mbar[Stages + i].arrive()
    #
    # loop:
    #   wait_for_mma()   # self.mbar[index()].wait(phase())
    #   use accumulator
    #   tmem_arrive()    # self.mbar[Stages + index()].arrive(), step()
    @always_inline
    fn wait_for_tmem(self):
        """
        Wait for the accumulator tmem to finish being read.
        """
        self.mbar[UInt32(Self.pipeline_stages) + self.pipeline.index()].wait(
            self.pipeline.phase()
        )

    @always_inline
    fn wait_for_mma(self, c_base: Self.c_t) -> Self.c_t:
        """
        Wait for the accumulator tmem to finish being read.
        """
        var idx: UInt32 = self.pipeline.index()
        self.mbar[idx].wait(self.pipeline.phase())
        return c_base[idx]

    @always_inline
    fn tmem_arrive_init(self):
        @parameter
        for i in range(Self.pipeline_stages):
            _ = self.mbar[Self.pipeline_stages + i].arrive()

    @always_inline
    fn tmem_arrive(mut self):
        """
        Indicate that the accumulator is ready to be updated.
        """
        _ = self.mbar[
            UInt32(Self.pipeline_stages) + self.pipeline.index()
        ].arrive()
        self.pipeline.step()


struct SM100TensorAccumulatorTS[
    operand_type: DType,
    accum_type: DType,
    MMA_M: Int,
    MMA_N: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    num_softmax_threads: Int,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    cta_group: Int = 1,
](TrivialRegisterPassable):
    comptime operand_t: DType = Self.operand_type
    comptime accum_t: DType = Self.accum_type

    comptime MMA_K = 16
    comptime smem_ptr_t = UnsafePointer[
        Scalar[Self.operand_t],
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ]

    comptime num_m_mmas = Self.BM // Self.MMA_M
    comptime num_n_mmas = Self.BN // Self.MMA_N
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime c_frag_size = Self.MMA_M * Self.MMA_N // Self.num_softmax_threads
    comptime a_frag_size = Self.MMA_M * Self.MMA_K // Self.num_softmax_threads
    comptime num_m_blocks_per_warp = 2 * Self.BM // Self.num_softmax_threads
    comptime ab_t: DescriptorPairTS = UMMADescriptorTS[
        Self.operand_t,
        Self.num_m_blocks_per_warp,
        Self.num_n_mmas,
        MMA_M = Self.BM // Self.num_m_blocks_per_warp,
        MMA_N = Self.BK,
        MMA_K = Self.MMA_K,
        consumer_group_size = Self.num_softmax_threads,
    ]
    comptime a_t: WriteableMMAOperandDescriptor = Self.ab_t.a_t
    comptime b_t: MMAOperandDescriptor = Self.ab_t.b_t

    comptime b_offset = MMAOperandOffsetFn[
        Self.operand_t,
        Self.BN,
        Self.BK,
        Self.swizzle_b,
        Self.transpose_b,
        Self.MMA_N,
        Self.MMA_K,
    ]()
    comptime c_t: AccumulatorTile = TMemAccumulator[
        Self.accum_t,
        Self.BM // Self.num_m_blocks_per_warp,
        Self.MMA_N,
        Self.num_m_blocks_per_warp,
        Self.num_n_mmas,
        Self.num_softmax_threads,
    ]

    comptime idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype = DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b = Self.transpose_b,
    ]()

    var mbar: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space = AddressSpace.SHARED
    ]
    var phase: UInt32

    @staticmethod
    @always_inline
    fn check_constraints():
        comptime assert (Self.BM % Self.MMA_M) == 0, (
            "BM, MMA_M = " + String(Self.BM) + ", " + String(Self.MMA_M)
        )
        comptime assert ((Self.BN % Self.MMA_N) == 0) and (
            Self.num_n_mmas > 0
        ), ("BN, MMA_N = " + String(Self.BN) + ", " + String(Self.MMA_N))
        comptime assert ((Self.BK % Self.MMA_K) == 0) and (
            Self.num_k_mmas > 0
        ), ("BK, MMA_K = " + String(Self.BK) + ", " + String(Self.MMA_K))

    @always_inline
    fn __init__(
        out self,
        smem: UnsafePointer[
            SharedMemBarrier, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
    ):
        Self.check_constraints()
        self.mbar = smem
        self.phase = 0

    @always_inline
    fn init(self):
        self.mbar[0].init()
        self.mbar[1].init(Int32(Self.num_softmax_threads))

    @staticmethod
    @always_inline
    fn a_mma_descriptor(a_tmem: UInt32) -> Self.ab_t.a_t:
        Self.check_constraints()
        return Self.ab_t.a_t(a_tmem)

    @staticmethod
    @always_inline
    fn b_mma_descriptor[
        dtype_b: DType
    ](
        p_b: UnsafePointer[
            Scalar[dtype_b], MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
    ) -> Self.ab_t.b_t:
        Self.check_constraints()
        comptime b_canonical_layout = Self.b_offset.canonical_layout
        comptime b_type = Self.operand_t
        comptime b_stride01 = b_canonical_layout[0].stride[1].value()
        comptime b_stride11 = b_canonical_layout[1].stride[1].value()
        comptime bSBO = (
            b_stride01 if Self.transpose_b else b_stride11
        ) * size_of[b_type]()
        comptime bLBO = (
            b_stride11 if Self.transpose_b else b_stride01
        ) * size_of[b_type]()

        return MMASmemDescriptor.create[bSBO, bLBO, Self.swizzle_b](p_b)

    @always_inline
    fn mma(
        self,
        a: Self.a_t,
        b: Self.b_t,
        c: Self.c_t,
        c_scale: UInt32,
    ):
        @parameter
        for k_mma in range(Self.num_k_mmas):

            @parameter
            for m_mma in range(Self.num_m_mmas):
                a_tmem = a.offset[m_mma=m_mma, k_mma=k_mma]()

                @parameter
                for n_mma in range(Self.num_n_mmas):
                    c_tmem = c.offset[m_mma=m_mma, n_mma=n_mma]()
                    comptime b_offset = Self.b_offset.layout(
                        IntTuple(Self.MMA_N * n_mma, Self.MMA_K * k_mma)
                    ) * size_of[Self.operand_t]()
                    b_desc = b + b_offset

                    @parameter
                    if k_mma == 0:
                        mma[Self.cta_group](
                            a_tmem,
                            b_desc,
                            c_tmem,
                            Self.idesc,
                            c_scale,
                        )
                    else:
                        mma[Self.cta_group, c_scale=1](
                            a_tmem, b_desc, c_tmem, Self.idesc
                        )
        mma_arrive(self.mbar)

    # the mma thread
    # loop:
    #   wait_for_tmem()   # self.mbar[1].wait(self.phase), self.phase ^= 1
    #   mma()             # self.mbar[0].arrive()
    #
    # the softmax thread
    # tmem_arrive()       # self.mbar[1].arrive()
    #
    # loop:
    #   wait_for_mma()    # self.mbar[0].wait(self.phase), self.phase ^= 1
    #   scale output, write P
    #   tmem_arrive()     # self.mbar[1].arrive()
    @always_inline
    fn wait(mut self, idx: UInt32):
        # update the phase before waiting
        var old_phase: UInt32 = self.phase
        self.phase = old_phase ^ 1
        self.mbar[idx].wait(old_phase)

    @always_inline
    fn wait_for_mma(mut self):
        """
        Wait for the mma to be complete.
        """
        self.wait(0)

    @always_inline
    fn wait_for_tmem(mut self):
        """
        Wait for the `output` and `A` tmem to be ready.
        """
        self.wait(1)

    @always_inline
    fn tmem_arrive(self):
        """
        Indicate that the accumulator and the tensor memory arguments
        are ready for the MMA to begin.
        """
        _ = self.mbar[1].arrive()


@always_inline
fn mha_sm100_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    output_type: DType,
    MaxPromptLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    //,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    sink: Bool,
    _is_cache_length_accurate: Bool,
](
    output: DeviceBuffer[output_type],
    q_arg: DeviceBuffer[q_type],
    k: KVType,
    v: KVType,
    num_rows_q: Int,
    mask: MaskType,
    score_mod: ScoreModType,
    valid_length: DeviceBuffer[DType.uint32],
    max_prompt_len_arg: MaxPromptLenType,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ],
    batch_size_arg: Int,
    partition: PartitionType,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin]
    ],
) raises:
    comptime decoding: Bool = MaxPromptLenType.static_value.or_else(0) == 1
    comptime new_config = MHAConfig[config.dtype](
        config.num_heads,
        config.depth,
        num_queries_per_block=Optional[UInt](64),
        num_keys_per_block=Optional[UInt](config.num_keys_per_block),
        BK=Optional[UInt](config.BK),
    ) if decoding else config
    comptime BM = new_config.block_m()
    comptime BK = new_config.padded_depth
    comptime assert BM % 64 == 0, "SM90 requires BM%64==0, but BM==" + String(
        BM
    )
    comptime assert (
        BK % 64 == 0
    ), "B200 requires BK%64 as it uses 128B swizzles, but BK==" + String(BK)
    comptime BN = new_config.block_n()
    # add the number of producer threads (i.e. 1 WARP_GROUP_SIZE)
    comptime num_threads = new_config.num_threads[True]()
    comptime assert num_threads % 128 == 0, "num_threads = " + String(
        num_threads
    )
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match for FA3."
    q = rebind[UnsafePointer[Scalar[KVType.dtype], MutAnyOrigin]](q_arg)

    # Persistent kernels not currently supported with partitioning
    # This doesn't seem useful: we partition to make SMs more busy,
    # implying we don't have enough to make them persistent.
    # This also requires some tricky control flow handling to support,
    # which we haven't added yet.
    comptime assert new_config.algorithm == FlashAttentionAlgorithm(3)

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)
    var max_prompt_len: UInt32 = max_prompt_len_arg.as_uint32()
    var max_num_prompt_tiles: UInt32 = ceildiv(max_prompt_len, UInt32(BM))

    comptime num_scheduler_heads = config.num_heads // UInt(
        group
    ) if decoding else config.num_heads
    # if decoding,
    comptime scheduler_tile_shape = 1 if decoding else BM
    comptime swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    q_tma_op = rebind[
        QTMATile[
            KVType.dtype,
            swizzle_mode,
            BM = Int(new_config.block_m()),
            depth = Int(new_config.depth),
            group=group,
            decoding = _is_decoding[MaxPromptLenType](),
        ]
    ](
        q_tma[
            swizzle_mode,
            BM = Int(BM),
            depth = Int(new_config.depth),
            q_num_heads = Int(new_config.num_heads),
            group=group,
            decoding=decoding,
        ](ctx, q, num_rows_q)
    )
    k_tma_op = k.create_tma_tile[
        swizzle_mode,
        BN = Int(new_config.block_n()),
        depth = Int(new_config.depth),
        BK = Int(new_config.padded_depth),
    ](ctx)
    v_tma_op = v.create_tma_tile[
        swizzle_mode,
        BN = Int(new_config.block_n()),
        depth = Int(new_config.depth),
        BK = Int(new_config.padded_depth),
    ](ctx)

    comptime SchedulerType = TransientScheduler[
        UInt32(scheduler_tile_shape), UInt32(num_scheduler_heads)
    ]
    var scheduler: SchedulerType = SchedulerType()

    @parameter
    if sink:
        comptime SinkType = NonNullPointer[KVType.dtype]
        var sink_ptr: SinkType = {
            rebind[UnsafePointer[Scalar[KVType.dtype], ImmutAnyOrigin]](
                sink_weights.value().ptr
            )
        }
        _mha_sm100_kv_input_row_offset_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVType,
            output_type=output_type,
            MaxSeqLenType=MaxPromptLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=new_config,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            output,
            k,
            scale,
            batch_size,
            max_prompt_len_arg,
            max_cache_valid_length,
            valid_length,
            kv_input_row_offsets,
            sink_ptr,
            partition,
            mask,
            score_mod,
            ctx,
        )
    else:
        comptime SinkType = NullPointer[KVType.dtype]
        comptime sink_ptr: SinkType = {}
        _mha_sm100_kv_input_row_offset_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVType,
            output_type=output_type,
            MaxSeqLenType=MaxPromptLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=new_config,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            output,
            k,
            scale,
            batch_size,
            max_prompt_len_arg,
            max_cache_valid_length,
            valid_length,
            kv_input_row_offsets,
            sink_ptr,
            partition,
            mask,
            score_mod,
            ctx,
        )


@always_inline
fn _mha_sm100_kv_input_row_offset_dispatch[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    SinkType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = Int(config.block_m()),
        depth = Int(config.depth),
        group=group,
        decoding = _is_decoding[MaxSeqLenType](),
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = Int(config.block_n()),
        BK = Int(config.padded_depth),
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = Int(config.block_n()),
        BK = Int(config.padded_depth),
    ],
    o_ptr_arg: DeviceBuffer[output_type],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: DeviceBuffer[DType.uint32],
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ],
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    score_mod: ScoreModType,
    ctx: DeviceContext,
) raises:
    comptime KVRowOffsetsNonNull = NonNullPointer[DType.uint32]
    comptime KVRowOffsetsNull = NullPointer[DType.uint32]
    if kv_input_row_offsets:
        var kv_row_offsets: KVRowOffsetsNonNull = {
            kv_input_row_offsets.value().ptr
        }
        _mha_sm100_valid_length_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=config,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            KVRowOffsetsType=KVRowOffsetsNonNull,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_length,
            kv_row_offsets,
            sink_weights,
            partition,
            mask,
            score_mod,
            ctx,
        )
    else:
        var kv_row_offsets: KVRowOffsetsNull = {}
        _mha_sm100_valid_length_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=config,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            KVRowOffsetsType=KVRowOffsetsNull,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_length,
            kv_row_offsets,
            sink_weights,
            partition,
            mask,
            score_mod,
            ctx,
        )


@always_inline
fn _mha_sm100_valid_length_dispatch[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = Int(config.block_m()),
        depth = Int(config.depth),
        group=group,
        decoding = _is_decoding[MaxSeqLenType](),
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = Int(config.block_n()),
        BK = Int(config.padded_depth),
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = Int(config.block_n()),
        BK = Int(config.padded_depth),
    ],
    o_ptr_arg: DeviceBuffer[output_type],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: DeviceBuffer[DType.uint32],
    kv_input_row_offsets: KVRowOffsetsType,
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    score_mod: ScoreModType,
    ctx: DeviceContext,
) raises:
    @parameter
    if ragged:
        comptime ValidLengthType = NonNullPointer[DType.uint32]
        var valid_len: ValidLengthType = {valid_length}
        _mha_sm100_enqueue[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=config,
            group=group,
            use_score_mod=use_score_mod,
            SinkType=SinkType,
            ValidLengthType=ValidLengthType,
            KVRowOffsetsType=KVRowOffsetsType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_len,
            kv_input_row_offsets,
            sink_weights,
            partition,
            mask,
            score_mod,
            ctx,
        )
    else:
        comptime ValidLengthType = NullPointer[DType.uint32]
        var valid_len: ValidLengthType = {}
        _mha_sm100_enqueue[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=config,
            group=group,
            use_score_mod=use_score_mod,
            SinkType=SinkType,
            ValidLengthType=ValidLengthType,
            KVRowOffsetsType=KVRowOffsetsType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_len,
            kv_input_row_offsets,
            sink_weights,
            partition,
            mask,
            score_mod,
            ctx,
        )


@always_inline
fn _mha_sm100_enqueue[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = Int(config.block_m()),
        depth = Int(config.depth),
        group=group,
        decoding = _is_decoding[MaxSeqLenType](),
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = Int(config.block_n()),
        BK = Int(config.padded_depth),
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = Int(config.block_n()),
        BK = Int(config.padded_depth),
    ],
    o_ptr_arg: DeviceBuffer[output_type],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: ValidLengthType,  # OptionalPointer[DType.uint32]
    kv_input_row_offsets: KVRowOffsetsType,  # OptionalPointer[DType.uint32],
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    score_mod: ScoreModType,
    ctx: DeviceContext,
) raises:
    # the pack contains all possibly 0-sized objects
    comptime kernel_sm100 = _mha_sm100[
        KVLUTType,
        output_type,
        MaskType,
        ScoreModType,
        SchedulerType,
        config,
        group,
        use_score_mod,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        _is_cache_length_accurate,
        MaxSeqLenType,
        PartitionType,
        swizzle_mode,
    ]
    comptime PackType = Pack[
        MaskType,
        ScoreModType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxSeqLenType,
        PartitionType,
    ]
    var pack: PackType = {
        mask,
        score_mod,
        scheduler,
        valid_length,
        sink_weights,
        kv_input_row_offsets,
        max_seq_len,
        partition,
    }

    var max_num_prompt_tiles: UInt32 = ceildiv(
        max_seq_len.as_uint32(), UInt32(config.block_m())
    )
    var block_x: UInt32 = max_num_prompt_tiles * partition.num_partitions()

    comptime max_tmem_cols = 512
    comptime BN = config.block_n()
    comptime num_s = (
        max_tmem_cols - (Int(BN) // 2) - Int(config.padded_depth)
    ) // Int(BN)
    # we add smem use for SharedMemBarrier synchronization
    # 2*8 for mma mbars
    comptime extra_B200_smem = (2 * num_s + 3) * 8
    comptime smem_use = config.shared_mem_bytes[True, sm_90=True]() + UInt(
        extra_B200_smem
    )
    comptime num_threads = config.num_threads[True]()
    comptime decoding = _is_decoding[MaxSeqLenType]()
    logger.info("------ Dispatching to SM100 FMHA-1Q ------")
    logger.info(
        "QKV Type: ",
        KVLUTType.dtype,
        "Depth:",
        config.depth,
        "Number of Q // KV Heads:",
        config.num_heads,
        "//",
        config.num_heads // UInt(group),
        "Batch Size:",
        batch_size,
        "Num Partitions:" if decoding else "Max Num Prompt Tiles:",
        partition.num_partitions() if decoding else max_num_prompt_tiles,
    )
    ctx.enqueue_function[kernel_sm100, kernel_sm100](
        q_tma_op,
        k_tma_op,
        v_tma_op,
        o_ptr_arg,
        kv_lut,
        scale,
        batch_size,
        num_keys_arg,
        pack,
        grid_dim=SchedulerType.grid_dim(batch_size, block_x),
        block_dim=(Int(num_threads), 1, 1),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


@__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads[True]())
    )
)
fn _mha_sm100[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = Int(config.block_m()),
        depth = Int(config.depth),
        group=group,
        decoding = _is_decoding[MaxSeqLenType](),
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = Int(config.block_n()),
        BK = Int(config.padded_depth),
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = Int(config.block_n()),
        BK = Int(config.padded_depth),
    ],
    o_ptr_arg: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    num_keys_arg: UInt32,
    pack: Pack[
        MaskType,
        ScoreModType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxSeqLenType,
        PartitionType,
    ],
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    comptime kv_type = KVLUTType.dtype
    comptime assert kv_type == config.dtype
    comptime decoding: Bool = _is_decoding[MaxSeqLenType]()

    comptime simd_size: Int = simd_width_of[kv_type]()

    comptime num_softmax_threads: Int = Int(config.num_consumer_threads())
    comptime num_softmax_warps = num_softmax_threads // 32

    comptime cta_group = 1
    comptime BM: Int = Int(config.block_m())
    comptime BN: Int = Int(config.block_n())
    comptime BK: Int = Int(config.padded_depth)
    comptime depth: Int = Int(config.depth)
    comptime padded_depth: Int = Int(config.padded_depth)
    # comptime mma_shape = Index(64, depth, 16)
    # comptime mma_shape = Index(128 if (BM % 128) == 0 else 64, depth, 16)
    # MMA_M here is defined as per-warp
    # comptime MMA_M = 64
    comptime MMA_M: Int = 128 if (BM % 128) == 0 else 64
    comptime MMA_N0: Int = BN
    comptime MMA_N1: Int = Int(config.padded_depth)
    comptime MMA_K: Int = 16
    # comptime WM = BM // num_softmax_warps
    # comptime WN = BN
    # comptime num_m_mmas = BM // MMA_M  # WM // MMA_M
    # mmas are now handled separately from in-register processing
    # in-register processing is divided up by warps, mmas are not
    comptime num_row_fragments = num_softmax_threads // 128
    comptime assert (32 % num_row_fragments) == 0
    comptime row_fragment_size = min(32 // num_row_fragments, BM // 4)
    comptime assert num_row_fragments * row_fragment_size <= 32
    comptime WM = row_fragment_size
    # if we have BM = 128, then we have
    # a 16x(BN//8) grid of 8x4<1x2>
    # this gives us 16 blocks to partition among rows.
    # Because we can load a minimum of 16 lanes at a time,
    # we prefer at least 2x blocks per warp, meaning we
    # can divide up to 8 ways.
    comptime num_m_blocks_per_warp = BM // (16 * num_softmax_warps)
    # before we had num_m_mmas * MMA_M = BM
    # now, we have num_m_blocks_per_warp * 16*num_softmax_warps == BM
    # num_m_blocks_per_warp is like `num_m_mmas`, but for non-mma consumers.
    comptime assert num_m_blocks_per_warp * 16 == WM
    #
    # The following constraint is effectively equivalent to
    # BM == 128 or BM == 64
    # If 32 // num_row_fragments is smaller, we have
    # 32*128*num_softmax_warps // num_softmax_threads == BM
    # 128*num_softmax_threads // num_softmax_threads == BM
    # 128 == BM
    # Or if BM // 4 is smaller, we have
    # BM // 4 * num_softmax_warps == BM
    # num_softmax_threads*BM // (4 * 32) == BM
    # num_softmax_threads == 128
    # 32*128 // BM == num_softmax_threads
    comptime assert WM * num_softmax_warps == BM
    # The above should also be true because:
    # num_softmax_warps = BM // (16 * num_m_blocks_per_warp)
    # -> BM // WM = BM // (16 * num_m_blocks_per_warp)
    # -> WM = (16 * num_m_blocks_per_warp)
    comptime num_m_mmas = 1
    comptime num_n_mmas = 1
    comptime num_k_mmas = BK // MMA_K
    # comptime num_warps_m = BM // WM  # 4 * num_softmax
    # comptime num_warps_n = BN // WN  # 1
    comptime num_heads: Int = Int(config.num_heads)
    # num_softmax_threads ignores the producers
    # actual number of threads is num_softmax_threads + 128
    comptime pipeline_stages = Int(config.num_pipeline_stages)
    var tid = UInt32(thread_idx.x)
    # warp group idx concept is still useful for sm100
    # because sets of 4 warps access tmem;
    # warp group idx gives index into sets of 16 lanes
    var warp_group_idx: UInt32 = warp.broadcast(tid // 128)
    # warp_group_tid = tid % 128
    comptime accum_type = get_accum_type[kv_type]()
    comptime assert (
        accum_type.is_floating_point()
    ), "accum_type must be floating point"
    comptime max_tmem_cols = 512
    comptime num_s = (max_tmem_cols - (MMA_N0 // 2) - MMA_N1) // MMA_N0
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        kv_type,
        accum_type,
        MMA_M=MMA_M,  # 128
        MMA_N=MMA_N0,  # BN
        BM=BM,  # 128
        BN=BN,  # BN
        BK=BK,  # depth
        compute_BK = align_up(depth, 16),
        num_softmax_threads=num_softmax_threads,
        swizzle_a=swizzle_mode,
        swizzle_b=swizzle_mode,
        transpose_b=True,
        pipeline_stages=num_s,
    ]
    # Second WGMMA is a
    # BM x BN tile of p_frag @ BN x depth tile of V
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        kv_type,
        accum_type,
        MMA_M=MMA_M,
        MMA_N=MMA_N1,  # depth
        BM=BM,
        BN=MMA_N1,  # depth
        BK=BN,  # BN
        num_softmax_threads=num_softmax_threads,
        swizzle_b=swizzle_mode,
        transpose_b=False,
    ]
    mask = pack.mask
    score_mod = pack.score_mod
    scheduler = pack.scheduler
    valid_length = pack.valid_length
    sink_weights = pack.sink_weights
    kv_input_row_offsets = pack.kv_input_row_offsets
    max_seq_len = pack.max_seq_len
    partition = pack.partition

    # var warp_x: UInt32 = warp_id % num_warps_n

    # first umma is BM x BK @ BK x BN
    # The entire query block (BM x depth) is tiled in shared memory.
    comptime q_smem_size = BM * padded_depth
    q_smem = external_memory[
        Scalar[kv_type],
        address_space = AddressSpace.SHARED,
        alignment=128,
        name="mha_dynamic_shared_memory",
    ]()

    # We have `num_pipeline_stages` instances of each
    comptime kv_smem_size = config.kv_smem_size(True)
    kv_smem = q_smem + q_smem_size

    # var head_idx: UInt32 = block_idx.y
    # var q_tile_idx: UInt32 = block_idx.x

    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    # p_frag_size is 2 * WM//8 * MMA_N//8
    # that is, we have a (WM//8) x (MMA_N//8) grid of 8x4<1x2> blocks
    # Each such block has 2 elements.
    comptime p_frag_size = BM * MMA_N0 // (
        num_softmax_threads * num_m_blocks_per_warp
    )
    comptime o_frag_size = BM * MMA_N1 // (
        num_softmax_threads * num_m_blocks_per_warp
    )
    comptime assert p_frag_size == 2 * (WM // 8) * (MMA_N0 // 8)
    comptime assert o_frag_size == 2 * (WM // 8) * (MMA_N1 // 8)
    comptime frag_simdwidth = 2
    comptime assert (
        BN * num_k_mmas * BM * MMA_K
        == BK
        * num_n_mmas
        * p_frag_size
        * num_softmax_threads
        * num_m_blocks_per_warp
    )

    comptime num_row_blocks_per_mma = 2
    # a umma.m64n32k16 `D` fragment looks like
    #
    # 0,1  4,5   8, 9  12,13
    # 2,3  6,7  10,11  14,15
    #
    # Each row/column has `p_frag_simdwidth`-sized vectors
    # (e.g. `4,5` is of size 2 = p_frag_simdwidth)
    # We have `num_row_blocks_per_mma` rows.
    # The total number of elements (16) equals `p_frag_size`.
    # The number of columns equals
    # `p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth)`
    #
    # This gives us the layout:
    #
    # Note the ordering of strides:
    # ((1, 3), (0, 2, 4))
    # comptime output_layout = Layout(
    #     IntTuple(
    #         IntTuple(num_row_blocks_per_mma, num_m_blocks_per_warp),
    #         IntTuple(
    #             p_frag_simdwidth,
    #             p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth),
    #             num_n_mmas,
    #         ),
    #     ),
    #     IntTuple(
    #         IntTuple(p_frag_simdwidth, p_frag_size),
    #         IntTuple(1, 2 * p_frag_simdwidth, num_m_blocks_per_warp * p_frag_size),
    #     ),
    # )
    # Vectorizing the layout:
    comptime element_layout = Layout.row_major(1, frag_simdwidth)
    comptime vec_output_row_shape = IntTuple(num_row_blocks_per_mma, num_m_mmas)
    comptime p_vec_output_layout = Layout(
        IntTuple(
            vec_output_row_shape,
            IntTuple(
                p_frag_size // (num_row_blocks_per_mma * frag_simdwidth),
                num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(frag_simdwidth, p_frag_size),
            IntTuple(
                num_row_blocks_per_mma * frag_simdwidth,
                num_m_mmas * p_frag_size,
            ),
        ),
    )
    comptime o_vec_output_layout = Layout(
        IntTuple(
            vec_output_row_shape,
            IntTuple(
                o_frag_size // (num_row_blocks_per_mma * frag_simdwidth),
                num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(frag_simdwidth, o_frag_size),
            IntTuple(
                num_row_blocks_per_mma * frag_simdwidth,
                num_m_mmas * o_frag_size,
            ),
        ),
    )
    comptime num_rows_per_warp = p_vec_output_layout[0].size()
    comptime num_cols_p = p_vec_output_layout[1].size()
    comptime num_cols_output = o_vec_output_layout[1].size()

    # Rowwise max and sum for online softmax
    comptime accum_simd_width = simd_width_of[accum_type]()
    comptime row_alignment = align_of[SIMD[accum_type, accum_simd_width]]()
    # Account for group query.
    comptime kv_num_heads = num_heads // group

    # var lane_predicate = elect_one_sync() # not needed with async_copy

    comptime mma_thread_layout = Layout.row_major(8, 4)
    comptime ragged = not ValidLengthType.is_null

    # Handle sink_weights
    var sink_weights_ptr = UnsafePointer[Scalar[kv_type], ImmutAnyOrigin]()

    @parameter
    if not SinkType.is_null:
        sink_weights_ptr = rebind[
            UnsafePointer[Scalar[kv_type], ImmutAnyOrigin]
        ](sink_weights.value())

    # actually 16 byte alignment
    produced_mbar_kv = (kv_smem + kv_smem_size).bitcast[SharedMemBarrier]()
    consumed_mbar_kv = produced_mbar_kv + pipeline_stages  # 16
    mma_mbar = consumed_mbar_kv + pipeline_stages  # 16
    umma_0 = UMMA0Type(mma_mbar)  # needs num_s
    umma_1 = UMMA1Type(mma_mbar + 2 * num_s)
    ptr_tmem_addr = (mma_mbar + 2 * num_s + 2).bitcast[UInt32]()  # 8

    comptime USE_TMA = True
    # https://github.com/Dao-AILab/flash-attention/blob/3b5047d2ce742848f45d44b143d511f211eba2d2/hopper/flash_fwd_kernel_sm90.h#L81-L82
    # comptime num_producer_regs = 56 if num_softmax_warps == 4 else (
    #     (24 if USE_TMA else 56) if num_softmax_warps == 8 else 32
    # )
    # comptime num_softmax_regs = 256 if num_softmax_warps == 4 else (
    #     (240 if USE_TMA else 224) if num_softmax_warps == 8 else 160
    # )
    comptime num_producer_regs = 56
    comptime num_softmax_regs = 224

    # constructing calls barrier() if static
    var tile_summary = MHATileSummary[ValidLengthType](
        batch_size,
        ceildiv(max_seq_len.as_uint32(), UInt32(BM))
        * partition.num_partitions(),
        valid_length,
        max_seq_len.as_uint32(),
    )
    var state: MHATileState = scheduler.initial_state(
        ptr_tmem_addr + 2, tile_summary
    )

    # The persistent kernels limit the grid size.
    # initial_seq_info = scheduler.unsafe_get_current_work_info(tile_summary, state)

    initial_seq_info = scheduler.unsafe_seq_info(tile_summary, state)
    comptime assert not SchedulerType.may_advance

    @parameter
    if not decoding:
        if not initial_seq_info.is_valid():
            return

    if tid == 0:

        @parameter
        for i in range(pipeline_stages):
            # until we can use TMA, we need 128 producers working on async copies
            produced_mbar_kv[i].init(1)
            consumed_mbar_kv[i].init(Int32(num_softmax_threads))
        umma_0.init()
        umma_1.init()

    comptime PositionType = MHAPosition[
        BM,
        BN,
        depth,
        padded_depth,
        num_heads,
        group,
        decoding,
    ]

    @parameter
    @always_inline
    fn get_position(seq_info: SeqInfo) -> PositionType:
        return _get_position[
            BM,
            BN,
            depth,
            padded_depth,
            num_heads,
            group,
            ragged,
            _is_cache_length_accurate,
        ](
            seq_info,
            kv_lut,
            max_seq_len,
            num_keys_arg,
            kv_input_row_offsets,
        )

    var position: PositionType = get_position(initial_seq_info)
    startend = position.get_start_and_end_for_partitions[
        page_size = KVLUTType.page_size
    ](partition, mask)
    var kv_tile_start_row: UInt32 = startend[0]
    var end: UInt32 = startend[1]

    comptime assert num_s > 0

    barrier()
    # For intra-warp overlap, we initiate ummas as
    # Q @ K_0, Q @ K_1, P_0 @ V_0, Q @ K_2, P_1 @ V_1, ...
    # ..., Q @ K_{N-1}, P_{N-2} @ V_{N-2}, P_{N-1} @ V_{N-1}
    #
    # Due to this, we can overlap ummas and softmax calculations.
    if warp_group_idx == 0:
        # producer
        warpgroup_reg_dealloc[num_producer_regs]()
        if tid == 96:  # thread 0 of warp id 3
            produce[
                swizzle_mode,
                pipeline_stages=pipeline_stages,
                ragged=ragged,
                _is_cache_length_accurate=_is_cache_length_accurate,
            ](
                q_tma_op,
                k_tma_op,
                v_tma_op,
                q_smem,
                kv_smem,
                produced_mbar_kv,
                consumed_mbar_kv,
                {},
                {},
                kv_lut,
                position,
                partition,
                scheduler,
                mask,
                tile_summary,
                state,
                max_seq_len,
                num_keys_arg,
                kv_input_row_offsets,
            )
        elif warp_id() == 0:  # warp id == 0: Q @ K'
            startend = position.get_start_and_end_for_partitions[
                page_size = KVLUTType.page_size
            ](partition, mask)
            var kv_tile_start_row: UInt32 = startend[0]
            var end: UInt32 = startend[1]

            @parameter
            if PartitionType.do_partition:
                # we exit before allocating so we don't need to deallocate
                if kv_tile_start_row >= end:
                    return

            comptime tmem_cols = num_s * MMA_N0 + (MMA_N0 // 2) + MMA_N1
            comptime assert tmem_cols <= max_tmem_cols
            tcgen05_alloc[cta_group](ptr_tmem_addr, max_tmem_cols)

            qk_desc = UMMA0Type.mma_descriptors(q_smem, kv_smem)

            named_barrier[Int32(num_softmax_threads + 2 * WARP_SIZE)]()
            if tid != 0:
                return
            q_desc = qk_desc.get_a()
            k_desc = qk_desc.get_b()
            var tmem_addr: UInt32 = ptr_tmem_addr[0]
            var s_tmem: UInt32 = tmem_addr
            var o_tmem: UInt32 = tmem_addr + UInt32(MMA_N0 * num_s)
            var p_tmem: UInt32 = (
                tmem_addr + UInt32(MMA_N0 * num_s) + UInt32(MMA_N1)
            )
            s_accumulator = UMMA0Type.c_t(s_tmem)

            @parameter
            @always_inline
            fn q_mul_k(read_idx: UInt32, read_phase: UInt32):
                q = q_desc
                k = k_desc + Int(
                    UInt32(BN * Int(config.padded_depth) * size_of[kv_type]())
                    * read_idx
                )
                umma_0.wait_for_tmem()
                produced_mbar_kv[read_idx].wait(read_phase)

                umma_0.mma(
                    rebind[UMMA0Type.a_t](q),
                    rebind[UMMA0Type.b_t](k),
                    s_accumulator,
                    0,
                )

            var mask_status: TileMaskStatus
            while True:
                mask_status = position.mask_status(mask, kv_tile_start_row)
                if mask_status != TileMaskStatus.FULL_MASK:
                    break
                kv_tile_start_row += UInt32(BN)

            kv_pipeline_states = PipelineState[pipeline_stages]()
            s_pipeline_states = PipelineState[pipeline_stages]()
            q_mul_k(
                kv_pipeline_states.index(),
                kv_pipeline_states.phase(),
            )
            kv_pipeline_states.step()
            # Consumption order:
            # Preheader: Q0, K0
            # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
            # Exit: V{-1}
            while True:
                # this loops over num_keys
                kv_tile_start_row += UInt32(BN)
                if kv_tile_start_row >= end:
                    break
                # this loops over num_keys
                mask_status = position.mask_status(mask, kv_tile_start_row)
                if mask_status == TileMaskStatus.FULL_MASK:
                    continue

                # new pipeline states
                # start ummas
                q_mul_k(
                    kv_pipeline_states.index(), kv_pipeline_states.phase()
                )  # can't rw `p_reg_tile`
                kv_pipeline_states.step()
                kv_pipeline_states.step()

        elif warp_id() == 1:  # warp id 1: P @ V
            startend = position.get_start_and_end_for_partitions[
                page_size = KVLUTType.page_size
            ](partition, mask)
            var kv_tile_start_row: UInt32 = startend[0]
            var end: UInt32 = startend[1]

            @parameter
            if PartitionType.do_partition:
                if kv_tile_start_row >= end:
                    return

            named_barrier[Int32(num_softmax_threads + 2 * WARP_SIZE)]()
            var tmem_addr: UInt32 = ptr_tmem_addr[0]
            if tid == 32:
                var s_tmem: UInt32 = tmem_addr
                var o_tmem: UInt32 = tmem_addr + UInt32(MMA_N0 * num_s)
                var p_tmem: UInt32 = (
                    tmem_addr + UInt32(MMA_N0 * num_s) + UInt32(MMA_N1)
                )
                p_desc = UMMA1Type.a_mma_descriptor(p_tmem)
                v_desc = UMMA1Type.b_mma_descriptor(kv_smem)
                output_accumulator = UMMA1Type.c_t(o_tmem)

                @parameter
                @always_inline("nodebug")
                fn p_mul_v(
                    read_idx: UInt32,
                    read_phase: UInt32,
                    scale_c: UInt32,
                    kv_row: UInt32,
                ):
                    comptime offset_elems_per = BN * Int(config.padded_depth)
                    comptime offset_bytes_per = offset_elems_per * size_of[
                        kv_type
                    ]()
                    v = v_desc + Int(UInt32(offset_bytes_per) * read_idx)
                    produced_mbar_kv[read_idx].wait(read_phase)
                    umma_1.wait_for_tmem()
                    umma_1.mma(
                        rebind[UMMA1Type.a_t](p_desc),
                        rebind[UMMA1Type.b_t](v),
                        output_accumulator,
                        scale_c,
                    )

                var mask_status: TileMaskStatus
                while True:
                    mask_status = position.mask_status(mask, kv_tile_start_row)
                    if mask_status != TileMaskStatus.FULL_MASK:
                        break
                    kv_tile_start_row += UInt32(BN)

                kv_pipeline_states = PipelineState[pipeline_stages]()
                var read_idx_q: UInt32 = kv_pipeline_states.index()
                kv_pipeline_states.step()

                var output_scale: UInt32 = 0
                # Consumption order:
                # Preheader: Q0, K0
                # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
                # Exit: V{-1}
                while True:
                    # this loops over num_keys
                    kv_tile_start_row += UInt32(BN)
                    if kv_tile_start_row >= end:
                        break
                    # this loops over num_keys
                    mask_status = position.mask_status(mask, kv_tile_start_row)
                    if mask_status == TileMaskStatus.FULL_MASK:
                        continue
                    # copy new pfrag, used by `p_mul_v` on next iter
                    # start ummas
                    kv_pipeline_states.step()
                    var read_idx_v: UInt32 = kv_pipeline_states.index()
                    p_mul_v(
                        read_idx_v, kv_pipeline_states.phase(), output_scale, 0
                    )  # can't rw output or pfrag
                    output_scale = 1
                    kv_pipeline_states.step()

                p_mul_v(
                    kv_pipeline_states.index(),
                    kv_pipeline_states.phase(),
                    output_scale,
                    kv_tile_start_row,
                )
            tcgen05_release_allocation_lock[cta_group]()
            tcgen05_dealloc[cta_group](tmem_addr, max_tmem_cols)

    else:  # softmax
        warpgroup_reg_alloc[num_softmax_regs]()

        # arrive to unblock the producers
        # TODO: skip this by not waiting on the first set
        @parameter
        for i in range(pipeline_stages):
            _ = consumed_mbar_kv[i].arrive()
        umma_0.tmem_arrive_init()

        var warp_id: UInt32 = warp.broadcast((tid - 128) // UInt32(WARP_SIZE))

        # Coordinates of the current warp.
        var elect_one_warp = warp_id == 0

        var lane = UInt32(lane_id())

        var warp_y: UInt32 = warp_id  # // num_warps_n

        @parameter
        if num_softmax_threads > 128:
            warp_y = 2 * (warp_y % 4) + (warp_y // 4)
        comptime warp_x: UInt32 = 0
        comptime assert num_softmax_warps == 4 or num_softmax_warps == 8

        # Mask global memory iterator.

        mask_warp_row = warp_y * UInt32(WM)
        var scale_log2e: Scalar[accum_type] = (
            scale.cast[accum_type]() if use_score_mod
            or MaskType.apply_log2e_after_mask else scale.cast[accum_type]()
            * log2e
        )

        # layout is
        # shape  = (2, num_m_blocks_per_warp) x (2, num_n_mmas)
        # stride = (2, 4*num_n_mmas) x (1, 4)

        rowmax = LayoutTensor[
            UMMA0Type.accum_t,
            Layout.row_major(num_rows_per_warp),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()
        rowsum = LayoutTensor[
            UMMA0Type.accum_t,
            Layout.row_major(num_rows_per_warp),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()
        comptime VecPType = LayoutTensor[
            accum_type,
            p_vec_output_layout,
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout=element_layout,
        ]
        comptime VecOType = LayoutTensor[
            accum_type,
            o_vec_output_layout,
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout=element_layout,
        ]

        p_reg_tile = UMMA0Type.c_t.allocate_register_tile()
        output_reg_tile = UMMA1Type.c_t.allocate_register_tile()

        @parameter
        @always_inline
        fn vectorize_p_reg_tile(
            out result: VecPType,
        ):
            result = {p_reg_tile.ptr}

        @parameter
        @always_inline
        fn vectorize_o_reg_tile(
            out result: VecOType,
        ):
            result = {output_reg_tile.ptr}

        @parameter
        @always_inline
        fn apply_mask(
            position: PositionType,
            mask_status: TileMaskStatus,
            kv_tile_start_row: UInt32,
        ):
            var max_len: UInt32 = (
                num_keys_arg if decoding else max_seq_len.as_uint32()
            )
            _apply_mask[WM, MMA_N0, num_m_mmas, num_n_mmas, use_score_mod](
                mask_warp_row,
                position,
                lane,
                max_len,
                scale_log2e,
                kv_tile_start_row,
                mask,
                mask_status,
                score_mod,
                vectorize_p_reg_tile(),
            )

        @parameter
        @always_inline
        fn scale(correction: type_of(rowmax), vout: VecOType):
            # Correct output
            # We could avoid this on the first iter
            # if we specialize and unswitch on `first_iter`
            # otherwise, the branch requires synchronization
            @parameter
            for row in range(num_rows_per_warp):
                c = SIMD[accum_type, element_layout.size()](
                    rebind[Scalar[accum_type]](correction[row])
                )

                @parameter
                for col in range(num_cols_output):
                    vout[row, col] = vout[row, col] * c

        @always_inline
        fn elementwise_reciprocal(
            old_rowsum: type_of(rowsum), new_rowsum: type_of(rowsum)
        ):
            # new_rowsum, old_rowsum = 1/old_rowsum, new_rowsum
            @parameter
            for row in range(num_rows_per_warp):
                old = old_rowsum[row]
                new = new_rowsum[row]
                new_rowsum[row] = recip(old)[0]
                old_rowsum[row] = new

        @parameter
        @always_inline
        fn write_output(
            position: PositionType,
            rowsum_inv: type_of(rowsum),
            vout: VecOType,
        ):
            # Apply softmax denumerator.
            @parameter
            for row in range(num_rows_per_warp):
                rs_inv = vout.element_type(rowsum_inv[row][0])

                @parameter
                for col in range(num_cols_output):
                    vout[row, col] = vout[row, col] * rs_inv

            var output_ptr: UnsafePointer[
                Scalar[output_type], MutAnyOrigin
            ] = o_ptr_arg

            @parameter
            if decoding and PartitionType.do_partition:
                output_ptr = output_ptr + (
                    UInt32(depth * num_heads)
                    * batch_size
                    * position.prompt_offset
                )
            output_gmem_tile = position.q_out_gmem_tensor(output_ptr)

            # Write to global memory.
            comptime assert (
                output_type.is_half_float()
            ), "we don't support Float32 output"
            comptime assert size_of[kv_type]() == size_of[output_type]()
            comptime swizzle = make_swizzle[
                num_rows = WM // 2, row_size=BN, access_size=8
            ]()
            # Reuse a_smem for c tile in smem
            comptime q_tile_size: UInt32 = q_smem_size // 2
            accum_smem_tile = LayoutTensor[
                output_type,
                Layout.row_major(BM, Int(config.padded_depth)),
                address_space = AddressSpace.SHARED,
            ]((q_smem).bitcast[Scalar[output_type]]())
            accum_smem_warp_tile = accum_smem_tile.tile[WM, BN](
                Int(warp_y), Int(warp_x)
            )

            # ensure all threads have finished reading `q_smem`
            named_barrier[Int32(num_softmax_threads)]()
            copy_local_to_shared[
                thread_layout=mma_thread_layout, swizzle=swizzle
            ](
                accum_smem_warp_tile.vectorize[1, 2](),
                UMMA1Type.c_t.rows_of_frags(output_reg_tile)
                .vectorize[1, 2]()
                .transpose(),
            )
            # Guard writing to shared memory.
            named_barrier[Int32(num_softmax_threads)]()
            # Vectorized copy from shared to global memory, during which every 2 FP32
            # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
            # vector and stored using 16B store instruction.
            copy_sram_to_dram[
                thread_layout = Layout.row_major(
                    num_softmax_threads * simd_size // depth,
                    depth // simd_size,
                ),
                swizzle=swizzle,
            ](
                output_gmem_tile.vectorize[1, simd_size](),
                accum_smem_tile.vectorize[1, simd_size](),
            )

        @parameter
        if (
            decoding and PartitionType.do_partition
        ):  # we may have an empty partition
            if kv_tile_start_row >= end:
                if thread_idx.x % 4 == 0 and thread_idx.x < UInt(
                    4 * min(group, 8) + 128
                ):
                    exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                        partition, batch_size
                    )
                    var q_heads = get_q_head_idx(position, lane)

                    @parameter
                    for i in range(q_heads.size):
                        var q_head_idx = q_heads[i]
                        exp_sum_ptr[q_head_idx] = Scalar[
                            PartitionType.accum_dtype
                        ](0)
                        qk_max_ptr[q_head_idx] = min_or_neg_inf[
                            PartitionType.accum_dtype
                        ]()

                write_output(position, rowsum, vectorize_o_reg_tile().fill(0))
                return

        named_barrier[Int32(num_softmax_threads + 2 * WARP_SIZE)]()
        var tmem_addr = ptr_tmem_addr[0]

        @parameter
        if num_softmax_warps > 4:
            if warp_group_idx != 1:  # elect_one_warp will be false
                tmem_addr += 1 << 20
        var s_tmem: UInt32 = tmem_addr
        var o_tmem: UInt32 = tmem_addr + UInt32(MMA_N0 * num_s)
        var p_tmem: UInt32 = tmem_addr + UInt32(MMA_N0 * num_s) + UInt32(MMA_N1)
        p_accumulator = UMMA0Type.c_t(s_tmem)
        p_desc = UMMA1Type.a_mma_descriptor(p_tmem)
        output_accumulator = UMMA1Type.c_t(o_tmem)
        v_desc = UMMA1Type.b_mma_descriptor(kv_smem)

        @parameter
        @always_inline
        fn wait_for_q_mul_k(read_idx: UInt32):
            p_acc = umma_0.wait_for_mma(p_accumulator)  # P is available
            _ = consumed_mbar_kv[read_idx].arrive()
            p_acc.copy_to(p_reg_tile)
            umma_0.tmem_arrive()

        @parameter
        @always_inline
        fn wait_for_p_mul_v(read_idx: UInt32):
            umma_1.wait_for_mma()  # output is available
            _ = consumed_mbar_kv[read_idx].arrive()
            output_accumulator.copy_to(output_reg_tile)

        var mask_status: TileMaskStatus
        while True:
            mask_status = position.mask_status(mask, kv_tile_start_row)
            if mask_status != TileMaskStatus.FULL_MASK:
                break
            kv_tile_start_row += UInt32(BN)

        kv_pipeline_states = PipelineState[pipeline_stages]()
        # q_mul_k must wait on fetching q and k
        # therefore, we find `kv_tile_start_row` first.
        var read_idx_q: UInt32 = kv_pipeline_states.index()
        # q_mul_k(
        #     read_idx_q,
        #     kv_pipeline_states.phase(),
        # )
        kv_pipeline_states.step()

        wait_for_q_mul_k(read_idx_q)
        apply_mask(position, mask_status, kv_tile_start_row)

        @parameter
        if not SinkType.is_null:
            # Include sink_weights in rowmax computation if present
            var q_head_indices = get_q_head_idx(position, lane)

            @parameter
            if decoding:

                @parameter
                for i in range(q_head_indices.size):
                    var head_idx = q_head_indices[i]
                    var sink_weight = sink_weights_ptr[head_idx] * log2e
                    rowmax[i] = sink_weight.cast[accum_type]()
            else:
                var sink_weight = sink_weights_ptr[q_head_indices[0]] * log2e

                @parameter
                for i in range(num_rows_per_warp):
                    rowmax[i] = sink_weight.cast[accum_type]()

        # Compute initial rowmax
        var attention_rowmax = _rowmax_online_softmax[
            1, mma_thread_layout, use_exp2=True
        ](vectorize_p_reg_tile(), rowmax, init_rowmax=SinkType.is_null)

        rowmax.copy_from(attention_rowmax)

        comptime assert p_vec_output_layout.size() > 0, "layout: " + String(
            p_vec_output_layout
        )

        # Compute rowsum
        var attention_rowsum = _rowsum[mma_thread_layout](
            vectorize_p_reg_tile()
        )

        # Add sink weight contribution to rowsum
        @parameter
        if not SinkType.is_null:
            var q_head_indices = get_q_head_idx(position, lane)

            @parameter
            if decoding:

                @parameter
                for i in range(q_head_indices.size):
                    var head_idx = q_head_indices[i]
                    var sink_weight = (
                        sink_weights_ptr[head_idx].cast[accum_type]() * log2e
                    )
                    var sink_contribution = exp2(sink_weight - rowmax[i])
                    attention_rowsum[i] += sink_contribution[0]
            else:
                var sink_weight = (
                    sink_weights_ptr[q_head_indices[0]].cast[accum_type]()
                    * log2e
                )

                @parameter
                for i in range(num_rows_per_warp):
                    # Compute exp2((sink_weight - rowmax[j]) * log2e)
                    var sink_contribution = exp2(sink_weight - rowmax[i])
                    attention_rowsum[i] += sink_contribution[0]

        rowsum.copy_from(attention_rowsum)

        # var output_scale: UInt32 = 0
        # Consumption order:
        # Preheader: Q0, K0
        # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
        # Exit: V{-1}
        while True:
            # this loops over num_keys
            kv_tile_start_row += UInt32(BN)
            if kv_tile_start_row >= end:
                break
            # this loops over num_keys
            mask_status = position.mask_status(mask, kv_tile_start_row)
            if mask_status == TileMaskStatus.FULL_MASK:
                continue
            # copy new pfrag, used by `p_mul_v` on next iter
            p_desc.copy_from(UMMA0Type.c_t.rows_of_frags(p_reg_tile))
            umma_1.tmem_arrive()

            # new pipeline states
            var read_idx_q: UInt32 = kv_pipeline_states.index()
            # start ummas
            # q_mul_k(
            #     read_idx_q, kv_pipeline_states.phase()
            # )  # can't rw `p_reg_tile`
            kv_pipeline_states.step()
            var read_idx_v: UInt32 = kv_pipeline_states.index()
            # p_mul_v(
            #     read_idx_v, kv_pipeline_states.phase(), output_scale
            # )  # can't rw output or pfrag
            # output_scale = 1
            kv_pipeline_states.step()
            wait_for_q_mul_k(read_idx_q)

            apply_mask(position, mask_status, kv_tile_start_row)
            # Compute rowmax for current scores
            var current_rowmax = _rowmax_online_softmax[
                1, mma_thread_layout, use_exp2=True
            ](vectorize_p_reg_tile(), rowmax, False)

            score_frag_rowmax = current_rowmax
            score_frag_rowsum = rebind[type_of(rowsum)](
                _rowsum[mma_thread_layout](vectorize_p_reg_tile())
            )

            _online_softmax_correction[use_exp2=True](rowmax, score_frag_rowmax)
            # rowmax now holds score_frag_rowmax
            # score_frag_rowmax now holds the correction

            @parameter
            for i in range(num_rows_per_warp):
                rowsum[i] = (
                    rowsum[i] * score_frag_rowmax[i] + score_frag_rowsum[i]
                )

            wait_for_p_mul_v(read_idx_v)  # can rw output and pfrag
            scale(score_frag_rowmax, vectorize_o_reg_tile())  # scale output
            output_accumulator.copy_from(output_reg_tile)

        p_desc.copy_from(UMMA0Type.c_t.rows_of_frags(p_reg_tile))
        umma_1.tmem_arrive()

        # p_mul_v(
        #     kv_pipeline_states.index(),
        #     kv_pipeline_states.phase(),
        #     output_scale,
        # )

        @parameter
        if decoding and PartitionType.do_partition:
            # Only the first thread of each row
            if thread_idx.x % 4 == 0 and thread_idx.x < UInt(
                4 * min(group, 8) + 128
            ):
                exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                    partition, batch_size
                )
                var q_heads = get_q_head_idx(position, lane)

                @parameter
                for i in range(q_heads.size):
                    var q_head_idx = q_heads[i]
                    exp_sum_ptr[q_head_idx] = rebind[
                        Scalar[PartitionType.accum_dtype]
                    ](rowsum[i])
                    qk_max_ptr[q_head_idx] = rebind[
                        Scalar[PartitionType.accum_dtype]
                    ](rowmax[i])

        @parameter
        for row in range(num_rows_per_warp):
            rowsum[row] = recip(rowsum[row])[0]
        umma_1.wait_for_mma()

        output_accumulator.copy_to(output_reg_tile)
        comptime assert type_of(output_reg_tile).layout[1].size() > 1, (
            "output_reg_tile.layout = "
            + String(type_of(output_reg_tile).layout)
            + "\n"
        )
        write_output(position, rowsum, vectorize_o_reg_tile())
        # don't arrive

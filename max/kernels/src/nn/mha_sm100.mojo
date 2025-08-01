# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from algorithm.functional import unswitch
from buffer import NDBuffer
from collections import OptionalReg
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    lane_id,
    thread_idx,
)
from gpu import WARP_SIZE
from gpu.cluster import elect_one_sync
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import B200
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory
from gpu.mma import MMAOperandDescriptor
from gpu.mma_sm100 import (
    UMMAInsDescriptor,
    MMASmemDescriptor,
    UMMAKind,
    mma_arrive,
    mma,
)
from gpu.sync import named_barrier
from gpu.tcgen05 import (
    tcgen05_alloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_st,
    tcgen05_store_wait,
    tcgen05_release_allocation_lock,
    tcgen05_dealloc,
)
from layout.int_tuple import IntTuple
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy_local_to_shared,
    copy_sram_to_dram,
    cp_async_k_major,
)
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tma_async import PipelineState, SharedMemBarrier
from math import ceildiv, recip
from math.constants import log2e
from memory import stack_allocation
from memory import stack_allocation, bitcast
from nn.mha_fa3_utils import (
    MHAPosition,
    _get_position,
    _produce,
    _apply_mask,
    valid_length_managed_tensor_slice_to_ndbuffer,
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
    get_start_and_end_for_partitions,
)
from nn.softmax import (
    _online_softmax_correction,
    _rowmax_online_softmax,
    _rowsum,
)
from sys import alignof, simdwidthof, sizeof
from sys import sizeof
from tensor_internal import ManagedTensorSlice
from utils.index import Index
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple
import gpu.warp as warp


struct RegisterAccumulatorDescription:
    var num_mmas: Int
    var frag_size: Int

    @always_inline
    fn __init__(out self, num_mmas: Int, frag_size: Int):
        self.num_mmas = num_mmas
        self.frag_size = frag_size


# consumer_group_size equals
# sm90: 128 (warp group size)
# sm100: num_consumer_threads
@register_passable("trivial")
struct RegisterAccumulatorLayout[
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    consumer_group_size: Int,
    *,
    frag_simdwidth: Int = 2,
]:
    alias frag_size: Int = MMA_M * MMA_N // consumer_group_size
    alias num_row_blocks_per_mma = 2
    alias element_layout: Layout = Layout.row_major(1, Self.frag_simdwidth)
    alias rows_of_frags_layout: Layout = Layout.row_major(
        num_m_mmas * num_n_mmas, Self.frag_size
    )
    alias vec_output_layout: Layout = Layout(
        IntTuple(
            IntTuple(Self.num_row_blocks_per_mma, num_m_mmas),
            IntTuple(
                Self.frag_size
                // (Self.num_row_blocks_per_mma * Self.frag_simdwidth),
                num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(Self.frag_simdwidth, Self.frag_size),
            IntTuple(
                Self.num_row_blocks_per_mma * Self.frag_simdwidth,
                num_m_mmas * Self.frag_size,
            ),
        ),
    )
    constrained[
        Self.vec_output_layout.size() > 0,
        "layout: " + String(Self.vec_output_layout),
    ]()

    @staticmethod
    @always_inline
    fn description() -> RegisterAccumulatorDescription:
        return RegisterAccumulatorDescription(
            num_m_mmas * num_n_mmas, Self.frag_size
        )


@register_passable("trivial")
struct MMAOperandOffsetFn[
    dtype: DType,
    BMN: Int,
    BK: Int,
    swizzle: TensorMapSwizzle,
    is_k_major: Bool,
    WMMA_MN: Int,
    WMMA_K: Int,
]:
    alias layout = tile_layout_k_major[
        dtype, BMN, BK, swizzle
    ]() if is_k_major else tile_layout_mn_major[dtype, BMN, BK, swizzle]()
    alias layout_size: Int = Self.layout.size()

    alias canonical_K = swizzle.bytes() // sizeof[
        dtype
    ]() if swizzle != TensorMapSwizzle.SWIZZLE_NONE else BK
    alias canonical_layout_flat = tile_layout_k_major[
        dtype, BMN, Self.canonical_K, swizzle
    ]() if is_k_major else Self.layout
    alias canonical_layout = tile_to_descriptor[
        dtype, Self.canonical_layout_flat, is_k_major
    ]()
    alias canonical_layout_size = Self.canonical_layout.size()

    @always_inline
    fn __init__(out self):
        pass


@register_passable("trivial")
trait DescriptorPair:
    alias a_t: MMAOperandDescriptor
    alias b_t: MMAOperandDescriptor

    @always_inline
    fn get_a(self) -> a_t:
        ...

    @always_inline
    fn get_b(self) -> b_t:
        ...


@register_passable("trivial")
trait WriteableMMAOperandDescriptor:
    @always_inline
    fn copy_from[
        src_type: DType, src_layout: Layout, src_element_layout: Layout, //
    ](
        self,
        src: LayoutTensor[
            src_type,
            src_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout=src_element_layout,
        ],
    ):
        ...


@register_passable("trivial")
trait DescriptorPairTS:
    alias a_t: WriteableMMAOperandDescriptor
    alias b_t: MMAOperandDescriptor

    @always_inline
    fn get_a(self) -> a_t:
        ...

    @always_inline
    fn get_b(self) -> b_t:
        ...


fn local_tensor_type[
    dtype: DType, layout: Layout, element_layout: Layout
](
    out dummy_arg: LayoutTensor[
        dtype,
        layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=element_layout,
    ]
):
    dummy_arg = __type_of(dummy_arg)(
        UnsafePointer[Scalar[dtype], address_space = AddressSpace.LOCAL]()
    )


@register_passable("trivial")
trait AccumulatorTile(Copyable, Movable):
    alias dtype: DType
    alias element_layout: Layout
    alias vec_output_layout: Layout
    alias rows_of_frags_layout: Layout

    @staticmethod
    @always_inline
    fn _empty_tensor() -> (
        __type_of(local_tensor_type[dtype, vec_output_layout, element_layout]())
    ):
        ...

    @staticmethod
    @always_inline
    fn rows_of_frags(
        src: __type_of(Self._empty_tensor()),
        out res: LayoutTensor[
            Self.dtype,
            Self.rows_of_frags_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ],
    ):
        ...

    @always_inline
    fn allocate_register_tile(
        self,
        out res: __type_of(Self._empty_tensor()),
    ):
        ...

    @always_inline
    fn copy_from(
        self,
        src: __type_of(Self._empty_tensor()),
    ):
        ...

    @always_inline
    fn copy_to(
        self,
        dst: __type_of(Self._empty_tensor()),
    ):
        ...


# We have a 4-step process:
# 1 Describe
# 2 Initialize
# 3 Barrier
# 4 Can use
@register_passable("trivial")
trait AsyncTensorAccumulator:
    alias operand_t: DType
    alias accum_t: DType
    alias ab_t: DescriptorPair
    alias a_t: MMAOperandDescriptor
    alias b_t: MMAOperandDescriptor
    alias c_t: AccumulatorTile

    @always_inline
    fn __init__(
        out self,
        smem: UnsafePointer[
            SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=8
        ],
    ):
        ...

    @staticmethod
    @always_inline
    fn mma_descriptors[
        dtype_a: DType, dtype_b: DType
    ](
        p_a: UnsafePointer[
            Scalar[dtype_a], address_space = AddressSpace.SHARED
        ],
        p_b: UnsafePointer[
            Scalar[dtype_b], address_space = AddressSpace.SHARED
        ],
    ) -> ab_t:
        ...

    @always_inline
    fn mma(self, a: a_t, b: b_t, c: c_t, c_scale: UInt32, wg_idx: UInt32 = 0):
        ...

    @always_inline
    fn wait_group[wgmma_left_in_flight: Int = 0](mut self):
        ...


@register_passable("trivial")
trait AsyncTensorAccumulatorTS:
    alias operand_t: DType
    alias accum_t: DType
    alias ab_t: DescriptorPairTS
    alias a_t: WriteableMMAOperandDescriptor
    alias b_t: MMAOperandDescriptor
    alias c_t: AccumulatorTile

    @always_inline
    fn __init__(
        out self,
        smem: UnsafePointer[
            SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=8
        ],
    ):
        ...

    @always_inline
    fn mma(self, a: a_t, b: b_t, c: c_t, c_scale: UInt32, wg_idx: UInt32 = 0):
        ...

    @always_inline
    fn wait_group[wgmma_left_in_flight: Int = 0](mut self):
        ...


@register_passable("trivial")
struct UMMADescriptorSS[operand_type: DType](DescriptorPair):
    alias operand_t = operand_type
    alias a_t = MMASmemDescriptor
    alias b_t = MMASmemDescriptor

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
    alias linear = _tmem_offset(
        dtype.sizeof(), MMA_N=MMA_N, m_mma=m_mma, n_mma=n_mma
    )
    return linear


@register_passable("trivial")
struct TMemAccumulator[
    dtype_: DType,
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    num_consumer_threads: Int,
](AccumulatorTile):
    alias dtype: DType = dtype_
    alias layout_t = RegisterAccumulatorLayout[
        MMA_M, MMA_N, num_m_mmas, num_n_mmas, num_consumer_threads
    ]
    alias vec_output_layout = Self.layout_t.vec_output_layout
    alias element_layout = Self.layout_t.element_layout
    alias rows_of_frags_layout = Self.layout_t.rows_of_frags_layout
    alias frag_size = Self.layout_t.frag_size

    alias tmem_addr_t = UInt32
    var tmem_addr: Self.tmem_addr_t

    @always_inline
    fn __init__(out self, tmem_addr: Self.tmem_addr_t):
        Self.check_constraints()
        self.tmem_addr = tmem_addr

    @staticmethod
    @always_inline
    fn _empty_tensor() -> (
        __type_of(
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
    @staticmethod
    fn check_constraints():
        constrained[
            MMA_M > 0,
            "MMA_M = "
            + String(MMA_M)
            + "\nMMA_N = "
            + String(MMA_N)
            + "\nnum_m_mmas = "
            + String(num_m_mmas)
            + "\nnum_n_mmas = "
            + String(num_n_mmas)
            + "\n",
        ]()
        constrained[
            MMA_N > 0,
            "MMA_M = "
            + String(MMA_M)
            + "\nMMA_N = "
            + String(MMA_N)
            + "\nnum_m_mmas = "
            + String(num_m_mmas)
            + "\nnum_n_mmas = "
            + String(num_n_mmas)
            + "\n",
        ]()
        constrained[
            num_m_mmas > 0,
            "MMA_M = "
            + String(MMA_M)
            + "\nMMA_N = "
            + String(MMA_N)
            + "\nnum_m_mmas = "
            + String(num_m_mmas)
            + "\nnum_n_mmas = "
            + String(num_n_mmas)
            + "\n",
        ]()
        constrained[
            num_n_mmas > 0,
            "MMA_M = "
            + String(MMA_M)
            + "\nMMA_N = "
            + String(MMA_N)
            + "\nm_mma = "
            + String(num_m_mmas)
            + "\nnum_n_mmas = "
            + String(num_n_mmas)
            + "\n",
        ]()

    @always_inline
    fn offset[m_mma: Int, n_mma: Int](self) -> Self.tmem_addr_t:
        Self.check_constraints()

        @parameter
        if m_mma == 0 and n_mma == 0:
            return self.tmem_addr
        else:
            alias linear = _tmem_offset[
                Self.dtype, MMA_N=MMA_N, m_mma=m_mma, n_mma=n_mma
            ]()

            return self.tmem_addr + linear

    @staticmethod
    @always_inline
    fn rows_of_frags(
        src: __type_of(Self._empty_tensor()),
        out res: LayoutTensor[
            Self.dtype,
            Self.rows_of_frags_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ],
    ):
        Self.check_constraints()
        res = __type_of(res)(src.ptr)

    @always_inline
    fn allocate_register_tile(
        self,
        out res: __type_of(Self._empty_tensor()),
    ):
        constrained[
            Self.vec_output_layout[0].size() > 0,
            "layout: "
            + String(Self.vec_output_layout)
            + "\nnum_m_mmas = "
            + String(num_m_mmas),
        ]()
        constrained[
            Self.vec_output_layout[1].size() > 0,
            "layout: " + String(Self.vec_output_layout),
        ]()
        res = __type_of(res).stack_allocation()

    @always_inline
    fn copy_from(
        self,
        src: __type_of(Self._empty_tensor()),
    ):
        frags = Self.rows_of_frags(src).vectorize[1, Self.frag_size]()
        alias dtype_size = sizeof[Self.dtype]()
        constrained[dtype_size == 4]()
        alias frag_size_b32 = Self.frag_size * dtype_size // 4
        # 16 x 256b results in repeated 8x4<1x2> pattern
        # each repetition thus fills 8 columns
        # and writes 4 values per thread.
        alias repeat = frag_size_b32 // 4

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                alias mma_id = n_mma * num_m_mmas + m_mma
                alias tmem_offset = _tmem_offset(
                    dtype_size,
                    MMA_N=MMA_N,
                    m_mma=m_mma,
                    n_mma=n_mma,
                )
                tmem = self.tmem_addr + tmem_offset
                frag = bitcast[DType.uint32, frag_size_b32](frags[mma_id, 0])
                # 16 x 256b results in repeated 8x4 matrix of <1,2> vector pattern
                tcgen05_st[
                    datapaths=16,  # first dimension of the shape
                    bits=256,  # second dimension of the shape
                    repeat=repeat,
                    pack=False,
                ](tmem, frag)
        tcgen05_store_wait()
        named_barrier[Self.num_consumer_threads]()

    @always_inline
    fn copy_to(
        self,
        dst: __type_of(Self._empty_tensor()),
    ):
        frags = Self.rows_of_frags(dst).vectorize[1, Self.frag_size]()
        alias dtype_size = sizeof[Self.dtype]()
        constrained[dtype_size == 4]()
        alias frag_size_b32 = (Self.frag_size * dtype_size) // 4
        # 16 x 256b results in repeated 8x4<1x2> pattern
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        alias repeat = frag_size_b32 // 4
        constrained[
            Self.vec_output_layout.size() * Self.element_layout.size()
            == __type_of(dst).layout.size()
            * __type_of(dst).element_layout.size()
        ]()
        constrained[num_m_mmas * num_n_mmas == __type_of(frags).layout.size()]()

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                alias mma_id = n_mma * num_m_mmas + m_mma
                alias tmem_offset = _tmem_offset(
                    dtype_size,
                    MMA_N=MMA_N,
                    m_mma=m_mma,
                    n_mma=n_mma,
                )
                tmem = self.tmem_addr + tmem_offset
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


@register_passable("trivial")
struct TMemOperand[
    dtype: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    num_consumer_threads: Int,
](WriteableMMAOperandDescriptor):
    var tmem_addr: UInt32

    alias reg_layout = RegisterAccumulatorLayout[
        MMA_M, MMA_N, num_m_mmas, num_n_mmas, num_consumer_threads
    ]
    alias frag_size = Self.reg_layout.frag_size
    alias vec_output_layout = Self.reg_layout.vec_output_layout
    alias reg_tile_t = __type_of(
        local_tensor_type[
            dtype, Self.vec_output_layout, Self.reg_layout.element_layout
        ]()
    )

    @always_inline
    fn __init__(out self, tmem_addr: UInt32):
        self.tmem_addr = tmem_addr

    @always_inline
    fn offset[m_mma: Int, k_mma: Int](self) -> UInt32:
        constrained[MMA_M > 0, "MMA_M = " + String(MMA_M) + "\n"]()
        constrained[MMA_K > 0, "MMA_K = " + String(MMA_K) + "\n"]()

        @parameter
        if m_mma == 0 and k_mma == 0:
            return self.tmem_addr
        else:
            alias linear = _tmem_offset[
                DType.bfloat16, MMA_N=MMA_K, m_mma=m_mma, n_mma=k_mma
            ]()
            return self.tmem_addr + linear

    @always_inline
    fn copy_from[
        src_type: DType,
        src_layout: Layout,
        src_element_layout: Layout, //,
    ](
        self,
        src: LayoutTensor[
            src_type,
            src_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout=src_element_layout,
        ],
    ):
        # src has row of frags layout
        alias num_frags = src_layout[0].size()
        constrained[num_frags == num_m_mmas * num_n_mmas]()
        constrained[num_n_mmas == 1]()
        constrained[
            Self.frag_size == src_layout[1].size(),
            "Self.frag_size = "
            + String(Self.frag_size)
            + "\nsrc_layout = "
            + String(src_layout),
        ]()
        constrained[src_element_layout.size() == 1]()
        alias src_size = sizeof[src_type]()
        alias dst_size = sizeof[dtype]()
        alias frag_size_b32 = (Self.frag_size * dst_size) // 4
        # 16 x 256b results in repeated 8x4<1xN> pattern, where
        alias N = 32 // (4 * src_size)
        alias bytes = 4 * dst_size * N
        alias bits = 8 * bytes
        # e.g., N = 2 for fp32
        #
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        # width == (repeat * bits * datapaths) // (32 * 32)
        alias repeat = 64 * frag_size_b32 // bits
        # We need to reshape into a row of frags
        constrained[
            num_m_mmas * num_n_mmas * Self.frag_size
            == src_layout.size() * src_element_layout.size()
        ]()
        frags = LayoutTensor[
            src_type,
            Layout(IntTuple(num_m_mmas * num_n_mmas), IntTuple(Self.frag_size)),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout = Layout.row_major(Self.frag_size),
        ](src.ptr)
        # frags = src.vectorize[1, Self.frag_size]()
        # assume src loaded with 256 bits
        constrained[src_size >= dst_size]()
        constrained[num_m_mmas == 1]()
        constrained[num_n_mmas == 1]()

        @parameter
        for m_mma in range(num_m_mmas):
            tmem = self.offset[m_mma, 0]()
            frag = bitcast[DType.uint32, frag_size_b32](
                frags[m_mma].cast[dtype]()
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
        named_barrier[Self.num_consumer_threads]()

    @always_inline
    fn copy_to[
        dst_type: DType,
        dst_layout: Layout,
        dst_element_layout: Layout, //,
    ](
        self,
        dst: LayoutTensor[
            dst_type,
            dst_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout=dst_element_layout,
        ],
    ):
        # src has row of frags layout
        alias num_frags = dst_layout[0].size()
        constrained[num_frags == num_m_mmas * num_n_mmas]()
        constrained[Self.frag_size == dst_layout[1].size()]()
        constrained[dst_element_layout.size() == 1]()
        constrained[sizeof[dst_type]() == 4]()
        # 16 x 256b results in repeated 8x4<1x2> pattern
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        alias src_size = sizeof[dtype]()
        alias dst_size = sizeof[dst_type]()
        alias frag_size_b32 = (Self.frag_size * src_size) // 4
        # 16 x 256b results in repeated 8x4<1xN> pattern, where
        alias N = 32 // (4 * dst_size)
        alias bytes = 4 * src_size * N
        alias bits = 8 * bytes
        # e.g., N = 2 for fp32
        #
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        # width == (repeat * bits * datapaths) // (32 * 32)
        alias repeat = 64 * frag_size_b32 // bits
        #
        frags = dst.vectorize[1, Self.frag_size]()
        # assume src loaded with 256 bits
        constrained[src_size <= dst_size]()
        constrained[num_n_mmas == 1]()

        @parameter
        for m_mma in range(num_m_mmas):
            tmem = self.offset[m_mma, 0]()
            # 16 x 256b results in repeated 8x4<1x2> pattern
            frags[m_mma, 0] = rebind[
                SIMD[dst_type, __type_of(frags).element_size]
            ](
                bitcast[dtype, Self.frag_size](
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


@register_passable("trivial")
struct UMMADescriptorTS[
    operand_type: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    *,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    consumer_group_size: Int,
](DescriptorPairTS):
    alias operand_t = operand_type
    alias a_t = TMemOperand[
        operand_type,
        num_m_mmas,
        num_n_mmas,
        MMA_M,
        MMA_N,
        MMA_K,
        consumer_group_size,
    ]
    alias b_t = MMASmemDescriptor

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


@register_passable("trivial")
struct SM100TensorAccumulatorSS[
    operand_type: DType,
    accum_type: DType,
    MMA_M: Int,
    MMA_N: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    num_consumer_threads: Int,
    swizzle_a: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    cta_group: Int = 1,
](AsyncTensorAccumulator):
    alias operand_t: DType = operand_type
    alias accum_t: DType = accum_type

    alias MMA_K = 16

    alias num_m_mmas = BM // MMA_M
    alias num_n_mmas = BN // MMA_N
    alias num_k_mmas = BK // Self.MMA_K

    alias num_m_blocks_per_warp = 2 * BM // num_consumer_threads

    alias smem_ptr_t = UnsafePointer[
        Scalar[Self.operand_t], address_space = AddressSpace.SHARED
    ]

    alias a_offset = MMAOperandOffsetFn[
        Self.operand_t, BM, BK, swizzle_a, True, Self.MMA_M, Self.MMA_K
    ]()
    alias b_offset = MMAOperandOffsetFn[
        Self.operand_t, BN, BK, swizzle_b, transpose_b, MMA_N, Self.MMA_K
    ]()

    alias idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype = DType.uint32](MMA_M, MMA_N),
        transpose_b=transpose_b,
    ]()

    alias ab_t: DescriptorPair = UMMADescriptorSS[Self.operand_t]
    alias a_t: MMAOperandDescriptor = Self.ab_t.a_t
    alias b_t: MMAOperandDescriptor = Self.ab_t.b_t
    alias c_t: AccumulatorTile = TMemAccumulator[
        Self.accum_t,
        BM // Self.num_m_blocks_per_warp,
        MMA_N,
        Self.num_m_blocks_per_warp,
        Self.num_n_mmas,
        num_consumer_threads,
    ]

    var mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=8
    ]
    var phase: UInt32

    @always_inline
    @staticmethod
    fn check_constraints():
        constrained[
            (BM % MMA_M) == 0,
            "BM, MMA_M = " + String(BM) + ", " + String(MMA_M),
        ]()
        constrained[
            ((BN % MMA_N) == 0) and (Self.num_n_mmas > 0),
            "BN, MMA_N = " + String(BN) + ", " + String(MMA_N),
        ]()
        constrained[
            ((BK % Self.MMA_K) == 0) and (Self.num_k_mmas > 0),
            "BK, MMA_K = " + String(BK) + ", " + String(Self.MMA_K),
        ]()

    @always_inline
    fn __init__(
        out self,
        smem: UnsafePointer[
            SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=8
        ],
    ):
        Self.check_constraints()
        self.mbar = smem
        self.phase = 0

    @always_inline
    fn init(self):
        self.mbar[0].init()

    # @staticmethod
    # @always_inline
    # fn accumulator_sizes[N: Int](sizes: StaticTuple[Tuple[DType,Layout],N]) -> StaticTuple[Tuple[DType,Layout],N+1]:
    #     pass

    @staticmethod
    @always_inline
    fn mma_descriptors[
        dtype_a: DType, dtype_b: DType
    ](
        p_a: UnsafePointer[
            Scalar[dtype_a], address_space = AddressSpace.SHARED
        ],
        p_b: UnsafePointer[
            Scalar[dtype_b], address_space = AddressSpace.SHARED
        ],
    ) -> Self.ab_t:
        Self.check_constraints()
        alias a_canonical_layout = Self.a_offset.canonical_layout
        alias a_type = Self.operand_t
        alias aSBO = a_canonical_layout[0].stride[1].value() * sizeof[a_type]()
        alias aLBO = a_canonical_layout[1].stride[1].value() * sizeof[a_type]()
        adesc_base = MMASmemDescriptor.create[aSBO, aLBO, swizzle_a](p_a)

        alias b_canonical_layout = Self.b_offset.canonical_layout
        alias b_type = Self.operand_t
        alias b_stride01 = b_canonical_layout[0].stride[1].value()
        alias b_stride11 = b_canonical_layout[1].stride[1].value()
        alias bSBO = (b_stride01 if transpose_b else b_stride11) * sizeof[
            b_type
        ]()
        alias bLBO = (b_stride11 if transpose_b else b_stride01) * sizeof[
            b_type
        ]()
        bdesc_base = MMASmemDescriptor.create[bSBO, bLBO, swizzle_b](p_b)

        return Self.ab_t(adesc_base, bdesc_base)

    @always_inline
    fn mma(
        self,
        a: Self.a_t,
        b: Self.b_t,
        c: Self.c_t,
        scale_c: UInt32,
        wg_idx: UInt32 = 0,  # FIXME: remove wgidx arg
    ):
        if thread_idx.x != 128:
            return

        @parameter
        for k_mma in range(Self.num_k_mmas):

            @parameter
            for m_mma in range(Self.num_m_mmas):
                alias a_offset = Self.a_offset.layout(
                    IntTuple(Self.MMA_M * m_mma, Self.MMA_K * k_mma)
                )
                alias a_offset_bytes = a_offset * sizeof[Self.operand_t]()
                a_desc = a + a_offset_bytes

                @parameter
                for n_mma in range(Self.num_n_mmas):
                    c_tmem = c.offset[m_mma, n_mma]()

                    alias b_offset = Self.b_offset.layout(
                        IntTuple(Self.MMA_N * n_mma, Self.MMA_K * k_mma)
                    ) * sizeof[Self.operand_t]()
                    b_desc = b + b_offset

                    @parameter
                    if k_mma == 0:
                        mma[cta_group](
                            a_desc,
                            b_desc,
                            c_tmem,
                            Self.idesc,
                            scale_c,
                        )
                    else:
                        mma[cta_group, c_scale=1](
                            a_desc, b_desc, c_tmem, Self.idesc
                        )

        mma_arrive(self.mbar)

    @always_inline
    fn wait_group[wgmma_left_in_flight: Int = 0](mut self):
        self.mbar[0].wait(self.phase)
        self.phase ^= 1


@register_passable("trivial")
struct SM100TensorAccumulatorTS[
    operand_type: DType,
    accum_type: DType,
    MMA_M: Int,
    MMA_N: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    num_consumer_threads: Int,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    cta_group: Int = 1,
](AsyncTensorAccumulatorTS):
    alias operand_t: DType = operand_type
    alias accum_t: DType = accum_type

    alias MMA_K = 16
    alias smem_ptr_t = UnsafePointer[
        Scalar[Self.operand_t], address_space = AddressSpace.SHARED
    ]

    alias num_m_mmas = BM // Self.MMA_M
    alias num_n_mmas = BN // MMA_N
    alias num_k_mmas = BK // Self.MMA_K
    alias c_frag_size = Self.MMA_M * MMA_N // num_consumer_threads
    alias a_frag_size = Self.MMA_M * Self.MMA_K // num_consumer_threads
    alias num_m_blocks_per_warp = 2 * BM // num_consumer_threads
    alias ab_t: DescriptorPairTS = UMMADescriptorTS[
        Self.operand_t,
        Self.num_m_blocks_per_warp,
        Self.num_n_mmas,
        MMA_M = BM // Self.num_m_blocks_per_warp,
        MMA_N=BK,
        MMA_K = Self.MMA_K,
        consumer_group_size=num_consumer_threads,
    ]
    alias a_t: WriteableMMAOperandDescriptor = Self.ab_t.a_t
    alias b_t: MMAOperandDescriptor = Self.ab_t.b_t

    alias b_offset = MMAOperandOffsetFn[
        Self.operand_t, BN, BK, swizzle_b, transpose_b, MMA_N, Self.MMA_K
    ]()
    alias c_t: AccumulatorTile = TMemAccumulator[
        Self.accum_t,
        Self.BM // Self.num_m_blocks_per_warp,
        Self.MMA_N,
        Self.num_m_blocks_per_warp,
        Self.num_n_mmas,
        num_consumer_threads,
    ]

    alias idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype = DType.uint32](MMA_M, MMA_N),
        transpose_b=transpose_b,
    ]()

    var mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=8
    ]
    var phase: UInt32

    @staticmethod
    @always_inline
    fn check_constraints():
        constrained[
            (BM % MMA_M) == 0,
            "BM, MMA_M = " + String(BM) + ", " + String(MMA_M),
        ]()
        constrained[
            ((BN % MMA_N) == 0) and (Self.num_n_mmas > 0),
            "BN, MMA_N = " + String(BN) + ", " + String(MMA_N),
        ]()
        constrained[
            ((BK % Self.MMA_K) == 0) and (Self.num_k_mmas > 0),
            "BK, MMA_K = " + String(BK) + ", " + String(Self.MMA_K),
        ]()

    @always_inline
    fn __init__(
        out self,
        smem: UnsafePointer[
            SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=8
        ],
    ):
        Self.check_constraints()
        self.mbar = smem
        self.phase = 0

    @always_inline
    fn init(self):
        self.mbar[0].init()

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
            Scalar[dtype_b], address_space = AddressSpace.SHARED
        ],
    ) -> Self.ab_t.b_t:
        Self.check_constraints()
        alias b_canonical_layout = Self.b_offset.canonical_layout
        alias b_type = Self.operand_t
        alias b_stride01 = b_canonical_layout[0].stride[1].value()
        alias b_stride11 = b_canonical_layout[1].stride[1].value()
        alias bSBO = (b_stride01 if transpose_b else b_stride11) * sizeof[
            b_type
        ]()
        alias bLBO = (b_stride11 if transpose_b else b_stride01) * sizeof[
            b_type
        ]()

        return MMASmemDescriptor.create[bSBO, bLBO, swizzle_b](p_b)

    @always_inline
    fn mma(
        self,
        a: Self.a_t,
        b: Self.b_t,
        c: Self.c_t,
        c_scale: UInt32,
        wg_idx: UInt32 = 0,
    ):
        if thread_idx.x != 128:
            return

        @parameter
        for k_mma in range(Self.num_k_mmas):

            @parameter
            for m_mma in range(Self.num_m_mmas):
                a_tmem = a.offset[m_mma=m_mma, k_mma=k_mma]()

                @parameter
                for n_mma in range(Self.num_n_mmas):
                    c_tmem = c.offset[m_mma=m_mma, n_mma=n_mma]()
                    alias b_offset = Self.b_offset.layout(
                        IntTuple(Self.MMA_N * n_mma, Self.MMA_K * k_mma)
                    ) * sizeof[Self.operand_t]()
                    b_desc = b + b_offset

                    @parameter
                    if k_mma == 0:
                        mma[cta_group](
                            a_tmem,
                            b_desc,
                            c_tmem,
                            Self.idesc,
                            c_scale,
                        )
                    else:
                        mma[cta_group, c_scale=1](
                            a_tmem, b_desc, c_tmem, Self.idesc
                        )
        mma_arrive(self.mbar)

    @always_inline
    fn wait_group[wgmma_left_in_flight: Int = 0](mut self):
        self.mbar[0].wait(self.phase)
        self.phase ^= 1


@always_inline
fn mha_sm100_dispatch[
    kv_t: MHAOperand,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    type: DType,
    output_type: DType,
    max_prompt_len_t: OptionallyStaticInt,
    partition_t: MHAPartitionScheme, //,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    output: UnsafePointer[Scalar[output_type]],
    q: UnsafePointer[Scalar[type]],
    k: kv_t,
    v: kv_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    max_prompt_len_arg: max_prompt_len_t,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
    batch_size_arg: Int,
    partition: partition_t,
    ctx: DeviceContext,
) raises:
    alias decoding: Bool = max_prompt_len_t.static_value.or_else(0) == 1
    alias new_config = MHAConfig(
        config.type,
        config.num_heads,
        config.depth,
        OptionalReg[UInt](64),
        OptionalReg[UInt](config.num_keys_per_block),
        OptionalReg[UInt](config.BK),
    ) if decoding else config
    alias BM = new_config.block_m()
    alias BK = new_config.depth
    constrained[
        BM % 64 == 0,
        "SM90 requires BM%64==0, but BM==",
        String(BM),
    ]()
    constrained[
        BK % 64 == 0,
        "B200 requires BK%64 as it uses 128B swizzles, but BK==",
        String(BK),
    ]()
    alias BN = new_config.block_n()
    # we add smem use for SharedMemBarrier synchronization
    alias extra_b200 = 32
    alias smem_use = new_config.shared_mem_bytes[True, sm_90=True]() + 32
    # add the number of producer threads (i.e. 1 WARP_GROUP_SIZE)
    alias num_threads = new_config.num_threads[True]()
    constrained[
        num_threads % 128 == 0, "num_threads = " + String(num_threads)
    ]()

    # Persistent kernels not currently supported with partitioning
    # This doesn't seem useful: we partition to make SMs more busy,
    # implying we don't have enough to make them persistent.
    # This also requires some tricky control flow handling to support,
    # which we haven't added yet.
    constrained[new_config.algorithm == FlashAttentionAlgorithm(3)]()

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)
    var max_prompt_len: UInt32 = max_prompt_len_arg.as_uint32()
    var max_num_prompt_tiles: UInt32 = ceildiv(max_prompt_len, BM)
    var block_x: UInt32 = max_num_prompt_tiles * partition.num_partitions()

    alias num_scheduler_heads = config.num_heads // group if decoding else config.num_heads
    # if decoding,
    alias scheduler_tile_shape = 1 if decoding else BM

    alias accum_type = get_accum_type[config.type]()

    alias scheduler_t = TransientScheduler[
        scheduler_tile_shape, num_scheduler_heads
    ]
    alias kernel_sm100 = _mha_sm100[
        new_config.type,
        kv_t,
        output_type,
        mask_t,
        score_mod_t,
        scheduler_t,
        new_config,
        group=group,
        use_score_mod=use_score_mod,
        ragged=ragged,
        _is_cache_length_accurate=_is_cache_length_accurate,
        max_seq_len_t=max_prompt_len_t,
        partition_t=partition_t,
    ]
    var scheduler: scheduler_t = scheduler_t()
    gd = scheduler_t.grid_dim(batch_size, block_x)

    @parameter
    if max_prompt_len_t.static_value:

        @parameter
        if partition_t.do_partition:
            ctx.enqueue_function[kernel_sm100](
                q,
                k,
                v,
                output,
                scale,
                batch_size,
                max_cache_valid_length,
                valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                kv_input_row_offsets,
                partition,
                mask_functor,
                score_mod_functor,
                grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                block_dim=(Int(num_threads), 1, 1),
                shared_mem_bytes=Int(smem_use),
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    smem_use
                ),
            )
        else:
            ctx.enqueue_function[kernel_sm100](
                q,
                k,
                v,
                output,
                scale,
                batch_size,
                max_cache_valid_length,
                valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                kv_input_row_offsets,
                mask_functor,
                score_mod_functor,
                grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                block_dim=(Int(num_threads), 1, 1),
                shared_mem_bytes=Int(smem_use),
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    smem_use
                ),
            )

    else:

        @parameter
        if partition_t.do_partition:
            ctx.enqueue_function[kernel_sm100](
                q,
                k,
                v,
                output,
                scale,
                batch_size,
                max_prompt_len,
                max_cache_valid_length,
                valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                kv_input_row_offsets,
                partition,
                mask_functor,
                score_mod_functor,
                grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                block_dim=(Int(num_threads), 1, 1),
                shared_mem_bytes=Int(smem_use),
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    smem_use
                ),
            )
        else:
            ctx.enqueue_function[kernel_sm100](
                q,
                k,
                v,
                output,
                scale,
                batch_size,
                max_prompt_len,
                max_cache_valid_length,
                valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                kv_input_row_offsets,
                mask_functor,
                score_mod_functor,
                grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                block_dim=(Int(num_threads), 1, 1),
                shared_mem_bytes=Int(smem_use),
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    smem_use
                ),
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        config.num_threads[True]()
    )
)
fn _mha_sm100[
    q_type: DType,
    kv_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    scheduler_t: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
    max_seq_len_t: OptionallyStaticInt,
    partition_t: MHAPartitionScheme,
](
    scheduler: scheduler_t,
    q_ptr_arg: UnsafePointer[Scalar[q_type]],
    k: kv_t,
    v: kv_t,
    output_ptr_arg: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: max_seq_len_t,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
    partition: partition_t,
    mask: mask_t,
    score_mod: score_mod_t,
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 Frist bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    alias k_type = kv_t.dtype
    alias v_type = kv_t.dtype
    constrained[q_type == k_type and k_type == v_type]()
    alias decoding: Bool = _is_decoding[max_seq_len_t]()

    alias simd_size: Int = simdwidthof[q_type]()

    alias num_consumer_threads: Int = config.num_consumer_threads()
    alias num_consumer_warps = num_consumer_threads // 32

    alias cta_group = 1
    alias BM: Int = config.block_m()
    alias BN: Int = config.block_n()
    alias BK: Int = config.depth
    alias depth = config.depth
    # alias mma_shape = Index(64, depth, 16)
    # alias mma_shape = Index(128 if (BM % 128) == 0 else 64, depth, 16)
    # MMA_M here is defined as per-warp
    # alias MMA_M = 64
    alias MMA_M = 128 if (BM % 128) == 0 else 64
    alias MMA_N0 = BN
    alias MMA_N1 = depth
    alias MMA_K = 16
    # alias WM = BM // num_consumer_warps
    # alias WN = BN
    # alias num_m_mmas = BM // MMA_M  # WM // MMA_M
    # mmas are now handled separately from in-register processing
    # in-register processing is divided up by warps, mmas are not
    alias num_row_fragments = num_consumer_threads // 128
    constrained[(32 % num_row_fragments) == 0]()
    alias row_fragment_size = min(32 // num_row_fragments, BM // 4)
    constrained[num_row_fragments * row_fragment_size <= 32]()
    alias WM = row_fragment_size
    # if we have BM = 128, then we have
    # a 16x(BN//8) grid of 8x4<1x2>
    # this gives us 16 blocks to partition among rows.
    # Because we can load a minimum of 16 lanes at a time,
    # we prefer at least 2x blocks per warp, meaning we
    # can divide up to 8 ways.
    alias num_m_blocks_per_warp = BM // (16 * num_consumer_warps)
    # before we had num_m_mmas * MMA_M = BM
    # now, we have num_m_blocks_per_warp * 16*num_consumer_warps == BM
    # num_m_blocks_per_warp is like `num_m_mmas`, but for non-mma consumers.
    constrained[num_m_blocks_per_warp * 16 == WM]()
    #
    # The following constraint is effectively equivalent to
    # BM == 128 or BM == 64
    # If 32 // num_row_fragments is smaller, we have
    # 32*128*num_consumer_warps // num_consumer_threads == BM
    # 128*num_consumer_threads // num_consumer_threads == BM
    # 128 == BM
    # Or if BM // 4 is smaller, we have
    # BM // 4 * num_consumer_warps == BM
    # num_consumer_threads*BM // (4 * 32) == BM
    # num_consumer_threads == 128
    # 32*128 // BM == num_consumer_threads
    constrained[WM * num_consumer_warps == BM]()
    # The above should also be true because:
    # num_consumer_warps = BM // (16 * num_m_blocks_per_warp)
    # -> BM // WM = BM // (16 * num_m_blocks_per_warp)
    # -> WM = (16 * num_m_blocks_per_warp)
    alias num_m_mmas = 1
    alias num_n_mmas = 1
    alias num_k_mmas = BK // MMA_K
    # alias num_warps_m = BM // WM  # 4 * num_consumer
    # alias num_warps_n = BN // WN  # 1
    alias num_heads = config.num_heads
    # num_consumer_threads ignores the producers
    # actual number of threads is num_consumer_threads + 128
    alias pipeline_stages = Int(config.num_pipeline_stages)
    var tid: UInt32 = thread_idx.x
    # warp group idx concept is still useful for sm100
    # because sets of 4 warps access tmem;
    # warp group idx gives index into sets of 16 lanes
    var warp_group_idx: UInt32 = warp.broadcast(tid // 128)
    # warp_group_tid = tid % 128
    alias q_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias k_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias v_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias wgmma_0_t = SM100TensorAccumulatorSS[
        config.type,
        accum_type,
        MMA_M=MMA_M,  # 128
        MMA_N=MMA_N0,  # 64
        BM=BM,  # 128
        BN=BN,  # 128
        BK=BK,  # depth # 256
        num_consumer_threads=num_consumer_threads,
        swizzle_a=q_swizzle,
        swizzle_b=k_swizzle,
        transpose_b=True,
    ]
    # Second WGMMA is a
    # BM x BN tile of p_frag @ BN x depth tile of V
    alias wgmma_1_t = SM100TensorAccumulatorTS[
        config.type,
        accum_type,
        MMA_M=MMA_M,
        MMA_N=MMA_N1,  # depth
        BM=BM,
        BN=MMA_N1,  # depth
        BK=BN,
        num_consumer_threads=num_consumer_threads,
        swizzle_b=v_swizzle,
        transpose_b=False,
    ]

    # var warp_x: UInt32 = warp_id % num_warps_n

    # first wgmma is BM x BK @ BK x BN
    alias q_smem_layout = tile_layout_k_major[
        DType.bfloat16, BM, BK, swizzle_mode=q_swizzle
    ]()
    alias k_smem_layout = tile_layout_k_major[
        DType.bfloat16, BN, BK, swizzle_mode=k_swizzle
    ]()
    # second wgmma is BM x BN @ BN x BK
    alias v_smem_layout = tile_layout_mn_major[
        DType.bfloat16, BN, BK, swizzle_mode=v_swizzle
    ]()

    # The entire query block (BM x depth) is tiled in shared memory.
    alias q_smem_size = config.q_smem_size(True, False)
    q_smem = rebind[
        UnsafePointer[
            Scalar[q_type], address_space = AddressSpace.SHARED, alignment=16
        ]
    ](
        external_memory[
            Scalar[q_type],
            address_space = AddressSpace.SHARED,
            alignment=16,
            name="mha_dynamic_shared_memory",
        ]()
    )
    q_smem_iter = LayoutTensorIter[
        q_type,
        q_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment = q_smem.alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    q_smem_layout,
                    MutableAnyOrigin,
                    address_space = AddressSpace.SHARED,
                    alignment = q_smem.alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )
    # We have `num_pipeline_stages` instances of each
    alias kv_smem_size = config.kv_smem_size(True)
    kv_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()

    # var head_idx: UInt32 = block_idx.y
    # var q_tile_idx: UInt32 = block_idx.x

    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    alias accum_type = get_accum_type[q_type]()
    # p_frag_size is 2 * WM//8 * MMA_N//8
    # that is, we have a (WM//8) x (MMA_N//8) grid of 8x4<1x2> blocks
    # Each such block has 2 elements.
    alias p_frag_size = BM * MMA_N0 // (
        num_consumer_threads * num_m_blocks_per_warp
    )
    alias o_frag_size = BM * MMA_N1 // (
        num_consumer_threads * num_m_blocks_per_warp
    )
    constrained[p_frag_size == 2 * (WM // 8) * (MMA_N0 // 8)]()
    constrained[o_frag_size == 2 * (WM // 8) * (MMA_N1 // 8)]()
    alias frag_simdwidth = 2
    alias a_frag_size = BM * MMA_K // (
        num_consumer_threads * num_m_blocks_per_warp
    )
    constrained[
        BN * num_k_mmas * a_frag_size == BK * num_n_mmas * p_frag_size
    ]()
    #
    alias frag_ratio = p_frag_size // a_frag_size  # MMA_N // MMA_K

    alias p_reg_tile_layout = Layout.row_major(
        num_m_blocks_per_warp * num_n_mmas, p_frag_size
    )
    alias o_reg_tile_layout = Layout.row_major(
        num_m_blocks_per_warp * num_n_mmas, o_frag_size
    )
    alias num_row_blocks_per_mma = 2
    # a wgmma.m64n32k16 `D` fragment looks like
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
    # alias output_layout = Layout(
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
    alias element_layout = Layout.row_major(1, frag_simdwidth)
    alias vec_output_row_shape = IntTuple(num_row_blocks_per_mma, num_m_mmas)
    alias p_vec_output_layout = Layout(
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
    alias o_vec_output_layout = Layout(
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
    alias num_rows_per_warp = p_vec_output_layout[0].size()
    alias num_cols_p = p_vec_output_layout[1].size()
    alias num_cols_output = o_vec_output_layout[1].size()

    # Rowwise max and sum for online softmax
    alias accum_simd_width = simdwidthof[accum_type]()
    alias row_alignment = alignof[SIMD[accum_type, accum_simd_width]]()
    # Account for group query.
    alias kv_num_heads = num_heads // group

    alias q_num_vecs: Int = BM * BK // simd_size

    alias async_copy_q_layout = Layout.row_major(
        min(num_consumer_threads, q_num_vecs) * simd_size // BK, BK // simd_size
    )

    # var lane_predicate = elect_one_sync() # not needed with async_copy

    alias mma_thread_layout = Layout.row_major(8, 4)

    # actually 16 byte alignment
    produced_mbar_kv = (
        (kv_smem + kv_smem_size)
        .bitcast[SharedMemBarrier]()
        .static_alignment_cast[8]()
    )
    consumed_mbar_kv = produced_mbar_kv + pipeline_stages  # 16
    mma_mbar = consumed_mbar_kv + pipeline_stages  # 16
    wgmma_0 = wgmma_0_t(mma_mbar)
    wgmma_1 = wgmma_1_t(mma_mbar + 2)
    ptr_tmem_addr = (mma_mbar + 3).bitcast[UInt32]()  # 8

    alias USE_TMA = False
    # https://github.com/Dao-AILab/flash-attention/blob/3b5047d2ce742848f45d44b143d511f211eba2d2/hopper/flash_fwd_kernel_sm90.h#L81-L82
    alias num_producer_regs = 56 if num_consumer_warps == 4 else (
        (24 if USE_TMA else 56) if num_consumer_warps == 8 else 32
    )
    alias num_consumer_regs = 256 if num_consumer_warps == 4 else (
        (240 if USE_TMA else 224) if num_consumer_warps == 8 else 160
    )
    # alias num_producer_regs = 56
    # alias num_consumer_regs = 224

    # constructing calls barrier() if static
    var tile_summary = MHATileSummary(
        batch_size,
        ceildiv(max_seq_len.as_uint32(), BM) * partition.num_partitions(),
        valid_length,
        max_seq_len.as_uint32(),
    )
    var state: MHATileState = scheduler.initial_state(
        ptr_tmem_addr + 2, tile_summary
    )

    # returns `true` if we are done
    @parameter
    @always_inline
    fn advance[
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](pipeline_idx: UInt32) -> OptionalReg[SeqInfo]:
        return scheduler.advance[ragged, producer, sync](
            tile_summary, state, pipeline_idx
        )

    # The persistent kernels limit the grid size.
    # initial_seq_info = scheduler.unsafe_get_current_work_info(tile_summary, state)

    initial_seq_info = scheduler.unsafe_seq_info[ragged](tile_summary, state)
    constrained[not scheduler_t.may_advance]()

    @parameter
    if not decoding:
        if not initial_seq_info.is_valid():
            return

    if tid == 0:

        @parameter
        for i in range(pipeline_stages):
            # until we can use TMA, we need 128 producers working on async copies
            produced_mbar_kv[i].init(128)
            consumed_mbar_kv[i].init(num_consumer_threads)

    alias position_t = MHAPosition[BM, BN, depth, num_heads, group, decoding]

    @parameter
    @always_inline
    fn get_position(seq_info: SeqInfo) -> position_t:
        return _get_position[config, group, ragged, _is_cache_length_accurate](
            seq_info,
            k,
            max_seq_len,
            num_keys_arg,
            valid_length,
            kv_input_row_offsets,
        )

    var position: position_t = get_position(initial_seq_info)

    q_pipeline_state = PipelineState[2]()

    barrier()
    # For intra-warp overlap, we initiate wgmmas as
    # Q @ K_0, Q @ K_1, P_0 @ V_0, Q @ K_2, P_1 @ V_1, ...
    # ..., Q @ K_{N-1}, P_{N-2} @ V_{N-2}, P_{N-1} @ V_{N-1}
    #
    # Due to this, we can overlap wgmmas and softmax calculations.
    if warp_group_idx == 0:
        # producer
        warpgroup_reg_dealloc[num_producer_regs]()
        write_pipeline_states = PipelineState[pipeline_stages]()

        @parameter
        if partition_t.do_partition:
            start, end = position.get_start_and_end_for_partitions[BN=BN](
                partition
            )
            if start >= end:
                return

        # note that Q does not wait or arrive...
        # it assumes you can use K's
        @parameter
        @always_inline("nodebug")
        fn produce_q(
            position: position_t,
            pipeline_idx: UInt32,
        ):
            # Query global memory iterator
            q_gmem_block = position.q_out_gmem_tensor(q_ptr_arg)
            q_smem_sub = q_smem_iter.next_unsafe(Int(pipeline_idx))[]

            # these copies get committed with the first `K`
            cp_async_k_major(q_smem_sub, q_gmem_block)

        alias kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(kv_num_heads * depth), 1),
        )

        @parameter
        @always_inline("nodebug")
        fn k_tile(
            idx: UInt32,
            out k_smem: LayoutTensor[
                kv_t.dtype,
                k_smem_layout,
                MutableAnyOrigin,
                address_space = AddressSpace.SHARED,
                layout_int_type = DType.int32,
                linear_idx_type = DType.int32,
            ],
        ):
            alias sz = BN * depth
            k_smem = __type_of(k_smem)(kv_smem + sz * idx)

        @parameter
        @always_inline("nodebug")
        fn v_tile(
            idx: UInt32,
            out v_smem: LayoutTensor[
                kv_t.dtype,
                v_smem_layout,
                MutableAnyOrigin,
                address_space = AddressSpace.SHARED,
                layout_int_type = DType.int32,
                linear_idx_type = DType.int32,
            ],
        ):
            alias sz = BN * depth
            v_smem = __type_of(v_smem)(
                rebind[
                    UnsafePointer[
                        Scalar[v_type], address_space = AddressSpace.SHARED
                    ]
                ](kv_smem)
                + sz * idx
            )

        @parameter
        @always_inline("nodebug")
        fn produce_k[
            wait: Bool
        ](
            mut state: PipelineState[pipeline_stages],
            kv_tile_start_row: UInt32,
            position: position_t,
        ):
            var write_idx: UInt32 = state.index()
            var write_phase: UInt32 = state.phase()
            _produce[kv_num_heads, axis=1, wait=wait](
                write_idx,
                write_phase,
                kv_tile_start_row,
                position,
                consumed_mbar_kv,
                produced_mbar_kv,
                k,
                k_tile(write_idx),
            )
            state.step()

        @parameter
        @always_inline("nodebug")
        fn produce_v[
            wait: Bool
        ](
            mut state: PipelineState[pipeline_stages],
            kv_tile_start_row: UInt32,
            position: position_t,
        ):
            var write_idx: UInt32 = state.index()
            var write_phase: UInt32 = state.phase()
            _produce[kv_num_heads, axis=0, wait=wait](
                write_idx,
                write_phase,
                kv_tile_start_row,
                position,
                consumed_mbar_kv,
                produced_mbar_kv,
                v,
                v_tile(write_idx),
            )
            state.step()

        produce_q(position, q_pipeline_state.index())

        start, end = position.get_start_and_end_for_partitions[BN=BN](partition)
        var kv_tile_start_row: UInt32 = start

        while (
            position.mask_status(mask, kv_tile_start_row)
            == TileMaskStatus.FULL_MASK
        ):
            kv_tile_start_row += BN

        produce_k[False](write_pipeline_states, kv_tile_start_row, position)

        var kv_tile_start_row_prev: UInt32 = kv_tile_start_row

        # wait to flip phase, but only bother after producing
        # there isn't any memory we can throttle
        # the order of the consumer's arrivals determines the
        # order of the producer's waits.
        # few_keys = num_keys <= BN

        # Process work with the tile size until there's not enough remaining work
        # to fit in a tile.
        # Production order:
        # Preheader: Q0, K0
        # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
        # Exit: V{-1}
        while True:
            # this loops over num_keys
            kv_tile_start_row += BN
            if kv_tile_start_row >= end:
                break

            if (
                position.mask_status(mask, kv_tile_start_row)
                == TileMaskStatus.FULL_MASK
            ):
                continue
            produce_k[True](
                write_pipeline_states,
                kv_tile_start_row,
                position,
            )
            produce_v[True](
                write_pipeline_states,
                kv_tile_start_row_prev,
                position,
            )
            # cache old
            kv_tile_start_row_prev = kv_tile_start_row

        produce_v[True](
            write_pipeline_states,
            kv_tile_start_row_prev,
            position,
        )

    else:
        warpgroup_reg_alloc[num_consumer_regs]()

        # arrive to unblock the producers
        # TODO: skip this by not waiting on the first set
        @parameter
        for i in range(pipeline_stages):
            _ = consumed_mbar_kv[i].arrive()

        var warp_id: UInt32 = warp.broadcast((tid - 128) // WARP_SIZE)

        # Coordinates of the current warp.
        var elect_one_warp = warp_id == 0

        alias max_tmem_cols = 512
        alias tmem_cols = MMA_N0 + (MMA_N0 // 2) + MMA_N1
        constrained[tmem_cols <= max_tmem_cols]()
        if elect_one_warp:
            # we still use max_tmem_cols, as it must be a power of 2
            tcgen05_alloc[cta_group](ptr_tmem_addr, max_tmem_cols)
            wgmma_0.init()
            wgmma_1.init()

        qk_desc = wgmma_0_t.mma_descriptors(q_smem_iter.ptr, kv_smem)

        q_desc = qk_desc.get_a()
        k_desc = qk_desc.get_b()

        v_desc = wgmma_1_t.b_mma_descriptor(kv_smem)
        var lane: UInt32 = lane_id()
        var elect_one_thread = thread_idx.x == 128

        var warp_y: UInt32 = warp_id  # // num_warps_n

        @parameter
        if num_consumer_threads > 128:
            warp_y = 2 * (warp_y % 4) + (warp_y // 4)
        alias warp_x: UInt32 = 0
        named_barrier[num_consumer_threads]()

        var tmem_addr = ptr_tmem_addr[0]

        constrained[num_consumer_warps == 4 or num_consumer_warps == 8]()

        @parameter
        if num_consumer_warps > 4:
            if warp_group_idx != 1:  # elect_one_warp will be false
                tmem_addr += 1 << 20
        var s_tmem: UInt32 = tmem_addr
        var o_tmem: UInt32 = tmem_addr + MMA_N0
        var p_tmem: UInt32 = tmem_addr + MMA_N0 + MMA_N1

        p_desc = wgmma_1_t.a_mma_descriptor(p_tmem)

        # layout is
        # shape  = (2, num_m_blocks_per_warp) x (2, num_n_mmas)
        # stride = (2, 4*num_n_mmas) x (1, 4)
        p_accumulator = wgmma_0_t.c_t(s_tmem)
        alias p_accumulator_t = __type_of(p_accumulator)
        alias p_reg_t = __type_of(p_accumulator_t._empty_tensor())

        output_accumulator = wgmma_1_t.c_t(o_tmem)
        alias output_accumulator_t = __type_of(output_accumulator)
        alias output_reg_t = __type_of(output_accumulator_t._empty_tensor())

        p_reg_tile = p_accumulator.allocate_register_tile()
        output_reg_tile = output_accumulator.allocate_register_tile()

        @parameter
        @always_inline
        fn vectorize_p_reg_tile(
            out result: LayoutTensor[
                accum_type,
                p_vec_output_layout,
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
                element_layout=element_layout,
            ],
        ):
            result = __type_of(result)(p_reg_tile.ptr)

        @parameter
        @always_inline
        fn vectorize_o_reg_tile(
            out result: LayoutTensor[
                accum_type,
                o_vec_output_layout,
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
                element_layout=element_layout,
            ],
        ):
            result = __type_of(result)(output_reg_tile.ptr)

        rowmax = LayoutTensor[
            __type_of(p_reg_tile).dtype,
            Layout.row_major(num_rows_per_warp),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()
        rowsum = LayoutTensor[
            __type_of(p_reg_tile).dtype,
            Layout.row_major(num_rows_per_warp),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        # Mask global memory iterator.

        mask_warp_row = warp_y * WM
        var scale_log2e: Scalar[accum_type] = (
            scale.cast[accum_type]() if use_score_mod
            or mask_t.apply_log2e_after_mask else scale.cast[accum_type]()
            * log2e
        )

        @parameter
        @always_inline
        fn q_mul_k(read_idx: UInt32, read_phase: UInt32, q_idx: UInt32):
            q = q_desc + Int(BM * BK * sizeof[q_type]() * q_idx)
            k = k_desc + Int(BN * depth * sizeof[k_type]() * read_idx)
            produced_mbar_kv[read_idx].wait(read_phase)

            wgmma_0.mma(
                rebind[wgmma_0_t.a_t](q),
                rebind[wgmma_0_t.b_t](k),
                p_accumulator,
                0,
            )

        @parameter
        @always_inline("nodebug")
        fn p_mul_v(read_idx: UInt32, read_phase: UInt32, scale_c: UInt32):
            v = v_desc + Int(BN * depth * sizeof[v_type]() * read_idx)
            produced_mbar_kv[read_idx].wait(read_phase)
            wgmma_1.mma(
                rebind[wgmma_1_t.a_t](p_desc),
                rebind[wgmma_1_t.b_t](v),
                output_accumulator,
                scale_c,
            )

        @parameter
        @always_inline
        fn wait_for_q_mul_k[wgmma_left_in_flight: Int](read_idx: UInt32):
            wgmma_0.wait_group[wgmma_left_in_flight]()  # P is available
            _ = consumed_mbar_kv[read_idx].arrive()
            p_accumulator.copy_to(p_reg_tile)

        @parameter
        @always_inline
        fn wait_for_p_mul_v(read_idx: UInt32):
            wgmma_1.wait_group[0]()  # output is available
            _ = consumed_mbar_kv[read_idx].arrive()
            output_accumulator.copy_to(output_reg_tile)

        @parameter
        @always_inline
        fn apply_mask(
            position: position_t,
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
        fn scale_output(correction: __type_of(rowmax)):
            vout = vectorize_o_reg_tile()

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
            old_rowsum: __type_of(rowsum), new_rowsum: __type_of(rowsum)
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
            position: position_t,
            q_idx: UInt32,
            rowsum_inv: __type_of(rowsum),
        ):
            vout = vectorize_o_reg_tile()

            # Apply softmax denumerator.
            @parameter
            for row in range(num_rows_per_warp):
                rs_inv = vout.element_type(rowsum_inv[row][0])

                @parameter
                for col in range(num_cols_output):
                    vout[row, col] = vout[row, col] * rs_inv

            var output_ptr: UnsafePointer[Scalar[output_type]] = output_ptr_arg

            @parameter
            if decoding and partition_t.do_partition:
                output_ptr = output_ptr.offset(
                    depth * num_heads * batch_size * position.prompt_offset
                )
            output_gmem_tile = position.q_out_gmem_tensor(output_ptr)

            # Write to global memory.
            constrained[
                output_type.is_half_float(), "we don't support Float32 output"
            ]()
            constrained[sizeof[q_type]() == sizeof[output_type]()]()
            alias swizzle = make_swizzle[
                num_rows = WM // 2, row_size=BN, access_size=8
            ]()
            # Reuse a_smem for c tile in smem
            alias q_tile_size: UInt32 = q_smem_size // 2
            accum_smem_tile = LayoutTensor[
                output_type,
                Layout.row_major(BM, depth),
                address_space = AddressSpace.SHARED,
            ]((q_smem + q_idx * q_tile_size).bitcast[Scalar[output_type]]())
            accum_smem_warp_tile = accum_smem_tile.tile[WM, BN](
                Int(warp_y), Int(warp_x)
            )

            # ensure all threads have finished reading `q_smem`
            named_barrier[num_consumer_threads]()
            copy_local_to_shared[
                thread_layout=mma_thread_layout, swizzle=swizzle
            ](
                accum_smem_warp_tile.vectorize[1, 2](),
                output_accumulator_t.rows_of_frags(output_reg_tile)
                .vectorize[1, 2]()
                .transpose(),
            )
            # Guard writing to shared memory.
            named_barrier[num_consumer_threads]()
            # Vectorized copy from shared to global memory, during which every 2 FP32
            # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
            # vector and stored using 16B store instruction.
            copy_sram_to_dram[
                thread_layout = Layout.row_major(
                    num_consumer_threads * simd_size // depth,
                    depth // simd_size,
                ),
                swizzle=swizzle,
            ](
                output_gmem_tile.vectorize[1, simd_size](),
                accum_smem_tile.vectorize[1, simd_size](),
            )

        start, end = position.get_start_and_end_for_partitions[BN=BN](partition)

        @parameter
        if (
            decoding and partition_t.do_partition
        ):  # we may have an empty partition
            if start >= end:
                if thread_idx.x % 4 == 0 and thread_idx.x < 4 * group + 128:
                    exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                        partition, batch_size
                    )
                    var q_head_idx = position.head_idx * group + lane // 4
                    exp_sum_ptr[q_head_idx] = Scalar[partition_t.accum_dtype](0)
                    qk_max_ptr[q_head_idx] = min_or_neg_inf[
                        partition_t.accum_dtype
                    ]()

                if elect_one_warp:
                    tcgen05_release_allocation_lock[cta_group]()
                    tcgen05_dealloc[cta_group](tmem_addr, max_tmem_cols)
                write_output(position, q_pipeline_state.index(), rowsum)
                return
        var kv_tile_start_row: UInt32 = start
        var mask_status: TileMaskStatus
        while True:
            mask_status = position.mask_status(mask, kv_tile_start_row)
            if mask_status != TileMaskStatus.FULL_MASK:
                break
            kv_tile_start_row += BN

        read_pipeline_states = PipelineState[pipeline_stages]()
        # q_mul_k must wait on fetching q and k
        # therefore, we find `kv_tile_start_row` first.
        var read_idx_q: UInt32 = read_pipeline_states.index()
        q_mul_k(
            read_idx_q,
            read_pipeline_states.phase(),
            q_pipeline_state.index(),
        )
        read_pipeline_states.step()
        wait_for_q_mul_k[0](read_idx_q)
        apply_mask(
            position,
            mask_status,
            kv_tile_start_row,
        )
        rowmax.copy_from(
            _rowmax_online_softmax[1, mma_thread_layout, use_exp2=True](
                vectorize_p_reg_tile(), rowmax, init_rowmax=True
            )
        )
        constrained[
            p_vec_output_layout.size() > 0,
            "layout: " + String(p_vec_output_layout),
        ]()
        rowsum.copy_from(_rowsum[mma_thread_layout](vectorize_p_reg_tile()))

        var q_idx_old: UInt32 = q_pipeline_state.index()
        var q_phase_old: UInt32 = q_pipeline_state.phase()
        var output_scale: UInt32 = 0
        # Consumption order:
        # Preheader: Q0, K0
        # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
        # Exit: V{-1}
        while True:
            # this loops over num_keys
            kv_tile_start_row += BN
            if kv_tile_start_row >= end:
                break
            # this loops over num_keys
            mask_status = position.mask_status(mask, kv_tile_start_row)
            if mask_status == TileMaskStatus.FULL_MASK:
                continue
            # copy new pfrag, used by `p_mul_v` on next iter
            p_desc.copy_from(p_accumulator_t.rows_of_frags(p_reg_tile))

            # new pipeline states
            var read_idx_q: UInt32 = read_pipeline_states.index()
            # start wgmmas
            q_mul_k(
                read_idx_q,
                read_pipeline_states.phase(),
                q_pipeline_state.index(),
            )  # can't rw `p_reg_tile`
            read_pipeline_states.step()
            var read_idx_v: UInt32 = read_pipeline_states.index()
            p_mul_v(
                read_idx_v, read_pipeline_states.phase(), output_scale
            )  # can't rw output or pfrag
            output_scale = 1
            read_pipeline_states.step()
            p_reg_tensor = wait_for_q_mul_k[1](
                read_idx_q
            )  # can rw `p_reg_tile`

            apply_mask(
                position,
                mask_status,
                kv_tile_start_row,
            )
            score_frag_rowmax = _rowmax_online_softmax[
                1, mma_thread_layout, use_exp2=True
            ](vectorize_p_reg_tile(), rowmax, False)
            score_frag_rowsum = rebind[__type_of(rowsum)](
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
            scale_output(score_frag_rowmax)  # scale output
            output_accumulator.copy_from(output_reg_tile)

        p_desc.copy_from(p_accumulator_t.rows_of_frags(p_reg_tile))

        p_mul_v(
            read_pipeline_states.index(),
            read_pipeline_states.phase(),
            output_scale,
        )

        @parameter
        if decoding and partition_t.do_partition:
            if thread_idx.x % 4 == 0 and thread_idx.x < 4 * group + 128:
                exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                    partition, batch_size
                )
                var q_head_idx = position.head_idx * group + lane // 4
                exp_sum_ptr[q_head_idx] = rebind[
                    Scalar[partition_t.accum_dtype]
                ](rowsum[0])
                qk_max_ptr[q_head_idx] = rebind[
                    Scalar[partition_t.accum_dtype]
                ](rowmax[0])

        @parameter
        for row in range(num_rows_per_warp):
            rowsum[row] = recip(rowsum[row])[0]
        wgmma_1.wait_group()

        output_accumulator.copy_to(output_reg_tile)
        constrained[
            __type_of(output_reg_tile).layout[1].size() > 1,
            "output_reg_tile.layout = "
            + String(__type_of(output_reg_tile).layout)
            + "\n",
        ]()
        if elect_one_warp:
            tcgen05_release_allocation_lock[cta_group]()
            tcgen05_dealloc[cta_group](tmem_addr, max_tmem_cols)
        write_output(position, q_pipeline_state.index(), rowsum)
        # don't arrive

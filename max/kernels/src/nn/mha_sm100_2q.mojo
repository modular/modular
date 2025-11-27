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

from memory import LegacyUnsafePointer as UnsafePointer
from collections import OptionalReg
from math import ceildiv, exp2, recip, align_up, align_down, gcd, iota
from math.constants import log2e
from sys import align_of, simd_width_of, size_of
import gpu.warp as warp
from bit import prev_power_of_two, pop_count
from buffer import NDBuffer
from collections import OptionalReg
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx,
    block_idx,
    warp_id,
)
from gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from gpu.cluster import elect_one_sync
from gpu.host import DeviceContext, FuncAttribute, DeviceBuffer
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory
from gpu.mma import MMAOperandDescriptor
from gpu.mma_sm100 import (
    UMMAInsDescriptor,
    UMMAKind,
    mma_arrive,
    mma,
)
from gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from gpu.memory import fence_async_view_proxy
from gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from gpu.tcgen05 import (
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
from gpu.primitives.warp import _vote_nvidia_helper
from layout.int_tuple import IntTuple, UNKNOWN_VALUE
from layout.layout import Layout, blocked_product
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
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMANestedTensorTile,
    RaggedTensorMap,
)
from logger import Logger
from memory import bitcast
from nn.mha_fa3_utils import (
    get_seq_info,
    MHAPosition,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    output_reg_to_smem_st_matrix,
    Pack,
    PositionSummary,
    produce,
    q_out_tma,
    QTMATile,
)
from nn.mha_mask import MHAMask, TileMaskStatus, MASK_VALUE
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
from utils.index import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple
from linalg.arch.sm100.mma import smem_descriptor

from sys import size_of, bit_width_of
from sys._assembly import inlined_assembly
from sys.info import _has_blackwell_tcgen05

comptime logger = Logger()

comptime LocalTensor[
    dtype: DType, layout: Layout, element_layout: Layout = Layout(1, 1)
] = LayoutTensor[
    dtype,
    layout,
    MutAnyOrigin,
    address_space = AddressSpace.LOCAL,
    element_layout=element_layout,
]
comptime SharedMemTensor[dtype: DType, layout: Layout] = LayoutTensor[
    dtype,
    layout,
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
    layout_int_type = DType.int32,
    linear_idx_type = DType.int32,
    alignment=128,
]
comptime SharedMemPointer[type: AnyType] = UnsafePointer[
    type, address_space = AddressSpace.SHARED
]
comptime MBarType = SharedMemPointer[SharedMemBarrier]


fn extract_power_of_two(N: Int, i: Int) -> Int:
    pt = prev_power_of_two(N)
    rem = N
    for _ in range(i):
        rem -= pt
        pt = prev_power_of_two(rem)
    return pt


fn cumulative_power_of_two(N: Int, i: Int) -> Int:
    acc = 0
    rem = N
    for _ in range(i):
        pt = prev_power_of_two(rem)
        acc += pt
        rem -= pt
    return acc


# Final call is with `pow_two == 0` (which isn't a power of 2)
# to enable use of this function with pipelining.
@always_inline("nodebug")
fn break_into_powers_of_two[
    origins: OriginSet, //,
    func: fn[pow_two: Int, offset: Int] () capturing [origins] -> None,
    N: Int,
    *,
    max_value: Int = 128,
]():
    comptime power_of_two = prev_power_of_two(min(max_value, N))

    @parameter
    for offset in range(0, N, power_of_two):
        comptime iter_size = min(N - offset, power_of_two)

        @parameter
        if iter_size == power_of_two:
            func[power_of_two, offset]()
        else:

            @parameter
            for j in range(pop_count(iter_size)):
                comptime pow_two = extract_power_of_two(iter_size, j)
                comptime coffset = offset + cumulative_power_of_two(
                    iter_size, j
                )
                func[pow_two, coffset]()
    # final call for possible pipeline cleanup
    func[0, N]()


@register_passable("trivial")
struct STMatrixLayout[
    BM: Int,
    BN: Int,
    *,
    num_threads: Int,
    accum_type_size: Int,
]:
    """
    Layout for using `st_matrix` for writing the final accumulator to smem.
    """

    # We have a BM x BN tile
    #
    # The st_matrix layout wants to map it to threads in 16x8 blocks
    # shape  (2,8), (2,4)
    # stride (0,4), (0,1)
    # Layout = ((2,8),(2,4)):((0,4),(0,1))
    # Where `0` stride indicates that the same thread is repeated across these.
    # We also need a layout for this local memory, which we define here.

    # look at figure 108 https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-stmatrix-fragments

    # That first `2` is
    comptime num_row_blocks_per_mma = 2
    # The second `2` is
    comptime frag_simdwidth: Int = 2

    comptime thread_cols = 4
    # When using tcgen05 ld/st we must repeat across all columns:
    comptime repeat = Self.BN // (Self.thread_cols * Self.frag_simdwidth)

    comptime num_warpgroups = ceildiv(Self.num_threads, 128)
    # 2 = 32 // 16, i.e. we need to load 2 sets of 16
    comptime num_m_tiles_total = ceildiv(2 * Self.BM, 128)
    comptime num_m_tiles = Self.num_m_tiles_total // Self.num_warpgroups

    comptime frag_size = Self.BN * Self.num_row_blocks_per_mma // Self.thread_cols

    # layout of local memory
    # alias local_layout: Layout = Layout(
    #     IntTuple(IntTuple(Self.num_row_blocks_per_mma, Self.num_m_tiles),IntTuple(Self.frag_simdwidth, Self.repeat)),
    #     IntTuple(IntTuple(Self.frag_simdwidth, Self.frag_size),IntTuple(1, Self.num_row_blocks_per_mma*Self.frag_simdwidth)),
    # )
    comptime elements_per_repeat = Self.frag_simdwidth * Self.num_row_blocks_per_mma

    comptime vec_local_layout: Layout = Layout(
        IntTuple(
            IntTuple(Self.num_row_blocks_per_mma, Self.num_m_tiles),
            IntTuple(Self.repeat),
        ),
        IntTuple(
            IntTuple(
                Self.frag_simdwidth, Self.frag_size
            ),  # distance between vertical m tiles and local fragments
            IntTuple(
                Self.num_row_blocks_per_mma * Self.frag_simdwidth
            ),  # distance between bn repeats
        ),
    )
    comptime element_layout: Layout = Layout.row_major(1, Self.frag_simdwidth)
    comptime TensorType[dtype: DType] = LocalTensor[
        dtype, Self.vec_local_layout, Self.element_layout
    ]
    comptime row_of_frags_layout: Layout = Layout.row_major(
        Self.num_m_tiles, Self.frag_size
    )

    comptime bits_per_byte = 8
    comptime bits = Self.bits_per_byte * Self.frag_simdwidth * Self.thread_cols * Self.accum_type_size

    @always_inline
    fn __init__(out self):
        pass


@register_passable("trivial")
struct STMatrixOffsets[
    BM: Int,
    BN: Int,
    *,
    num_threads: Int,
    accum_type_size: Int,
    curr_repeat: Int,
    cumulative_repeat: Int,
    m_mma: Int,
]:
    comptime STLayout = STMatrixLayout[
        Self.BM,
        Self.BN,
        num_threads = Self.num_threads,
        accum_type_size = Self.accum_type_size,
    ]

    comptime tmem_col_offset = Self.cumulative_repeat * Self.STLayout.frag_simdwidth * Self.STLayout.thread_cols
    comptime tmem_row_offset = 16 * Self.m_mma
    comptime tmem_offset = (Self.tmem_row_offset << 16) + Self.tmem_col_offset
    comptime b32_per_repeat = Self.STLayout.elements_per_repeat * Self.accum_type_size // 4
    comptime local_frag_size_b32 = Self.curr_repeat * Self.b32_per_repeat
    comptime ptr_offset = Self.b32_per_repeat * (
        Self.STLayout.repeat * Self.m_mma + Self.cumulative_repeat
    )

    @always_inline
    fn __init__(out self):
        pass


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


@register_passable("trivial")
struct TMemTile[
    dtype_: DType,
    BM: Int,
    BN: Int,
]:
    comptime dtype: DType = Self.dtype_
    comptime dtype_size = size_of[Self.dtype]()
    # alias layout_t = STMatrixLayout[
    #     BM, BN, num_threads= num_threads
    # ]
    # alias vec_output_layout = Self.layout_t.vec_local_layout
    # alias element_layout = Self.layout_t.element_layout
    comptime num_m_tiles = Self.BM // 64

    var tmem_addr: UInt32

    @always_inline
    fn __init__(out self, tmem_addr: UInt32):
        self.tmem_addr = tmem_addr

    @always_inline
    fn __getitem__(self, i: UInt32) -> Self:
        return {self.tmem_addr + i * Self.BN}

    @always_inline
    fn offset[m_mma: Int, n_mma: Int](self) -> UInt32:
        @parameter
        if m_mma == 0 and n_mma == 0:
            return self.tmem_addr
        else:
            comptime linear = _tmem_offset[
                Self.dtype, MMA_N = Self.BN, m_mma=m_mma, n_mma=n_mma
            ]()

            return self.tmem_addr + linear

    @staticmethod
    @always_inline
    fn allocate_register_tile[
        *, num_threads: Int
    ](
        out res: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_type_size = Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        res = type_of(res).stack_allocation()

    @always_inline
    fn store_async[
        *, num_threads: Int
    ](
        self,
        src: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_type_size = Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        constrained[Self.dtype_size <= 4]()
        ptr = src.ptr.bitcast[UInt32]()
        comptime st_mat_layout = STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_type_size = Self.dtype_size,
        ]
        constrained[st_mat_layout.bits == 128 or st_mat_layout.bits == 256]()

        @parameter
        @always_inline
        fn store_fn[pow_two: Int, offset: Int]():
            # pow_two is current repeat, offset total so far
            @parameter
            if pow_two > 0:

                @parameter
                for m_mma in range(st_mat_layout.num_m_tiles):
                    comptime offsets = STMatrixOffsets[
                        Self.BM,
                        Self.BN,
                        num_threads=num_threads,
                        accum_type_size = Self.dtype_size,
                        curr_repeat=pow_two,
                        cumulative_repeat=offset,
                        m_mma=m_mma,
                    ]()
                    tmem = self.tmem_addr + offsets.tmem_offset
                    frag = ptr.load[width = offsets.local_frag_size_b32](
                        offsets.ptr_offset
                    )
                    # 16 x 256b results in repeated 8x4 matrix of <1,2> vector pattern
                    tcgen05_st[
                        datapaths=16,  # first dimension of the shape
                        bits = st_mat_layout.bits,  # second dimension of the shape
                        repeat=pow_two,
                        pack=False,
                    ](tmem, frag)

        comptime max_value = 64 if st_mat_layout.bits == 128 else 32
        break_into_powers_of_two[
            func=store_fn, N = st_mat_layout.repeat, max_value=max_value
        ]()

    @always_inline
    fn store[
        *, num_threads: Int
    ](
        self,
        src: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_type_size = Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        self.store_async[num_threads=num_threads](src)
        tcgen05_store_wait()
        named_barrier[num_threads]()

    @always_inline
    fn load_async_with_st_matrix_layout[
        *, num_threads: Int
    ](
        self,
        out dst: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_type_size = Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        constrained[
            Self.dtype_size <= 4,
            "Loading for st matrix requires elements to be <= 4 bytes.",
        ]()
        comptime st_mat_layout = STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_type_size = Self.dtype_size,
        ]()
        constrained[
            (st_mat_layout.num_m_tiles == 1)
            or (st_mat_layout.num_m_tiles == 2),
            "Only 1 or 2 m tiles are supported, but"
            " st_mat_layout.num_m_tiles == "
            + String(st_mat_layout.num_m_tiles),
        ]()
        comptime repeat = st_mat_layout.repeat
        comptime frag_size_b32 = st_mat_layout.frag_size * Self.dtype_size // 4

        dst = type_of(dst).stack_allocation()
        comptime load_dtype = DType.uint32
        # alias load_dtype = Self.dtype if Self.dtype_size == 4 else DType.uint32
        var ptr: UnsafePointer[
            Scalar[load_dtype], address_space = AddressSpace.LOCAL
        ]

        ptr = rebind[type_of(ptr)](dst.ptr)

        @parameter
        @always_inline
        fn load_fn[pow_two: Int, offset: Int]():
            constrained[pow_two + offset <= repeat]()

            @parameter
            if pow_two > 0:

                @parameter
                for m_mma in range(st_mat_layout.num_m_tiles):
                    comptime offsets = STMatrixOffsets[
                        Self.BM,
                        Self.BN,
                        num_threads=num_threads,
                        accum_type_size = Self.dtype_size,
                        curr_repeat=pow_two,
                        cumulative_repeat=offset,
                        m_mma=m_mma,
                    ]()
                    tmem = self.tmem_addr + offsets.tmem_offset
                    frag = tcgen05_ld[
                        datapaths=16,  # first dimension of the shape
                        bits = st_mat_layout.bits,  # second dimension of the shape
                        repeat=pow_two,
                        dtype=load_dtype,
                        pack=False,
                        width = offsets.local_frag_size_b32,
                    ](tmem)
                    ptr.store(offsets.ptr_offset, frag)

        comptime max_value = 64 if st_mat_layout.bits == 128 else 32
        break_into_powers_of_two[func=load_fn, N=repeat, max_value=max_value]()

    @always_inline
    fn load_async(
        self,
        out dst: LocalTensor[Self.dtype, Layout.row_major(Self.BN)],
    ):
        dst = type_of(dst).stack_allocation()
        comptime repeat = Self.dtype_size * Self.BN // 4
        comptime dtype = Self.dtype if Self.dtype_size == 4 else DType.uint32

        @parameter
        @always_inline
        fn load_fn[pow_two: Int, offset: Int]():
            @parameter
            if pow_two > 0:

                @parameter
                if dtype == Self.dtype:
                    frag0 = tcgen05_ld[
                        datapaths=32,  # first dimension of the shape
                        bits=32,  # second dimension of the shape
                        repeat=pow_two,
                        dtype = Self.dtype,
                        pack=False,
                        width=pow_two,
                    ](self.tmem_addr + offset)
                    dst.ptr.store(offset, frag0)
                else:
                    frag1 = tcgen05_ld[
                        datapaths=32,  # first dimension of the shape
                        bits=32,  # second dimension of the shape
                        repeat=pow_two,
                        dtype = DType.uint32,
                        pack=False,
                        width=pow_two,
                    ](self.tmem_addr + offset)
                    dst.ptr.bitcast[UInt32]().store[width=pow_two](
                        offset, frag1
                    )

        break_into_powers_of_two[func=load_fn, N=repeat, max_value=128]()

    @always_inline
    fn store_async[
        src_type: DType
    ](self, src: LocalTensor[src_type, Layout.row_major(Self.BN)]):
        @parameter
        @always_inline
        fn store_fn[pow_two: Int, offset: Int]():
            @parameter
            if pow_two > 0:
                var frag: SIMD[DType.uint32, pow_two * Self.dtype_size // 4]

                @parameter
                if src_type == Self.dtype:
                    frag = src.ptr.bitcast[UInt32]().load[
                        width = pow_two * Self.dtype_size // 4
                    ](offset)
                else:
                    comptime src_offset = offset
                    comptime src_frag = pow_two
                    frag = bitcast[
                        DType.uint32, pow_two * Self.dtype_size // 4
                    ](
                        src.ptr.load[width=src_frag](src_offset).cast[
                            Self.dtype
                        ]()
                    )
                tcgen05_st[
                    datapaths=32,  # first dimension of the shape
                    bits=32,  # second dimension of the shape
                    repeat = pow_two * Self.dtype_size // 4,
                    pack=False,
                ](self.tmem_addr + offset * Self.dtype_size // 4, frag)

        break_into_powers_of_two[func=store_fn, N = Self.BN, max_value=128]()

    @always_inline
    fn store[
        src_type: DType
    ](self, src: LocalTensor[src_type, Layout.row_major(Self.BN)]):
        self.store_async(src)
        tcgen05_store_wait()


@register_passable("trivial")
struct SM100TensorAccumulatorSS[
    operand_type: DType,
    accum_type: DType,
    MMA_M: Int,
    MMA_N: Int,
    BK: Int,
    *,
    swizzle_a: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    cta_group: Int = 1,
    num_stages: Int = 1,
]:
    # This performs C = A @ B
    # where A is BM x BK and B is BN x BK if k major, else BK x BN.
    # `BK` is broken into `num_stages` and pipelined.
    #
    # The complete multiplication of all stages produces an unweighted
    # score, which is the input of the `softmax`.
    # The benefit of setting `stages > 1` is that this can hide latency.
    comptime operand_t = Self.operand_type
    comptime operand_size = size_of[Self.operand_t]()
    comptime accum_t = Self.accum_type
    comptime MMA_K = 16
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime swizzle_granularity = max(
        Self.swizzle_a.bytes(), Self.swizzle_b.bytes()
    ) // size_of[Self.operand_t]()
    comptime padded_BK = align_up(Self.BK, Self.swizzle_granularity)
    comptime num_k_blocks = Self.padded_BK // Self.MMA_K
    comptime num_k_blocks_per_stage = Self.num_k_blocks // Self.num_stages

    comptime a_layout = tile_layout_k_major[
        Self.operand_t, align_up(Self.MMA_M, 8), Self.padded_BK, Self.swizzle_a
    ]()
    comptime b_layout = tile_layout_k_major[
        Self.operand_t, Self.MMA_N, Self.padded_BK, Self.swizzle_b
    ]() if Self.transpose_b else tile_layout_mn_major[
        Self.operand_t, Self.MMA_N, Self.padded_BK, Self.swizzle_b
    ]()

    comptime idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype = DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b = Self.transpose_b,
    ]()

    comptime AType = MMASmemDescriptorPair
    comptime BType = MMASmemDescriptorPair
    comptime CType = TMemTile[Self.accum_t, Self.MMA_M, Self.MMA_N]

    @staticmethod
    @always_inline
    fn mma[
        *, stage_idx: Int = 0
    ](
        a: Self.AType,
        b: Self.BType,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        constrained[stage_idx == 0]()
        bulk_mma[
            Self.a_layout,
            Self.b_layout,
            num_k_mmas = Self.num_k_mmas,
            operand_size = Self.operand_size,
        ](Self.idesc, a, b, c, c_scale, elect)


@register_passable("trivial")
struct SM100TensorAccumulatorTS[
    operand_type: DType,
    accum_type: DType,
    MMA_M: Int,
    MMA_N: Int,
    BK: Int,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    *,
    transpose_b: Bool = True,
    cta_group: Int = 1,
    num_stages: Int = 1,
    padded_BK: Int = BK,
]:
    comptime operand_t: DType = Self.operand_type
    comptime accum_t: DType = Self.accum_type

    comptime operand_size = size_of[Self.operand_type]()
    comptime swizzle_granularity = Self.swizzle_b.bytes() // Self.operand_size
    # alias MMA_N_padded = align_up(MMA_N, Self.swizzle_granularity)
    # BN here is depth
    comptime b_layout = tile_layout_k_major[
        Self.operand_t, Self.MMA_N, Self.BK, Self.swizzle_b
    ]() if Self.transpose_b else tile_layout_mn_major[
        Self.operand_t, Self.MMA_N, Self.BK, Self.swizzle_b
    ]()

    comptime MMA_K = 16
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime num_k_blocks = Self.padded_BK // Self.MMA_K
    comptime num_k_blocks_per_stage = Self.num_k_blocks // Self.num_stages

    comptime AType = TMemTile[Self.operand_type, Self.MMA_M, Self.BK]
    comptime BType = MMASmemDescriptorPair
    comptime CType = TMemTile[Self.accum_t, Self.MMA_M, Self.MMA_N]

    # B's descriptor contains stride info, so we should be
    # able to use `BN` here instead of `BN_padded`
    comptime idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype = DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b = Self.transpose_b,
    ]()

    @staticmethod
    @always_inline
    fn descriptor_a(a_tmem: UInt32) -> Self.AType:
        return {a_tmem}

    @staticmethod
    @always_inline
    fn mma[
        *, stage_idx: Int = 0
    ](a: UInt32, b: Self.BType, c: UInt32, *, c_scale: UInt32, elect: Int32):
        constrained[stage_idx == 0]()
        bulk_mma[
            Self.b_layout,
            num_k_mmas = Self.num_k_mmas,
            operand_size = Self.operand_size,
        ](Self.idesc, a, b, c, c_scale, elect)


@register_passable("trivial")
struct FA4Config:
    var MMA_M: Int
    var BM: Int
    var BN: Int
    var BK0: Int  # BK for MMA0
    var BK1: Int  # BK for MMA1
    var depth: Int
    var padded_depth: Int
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int
    comptime TMEM_S0: Int = 0
    var TMEM_S1: Int
    var TMEM_O0: Int
    var TMEM_O1: Int
    var TMEM_P0: Int
    var TMEM_P1: Int
    var TMEM_C0: Int
    var TMEM_C1: Int
    var tmem_used: Int
    var num_kv_stages: Int
    var num_mma_stages: Int
    var smem_used: Int
    var dtype_size: Int
    comptime num_threads: Int = 512  # 2x softmax, 1x correction, 1x other
    var split_m: Bool
    var swizzle_mode: TensorMapSwizzle

    comptime MMA_K = 16
    comptime sm100_smem_carveout = B200.shared_memory_per_multiprocessor - 1024
    comptime sm100_tmem_cols = 512
    comptime mbar_size = size_of[DType.int64]()
    comptime num_correction_cols = 1

    @always_inline
    fn num_qo(self) -> Int:
        return 2

    fn __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        depth: Int,
        dtype_size: Int,
        swizzle_mode: TensorMapSwizzle,
        page_size: Int,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.depth = depth
        self.split_m = depth > 128
        if self.split_m:
            self.BM = 128
            self.MMA_M = 64
        else:
            self.BM = 256
            self.MMA_M = 128
        self.dtype_size = dtype_size
        self.swizzle_mode = swizzle_mode
        swizzle_elems = swizzle_mode.bytes() // dtype_size
        self.padded_depth = align_up(depth, swizzle_elems)

        var smem_use = 4  # tmem
        if self.split_m:
            self.BN = min(
                256, align_down(Self.sm100_tmem_cols - depth, Self.MMA_K)
            )
            # TODO : delete this as soon as we define spliting BN across the pages
            if page_size % self.BN != 0:
                self.BN = prev_power_of_two(self.BN)
            self.TMEM_P0 = Self.TMEM_S0
            self.TMEM_O0 = Self.TMEM_S0 + self.BN
            self.TMEM_C0 = Self.TMEM_S0 + self.BN // 2

            self.TMEM_S1 = Self.TMEM_S0 + 16 << 16
            self.TMEM_P1 = self.TMEM_P0 + 16 << 16
            self.TMEM_O1 = self.TMEM_O0 + 16 << 16
            self.TMEM_C1 = self.TMEM_C0 + 16 << 16
            self.tmem_used = self.TMEM_O1 + depth
        else:
            # we use two q and o
            # determine BN via tmem:
            # 2*BN + 2*depth <= 512 -> BN + depth <= 256
            self.BN = min(
                256,
                align_down(
                    (Self.sm100_tmem_cols // 2 - self.padded_depth), Self.MMA_K
                ),
            )
            # TODO : delete this as soon as we define spliting BN across the pages
            if page_size % self.BN != 0:
                self.BN = prev_power_of_two(self.BN)
            self.TMEM_S1 = Self.TMEM_S0 + self.BN
            self.TMEM_P0 = Self.TMEM_S0
            self.TMEM_P1 = self.TMEM_S1
            self.TMEM_C0 = self.TMEM_P0 + self.BN // 2
            self.TMEM_C1 = self.TMEM_P1 + self.BN // 2
            self.TMEM_O0 = self.TMEM_S1 + self.BN
            self.TMEM_O1 = self.TMEM_O0 + self.padded_depth
            self.tmem_used = self.TMEM_O1 + self.padded_depth

        # We have the following resources that need smem barriers:
        # KV: num_kv_stages
        # S: 2
        # C: 2
        # O: 2
        # softmax order: 2
        # q: 1, for Q1 synchronization
        # 4 for `o_pipeline` (2 consumer + 2 producer)
        smem_use += (FA4MiscMBars.size + 4) * Self.mbar_size

        # We use the gcd here to ensure that both
        # depth//swizzle_elems and BN//MMA_K
        # can be evenly divided by the mma stages
        # TODO: Allow setting num_mma_stages > 1 and benchmark
        # self.num_mma_stages = gcd(
        #     self.padded_depth // swizzle_elems, self.BN // Self.MMA_K
        # )
        self.num_mma_stages = 1
        self.BK0 = self.padded_depth // self.num_mma_stages
        self.BK1 = self.BN // self.num_mma_stages
        # smem use is (NOTE: smem uses padded depth):
        # BM*depth*dtype_size + num_kv_stages*(2*mbar_size + BN*depth*dtype_size) <= smem_remaining
        # num_kv_stages <= (smem_remaining - 2*BM*depth*dtype_size) // (2*mbar_size + BN*depth*dtype_size)
        smem_use += self.BM * self.padded_depth * dtype_size
        smem_per_kv = (
            self.BN * self.padded_depth * dtype_size
            + 2 * Self.mbar_size * self.num_mma_stages
        )
        self.num_kv_stages = (
            Self.sm100_smem_carveout - smem_use
        ) // smem_per_kv
        # example values of (num_kv_stages * num_mma_stages)
        # depth= 64: (8 * 1) =  8
        # depth= 80: (3 * 2) =  6
        # depth=128: (5 * 2) = 10
        # depth=256: (1 * 4) =  4
        # The product gives the total number of stages
        smem_use += self.num_kv_stages * smem_per_kv
        self.smem_used = smem_use

    fn supported(self) -> Bool:
        return (
            self.depth >= 64
            and self.BN >= 64
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
        )

    fn use_tmem_for_correction(self) -> Bool:
        # if Self.TMEM_S0 == self.TMEM_P0, then we can place the correction
        # starting at `self.TMEM_P0 + self.BN//2`.
        # Otherwise, we check if we can place it after `O1`.
        return (Self.TMEM_S0 == self.TMEM_P0) or (
            (self.TMEM_O1 + self.depth + Self.num_correction_cols) <= 512
        )

    fn correction_smem_elements(self) -> Int:
        return (
            0 if self.use_tmem_for_correction() else self.BM
            * Self.num_correction_cols
        )

    fn num_active_warps_per_group(self) -> Int:
        return 4

    fn num_active_threads_per_group(self) -> Int:
        return WARP_SIZE * self.num_active_warps_per_group()


fn build_mma_ss(
    kind: String,
    layout_a: Layout,
    layout_b: Layout,
    *,
    operand_size: Int,
    num_k_mmas: Int,
) -> String:
    # Our code tries to extensively re-use registers so that the upper half
    # of the descriptors can be re-used.
    #
    # rda and rdb are the 64-bit smem descriptors.
    # %pj the jump-predicate.
    # %ps the scale-prediate.
    mma = """{
.reg .b64 %rda;
.reg .b64 %rdb;
.reg .s32 %ra;
.reg .s32 %rb;
.reg .pred %pj;
.reg .pred %ps;
setp.eq.s32 %pj, $6, 0;
@%pj bra skip;
"""
    tcgen05_mma = "tcgen05.mma.cta_group::1." + kind
    # prev_offset_a = 0
    # prev_offset_b = 0
    for k in range(num_k_mmas):
        if k == 0:  # set predicate based on c-scale
            mma += "mov.b64 %rda, {$7, $8};\n"
            mma += "mov.b64 %rdb, {$4, $5};\n"
            mma += "setp.ne.b32 %ps, $3, 0;\n"
        else:
            # define rda and rdb
            a_offset = (layout_a(IntTuple(0, 16 * k)) * operand_size) >> 4
            mma += String("add.s32 %ra, $7, ", a_offset, ";\n")
            b_offset = (layout_b(IntTuple(0, 16 * k)) * operand_size) >> 4
            mma += String("add.s32 %rb, $4, ", b_offset, ";\n")
            mma += "mov.b64 %rda, {%ra, $8};\n"
            mma += "mov.b64 %rdb, {%rb, $5};\n"
            if k == 1:  # set predicate to 1
                mma += "setp.ne.b32 %ps, 1, 0;\n"
        mma += tcgen05_mma + " [$0], %rda, %rdb, $2, {$1, $1, $1, $1}, %ps;\n"
    mma += "skip:\n}"
    return mma


fn build_mma_ts(
    kind: String,
    layout_b: Layout,
    *,
    operand_size: Int,
    num_k_mmas: Int,
) -> String:
    # Our code tries to extensively re-use registers so that the upper half
    # of the descriptors can be re-used.
    #
    # rda and rdb are the 64-bit smem descriptors.
    # %pj the jump-predicate.
    # %ps the scale-prediate.
    mma = """{
.reg .b64 %rdb;
.reg .s32 %rb;
.reg .pred %pj;
.reg .pred %ps;
setp.eq.s32 %pj, $6, 0;
@%pj bra skip;
"""
    tcgen05_mma = "tcgen05.mma.cta_group::1." + kind
    # prev_offset_a = 0
    # prev_offset_b = 0
    for k in range(num_k_mmas):
        if k == 0:  # set predicate based on c-scale
            mma += "mov.b64 %rdb, {$4, $5};\n"
            mma += "setp.ne.b32 %ps, $3, 0;\n"
        else:
            # define rda and rdb
            b_offset = (layout_b(IntTuple(0, 16 * k)) * operand_size) >> 4
            mma += String("add.s32 %rb, $4, ", b_offset, ";\n")
            mma += "mov.b64 %rdb, {%rb, $5};\n"
            if k == 1:  # set predicate to 1
                mma += "setp.ne.b32 %ps, 1, 0;\n"
        mma += String(
            tcgen05_mma,
            " [$0], [$",
            7 + k,
            "], %rdb, $2, {$1, $1, $1, $1}, %ps;\n",
        )
    mma += "skip:\n}"
    return mma


@always_inline
fn bulk_mma[
    kind: UMMAKind, //,
    layout_a: Layout,
    layout_b: Layout,
    *,
    num_k_mmas: Int,
    operand_size: Int,
](
    idesc: UMMAInsDescriptor[kind],
    a: MMASmemDescriptorPair,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
):
    comptime mma_string = build_mma_ss(
        String(kind),
        layout_a,
        layout_b,
        operand_size=operand_size,
        num_k_mmas=num_k_mmas,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r"](
        c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a.lo, a.hi
    )


@always_inline
fn bulk_mma[
    kind: UMMAKind, //,
    layout_b: Layout,
    *,
    num_k_mmas: Int,
    operand_size: Int,
](
    idesc: UMMAInsDescriptor[kind],
    a: UInt32,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
):
    constrained[num_k_mmas >= 1 and num_k_mmas <= 16]()
    comptime mma_string = build_mma_ts(
        String(kind),
        layout_b,
        operand_size=operand_size,
        num_k_mmas=num_k_mmas,
    )

    comptime constraints = "r,r,r,r,r,r,r" + ",r" * num_k_mmas
    comptime x = 4 * operand_size
    # fmt: off
    @parameter
    if num_k_mmas == 1:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a
        )
    elif num_k_mmas == 2:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x
        )
    elif num_k_mmas == 3:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x
        )
    elif num_k_mmas == 4:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x
        )
    elif num_k_mmas == 5:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x
        )
    elif num_k_mmas == 6:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x,a+5*x
        )
    elif num_k_mmas == 7:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x,a+5*x,a+6*x
        )
    elif num_k_mmas == 8:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x,a+5*x,a+6*x,a+7*x
        )
    elif num_k_mmas == 9:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x,a+5*x,a+6*x,a+7*x,a+8*x
        )
    elif num_k_mmas == 10:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x,a+5*x,a+6*x,a+7*x,a+8*x,a+9*x
        )
    elif num_k_mmas == 11:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x,a+5*x,a+6*x,a+7*x,a+8*x,a+9*x,a+10*x
        )
    elif num_k_mmas == 12:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x,a+5*x,a+6*x,a+7*x,a+8*x,a+9*x,a+10*x,a+11*x
        )
    elif num_k_mmas == 13:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x,a+5*x,a+6*x,a+7*x,a+8*x,a+9*x,a+10*x,a+11*x,a+12*x
        )
    elif num_k_mmas == 14:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x,a+5*x,a+6*x,a+7*x,a+8*x,a+9*x,a+10*x,a+11*x,a+12*x,a+13*x
        )
    elif num_k_mmas == 15:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x,a+5*x,a+6*x,a+7*x,a+8*x,a+9*x,a+10*x,a+11*x,a+12*x,a+13*x,a+14*x
        )
    else:
        inlined_assembly[mma_string, NoneType, constraints=constraints](
            c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a,a+x,a+2*x,a+3*x,a+4*x,a+5*x,a+6*x,a+7*x,a+8*x,a+9*x,a+10*x,a+11*x,a+12*x,a+13*x,a+14*x,a+15*x
        )
    # fmt: on


@always_inline
fn elect() -> Int32:
    return inlined_assembly[
        """{
            .reg .b32 %re;
            .reg .pred %pa;
            mov.s32 $0, 0;
            elect.sync %re|%pa, $1;
            @%pa mov.s32 $0, 1;
        }""",
        Int32,
        constraints="=r,r",
    ](-1)


@always_inline
fn elect_mma_arrive[
    cta_group: Int = 1
](
    mbar_ptr: UnsafePointer[address_space = AddressSpace.SHARED, *_, **_],
    elect: Int32,
):
    """Arrive at the mbar pointer for the MMA instruction.

    Parameters:
        cta_group: Number of ctas used by MMA.

    Args:
        mbar_ptr: Pointer to the mbar.
        elect: `elect()`.
    """

    constrained[
        cta_group in (1, 2),
        String("Unsupported cta group: ", cta_group),
    ]()

    comptime type = mbar_ptr.type
    constrained[size_of[type]() == 8, "mbar_ptr must be 8 bytes"]()

    inlined_assembly[
        """{
        .reg .pred %pb;
        setp.eq.s32  %pb, $1, 0;
        @%pb bra skip;
        tcgen05.commit.cta_group::"""
        + String(cta_group)
        + """.mbarrier::arrive::one.shared::cluster.b64 [$0];
        skip:
        }""",
        NoneType,
        constraints="r, r",
    ](Int32(Int(mbar_ptr)), elect)


@always_inline
fn maximum[
    dtype: DType, BN: Int, //, *, width: Int = 8
](x: LocalTensor[dtype, Layout.row_major(BN)]) -> SIMD[dtype, width]:
    constrained[BN % width == 0]()
    vx = x.vectorize[width]()
    acc = vx[0]

    # unroll (using SIMD) to break up dependency chain
    @parameter
    for i in range(1, BN // width):
        acc = max(acc, vx[i])
    # return acc.reduce_max()
    return rebind[SIMD[dtype, width]](acc)


@always_inline
fn maximum[
    dtype: DType,
    BN: Int,
    width: Int, //,
](
    x: LocalTensor[dtype, Layout.row_major(BN)], init: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    constrained[BN % width == 0]()
    vx = x.vectorize[width]()
    acc = rebind[vx.element_type](init)

    # unroll (using SIMD) to break up dependency chain
    @parameter
    for i in range(BN // width):
        acc = max(acc, vx[i])
    return rebind[SIMD[dtype, width]](acc)


@always_inline
fn sum[
    dtype: DType, BN: Int, //, *, width: Int = 8
](x: LocalTensor[dtype, Layout.row_major(BN)]) -> SIMD[dtype, 2]:
    constrained[BN % width == 0]()
    vx = x.vectorize[width]()
    acc = vx[0]

    # unroll (using SIMD) to break up dependency chain
    @parameter
    for i in range(1, BN // width):
        acc += vx[i]

    return acc.reduce_add[size_out=2]()
    # return rebind[SIMD[dtype,width]](acc)


@always_inline
fn mha_sm100_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    output_type: DType,
    MaxPromptLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme, //,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    sink: Bool,
    _is_cache_length_accurate: Bool,
](
    output: DeviceBuffer[output_type],
    q_arg: UnsafePointer[Scalar[q_type]],
    k: KVType,
    v: KVType,
    num_rows_q: Int,
    mask: MaskType,
    score_mod: ScoreModType,
    valid_length: UnsafePointer[UInt32],
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
    constrained[
        config.dtype == KVType.dtype and config.dtype == q_type,
        "config, kv, and q types must all match for FA3.",
    ]()
    comptime decoding: Bool = _is_decoding[MaxPromptLenType]()
    constrained[not decoding, "this implementation does not support decoding"]()
    comptime fa4_config = FA4Config(
        num_q_heads=Int(config.num_heads),
        group=group,
        depth=Int(config.depth),
        dtype_size=size_of[q_type](),
        swizzle_mode=config.swizzle_mode,
        page_size=KVType.page_size,
    )
    comptime swizzle_mode = fa4_config.swizzle_mode
    comptime BM = fa4_config.BM
    comptime BK = fa4_config.padded_depth
    constrained[
        BK % 64 == 0,
        "B200 requires BK%64 as it uses 128B swizzles, but BK==",
        String(BK),
    ]()
    comptime BN = fa4_config.BN
    comptime num_threads = fa4_config.num_threads
    q = rebind[UnsafePointer[Scalar[KVType.dtype]]](q_arg)

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)
    var max_prompt_len: UInt32 = max_prompt_len_arg.as_uint32()
    var max_num_prompt_tiles: UInt32 = ceildiv(max_prompt_len, BM)
    var block_x: UInt32 = max_num_prompt_tiles * partition.num_partitions()

    comptime out_depth = fa4_config.depth
    comptime out_num_heads = fa4_config.num_q_heads

    comptime max_descriptor_length = BM // 2

    comptime descriptor_shape = IndexList[3](
        1, max_descriptor_length, swizzle_mode.bytes() // size_of[output_type]()
    )

    comptime RaggedStoreType = RaggedTensorMap[
        output_type,
        descriptor_shape,
        1,
        swizzle_mode=swizzle_mode,
    ]

    var ragged_tma_store = RaggedStoreType(
        ctx,
        output.unsafe_ptr(),
        max_descriptor_length,
        out_depth * out_num_heads,
        ceildiv(num_rows_q, max_descriptor_length),
        out_depth,
        IndexList[1](out_num_heads),
        IndexList[1](out_depth),
    )

    q_tma_op = q_out_tma[
        swizzle_mode,
        BM = BM // 2,
        depth = fa4_config.depth,
        padded_depth = fa4_config.BK0,
        q_num_heads = fa4_config.num_q_heads,
        group = fa4_config.group,
        decoding=False,
    ](ctx, q, num_rows_q)
    k_tma_op = k.create_tma_tile[
        BN, fa4_config.BK0, swizzle_mode, is_k_major=True
    ](ctx)
    v_tma_op = v.create_tma_tile[
        fa4_config.BK1, fa4_config.padded_depth, swizzle_mode, is_k_major=False
    ](ctx)
    constrained[BM == 256]()
    comptime SchedulerType = TransientScheduler[BM, fa4_config.num_q_heads]
    var scheduler: SchedulerType = SchedulerType()

    @parameter
    if sink:
        comptime SinkType = NonNullPointer[KVType.dtype]
        var sink_ptr: SinkType = {
            rebind[UnsafePointer[Scalar[KVType.dtype]]](
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
            config=fa4_config,
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
            descriptor_shape=descriptor_shape,
            remaining_global_dim_rank=1,
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
            num_rows_q,
            ragged_tma_store,
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
            config=fa4_config,
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
            descriptor_shape=descriptor_shape,
            remaining_global_dim_rank=1,
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
            num_rows_q,
            ragged_tma_store,
        )


@always_inline
fn _mha_sm100_kv_input_row_offset_dispatch[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
    use_score_mod: Bool,
    ragged: Bool,
    SinkType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
    descriptor_shape: IndexList[3],
    remaining_global_dim_rank: Int,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = config.BM // 2,
        depth = config.BK0,
        group = config.group,
        decoding=False,
    ],
    k_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BN,
        config.BK0,
        swizzle_mode,
        is_k_major=True,
    ],
    v_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BK1,
        config.padded_depth,
        swizzle_mode,
        is_k_major=False,
    ],
    o_ptr_arg: DeviceBuffer[output_type],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: UnsafePointer[UInt32],
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
    num_rows_q: Int,
    ragged_tma_store: RaggedTensorMap[
        output_type,
        descriptor_shape,
        remaining_global_dim_rank,
        swizzle_mode=swizzle_mode,
    ],
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
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            KVRowOffsetsType=KVRowOffsetsNonNull,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
            descriptor_shape=descriptor_shape,
            remaining_global_dim_rank=remaining_global_dim_rank,
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
            num_rows_q,
            ragged_tma_store,
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
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            KVRowOffsetsType=KVRowOffsetsNull,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
            descriptor_shape=descriptor_shape,
            remaining_global_dim_rank=remaining_global_dim_rank,
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
            num_rows_q,
            ragged_tma_store,
        )


@always_inline
fn _mha_sm100_valid_length_dispatch[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
    use_score_mod: Bool,
    ragged: Bool,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
    descriptor_shape: IndexList[3],
    remaining_global_dim_rank: Int,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = config.BM // 2,
        depth = config.BK0,
        group = config.group,
        decoding=False,
    ],
    k_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BN,
        config.BK0,
        swizzle_mode,
        is_k_major=True,
    ],
    v_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BK1,
        config.padded_depth,
        swizzle_mode,
        is_k_major=False,
    ],
    o_ptr_arg: DeviceBuffer[output_type],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: UnsafePointer[UInt32],
    kv_input_row_offsets: KVRowOffsetsType,
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    score_mod: ScoreModType,
    ctx: DeviceContext,
    num_rows_q: Int,
    ragged_tma_store: RaggedTensorMap[
        output_type,
        descriptor_shape,
        remaining_global_dim_rank,
        swizzle_mode=swizzle_mode,
    ],
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
            use_score_mod=use_score_mod,
            SinkType=SinkType,
            ValidLengthType=ValidLengthType,
            KVRowOffsetsType=KVRowOffsetsType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
            descriptor_shape=descriptor_shape,
            remaining_global_dim_rank=remaining_global_dim_rank,
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
            num_rows_q=num_rows_q,
            ragged_tma_store=ragged_tma_store,
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
            use_score_mod=use_score_mod,
            SinkType=SinkType,
            ValidLengthType=ValidLengthType,
            KVRowOffsetsType=KVRowOffsetsType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
            descriptor_shape=descriptor_shape,
            remaining_global_dim_rank=remaining_global_dim_rank,
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
            num_rows_q=num_rows_q,
            ragged_tma_store=ragged_tma_store,
        )


@always_inline
fn _mha_sm100_enqueue[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
    descriptor_shape: IndexList[3],
    remaining_global_dim_rank: Int,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = config.BM // 2,
        depth = config.BK0,
        group = config.group,
        decoding=False,
    ],
    k_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BN,
        config.BK0,
        swizzle_mode,
        is_k_major=True,
    ],
    v_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BK1,
        config.padded_depth,
        swizzle_mode,
        is_k_major=False,
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
    num_rows_q: Int,
    ragged_tma_store: RaggedTensorMap[
        output_type,
        descriptor_shape,
        remaining_global_dim_rank,
        swizzle_mode=swizzle_mode,
    ],
) raises:
    # the pack contains all possibly 0-sized objects
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
        max_seq_len.as_uint32(), config.BM
    )
    var block_x: UInt32 = max_num_prompt_tiles * partition.num_partitions()
    logger.info("------ Dispatching to SM100 FMHA-2Q ------")
    logger.info(
        "QKV Type:",
        KVLUTType.dtype,
        "Depth:",
        config.depth,
        "Number of Q // KV Heads:",
        config.num_q_heads,
        "//",
        config.num_kv_heads,
        "Batch Size:",
        batch_size,
        "Max Num Prompt Tiles:",
        max_num_prompt_tiles,
    )

    comptime num_threads = config.num_threads
    comptime smem_use = config.smem_used

    comptime kernel = SM100MHA2Q[
        KVLUTType,
        output_type,
        MaskType,
        ScoreModType,
        SchedulerType,
        config,
        use_score_mod,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        _is_cache_length_accurate,
        MaxSeqLenType,
        PartitionType,
        descriptor_shape,
        remaining_global_dim_rank,
    ].kernel

    ctx.enqueue_function_checked[kernel, kernel](
        q_tma_op,
        k_tma_op,
        v_tma_op,
        o_ptr_arg,
        ragged_tma_store,
        kv_lut,
        scale,
        batch_size,
        num_keys_arg,
        pack,
        grid_dim=SchedulerType.grid_dim(batch_size, block_x),
        block_dim=(Int(num_threads), 1, 1),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )


@register_passable("trivial")
struct KVPipeline[num_kv_stages: Int, num_mma_stages: Int]:
    """
    KVPipeline has `num_kv_stages * num_mma_stages` stages.
    `num_kv_stages` refers to how many `K` and `V` tiles we pipeline
    for performing the `S = Q@K'` and `O += P@V` MMAs.
    Each of these MMAs is broken up into `num_mma_stages` pipelined
    MMAs. We set `step=False` for all but the last MMA that completes
    the operation.
    An alternative implementation would separate the two, and potentially
    allow for more overall stages at the cost of slightly more bookkeeping.
    """

    comptime num_stages: Int = Self.num_kv_stages * Self.num_mma_stages

    # mbars are ordered in {producer, consumer} pairs
    var mbar: MBarType
    var state: PipelineState[Self.num_kv_stages]

    @always_inline
    fn __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    fn init(self):
        # Consumer & Producer mbars: arrived by 1 thread performing TMA/mma
        @parameter
        for i in range(2 * Self.num_stages):
            self.mbar[i].init(1)

    @always_inline
    fn producer_mbar[mma_stage: Int](self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.mbar + Self.num_mma_stages * idx + mma_stage

    @always_inline
    fn consumer_mbar[mma_stage: Int](self, idx: UInt32) -> MBarType:
        comptime const_offset = mma_stage + Self.num_stages
        return self.mbar + Self.num_mma_stages * idx + const_offset

    @always_inline
    fn consumer_mbar[mma_stage: Int](self) -> MBarType:
        return self.consumer_mbar[mma_stage](self.state.index())

    @always_inline("nodebug")
    fn producer_acquire[mma_stage: Int = Self.num_mma_stages - 1](self):
        """
        Returns the dynamic pipe idx.
        """
        self.consumer_mbar[mma_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_wait[mma_stage: Int = Self.num_mma_stages - 1](self):
        self.producer_mbar[mma_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_release[
        mma_stage: Int = Self.num_mma_stages - 1
    ](mut self, e: Int32):
        elect_mma_arrive(self.consumer_mbar[mma_stage](), e)

        @parameter
        if mma_stage == Self.num_mma_stages - 1:
            self.state.step()

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return 2 * Self.num_mma_stages * Self.num_kv_stages


@register_passable("trivial")
struct TMADestination[dtype: DType, layout: Layout]:
    var mbar: MBarType
    var smem: SharedMemTensor[Self.dtype, Self.layout]

    @always_inline
    fn __init__(
        out self, mbar: MBarType, smem: SharedMemTensor[Self.dtype, Self.layout]
    ):
        self.mbar = mbar
        self.smem = smem


@register_passable("trivial")
struct KVProducerPipeline[dtype: DType, config: FA4Config]:
    comptime KType = SharedMemTensor[
        Self.dtype,
        tile_layout_k_major[
            Self.dtype,
            Self.config.BN,
            Self.config.BK0,
            Self.config.swizzle_mode,
        ](),
    ]
    comptime VType = SharedMemTensor[
        Self.dtype,
        tile_layout_mn_major[
            Self.dtype,
            Self.config.padded_depth,
            Self.config.BK1,
            Self.config.swizzle_mode,
        ](),
    ]
    comptime KPairType = TMADestination[Self.dtype, Self.KType.layout]
    comptime VPairType = TMADestination[Self.dtype, Self.VType.layout]
    comptime kv_elements = Self.KType.layout.size()
    comptime kv_bytes = Self.kv_elements * size_of[Self.dtype]()
    comptime SMemType = SharedMemPointer[Scalar[Self.dtype]]

    var kv_pipeline: KVPipeline[
        Self.config.num_kv_stages, Self.config.num_mma_stages
    ]
    var smem: Self.SMemType

    @always_inline
    fn __init__(
        out self,
        mbar: MBarType,
        smem: Self.SMemType,
    ):
        constrained[
            Self.config.padded_depth % Self.config.num_mma_stages == 0
        ]()
        constrained[Self.config.BN % Self.config.num_mma_stages == 0]()
        constrained[Self.kv_elements == Self.VType.layout.size()]()
        self.kv_pipeline = {mbar}
        self.smem = smem
        self.kv_pipeline.state._phase = 1

    @always_inline
    fn __init__(
        out self,
        kv_pipeline: KVPipeline[
            Self.config.num_kv_stages, Self.config.num_mma_stages
        ],
        smem: Self.SMemType,
    ):
        constrained[
            Self.config.padded_depth % Self.config.num_mma_stages == 0
        ]()
        constrained[Self.config.BN % Self.config.num_mma_stages == 0]()
        constrained[Self.kv_elements == Self.VType.layout.size()]()
        self.kv_pipeline = kv_pipeline
        self.smem = smem
        self.kv_pipeline.state._phase = 1

    @always_inline
    fn init(self):
        """
        Only one of the producer or consumer should call `init()`.
        """
        self.kv_pipeline.init()

    @always_inline
    fn get_kv_smem[*, mma_stage: Int](self) -> Self.SMemType:
        comptime stage_offset = mma_stage * Self.config.padded_depth * Self.config.BN
        var dyn_offset: UInt32 = (
            Self.kv_elements * self.kv_pipeline.state.index()
        )
        return self.smem + stage_offset + dyn_offset

    @always_inline
    fn get_k[*, mma_stage: Int, expect: Bool = True](self) -> Self.KPairType:
        p_mbar = self.kv_pipeline.producer_mbar[mma_stage=mma_stage]()

        @parameter
        if expect:
            p_mbar[].expect_bytes(Self.kv_bytes)
        return {p_mbar, {self.get_kv_smem[mma_stage=mma_stage]()}}

    @always_inline
    fn get_v[*, mma_stage: Int](self) -> Self.VPairType:
        p_mbar = self.kv_pipeline.producer_mbar[mma_stage=mma_stage]()
        p_mbar[].expect_bytes(Self.kv_bytes)
        return {p_mbar, {self.get_kv_smem[mma_stage=mma_stage]()}}

    @always_inline
    fn acquire_kv[*, mma_stage: Int = Self.config.num_mma_stages - 1](self):
        self.kv_pipeline.producer_acquire[mma_stage]()

    @always_inline
    fn commit_kv_step(mut self):
        """
        Step the kv pipeline. The does not perform the commit on the mbars;
        that should be handled by the `tma_op.async_copy`.
        """
        self.kv_pipeline.state.step()


@register_passable("trivial")
struct KVConsumerPipeline[dtype: DType, config: FA4Config]:
    """
    Pipeline for managing the consumption of K and V.
    This follows the order of Tri Dao and Cutlass implementations
    (modulo any rotation of the ops through the iterations).

    We consume/produce in the following order:
        0. S0 <- Q0 @ Kn'
        1. O1 <- O1 + P1 @ V{n-1}
        2. S1 <- Q1 @ Kn'
        3. O0 <- O0 + P0 @ Vn

    Note that we have two MMA between calculating Si and consuming Pi,
    maximizing the overlap between MMAs and softmax calculation.
    Oi + Pi @ V also depends on the correction, which is computed
    asynchronously with the softmax in a correction warpgroup (as soon
    as the softmax writes the correction factor).

    # wait on K0
    S0 <- Q0 @ K0'
    S1 <- Q1 @ K0'
    # release K0
    # wait on V0
    O0 <- P0 @ V0
    for n in range(1,num_iters):
        # wait on Kn
        S0 <- Q0 @ Kn'
        O1 <- O1 + P1@V{n-1}
        # release V{n-1}
        S1 <- Q1 @ Kn'
        # release Kn
        # wait on Vn
        O0 <- P0 @ Vn
    O1 <- O1 + P1@V{num_iters-1}

    wK0, rK0, wV0
    wK1, rV0, rK1, wV1
    wK2, rV1, rK2, wV2
    wK3, rV2, rK3, wV3

    wKn(state)
    wK0(0), rK0(0), wV0(1)
    wK1(2), rV0(1), rK1(2), wV1(3)
    wK2(4), rV1(3), rK2(4), wV2(5)
    wK3(6), rV2(5), rK3(6), wV3(7)

    Rules:
        wK backs up and increments prior to waiting, except K0
        rK increments after releasing
        rV uses backup

    wK0(0), rK0(0), wV0(1)
    wK1(2), rV0(1), rK1(2), wV1(3)
    wK2(4), rV1(3), rK2(4), wV2(5)
    rV2(5)
    """

    comptime full_kv_bytes = Self.config.BN * Self.config.padded_depth * size_of[
        Self.dtype
    ]()
    comptime mma_kv_bytes = Self.config.BN * Self.config.BK0 * size_of[
        Self.dtype
    ]()

    var kv_pipeline: KVPipeline[
        Self.config.num_kv_stages, Self.config.num_mma_stages
    ]
    var k_smem_descriptor: MMASmemDescriptorPair
    var v_smem_descriptor: MMASmemDescriptorPair
    var v_pipeline_release_index: UInt32

    @always_inline
    fn __init__(
        out self,
        kv_pipeline: KVPipeline[
            Self.config.num_kv_stages, Self.config.num_mma_stages
        ],
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.kv_pipeline = kv_pipeline
        self.k_smem_descriptor = smem_descriptor[
            BMN = Self.config.BN,
            BK = Self.config.BK0,
            swizzle_mode = Self.config.swizzle_mode,
            is_k_major=True,
        ](smem)
        self.v_smem_descriptor = smem_descriptor[
            BMN = Self.config.padded_depth,
            BK = Self.config.BK1,
            swizzle_mode = Self.config.swizzle_mode,
            is_k_major=False,
        ](smem)
        self.v_pipeline_release_index = 0

    @always_inline
    fn __init__(
        out self,
        mbar: MBarType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        return self.__init__(
            KVPipeline[Self.config.num_kv_stages, Self.config.num_mma_stages](
                mbar
            ),
            smem,
        )

    @always_inline
    fn init(self):
        """
        Only one of the producer or consumer should call `init()`.
        """
        self.kv_pipeline.init()

    @always_inline("nodebug")
    fn wait[*, mma_stage: Int](self) -> UInt32:
        """
        Wait on `k` from the producer, and return the `k` smem descriptor.
        """

        comptime stage_offset = mma_stage * Self.mma_kv_bytes
        var dyn_offset: UInt32 = (
            Self.full_kv_bytes * self.kv_pipeline.state.index()
        )
        self.kv_pipeline.consumer_wait[mma_stage]()
        return dyn_offset + stage_offset

    @always_inline("nodebug")
    fn wait_k[
        *,
        mma_stage: Int = Self.config.num_mma_stages - 1,
        pre_increment: Bool = True,
    ](mut self) -> MMASmemDescriptorPair:
        """
        Wait on `k` from the producer, and return the `k` smem descriptor.
        If `pre-increment` is true.
        """

        @parameter
        if pre_increment and (mma_stage == 0):
            self.v_pipeline_release_index = self.kv_pipeline.state.index()
            self.kv_pipeline.state.step()
        return self.k_smem_descriptor + Int(self.wait[mma_stage=mma_stage]())

    @always_inline("nodebug")
    fn wait_v[
        *, mma_stage: Int = Self.config.num_mma_stages - 1
    ](self) -> MMASmemDescriptorPair:
        return self.v_smem_descriptor + Int(self.wait[mma_stage=mma_stage]())

    @always_inline("nodebug")
    fn release_k[
        *, mma_stage: Int = Self.config.num_mma_stages - 1
    ](mut self, e: Int32):
        """
        Must call `producer_commit` on the tmem resource before calling
        `consumer_release`.
        `release_k` does increment the pipeline step.
        """
        self.kv_pipeline.consumer_release[mma_stage](e)

    @always_inline("nodebug")
    fn release_v[
        *, mma_stage: Int = Self.config.num_mma_stages - 1
    ](self, e: Int32):
        """
        Must call `producer_commit` on the tmem resource before calling
        `consumer_release`.
        `release_v` does not increment the pipeline step.
        """
        elect_mma_arrive(
            self.kv_pipeline.consumer_mbar[mma_stage](
                self.v_pipeline_release_index
            ),
            e,
        )


@register_passable("trivial")
struct ProducerPipeline[number_of_stages: Int]:
    comptime num_stages: Int = Self.number_of_stages

    # mbars are ordered in {producer, consumer} pairs
    var mbar: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    fn __init__(out self, mbar: MBarType):
        self.mbar = mbar
        # Behavior:
        # mbar - initially phase 0
        # Producer - phase 1
        # Consumer - phase 0
        #
        # A `wait(phase)` blocks so long as
        # `mbar.phase != phase`.
        # Memory barriers are initialized with `mbar.phase = 0`.
        # Memory barrier phases flip after init-count arrivals.
        # Example with `num_stages = 1`.
        #
        # Producer:
        # p0. consumer_mbar.wait(phase=1)  # 1 != 0: falls through
        # p1. producer_mbar.commit()       # producer_mbar.phase=1
        # p2. step()                       # phase = 0
        # p3. consumer_mbar.wait(phase=0)  # 0 == 0: blocked until c1
        # p4. producer_mbar.commit()       # producer_mbar.phase=0
        # p5. step()
        # p6. consumer_mbar.wait(phase=1)
        # p7. producer_mbar.commit()       # producer_mbar.phase=1
        #
        # Consumer:
        # c0. producer_mbar.wait(phase=0)  # 0 == 0: blocked until p1
        # c1. consumer.release()           # consumer_mbar.phase=1
        # c2. step()                       # phase = 1
        # c3. producer_mbar.wait(phase=1)  # blocked until p4
        # c4. consumer.release()           # consumer_mbar.phase=0
        # c5. step()
        # c6. producer_mbar.wait(phase=0)
        # c7. consumer.release()           # consumer_mbar.phase=1
        #
        # The order of blocking/unblocking can be visualized as:
        # p0, p1, p2
        #     \-> c0, c1, c2
        #              \-> p3, p4, p5
        #                       \-> c3, c4, c5
        #                                \-> p6, p7
        #                                         \-> c6, c7
        #
        # The producer initializes phase to `1`
        # Thus, initial producer `wait`s fall through; only after
        # `number_of_stages` steps will we reset to `phase = 0`,
        # and thus begin waiting on the first set of `consumer` releases.
        self.state = {}  # {0, 1, 0}
        self.state._phase = 1

    @always_inline
    fn producer_mbar(self) -> MBarType:
        return self.mbar + self.state.index()

    @always_inline
    fn consumer_mbar(self) -> MBarType:
        return self.mbar + Self.number_of_stages + self.state.index()

    @always_inline("nodebug")
    fn acquire(self):
        self.consumer_mbar()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn commit(mut self):
        _ = self.producer_mbar()[].arrive()
        self.state.step()

    @always_inline("nodebug")
    fn commit_mma(self):
        mbar = self.producer_mbar()
        elect_mma_arrive(mbar, elect())

    @always_inline("nodebug")
    fn commit_mma(self, elect: Int32):
        mbar = self.producer_mbar()
        elect_mma_arrive(mbar, elect)

    @always_inline("nodebug")
    fn step(mut self):
        self.state.step()


@register_passable("trivial")
struct ConsumerPipeline[number_of_stages: Int]:
    comptime num_stages: Int = Self.number_of_stages

    # mbars are ordered in {producer, consumer} pairs
    var mbar: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    fn __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}
        # Consumer phase is initialized to `0`.
        # Producer phase is initialized to `1`.
        # See `ProducerPipeline.__init__` for details.

    @always_inline
    fn producer_mbar(self) -> MBarType:
        return self.mbar + self.state.index()

    @always_inline
    fn consumer_mbar(self) -> MBarType:
        return self.mbar + Self.number_of_stages + self.state.index()

    @always_inline("nodebug")
    fn wait(self):
        self.producer_mbar()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn release(mut self):
        _ = self.consumer_mbar()[].arrive()
        self.state.step()

    @always_inline("nodebug")
    fn step(mut self):
        self.state.step()


@register_passable("trivial")
struct MBarPipeline[number_of_stages: Int]:
    comptime num_stages: Int = Self.number_of_stages

    # mbars are ordered in {producer, consumer} pairs
    var mbar: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    fn __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    fn init[*, num_producer: UInt32 = 1, num_consumer: UInt32 = 1](self):
        @parameter
        for i in range(Self.number_of_stages):
            self.mbar[i].init(Int(num_producer))

        @parameter
        for i in range(Self.number_of_stages):
            self.mbar[i + Self.number_of_stages].init(Int(num_consumer))

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return 2 * Self.number_of_stages


@always_inline
fn apply_mask[
    dtype: DType,
    BN: Int,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait, //,
    *,
    use_score_mod: Bool,
    masked: Bool,
    last_iter: Bool,
    decoding: Bool = False,
](
    srow: LocalTensor[dtype, Layout.row_major(BN)],
    mask: MaskType,
    score_mod: ScoreModType,
    scale_log2e: Scalar[dtype],
    *,
    prompt_idx: UInt32,
    q_head_idx: UInt32,
    kv_tile_start_row: UInt32,
    max_seq_len: UInt32,
    num_keys: UInt32,
    score_row: UInt32,
):
    comptime simd_size = simd_width_of[dtype]()
    vs = srow.vectorize[simd_size]()

    @parameter
    for n in range(BN // simd_size):
        # score_col = mask_frag_col + j * 8
        s = vs[n]
        comptime frag_col = simd_size * n
        var score_col: UInt32 = kv_tile_start_row + frag_col

        @parameter
        if masked:
            # if thread_idx.x == 0:
            #     print("score_row score_col n =", score_row, score_col, n)

            s = mask.mask(
                IndexList[4, element_type = DType.uint32](
                    Int(prompt_idx),
                    Int(q_head_idx),
                    Int(score_row),
                    Int(score_col),
                ),
                s * scale_log2e,
            )
        else:  # if MaskType.apply_log2e_after_mask, this is scale only
            s *= scale_log2e

        @parameter
        if use_score_mod:
            s = (
                score_mod.score_mod(
                    IndexList[4, element_type = DType.uint32](
                        Int(prompt_idx),
                        Int(q_head_idx),
                        Int(score_row),
                        Int(score_col),
                    ),
                    s,
                    Int(max_seq_len),
                )
                * log2e
            )
        elif MaskType.apply_log2e_after_mask:
            s *= log2e

        var bound: IndexList[2, element_type = DType.uint32]

        @parameter
        if decoding:
            var coord: UInt32 = min(BN + kv_tile_start_row, num_keys)
            s = (
                iota[DType.uint32, vs.element_size](coord)
                .lt(score_col)
                .select(s, MASK_VALUE)
            )
        elif last_iter:
            s = (
                iota[DType.uint32, vs.element_size](score_col)
                .lt(num_keys)
                .select(s, MASK_VALUE)
            )

        vs[n] = s


@register_passable("trivial")
struct FA4MiscMBars:
    var mbar_base: MBarType
    comptime S0_offset = 0
    comptime S1_offset = 2
    comptime C0_offset = 4
    comptime C1_offset = 6
    comptime order_offset = 8
    comptime Q1SyncIdx = 10
    comptime size = Self.Q1SyncIdx + 1

    @always_inline
    fn __init__(out self, mbar_base: MBarType):
        self.mbar_base = mbar_base

    @always_inline
    fn init(self):
        # [0] producer 0
        # [1] consumer 0
        # [2] producer 1
        # [3] consumer 1
        @parameter
        for wg_idx in range(2):
            # S producer, produced by 1 UMMA
            self.mbar_base[2 * wg_idx].init(1)
            # S consumer, consumed by 128 softmax threads
            self.mbar_base[2 * wg_idx + 1].init(128)
            # C producer, produced by 128 softmax threads
            self.mbar_base[2 * wg_idx + Self.C0_offset].init(128)
            # C consumer, consumed by 128 correction threads
            self.mbar_base[2 * wg_idx + 1 + Self.C0_offset].init(128)
            # ordering is done by 128 softmax threads
            self.mbar_base[wg_idx + Self.order_offset].init(128)

        self.mbar_base[Self.Q1SyncIdx].init(1)

    @always_inline
    fn producer_s0(self) -> ProducerPipeline[1]:
        return {self.mbar_base}

    @always_inline
    fn producer_s1(self) -> ProducerPipeline[1]:
        return {self.mbar_base + Self.S1_offset}

    @always_inline
    fn consumer_s(self, wg_idx: UInt32) -> ConsumerPipeline[1]:
        return {self.mbar_base + 2 * wg_idx}

    @always_inline
    fn consumer_c0(self) -> ConsumerPipeline[1]:
        return {self.mbar_base + Self.C0_offset}

    @always_inline
    fn consumer_c1(self) -> ConsumerPipeline[1]:
        return {self.mbar_base + Self.C1_offset}

    @always_inline
    fn producer_c(self, wg_idx: UInt32) -> ProducerPipeline[1]:
        return {self.mbar_base + Self.C0_offset + 2 * wg_idx}

    @always_inline
    fn pipeline_order_wait(self, wg_idx: UInt32) -> MBarType:
        return {self.mbar_base + Self.order_offset + wg_idx}

    @always_inline
    fn pipeline_order_arrive(self, wg_idx: UInt32) -> MBarType:
        return {self.mbar_base + (Self.order_offset + 1) - wg_idx}

    @always_inline
    fn q1_wait_mbar(
        self,
    ) -> ref [
        self.mbar_base.origin, self.mbar_base.address_space
    ] SharedMemBarrier:
        return self.mbar_base[Self.Q1SyncIdx]

    @always_inline
    fn end(self) -> MBarType:
        return self.mbar_base + Self.size


@register_passable("trivial")
struct SM100MHA2Q[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    descriptor_shape: IndexList[3],
    remaining_global_dim_rank: Int,
]:
    comptime qkv_type = Self.KVLUTType.dtype
    comptime accum_type = get_accum_type[Self.qkv_type]()
    comptime simd_size: Int = simd_width_of[Self.qkv_type]()

    comptime cta_group = 1  # TODO: support 2
    comptime BM = Self.config.BM
    comptime BN = Self.config.BN
    comptime depth = Self.config.depth
    comptime padded_depth = Self.config.padded_depth
    comptime num_q_heads = Self.config.num_q_heads
    comptime group = Self.config.group
    comptime ragged = not Self.ValidLengthType.is_null
    comptime page_size = Self.KVLUTType.page_size

    comptime num_m_mmas = 2
    comptime MMA_M = Self.config.BM // Self.num_m_mmas
    comptime qo_elements = Self.padded_depth * Self.MMA_M
    comptime qkv_dt_size = size_of[Self.qkv_type]()

    comptime OPipelineType = MBarPipeline[2]  # x1 -> 4 barriers

    comptime num_mma_stages = Self.config.num_mma_stages

    # First MMA is
    # (BM x depth) @ (BN x depth)' -> (BM x BN)
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M = Self.MMA_M,  # generally 128
        MMA_N = Self.BN,
        BK = Self.depth,  # BK in memory depth
        swizzle_a = Self.config.swizzle_mode,
        swizzle_b = Self.config.swizzle_mode,
        transpose_b=True,
        num_stages = Self.num_mma_stages,
    ]
    # Second MMA is
    # (BM x BN) @ (BN x depth) -> (BM x depth)
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M = Self.MMA_M,
        MMA_N = Self.config.padded_depth,
        BK = Self.BN,
        swizzle_b = Self.config.swizzle_mode,
        transpose_b=False,
        num_stages = Self.num_mma_stages,
    ]

    comptime swizzle_granularity = Self.config.swizzle_mode.bytes() // Self.qkv_dt_size
    comptime k_elements: UInt32 = Self.swizzle_granularity * Self.config.BN
    comptime qo_bytes: UInt32 = Self.qkv_dt_size * Self.qo_elements
    comptime k_bytes: UInt32 = Self.qkv_dt_size * Self.k_elements
    comptime MMA_K = 16
    comptime v_bytes_per_mma: UInt32 = Self.qkv_dt_size * Self.MMA_K * Self.config.padded_depth

    comptime KVPipelineType = KVPipeline[
        Self.config.num_kv_stages, Self.config.num_mma_stages
    ]
    comptime PositionType = MHAPosition[
        Self.config.BM,
        Self.config.BN,
        Self.config.depth,
        Self.config.padded_depth,
        Self.config.num_q_heads,
        Self.config.group,
        _is_decoding[Self.MaxSeqLenType](),
    ]

    @staticmethod
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Self.config.num_threads
        )
    )
    fn kernel(
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            depth = Self.config.BK0,
            group = Self.config.group,
            decoding=False,
        ],
        k_tma_op: TMANestedTensorTile[
            Self.KVLUTType.dtype,
            Self.config.BN,
            Self.config.padded_depth,
            Self.config.swizzle_mode,
            is_k_major=True,
        ],
        v_tma_op: TMANestedTensorTile[
            Self.KVLUTType.dtype,
            Self.config.BN,
            Self.config.padded_depth,
            Self.config.swizzle_mode,
            is_k_major=False,
        ],
        o_ptr_arg: UnsafePointer[Scalar[Self.output_type]],
        ragged_tma_store: RaggedTensorMap[
            Self.output_type,
            Self.descriptor_shape,
            Self.remaining_global_dim_rank,
            swizzle_mode = Self.config.swizzle_mode,
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        batch_size: UInt32,
        num_keys_arg: UInt32,
        pack: Pack[
            Self.MaskType,
            Self.ScoreModType,
            Self.SchedulerType,
            Self.ValidLengthType,
            Self.SinkType,
            Self.KVRowOffsetsType,
            Self.MaxSeqLenType,
            Self.PartitionType,
        ],
    ):
        constrained[Self.MMA_M == 64 or Self.MMA_M == 128]()
        constrained[_is_decoding[Self.MaxSeqLenType]() == False]()
        constrained[
            Self.config.supported(),
            "depth = "
            + String(Self.config.depth)
            + "\nBN = "
            + String(Self.config.BN)
            + "\nnum_kv_stages = "
            + String(Self.config.num_kv_stages)
            + "\ntmem_used = "
            + String(Self.config.tmem_used)
            + "\nsmem_used = "
            + String(Self.config.smem_used),
        ]()
        constrained[
            not Self.SchedulerType.may_advance,
            "Persistent kernels not yet supported with FA4",
        ]()
        constrained[Self.UMMA0Type.num_stages == Self.UMMA1Type.num_stages]()

        mask = pack.mask
        score_mod = pack.score_mod
        scheduler = pack.scheduler
        valid_length = pack.valid_length
        sink_weights = pack.sink_weights
        kv_input_row_offsets = pack.kv_input_row_offsets
        max_seq_len = pack.max_seq_len
        partition = pack.partition

        comptime num_qo = Self.config.num_qo()
        # TODO: We may want to support num_qo>2 for depth=64?
        constrained[
            num_qo == 1 or num_qo == 2,
            "Currently only support num_qo == 1 or 2",
        ]()
        q_smem = external_memory[
            Scalar[Self.qkv_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        kv_smem = q_smem + Self.config.BM * Self.config.padded_depth
        comptime kv_total_stages = Self.config.num_kv_stages * Self.config.num_mma_stages
        comptime kv_smem_total_bytes = Self.config.padded_depth * Self.config.BN * kv_total_stages
        var correction_smem: SharedMemPointer[Scalar[Self.accum_type]] = (
            kv_smem + kv_smem_total_bytes
        ).bitcast[Scalar[Self.accum_type]]()
        var mbar_base: MBarType

        @parameter
        if Self.config.use_tmem_for_correction():
            mbar_base = correction_smem.bitcast[SharedMemBarrier]()
        else:
            mbar_base = (
                correction_smem + Self.config.correction_smem_elements()
            ).bitcast[SharedMemBarrier]()

        kv_pipeline = Self.KVPipelineType(mbar_base)
        mbar_base += Self.KVPipelineType.num_mbars()
        # O += P@V -> correction
        o_mbar = mbar_base  # 2, UMMA
        mbar_base += Self.OPipelineType.num_mbars()
        var misc_mbars: FA4MiscMBars = {mbar_base}
        # S = Q@K' -> softmax 0/1
        # softmax 0/1 -> correction
        # 4s (2 consumer, 2 producer)
        # 4c (2 consumer, 2 producer)
        # 2 softmax-order
        ptr_tmem_addr = misc_mbars.end().bitcast[UInt32]()

        # https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
        comptime num_reg_softmax = 184
        comptime num_reg_correction = 104
        comptime num_reg_other = 40

        constrained[
            not Self.PartitionType.do_partition,
            (
                "Neither partitioning nor decoding are supported by the 2-q"
                " implementation."
            ),
        ]()

        var warp_idx: UInt32 = warp.broadcast(warp_id())
        if warp_idx == 0:
            if elect() != 0:
                kv_pipeline.init()

                # o produced by 1 MMA, consumed by 128 correction
                @parameter
                for i in range(2):
                    o_mbar[i].init(1)  # producer
                    o_mbar[i + 2].init(WARPGROUP_SIZE)  # consumer
                misc_mbars.init()
        elif warp_idx == 1:
            tcgen05_alloc[Self.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )

        barrier()

        # warp group partitioning
        # Two QO:
        if warp_idx < 8:
            # softmax $warp_group_idx
            warpgroup_reg_alloc[num_reg_softmax]()
            var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                batch_size, max_seq_len, valid_length, partition
            )

            if not seq_info.is_valid():
                return

            var pos: PositionSummary = PositionSummary.create[
                ragged = Self.ragged,
                _is_cache_length_accurate = Self._is_cache_length_accurate,
            ](kv_lut, seq_info, num_keys_arg, kv_input_row_offsets, max_seq_len)

            Self.softmax(
                ptr_tmem_addr[0],
                warp_idx,
                misc_mbars,
                o_mbar,
                pos.score_row,
                seq_info,
                mask,
                pos.num_keys,
                scale.cast[Self.accum_type](),
                score_mod,
                max_seq_len.as_uint32(),
                o_ptr_arg,
                ragged_tma_store,
                q_smem.bitcast[Scalar[Self.output_type]](),
                sink_weights,
            )

        elif warp_idx < 12:
            # correction
            warpgroup_reg_dealloc[num_reg_correction]()

            var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                batch_size, max_seq_len, valid_length, partition
            )
            if not seq_info.is_valid():
                return
            var pos: PositionSummary = PositionSummary.create[
                ragged = Self.ragged,
                _is_cache_length_accurate = Self._is_cache_length_accurate,
            ](kv_lut, seq_info, num_keys_arg, kv_input_row_offsets, max_seq_len)
            Self.correction(
                ptr_tmem_addr[0],
                misc_mbars,
                o_mbar,
                pos.score_row,
                pos.num_keys,
                mask,
            )
        else:
            warpgroup_reg_dealloc[num_reg_other]()
            if warp_idx == 13:  # produce
                var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                    batch_size, max_seq_len, valid_length, partition
                )

                if not seq_info.is_valid():
                    return
                var pos: PositionSummary = PositionSummary.create[
                    ragged = Self.ragged,
                    _is_cache_length_accurate = Self._is_cache_length_accurate,
                ](
                    kv_lut,
                    seq_info,
                    num_keys_arg,
                    kv_input_row_offsets,
                    max_seq_len,
                )
                Self.load(
                    misc_mbars,
                    kv_pipeline,
                    pos.score_row,
                    pos.num_keys,
                    seq_info,
                    max_seq_len,
                    mask,
                    q_tma_op,
                    k_tma_op,
                    v_tma_op,
                    kv_lut,
                    q_smem,
                )

            elif warp_idx == 12:  # Q @ K', P @ V
                var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                    batch_size, max_seq_len, valid_length, partition
                )

                if not seq_info.is_valid():
                    tcgen05_release_allocation_lock[Self.cta_group]()
                    tcgen05_dealloc[Self.cta_group](
                        ptr_tmem_addr[0], Self.config.sm100_tmem_cols
                    )
                    return
                var pos: PositionSummary = PositionSummary.create[
                    ragged = Self.ragged,
                    _is_cache_length_accurate = Self._is_cache_length_accurate,
                ](
                    kv_lut,
                    seq_info,
                    num_keys_arg,
                    kv_input_row_offsets,
                    max_seq_len,
                )
                Self.mma(
                    ptr_tmem_addr[0],
                    misc_mbars,
                    kv_pipeline,
                    o_mbar,
                    pos.score_row,
                    pos.num_keys,
                    mask,
                    q_smem,
                )

    @staticmethod
    @always_inline
    fn mask_status(
        mask: Self.MaskType, score_row: UInt32, kv_row: UInt32
    ) -> TileMaskStatus:
        return mask.status(
            Index[dtype = DType.int32](
                Int(score_row),
                Int(kv_row),
            ),
            Index[dtype = DType.int32](Int(Self.BM), Int(Self.BN)),
        )

    @always_inline
    @staticmethod
    fn scale_write_output(
        local_row: UInt32,
        inv_row_sum: Scalar[Self.accum_type],
        o_smem: SharedMemPointer[Scalar[Self.output_type]],
        o_tmem: TMemTile[Self.accum_type, Self.BM // 2, Self.padded_depth],
        o_ptr: UnsafePointer[Scalar[Self.output_type]],
        ragged_tma_store: RaggedTensorMap[
            Self.output_type,
            Self.descriptor_shape,
            swizzle_mode = Self.config.swizzle_mode,
        ],
        warp_group_idx: UInt32,
        consumer_mbar: MBarType,
        current_seq: Int,
        num_output_rows: Int32,
    ):
        o = o_tmem.load_async_with_st_matrix_layout[
            num_threads=WARPGROUP_SIZE
        ]()
        comptime num_rows = o.layout[0].size()
        inv_row_sums = LocalTensor[
            Self.accum_type, Layout.row_major(num_rows)
        ].stack_allocation()
        lane = local_row % 32
        lane_row = lane // 4

        #  0  1  2  3
        #  4  5  6  7
        #  8  9 10 11
        # 12 13 14 15
        # 16 17 18 19
        # 20 21 22 23
        # 24 25 26 27
        # 28 29 30 31
        # lane 0 needs to get
        @parameter
        for i in range(num_rows):
            # lane // 4, lane // 4 + 8, lane // 4 + 16, lane // 4 + 24
            inv_row_sums[i] = warp.shuffle_idx(inv_row_sum, lane_row + 8 * i)

        tcgen05_load_wait()
        tcgen05_fence_before()
        _ = consumer_mbar[].arrive()

        @parameter
        for i in range(num_rows):
            irs = o.element_type(
                rebind[Scalar[Self.accum_type]](inv_row_sums[i])
            )

            @parameter
            for j in range(o.layout[1].size()):
                o[i, j] *= irs

        comptime swizzle = make_swizzle[
            Self.output_type, Self.config.swizzle_mode
        ]()

        comptime ST = STMatrixLayout[
            Self.BM // 2, Self.padded_depth, num_threads=WARPGROUP_SIZE
        ]

        var head = Int(block_idx.y)
        comptime last_dim = Self.descriptor_shape[2]

        constrained[
            Self.padded_depth % last_dim == 0,
            "padded_depth must be a multiple of last descriptor dimension",
        ]()
        comptime iters = Self.padded_depth // last_dim

        comptime smem_base_layout = Layout.row_major(Self.BM // 2, last_dim)
        comptime tiler_layout = Layout.row_major(1, iters)
        comptime smem_blocked_layout = blocked_product(
            smem_base_layout, tiler_layout, coalesce_output=True
        )

        accum_smem_tile = LayoutTensor[
            Self.output_type,
            smem_blocked_layout,
            address_space = AddressSpace.SHARED,
        ](o_smem)
        var warpy = local_row // 32

        @parameter
        for i in range(2):

            @parameter
            for j in range(iters):
                alias ofs = i * ST.frag_size + j * (ST.frag_size // iters)
                var rows_of_o_frags = LocalTensor[
                    Self.accum_type,
                    layout = Layout.row_major(1, ST.frag_size // iters),
                ](
                    o.ptr + ofs
                )  # all the repeats across n and m

                accum_smem_warp_tile = accum_smem_tile.tile[16, last_dim](
                    Int(2 * warpy + i), j
                )

                output_reg_to_smem_st_matrix[
                    BM=16,
                    padded_depth=last_dim,
                    swizzle=swizzle,
                    num_consumer=1,
                ](
                    lane,
                    local_warp_group_idx=0,
                    output_reg_tile=rows_of_o_frags,
                    accum_smem_tile=rebind[
                        LayoutTensor[
                            Self.output_type,
                            Layout.row_major(16, last_dim),
                            MutAnyOrigin,
                            address_space = AddressSpace.SHARED,
                        ]
                    ](accum_smem_warp_tile),
                )
        named_barrier[WARPGROUP_SIZE](Int32(warp_group_idx))

        ragged_tma_store.prefetch_descriptor()
        fence_async_view_proxy()

        # # first thread of each warp_group
        if thread_idx.x % 128 == 0:

            @parameter
            for itr in range(iters):
                var smem_tile = accum_smem_tile.tile[Self.BM // 2, last_dim](
                    0, itr
                )

                comptime sequence_length = Self.descriptor_shape[1]
                var tile_iter = smem_tile.tiled_iterator[
                    sequence_length, last_dim, axis=0
                ](0, 0)

                var coordinates = IndexList[
                    4
                ]()  # rest will be filled in by store_ragged_tile
                coordinates[0] = itr * last_dim
                coordinates[2] = head

                ragged_tma_store.store_ragged_tile[
                    using_max_descriptor_size=True
                ](
                    coordinates,
                    current_seq,
                    Int(num_output_rows),
                    tile_iter,
                )

            cp_async_bulk_commit_group()
        cp_async_bulk_wait_group[0]()

    @staticmethod
    @always_inline
    fn softmax(
        tmem_addr: UInt32,
        warp_idx: UInt32,
        mbars: FA4MiscMBars,
        o_mbar: MBarType,
        score_row: UInt32,
        seq_info: SeqInfo,
        mask: Self.MaskType,
        num_keys: UInt32,
        scale: Scalar[Self.accum_type],
        score_mod: Self.ScoreModType,
        max_seq_len: UInt32,
        o_ptr_arg: UnsafePointer[Scalar[Self.output_type]],
        ragged_tma_store: RaggedTensorMap[
            Self.output_type,
            Self.descriptor_shape,
            swizzle_mode = Self.config.swizzle_mode,
        ],
        o_smem: SharedMemPointer[Scalar[Self.output_type]],
        sink_weights: Self.SinkType,
    ):
        # FIXME: for depth 256
        var s_tmem: UInt32 = tmem_addr + Self.config.TMEM_S0

        var warp_group_idx: UInt32 = warp_idx // 4

        @parameter
        if Self.config.split_m:
            # split-M: second S is (+16 rows) in st-matrix space
            s_tmem += (16 << 16) * warp_group_idx
        else:
            # 2-Q path: S1 is at +BN columns
            s_tmem += Self.config.BN * warp_group_idx

        p_tmem = s_tmem
        c_tmem = p_tmem + Self.config.BN // 2
        s_tile = Self.UMMA0Type.CType(s_tmem)
        p_tile = Self.UMMA1Type.AType(p_tmem)

        pipeline_s = mbars.consumer_s(warp_group_idx)
        pipeline_c = mbars.producer_c(warp_group_idx)
        # TODO: order_s_wait/arrive
        order_s_wait = mbars.pipeline_order_wait(warp_group_idx)
        order_s_arrive = mbars.pipeline_order_arrive(warp_group_idx)
        var order_phase: UInt32 = 0

        var q_head_idx: UInt32 = seq_info.head_idx
        var tid: UInt32 = thread_idx.x
        var row: UInt32 = tid % 128
        var scale_log2e: Scalar[Self.accum_type] = scale

        @parameter
        if not (Self.use_score_mod or Self.MaskType.apply_log2e_after_mask):
            scale_log2e *= log2e

        @parameter
        @always_inline
        fn mask_row[
            BN: Int, //, masked: Bool, last_iter: Bool
        ](
            s: LocalTensor[Self.accum_type, Layout.row_major(BN)],
            kv_row: UInt32,
        ):
            apply_mask[
                decoding=False,
                use_score_mod = Self.use_score_mod,
                masked=masked,
                last_iter=last_iter,
            ](
                s,
                mask,
                score_mod,
                scale_log2e,
                prompt_idx=seq_info.prompt_idx,
                q_head_idx=q_head_idx,
                kv_tile_start_row=kv_row,
                max_seq_len=max_seq_len,
                num_keys=num_keys,
                score_row=score_row + tid,
            )

        # while waiting, offset output
        comptime splitBM = Self.BM // 2
        var num_output_rows = min(
            splitBM,
            Int32(seq_info.seq_len)
            - Int32(seq_info.prompt_offset)
            - Int32(warp_group_idx) * splitBM,
        )

        gmem_row = Self.PositionType.get_q_gmem_row[ragged = Self.ragged](
            seq_info, max_seq_len
        )
        gmem_col = seq_info.head_idx * Self.depth
        output_offset = Int(Self.depth * Self.num_q_heads) * Int(
            gmem_row
        ) + Int(gmem_col)
        var o_ptr: UnsafePointer[Scalar[Self.output_type]] = (
            o_ptr_arg
            + output_offset
            + warp_group_idx * (Self.PositionType.q_stride * splitBM)
        )

        pipeline_s.wait()
        tcgen05_fence_after()
        s = LocalTensor[
            Self.accum_type, Layout.row_major(Self.config.BN)
        ].stack_allocation()

        @parameter
        @always_inline
        fn load_mask_max[
            *, masked: Bool, last_iter: Bool
        ](kv_row: UInt32) -> Scalar[Self.accum_type]:
            # break up into sets of 32
            # minimize wait time by using smallest first
            comptime BM = Self.config.BM // 2
            comptime batch_size = 32
            comptime has_remainder = (Self.config.BN % batch_size) != 0
            comptime first_cols = (
                Self.config.BN % batch_size
            ) if has_remainder else batch_size
            s0 = TMemTile[Self.accum_type, BM, first_cols](s_tmem).load_async()
            tcgen05_load_wait()
            # if thread_idx.x == 0:
            #     print("s0[0:8] =", s0.vectorize[8]()[0])
            s1 = TMemTile[Self.accum_type, BM, batch_size](
                s_tmem + first_cols
            ).load_async()
            mask_row[masked=masked, last_iter=last_iter](s0, kv_row)
            # if thread_idx.x == 0:
            #     print("m0[0:8] =", s0.vectorize[8]()[0])
            vrow_max = maximum[width = Self.simd_size](s0)

            s.ptr.store(s0.ptr.load[width=first_cols]())
            # i = 0
            # offset0 = first_cols
            # offset1 = first_cols + batch_size
            # offset2 = first_cols + 2*batch_size
            # i = 1
            # offset0 = first_cols + 2*batch_size
            # offset1 = first_cols + 3*batch_size
            # offset2 = first_cols + 4*batch_size
            # i = 2
            # offset0 = first_cols + 4*batch_size
            # offset1 = first_cols + 5*batch_size
            # offset2 = first_cols + 6*batch_size
            comptime cols = Self.config.BN - first_cols + batch_size

            # Examples:
            # BN = 80, first_cols = 16, batch_size = 32
            # cols = 64; cols//64 = 1
            # (80-16+32)//64 = 1
            # 80 // 64 = 1
            # offsets = (16, 48, 80)
            #
            # BN = 96, first_cols = 32, batch_size = 32
            # cols = 64; cols//64 = 1
            # (96-32+32)//64 = 1
            # 96 // 64 = 1
            # offsets = (32, 64, 96)
            #
            # BN = 112, first_cols = 16, batch_size = 32
            # cols = 96; cols//64 = 1
            # (112-16+32)//64 = 2
            # 112 // 64 = 1
            # offsets = (16, 48, 80)
            # offsets = (80, 112, 144)
            #
            # BN = 128, first_cols = 32, batch_size = 32
            # cols = 96; cols//64 = 1
            # (128-32+32)//64 = 2
            # 128 // 64 = 2
            # offsets = (32, 64, 96)
            #
            # BN = 144, first_cols = 16, batch_size = 32
            # cols = 128; cols//64 = 2
            # (144-16+32)//64 = 2
            # 144 // 64 = 2
            # offsets = (16, 48, 80)
            # offsets = (80, 112, 144)
            #
            # BN = 160, first_cols = 32, batch_size = 32
            # cols = 128; cols//64 = 2
            # (160-32+32)//64 = 2
            # 160 // 64 = 2
            # offsets = (32, 64, 96)
            # offsets = (96, 128, 160)
            #
            # BN = 176, first_cols = 16, batch_size = 32
            # cols = 160; cols//64 = 2
            # (176-16+32)//64 = 3
            # 176 // 64 = 2
            # offsets = (16, 48, 80)
            # offsets = (80, 112, 144)
            # offsets = (144, 176, 208)
            @parameter
            for i in range(cols // (2 * batch_size)):
                comptime offset0 = first_cols + batch_size * (2 * i)
                comptime offset1 = first_cols + batch_size * (2 * i + 1)
                comptime offset2 = first_cols + batch_size * (2 * i + 2)

                tcgen05_load_wait()

                @parameter
                if offset1 >= Self.config.BN:
                    mask_row[masked=masked, last_iter=last_iter](
                        s1, kv_row + offset0
                    )
                    vrow_max = maximum(s1, vrow_max)
                    s.ptr.store(offset0, s1.ptr.load[width=batch_size]())
                else:
                    s2 = TMemTile[Self.accum_type, BM, batch_size](
                        s_tmem + offset1
                    ).load_async()
                    mask_row[masked=masked, last_iter=last_iter](
                        s1, kv_row + offset0
                    )
                    vrow_max = maximum(s1, vrow_max)
                    s.ptr.store(offset0, s1.ptr.load[width=batch_size]())
                    tcgen05_load_wait()

                    @parameter
                    if offset2 < Self.config.BN:
                        s1 = TMemTile[Self.accum_type, BM, batch_size](
                            s_tmem + offset2
                        ).load_async()
                    mask_row[masked=masked, last_iter=last_iter](
                        s2, kv_row + offset1
                    )
                    vrow_max = maximum(s2, vrow_max)
                    s.ptr.store(offset1, s2.ptr.load[width=batch_size]())

            return vrow_max.reduce_max()

        var kv_row: UInt32 = mask.start_column[
            Self.BM, Self.BN, Self.page_size
        ](score_row)
        comptime mask_sets = Self.MaskType.nonfull_sets[Self.BM, Self.BN]()
        comptime num_sets = len(mask_sets)
        var row_max: Scalar[Self.accum_type] = load_mask_max[
            masked=True, last_iter=True
        ](kv_row)
        var sink_weights_ptr = UnsafePointer[Scalar[Self.qkv_type]]()
        var sink_weight: Scalar[Self.accum_type]

        @parameter
        if not Self.SinkType.is_null:
            sink_weights_ptr = rebind[UnsafePointer[Scalar[Self.qkv_type]]](
                sink_weights.value()
            )
            var head_idx: UInt32 = seq_info.head_idx
            sink_weight = (
                sink_weights_ptr[head_idx].cast[Self.accum_type]() * log2e
            )
            row_max = max(row_max, sink_weight)
        else:
            sink_weights_ptr = UnsafePointer[Scalar[Self.qkv_type]]()
            sink_weight = 0.0

        @parameter
        @always_inline
        fn store_exp(
            row_max: Scalar[Self.accum_type],
        ) -> SIMD[Self.accum_type, 2]:
            comptime exp_simd = 2
            comptime vs_len = Self.config.BN // exp_simd  # 128 // 2 = 64
            comptime batch_size = 32
            comptime num_batch_iters = vs_len // batch_size
            comptime remainder = vs_len % batch_size
            constrained[num_batch_iters > 0]()
            comptime BatchTileType = TMemTile[
                Self.qkv_type, Self.config.BM // 2, batch_size * exp_simd
            ]
            comptime RemainderTileType = TMemTile[
                Self.qkv_type, Self.config.BM // 2, remainder * exp_simd
            ]
            constrained[(Self.config.BN % exp_simd) == 0]()

            vs = s.vectorize[exp_simd]()
            # We batch stores, e.g. use `tcgen_05.st.x32`.
            # If we have BN = 128, we would perform two such stores
            # (storing 64 elements as 32x bf16x2)
            #
            # Let `x` be the number of elements we add prior to storing.
            # If `x < 64`, with BN = 128, we have these live counts at
            # the two `tcgen_05.st.x32`:
            # 0. (BN - x) + 32
            # 1. (BN - x) + 32
            #
            # Thus, we can sum the first 32 elements, leaving the remaining 96
            # in registers until after we write.
            # The optimal solution for the number to do in advance is also
            # independent of the number of batches.
            comptime AccType = SIMD[Self.accum_type, exp_simd]
            var acc: AccType = exp2(rebind[AccType](vs[0]) - row_max)
            vs[0] = rebind[vs.element_type](acc)

            @parameter
            for i in range(1, batch_size // 2):
                vsi = exp2(rebind[AccType](vs[i]) - row_max)
                vs[i] = rebind[vs.element_type](vsi)
                acc += vsi

            # at this point, we need 32 fewer fp32 registers but 16 more u32
            @parameter
            for i in range(batch_size // 2, batch_size):
                vs[i] = exp2(vs[i] - row_max)

            BatchTileType(p_tmem).store(
                LocalTensor[
                    Self.accum_type, Layout.row_major(batch_size * exp_simd)
                ](s.ptr)
            )

            @parameter
            for b in range(1, num_batch_iters):
                comptime offset = batch_size * b

                @parameter
                for i in range(offset, offset + batch_size):
                    vs[i] = exp2(vs[i] - row_max)

                comptime el_offset = offset * exp_simd
                comptime tmem_offset = (
                    el_offset * size_of[Self.qkv_type]()
                ) // size_of[Self.accum_type]()
                BatchTileType(p_tmem + tmem_offset).store(
                    LocalTensor[
                        Self.accum_type, Layout.row_major(batch_size * exp_simd)
                    ](s.ptr + el_offset)
                )

            @parameter
            if remainder > 0:
                comptime offset = batch_size * num_batch_iters

                @parameter
                for i in range(offset, offset + remainder):
                    vs[i] = exp2(vs[i] - row_max)

                comptime el_offset = offset * exp_simd
                comptime tmem_offset = (
                    el_offset * size_of[Self.qkv_type]()
                ) // size_of[Self.accum_type]()
                RemainderTileType(p_tmem + tmem_offset).store(
                    LocalTensor[
                        Self.accum_type, Layout.row_major(remainder * exp_simd)
                    ](s.ptr + el_offset)
                )

            tcgen05_store_wait()
            tcgen05_fence_before()
            pipeline_s.release()
            # now we can sum the remaining elements of `acc`
            acc0 = vs[batch_size // 2]
            acc1 = vs[batch_size // 2 + 1]
            acc2 = vs[batch_size // 2 + 2] + vs[batch_size // 2 + 3]

            @parameter
            for i in range(batch_size // 2 + 4, vs_len, 4):
                acc += rebind[AccType](vs[i])
                acc0 += vs[i + 1]
                acc1 += vs[i + 2]
                acc2 += vs[i + 3]
            return (acc + rebind[AccType](acc0)) + rebind[AccType](acc1 + acc2)

        var row_sum: SIMD[Self.accum_type, 2] = store_exp(row_max)

        var o_phase: UInt32 = 0  # initial wait is phase 0

        @parameter
        if not Self.SinkType.is_null:
            row_sum[0] += exp2(sink_weight - row_max)

        # TODO: add ordering barriers to prevent overlap
        # between the two softmax warpgroups
        @parameter
        if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
            mask_ends = mask.masked_set_ends[
                BM = Self.BM, BN = Self.BN, page_size = Self.page_size
            ](score_row, num_keys)
            var decrement: Bool = True

            @parameter
            for i in range(num_sets):
                comptime mask_status = mask_sets[i]
                var iters: UInt32

                @parameter
                if i == 0:
                    iters = mask_ends[i]
                else:
                    iters = mask_ends[i] - mask_ends[i - 1]
                if decrement and iters > 0:
                    iters -= 1
                    decrement = False
                while iters != 0:
                    iters -= 1
                    kv_row += Self.config.BN
                    pipeline_s.wait()
                    # calculate rowmax
                    old_max = row_max
                    var new_row_max: Scalar[Self.accum_type]

                    # last_iter == (i + 1 == num_sets) and (i == 0)
                    # `i == 0` is runtime; for now, we set to `True`
                    # as this number of iterations is small
                    comptime last_iter: Bool = i + 1 == num_sets
                    comptime masked: Bool = mask_status == TileMaskStatus.PARTIAL_MASK
                    new_row_max = load_mask_max[
                        masked=masked, last_iter=last_iter
                    ](kv_row)
                    row_max = max(old_max, new_row_max)
                    correction = exp2(old_max - row_max)
                    pipeline_c.acquire()
                    tcgen05_st[
                        datapaths=32,
                        bits=32,
                        repeat=1,
                        pack=False,
                    ](c_tmem, correction)
                    pipeline_c.commit()
                    # update s->p
                    local_rowsum = store_exp(row_max)
                    row_sum = row_sum.fma(correction, local_rowsum)
                    o_phase ^= 1
        else:
            while True:
                kv_row += Self.config.BN
                if kv_row >= num_keys:
                    break
                mask_status = Self.mask_status(mask, score_row, kv_row)
                if mask_status == TileMaskStatus.FULL_MASK:
                    continue
                pipeline_s.wait()
                # calculate rowmax
                old_max = row_max
                var new_row_max: Scalar[Self.accum_type]
                if mask_status == TileMaskStatus.PARTIAL_MASK:
                    new_row_max = load_mask_max[masked=True, last_iter=True](
                        kv_row
                    )
                else:
                    new_row_max = load_mask_max[masked=False, last_iter=True](
                        kv_row
                    )
                row_max = max(old_max, new_row_max)
                correction = exp2(old_max - row_max)
                pipeline_c.acquire()
                tcgen05_st[
                    datapaths=32,
                    bits=32,
                    repeat=1,
                    pack=False,
                ](c_tmem, correction)
                pipeline_c.commit()
                # update s->p
                local_rowsum = store_exp(row_max)
                row_sum = row_sum.fma(correction, local_rowsum)
                o_phase ^= 1
        # Do the final correction and write
        inv_row_sum = recip(row_sum.reduce_add())
        o_tile = Self.UMMA1Type.CType(
            tmem_addr
            + Self.config.TMEM_O0
            + warp_group_idx * Self.config.padded_depth
        )
        # wait on the o_pipeline producer
        constrained[size_of[Self.output_type]() == size_of[Self.qkv_type]()]()
        if num_output_rows > 0:
            o_mbar[warp_group_idx].wait(o_phase)  # consumer wait
            tcgen05_fence_after()  # example 1
            # TODO: pass in a dedicated barrier that a q-writer can wait on in a persistent kernel?

            var start_seq = Self.PositionType.get_q_gmem_row[
                ragged = Self.ragged
            ](seq_info, max_seq_len)

            var wg_seq = Int(start_seq) + Int(warp_group_idx * (Self.BM // 2))

            Self.scale_write_output(
                row,
                inv_row_sum,
                o_smem
                + warp_group_idx
                * (Self.config.BM // 2 * Self.config.padded_depth),
                o_tile,
                o_ptr_arg,
                ragged_tma_store,
                warp_group_idx,
                o_mbar + 2 + warp_group_idx,  # consumer arrive
                wg_seq,
                num_output_rows,
            )
        named_barrier[2 * WARPGROUP_SIZE](2)
        if warp_idx == 0:
            tcgen05_release_allocation_lock[Self.cta_group]()
            tcgen05_dealloc[Self.cta_group](
                tmem_addr, Self.config.sm100_tmem_cols
            )

    @staticmethod
    @always_inline
    fn correction(
        tmem_addr: UInt32,
        mbars: FA4MiscMBars,
        o_mbar: MBarType,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
    ):
        constrained[size_of[Self.accum_type]() == 4]()

        o0_tmem = tmem_addr + Self.config.TMEM_O0
        o1_tmem = tmem_addr + Self.config.TMEM_O1
        c0_tmem = tmem_addr + Self.config.TMEM_C0
        c1_tmem = tmem_addr + Self.config.TMEM_C1

        pipeline_c0 = mbars.consumer_c0()
        pipeline_c1 = mbars.consumer_c1()
        pipeline_o = ConsumerPipeline[2](o_mbar)

        var iter_count: UInt32 = (
            mask.total_iters[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )

        comptime batch_size = 16
        # output is BM x depth
        comptime load_iters = Self.config.depth // (2 * batch_size)
        comptime load_remainder = Self.config.depth % (2 * batch_size)

        while iter_count != 0:
            iter_count -= 1

            @parameter
            for i in range(2):
                var c_tmem: UInt32

                @parameter
                if i == 0:
                    c_tmem = c0_tmem
                    pipeline_c0.wait()
                else:
                    c_tmem = c1_tmem
                    pipeline_c1.wait()

                # correct
                c_scalar = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=1,
                    dtype = Self.accum_type,
                    pack=False,
                    width=1,
                ](c_tmem)
                tcgen05_load_wait()

                @parameter
                if i == 0:
                    pipeline_c0.release()
                else:
                    pipeline_c1.release()

                change = _vote_nvidia_helper(c_scalar != 1) != 0
                pipeline_o.wait()
                if change:
                    # TODO: experiment with different batch sizes.
                    # The idea here is to both pipeline, and reduce peak register use.
                    constrained[load_iters > 1]()
                    constrained[Self.config.depth % batch_size == 0]()

                    var o_tmem: UInt32

                    @parameter
                    if i == 0:
                        o_tmem = o0_tmem
                    else:
                        o_tmem = o1_tmem

                    var o_b0: SIMD[Self.accum_type, batch_size]
                    var o_b1: SIMD[Self.accum_type, batch_size]
                    o_b0 = tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=batch_size,
                        dtype = Self.accum_type,
                        pack=False,
                        width=batch_size,
                    ](o_tmem)

                    @parameter
                    for b in range(load_iters):
                        tcgen05_load_wait()  # ob0 loaded
                        # BN=64 or BN=80, load_iters=2
                        # b=0
                        # b0_offset0=0
                        # b1_offset =16
                        # b0_offset1=32
                        # b=1
                        # b0_offset0=32
                        # b1_offset =48
                        # b0_offset1=64
                        comptime b0_offset0 = 2 * b * batch_size
                        comptime b1_offset = b0_offset0 + batch_size
                        comptime b0_offset1 = b1_offset + batch_size
                        o_b1 = tcgen05_ld[  # 0b1 start
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            dtype = Self.accum_type,
                            pack=False,
                            width=batch_size,
                        ](o_tmem + b1_offset)
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            pack=False,
                        ](o_tmem + b0_offset0, o_b0 * c_scalar)
                        tcgen05_load_wait()  # ob1 loaded

                        @parameter
                        if b0_offset1 + batch_size <= Self.config.depth:
                            o_b0 = tcgen05_ld[  # 0b0 start
                                datapaths=32,
                                bits=32,
                                repeat=batch_size,
                                dtype = Self.accum_type,
                                pack=False,
                                width=batch_size,
                            ](o_tmem + b0_offset1)
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            pack=False,
                        ](o_tmem + b1_offset, o_b1 * c_scalar)

                    @parameter
                    if load_remainder > 0:
                        tcgen05_load_wait()  # ob1 loaded
                        comptime offset = 2 * batch_size * load_iters
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=load_remainder,
                            pack=False,
                        ](o_tmem + offset, o_b0 * c_scalar)
                    tcgen05_store_wait()
                    tcgen05_fence_before()
                pipeline_o.release()

    @staticmethod
    @always_inline
    fn load(
        mbars: FA4MiscMBars,
        kv_pipeline_arg: Self.KVPipelineType,
        score_row: UInt32,
        num_keys: UInt32,
        seq_info: SeqInfo,
        max_seq_len: Self.MaxSeqLenType,
        mask: Self.MaskType,
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            depth = Self.config.BK0,
            group = Self.config.group,
            decoding=False,
        ],
        k_tma_op: TMANestedTensorTile[
            Self.KVLUTType.dtype,
            Self.config.BN,
            Self.config.padded_depth,
            Self.config.swizzle_mode,
            is_k_major=True,
        ],
        v_tma_op: TMANestedTensorTile[
            Self.KVLUTType.dtype,
            Self.config.BN,
            Self.config.padded_depth,
            Self.config.swizzle_mode,
            is_k_major=False,
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
        comptime KVPipeType = KVProducerPipeline[
            Self.KVLUTType.dtype, Self.config
        ]

        # If two-qo, we produce qkv in a pattern of
        # q0 & k0, q1, v0, k1, v1, k2, v2...
        comptime SMemTensor[layout: Layout] = SharedMemTensor[
            Self.KVLUTType.dtype, layout
        ]
        comptime QType = SMemTensor[type_of(q_tma_op).layout]
        comptime KType = SMemTensor[type_of(k_tma_op).layout]
        comptime VType = SMemTensor[type_of(v_tma_op).layout]
        constrained[
            QType.layout
            == tile_layout_k_major[
                Self.qkv_type,
                Self.config.BM // 2,
                Self.config.BK0,
                Self.config.swizzle_mode,
            ]()
        ]()
        constrained[KType.layout == KVPipeType.KType.layout]()
        constrained[VType.layout == KVPipeType.VType.layout]()

        var kv_col: UInt32 = kv_lut.col_idx(seq_info.head_idx // Self.group)

        comptime q_elements = (Self.config.BM // 2) * Self.config.BK0
        comptime q_bytes = size_of[Self.qkv_type]() * q_elements

        kv_smem = q_smem + Self.config.BM * Self.config.padded_depth
        var pipeline_kv: KVPipeType = {kv_pipeline_arg, kv_smem}

        var mbark0: KVPipeType.KPairType

        mbark0 = pipeline_kv.get_k[mma_stage=0, expect=False]()  # no wait
        var q_gmem_row: UInt32 = Self.PositionType.get_q_gmem_row[
            ragged = Self.ragged
        ](seq_info, max_seq_len)
        var q_col: UInt32 = seq_info.head_idx * Self.depth
        elect = elect() != 0
        # copy q0
        if elect:
            # Q0
            mbark0.mbar[].expect_bytes(pipeline_kv.kv_bytes + q_bytes)
            q_tma_op.async_copy(
                QType(q_smem),
                mbark0.mbar[],
                (UInt(q_col), UInt(q_gmem_row)),
            )
        var kv_row: UInt32 = mask.start_column[
            Self.BM, Self.BN, Self.page_size
        ](score_row)
        var kv_gmem_row: UInt32 = kv_lut.row_idx(seq_info.prompt_idx, kv_row)
        var iter_count: UInt32 = (
            mask.last_masked_set_end[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )
        # copy k0
        if elect:
            # K0
            k_tma_op.async_copy(
                mbark0.smem,
                mbark0.mbar[],
                (UInt(kv_col), UInt(kv_gmem_row)),
            )
        pipeline_kv.commit_kv_step()
        if elect:
            ref q1_mbar = mbars.q1_wait_mbar()
            q1_mbar.expect_bytes(q_bytes)
            # Q1
            q_tma_op.async_copy(
                QType(q_smem + q_elements),
                q1_mbar,
                (
                    UInt(q_col),
                    UInt(q_gmem_row + Self.config.BM // 2),
                ),
            )
        # copy v0
        if elect:
            mbarv0 = pipeline_kv.get_v[mma_stage=0]()
            v_tma_op.async_copy(
                mbarv0.smem,
                mbarv0.mbar[],
                (UInt(kv_col), UInt(kv_gmem_row)),
            )
        pipeline_kv.commit_kv_step()
        comptime check_mask = mask.nonfull_sets[Self.BM, Self.BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK
        # kv producer loop
        while iter_count != 0:
            iter_count -= 1
            kv_row += Self.config.BN

            @parameter
            if check_mask:
                if (
                    Self.mask_status(mask, score_row, kv_row)
                    == TileMaskStatus.FULL_MASK
                ):
                    continue
            kv_gmem_row = kv_lut.row_idx(seq_info.prompt_idx, kv_row)
            # produce k
            pipeline_kv.acquire_kv()
            if elect:
                mbarkn = pipeline_kv.get_k[mma_stage=0]()
                k_tma_op.async_copy(
                    mbarkn.smem,
                    mbarkn.mbar[],
                    (UInt(kv_col), UInt(kv_gmem_row)),
                )
            pipeline_kv.commit_kv_step()
            pipeline_kv.acquire_kv()
            if elect:
                mbarvn = pipeline_kv.get_v[mma_stage=0]()
                v_tma_op.async_copy(
                    mbarvn.smem,
                    mbarvn.mbar[],
                    (UInt(kv_col), UInt(kv_gmem_row)),
                )
            pipeline_kv.commit_kv_step()

    @staticmethod
    @always_inline
    fn descriptor_q(
        q_smem: SharedMemPointer[Scalar[Self.qkv_type]],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN = Self.config.BM // 2,
            BK = Self.config.BK0,
            swizzle_mode = Self.config.swizzle_mode,
            is_k_major=True,
        ](q_smem)

    @staticmethod
    @always_inline
    fn mma(
        tmem_addr: UInt32,
        mbars: FA4MiscMBars,
        kv_pipeline_arg: Self.KVPipelineType,
        o_mbar: MBarType,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
        comptime KVPipeType = KVConsumerPipeline[
            Self.KVLUTType.dtype, Self.config
        ]

        s0_tmem = tmem_addr + Self.config.TMEM_S0
        s1_tmem = tmem_addr + Self.config.TMEM_S1
        o0_tmem = tmem_addr + Self.config.TMEM_O0
        o1_tmem = tmem_addr + Self.config.TMEM_O1

        producer_s0 = mbars.producer_s0().mbar  # phase = 1
        consumer_s0 = producer_s0 + 1
        producer_s1 = mbars.producer_s1().mbar  # phase = 1
        consumer_s1 = producer_s1 + 1
        pipeline_o_initial = ProducerPipeline[2](o_mbar)  # phase = 1
        producer_o0 = pipeline_o_initial.mbar
        producer_o1 = producer_o0 + 1
        consumer_o0 = producer_o1 + 1
        consumer_o1 = consumer_o0 + 1

        comptime q0_size = (Self.config.BM // 2) * Self.config.padded_depth
        comptime q0_bytes = q0_size * size_of[Self.KVLUTType.dtype]()
        q0 = Self.descriptor_q(q_smem)
        q1 = q0 + q0_bytes
        kv_smem = q_smem + 2 * q0_size

        var pipeline_kv: KVPipeType = {kv_pipeline_arg, kv_smem}

        # We peel the first iteration, as we want to wait on q1
        var iter_count: UInt32 = (
            mask.total_iters[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )

        # Q_0 @ K_0'
        k0 = pipeline_kv.wait_k[mma_stage=0, pre_increment=False]()  # [kv0]
        e = elect()
        Self.UMMA0Type.mma(q0, k0, s0_tmem, elect=e, c_scale=0)
        elect_mma_arrive(producer_s0, e)
        # pipeline_s0.step()  # pipline_s0.phase = 0

        # Q_1 @ K_0'
        # pipeline_s1.producer_acquire()
        mbars.q1_wait_mbar().wait()  # wait on Q1
        # we don't need to wait on s1
        Self.UMMA0Type.mma(q1, k0, s1_tmem, elect=e, c_scale=0)
        elect_mma_arrive(producer_s1, e)

        pipeline_kv.release_k(e)  # [kv0]->kv1

        vlatest = pipeline_kv.wait_v[mma_stage=0]()  # [kv1]
        # For the first V tile in the current KV stage buffer:
        # Use the SAME base pointer you used for K (no manual offset).
        _ = consumer_s0[].wait(0)
        Self.UMMA1Type.mma(s0_tmem, vlatest, o0_tmem, elect=e, c_scale=0)
        elect_mma_arrive(producer_o0, e)
        var phase_s: UInt32 = 0
        var phase_o: UInt32 = 1

        var c_scale: UInt32 = 0
        # wait order
        # s0.wait(1)              # Q0@K0'
        # s1.wait(1)              # Q1@K0'
        # s0.wait(0), o0.wait(1)  # P0@V0
        # s1.wait(0), o1.wait(1)  # P1@V0

        while iter_count != 0:
            iter_count -= 1
            # Q_0 @ K_n'
            kn = pipeline_kv.wait_k[mma_stage=0]()  # kv_{2n-1}->[kv_{2n}]
            Self.UMMA0Type.mma(q0, kn, s0_tmem, elect=e, c_scale=0)
            elect_mma_arrive(producer_s0, e)

            # O_1 + P_1 @ V_{n-1}
            _ = consumer_o1[].wait(phase_o)
            # pipeline_o.acquire()
            _ = consumer_s1[].wait(phase_s)
            # pipeline_s1.acquire()
            Self.UMMA1Type.mma(
                s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
            )
            elect_mma_arrive(producer_o1, e)
            # pipeline_o.step()
            phase_o = phase_s
            c_scale = 1
            pipeline_kv.release_v(e)  # [kv_{2n-1}]

            # Q_1 @ K_n'
            Self.UMMA0Type.mma(q1, kn, s1_tmem, elect=e, c_scale=0)
            elect_mma_arrive(producer_s1, e)
            phase_s ^= 1

            pipeline_kv.release_k(e)  # [kv_{2n}]->kv_{2n+1}

            # O_0 + P_0 @ V_n
            vlatest = pipeline_kv.wait_v[mma_stage=0]()  # [kv_{2n+1}]
            _ = consumer_o0[].wait(phase_o)
            # pipeline_o.acquire()
            _ = consumer_s0[].wait(phase_s)
            # pipeline_s0.acquire()
            Self.UMMA1Type.mma(s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1)
            elect_mma_arrive(producer_o0, e)

        _ = consumer_o1[].wait(phase_o)
        _ = consumer_s1[].wait(phase_s)
        Self.UMMA1Type.mma(s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale)
        elect_mma_arrive(producer_o1, e)

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

from collections import OptionalReg
from math import ceildiv, exp2, recip, align_up, align_down, gcd, iota
from math.constants import log2e
from sys import simd_width_of, size_of, _RegisterPackType
import gpu.primitives.warp as warp
from bit import prev_power_of_two, pop_count
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx,
    block_idx,
    warp_id,
)
from gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from gpu.primitives.cluster import elect_one_sync
from gpu.host import DeviceContext, FuncAttribute, DeviceBuffer
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory, CacheEviction
from gpu.compute.mma import MMAOperandDescriptor
from gpu.compute.arch.mma_nvidia_sm100 import (
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
from gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)
from gpu.primitives.warp import _vote_nvidia_helper
from layout.int_tuple import IntTuple, UNKNOWN_VALUE
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout._layout import Layout as InternalLayout, row_major
from layout._tile_tensor import TileTensor
from layout._tile_tensor import stack_allocation as tt_stack_allocation
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    RaggedTMA3DTile,
)
from logger import Logger
from memory import bitcast
from nn.mha_fa3_utils import (
    get_seq_info,
    KVTMATile,
    kv_coord,
    MHAPosition,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    output_reg_to_smem_st_matrix,
    _LocalTT,
    _SharedMemTT,
    Pack,
    PositionSummary,
    produce,
    q_coord,
    q_tma,
    QTMATile,
)
from nn.mha_mask import MHAMask, TileMaskStatus, MASK_VALUE, MaskStrategy
from nn.mha_operand import MHAOperand, LayoutTensorMHAOperand
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
from utils.numerics import min_or_neg_inf
from utils.static_tuple import StaticTuple
from linalg.arch.sm100.mma import smem_descriptor
from kv_cache.types import swizzle_granularity

from sys import env_get_bool
from sys._assembly import inlined_assembly


comptime logger = Logger()

# TileTensor-based aliases for storage (native types)
comptime LocalTensor[
    dtype: DType,
    layout: InternalLayout,
] = TileTensor[
    dtype,
    InternalLayout[
        shape_types = layout.shape_types,
        stride_types = layout.stride_types,
    ],
    MutExternalOrigin,
    address_space = AddressSpace.LOCAL,
]
comptime SharedMemTensor[dtype: DType, layout: InternalLayout] = TileTensor[
    dtype,
    InternalLayout[
        shape_types = layout.shape_types,
        stride_types = layout.stride_types,
    ],
    MutExternalOrigin,
    address_space = AddressSpace.SHARED,
]

# Legacy LayoutTensor aliases for TMA/MMA API boundaries
comptime LocalLT[
    dtype: DType, layout: Layout, element_layout: Layout = Layout(1, 1)
] = LayoutTensor[
    dtype,
    layout,
    MutAnyOrigin,
    address_space = AddressSpace.LOCAL,
    element_layout=element_layout,
]
comptime SharedMemLT[dtype: DType, layout: Layout] = LayoutTensor[
    dtype,
    layout,
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
    layout_int_type = DType.int32,
    linear_idx_type = DType.int32,
    alignment=128,
]
comptime SharedMemPointer[type: AnyType] = UnsafePointer[
    type, MutAnyOrigin, address_space = AddressSpace.SHARED
]
comptime MBarType = SharedMemPointer[SharedMemBarrier]

comptime EnableForcedOrdering = env_get_bool[
    "FA4ForcedSoftmaxOrdering", False
]()
comptime EnableEarlyAdd = env_get_bool["FA4AddEarly", False]()


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
    origins: OriginSet,
    //,
    func: fn[pow_two: Int, offset: Int]() capturing[origins] -> None,
    N: Int,
    *,
    max_value: Int = 128,
]():
    comptime power_of_two = prev_power_of_two(min(max_value, N))

    comptime for offset in range(0, N, power_of_two):
        comptime iter_size = min(N - offset, power_of_two)

        comptime if iter_size == power_of_two:
            func[power_of_two, offset]()
        else:
            comptime for j in range(pop_count(iter_size)):
                comptime pow_two = extract_power_of_two(iter_size, j)
                comptime coffset = offset + cumulative_power_of_two(
                    iter_size, j
                )
                func[pow_two, coffset]()
    # final call for possible pipeline cleanup
    func[0, N]()


struct STMatrixLayout[
    BM: Int,
    BN: Int,
    *,
    num_threads: Int,
    accum_type_size: Int,
](TrivialRegisterPassable):
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
    comptime TensorType[dtype: DType] = LocalLT[
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


struct STMatrixOffsets[
    BM: Int,
    BN: Int,
    *,
    num_threads: Int,
    accum_type_size: Int,
    curr_repeat: Int,
    cumulative_repeat: Int,
    m_mma: Int,
](TrivialRegisterPassable):
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


struct TMemTile[
    dtype_: DType,
    BM: Int,
    BN: Int,
](TrivialRegisterPassable):
    comptime dtype: DType = Self.dtype_
    comptime dtype_size = size_of[Self.dtype]()
    comptime num_m_tiles = Self.BM // 64

    var tmem_addr: UInt32

    @always_inline
    fn __init__(out self, tmem_addr: UInt32):
        self.tmem_addr = tmem_addr

    @always_inline
    fn __getitem__(self, i: UInt32) -> Self:
        return {self.tmem_addr + i * UInt32(Self.BN)}

    @always_inline
    fn offset[m_mma: Int, n_mma: Int](self) -> UInt32:
        comptime if m_mma == 0 and n_mma == 0:
            return self.tmem_addr
        else:
            comptime linear = _tmem_offset[
                Self.dtype, MMA_N = Self.BN, m_mma=m_mma, n_mma=n_mma
            ]()

            return self.tmem_addr + UInt32(linear)

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
        comptime assert Self.dtype_size <= 4
        ptr = src.ptr.bitcast[UInt32]()
        comptime st_mat_layout = STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_type_size = Self.dtype_size,
        ]
        comptime assert st_mat_layout.bits == 128 or st_mat_layout.bits == 256

        @parameter
        @always_inline
        fn store_fn[pow_two: Int, offset: Int]():
            # pow_two is current repeat, offset total so far
            comptime if pow_two > 0:
                comptime for m_mma in range(st_mat_layout.num_m_tiles):
                    comptime offsets = STMatrixOffsets[
                        Self.BM,
                        Self.BN,
                        num_threads=num_threads,
                        accum_type_size = Self.dtype_size,
                        curr_repeat=pow_two,
                        cumulative_repeat=offset,
                        m_mma=m_mma,
                    ]()
                    tmem = self.tmem_addr + UInt32(offsets.tmem_offset)
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
        comptime assert (
            Self.dtype_size <= 4
        ), "Loading for st matrix requires elements to be <= 4 bytes."
        comptime st_mat_layout = STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_type_size = Self.dtype_size,
        ]()
        comptime assert (st_mat_layout.num_m_tiles == 1) or (
            st_mat_layout.num_m_tiles == 2
        ), (
            "Only 1 or 2 m tiles are supported, but"
            " st_mat_layout.num_m_tiles == "
            + String(st_mat_layout.num_m_tiles)
        )
        comptime repeat = st_mat_layout.repeat
        comptime frag_size_b32 = st_mat_layout.frag_size * Self.dtype_size // 4

        dst = type_of(dst).stack_allocation()
        comptime load_dtype = DType.uint32
        var ptr: UnsafePointer[
            Scalar[load_dtype], MutAnyOrigin, address_space = AddressSpace.LOCAL
        ]

        ptr = rebind[type_of(ptr)](dst.ptr)

        @parameter
        @always_inline
        fn load_fn[pow_two: Int, offset: Int]():
            comptime assert pow_two + offset <= repeat

            comptime if pow_two > 0:
                comptime for m_mma in range(st_mat_layout.num_m_tiles):
                    comptime offsets = STMatrixOffsets[
                        Self.BM,
                        Self.BN,
                        num_threads=num_threads,
                        accum_type_size = Self.dtype_size,
                        curr_repeat=pow_two,
                        cumulative_repeat=offset,
                        m_mma=m_mma,
                    ]()
                    tmem = self.tmem_addr + UInt32(offsets.tmem_offset)
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
        out dst: LocalTensor[Self.dtype, row_major[Self.BN]()],
    ):
        dst = tt_stack_allocation[
            dtype = Self.dtype, address_space = AddressSpace.LOCAL
        ](row_major[Self.BN]())
        comptime repeat = Self.dtype_size * Self.BN // 4
        comptime dtype = Self.dtype if Self.dtype_size == 4 else DType.uint32

        @parameter
        @always_inline
        fn load_fn[pow_two: Int, offset: Int]():
            comptime if pow_two > 0:
                comptime if dtype == Self.dtype:
                    frag0 = tcgen05_ld[
                        datapaths=32,  # first dimension of the shape
                        bits=32,  # second dimension of the shape
                        repeat=pow_two,
                        dtype = Self.dtype,
                        pack=False,
                        width=pow_two,
                    ](self.tmem_addr + UInt32(offset))
                    dst.ptr.store(offset, frag0)
                else:
                    frag1 = tcgen05_ld[
                        datapaths=32,  # first dimension of the shape
                        bits=32,  # second dimension of the shape
                        repeat=pow_two,
                        dtype = DType.uint32,
                        pack=False,
                        width=pow_two,
                    ](self.tmem_addr + UInt32(offset))
                    dst.ptr.bitcast[UInt32]().store[width=pow_two](
                        offset, frag1
                    )

        break_into_powers_of_two[func=load_fn, N=repeat, max_value=128]()

    @always_inline
    fn store_async[
        src_type: DType
    ](self, src: LocalTensor[src_type, row_major[Self.BN]()]):
        @parameter
        @always_inline
        fn store_fn[pow_two: Int, offset: Int]():
            comptime if pow_two > 0:
                var frag: SIMD[DType.uint32, pow_two * Self.dtype_size // 4]

                comptime if src_type == Self.dtype:
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
                ](self.tmem_addr + UInt32(offset * Self.dtype_size // 4), frag)

        break_into_powers_of_two[func=store_fn, N = Self.BN, max_value=128]()


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
](TrivialRegisterPassable):
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
    comptime num_k_mmas = ceildiv(Self.BK, Self.MMA_K)
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
    @always_inline("nodebug")
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
        comptime if Self.num_stages == 1:
            # Original single-stage behavior
            bulk_mma[
                Self.a_layout,
                Self.b_layout,
                num_k_mmas = Self.num_k_mmas,
                operand_size = Self.operand_size,
            ](Self.idesc, a, b, c, c_scale, elect)
        else:
            comptime k_batch_start = Self.num_k_blocks_per_stage * stage_idx
            comptime k_batch_end = min(
                Self.num_k_blocks_per_stage * (stage_idx + 1), Self.num_k_mmas
            )
            comptime k_offset = k_batch_start * Self.MMA_K
            # Offset both A and B descriptors by k_offset
            comptime a_byte_offset = (
                Self.a_layout(IntTuple(0, k_offset)) * Self.operand_size
            )
            comptime b_byte_offset = (
                Self.b_layout(IntTuple(0, k_offset)) * Self.operand_size
            )
            var scale: UInt32

            comptime if stage_idx == 0:
                scale = c_scale
            else:
                scale = 1
            bulk_mma[
                Self.a_layout,
                Self.b_layout,
                num_k_mmas = k_batch_end - k_batch_start,
                operand_size = Self.operand_size,
            ](
                Self.idesc,
                a + UInt32(a_byte_offset),
                b + UInt32(b_byte_offset),
                c,
                scale,
                elect,
            )


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
](TrivialRegisterPassable):
    comptime operand_t: DType = Self.operand_type
    comptime accum_t: DType = Self.accum_type

    comptime operand_size = size_of[Self.operand_type]()
    comptime swizzle_granularity = Self.swizzle_b.bytes() // Self.operand_size
    # BN here is depth
    comptime b_layout = tile_layout_k_major[
        Self.operand_t, Self.MMA_N, Self.BK, Self.swizzle_b
    ]() if Self.transpose_b else tile_layout_mn_major[
        Self.operand_t, Self.MMA_N, Self.BK, Self.swizzle_b
    ]()

    comptime MMA_K = 16
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime num_k_blocks = Self.padded_BK // Self.MMA_K
    comptime use_3_then_1_split: Bool = Self.num_stages == 2
    comptime num_k_blocks_per_stage = Self.num_k_blocks // (
        4 if Self.use_3_then_1_split else Self.num_stages
    )

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
    @always_inline("nodebug")
    fn mma[
        *, stage_idx: Int = 0
    ](a: UInt32, b: Self.BType, c: UInt32, *, c_scale: UInt32, elect: Int32):
        comptime if Self.num_stages == 1:
            # Original single-stage behavior
            bulk_mma[
                Self.b_layout,
                num_k_mmas = Self.num_k_mmas,
                operand_size = Self.operand_size,
            ](Self.idesc, a, b, c, c_scale, elect)
        else:
            comptime start = 3 * stage_idx if Self.use_3_then_1_split else stage_idx
            comptime end = stage_idx + 3 if Self.use_3_then_1_split else stage_idx + 1
            comptime k_batch_start = Self.num_k_blocks_per_stage * start
            comptime k_batch_end = min(
                Self.num_k_blocks_per_stage * end, Self.num_k_mmas
            )
            comptime k_offset = k_batch_start * Self.MMA_K
            # P (tmem) offset: move by stage_idx * k_per_stage columns
            # P is MMA_M x BK, so column offset is k_per_stage * dtype_size / 4 (in tmem units)
            comptime a_tmem_offset = (k_offset * Self.operand_size) // 4
            # V (smem) offset: move by stage_idx * k_per_stage rows
            comptime b_byte_offset = (
                Self.b_layout(IntTuple(0, k_offset)) * Self.operand_size
            )

            comptime if stage_idx == 0:
                scale = c_scale
            else:
                scale = 1
            bulk_mma[
                Self.b_layout,
                num_k_mmas = k_batch_end - k_batch_start,
                operand_size = Self.operand_size,
            ](
                Self.idesc,
                a + UInt32(a_tmem_offset),
                b + UInt32(b_byte_offset),
                c,
                scale,
                elect,
            )


struct FA4Config(TrivialRegisterPassable):
    var MMA_M: Int
    var BM: Int
    var BN: Int
    var BK0: Int  # BK for MMA0
    var BK1: Int  # BK for MMA1
    var depth: Int
    var padded_depth: Int  # align_up(depth, 64)
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
    var num_qk_stages: Int  # Stages for Q@K' (K loading pipelining)
    var num_pv_stages: Int  # Stages for P@V (P writing pipelining)
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
        is_mla: Bool = False,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.depth = depth
        self.split_m = depth > 128 and not is_mla
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
                    (Self.sm100_tmem_cols // 2 - self.padded_depth),
                    Self.MMA_K,
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
        # we need two per stage
        # Compute staging for Q@K' and P@V operations
        # num_qk_stages: Controls how K loading is pipelined for Q@K' MMA
        # num_pv_stages: Controls how P writing is pipelined for P@V MMA
        #
        # For Q@K': K can be loaded in stages, MMA starts after first stage arrives
        # For P@V: V must be complete, but P writing can be staged to unblock MMA sooner
        #
        # Divisibility constraints:
        # - num_qk_stages must divide padded_depth (for K column splitting)
        # - num_pv_stages must divide BN (for P column splitting)
        # - Both must respect MMA_K alignment (16 elements)
        #
        # Staging infrastructure:
        # - SM100TensorAccumulatorSS.mma and SM100TensorAccumulatorTS.mma support
        #   stage_idx parameter for processing in chunks when num_stages > 1
        # - KPipeline and VPipeline structs support separate K/V barrier management
        # - FA4MiscMBars is parameterized by num_pv_stages for S barriers
        # - load() loads K in num_qk_stages chunks with separate barriers per stage
        # - store_exp() writes P in num_pv_stages chunks with barriers per stage
        # - mma() loops over qk_stages for Q@K' and pv_stages for P@V
        #
        # Computed staging values:
        # - num_qk_stages: How many chunks to split K processing into for Q@K' MMA
        # - num_pv_stages: How many chunks to split P writing into for P@V MMA
        #
        if is_mla:
            self.num_qk_stages = 1
            self.num_pv_stages = 1
        else:
            # Q@K' staging is enabled: MMA processes K in num_qk_stages chunks,
            # allowing register pressure reduction and potential overlap.
            self.num_qk_stages = gcd(
                self.padded_depth // swizzle_elems,
                self.padded_depth // Self.MMA_K,
            )
            # P@V staging requires coordinated changes to store_exp and mma functions:
            # - store_exp must write P in stages and signal barriers per stage
            # - mma must wait for each P stage barrier before processing
            if self.BN % 32 != 0:
                self.num_pv_stages = 1
            elif self.BN % 3 == 0:
                self.num_pv_stages = 3
            else:
                self.num_pv_stages = 2

        var smem_use = 4
        # Compute misc_mbars fixed size (barriers that don't scale with num_kv_stages):
        # - S barriers: 2 * (1 + num_pv_stages) per warp group = 4 + 4*num_pv_stages
        # - C barriers: 4 (C0/C1 producer/consumer)
        # - Order barriers: 2 (only when EnableForcedOrdering)
        # - Q1Sync barriers: num_qk_stages
        # - O barriers: 4 (2 producer + 2 consumer)
        # Total fixed = 10 + order_barrier_count + 2*num_pv_stages + num_qk_stages
        comptime order_barrier_count: Int = 2 if EnableForcedOrdering else 0
        misc_mbars_fixed_size = (
            10
            + order_barrier_count
            + 2 * self.num_pv_stages
            + self.num_qk_stages
        )
        smem_use += misc_mbars_fixed_size * Self.mbar_size

        # BK0: K-dimension chunk size for Q@K' per stage
        self.BK0 = self.padded_depth // self.num_qk_stages
        # BK1: Full BN since V loading is not staged (V must be complete for P@V)
        self.BK1 = self.BN
        # smem use is (NOTE: smem uses padded depth):
        # BM*depth*dtype_size + num_kv_stages*(2*mbar_size + BN*depth*dtype_size) <= smem_remaining
        # num_kv_stages <= (smem_remaining - 2*BM*depth*dtype_size) // (2*mbar_size + BN*depth*dtype_size)
        smem_use += self.BM * self.padded_depth * dtype_size
        # Barriers per KV stage (K/V barriers scale with num_kv_stages):
        # - K loading: 2 * num_qk_stages (producer + consumer per stage)
        # - V loading: 2 (single stage, producer + consumer) if separate_kv
        # Note: For MLA (separate_kv=False), V barriers are 0
        smem_per_kv = (
            2 * self.BN * self.padded_depth * dtype_size
            + 2 * Self.mbar_size * self.num_qk_stages  # K barriers
            + 2 * Self.mbar_size  # V barriers (MHA, 1 stage)
        )
        self.num_kv_stages = (
            Self.sm100_smem_carveout - smem_use
        ) // smem_per_kv
        # Example staging values (when implemented):
        # depth= 64: num_qk_stages=1, num_pv_stages=2
        # depth=128: num_qk_stages=2, num_pv_stages=2
        # depth=256: num_qk_stages=4, num_pv_stages=2
        # Currently both are 1 until staged operations are implemented.
        smem_use += self.num_kv_stages * smem_per_kv
        # Add space for correction smem when not using tmem for correction
        smem_use += (
            self.BM * Self.num_correction_cols * size_of[DType.float32]()
        )
        self.smem_used = smem_use

    fn supported(self) -> Bool:
        return (
            self.depth >= 64
            and self.BN >= 64
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
        )

    fn correction_smem_elements(self) -> Int:
        return self.BM * Self.num_correction_cols

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
"""
    tcgen05_mma = "tcgen05.mma.cta_group::1." + kind
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
        mma += String("@%pj bra skip", k, ";")
        mma += tcgen05_mma + " [$0], %rda, %rdb, $2, {$1, $1, $1, $1}, %ps;\n"
        mma += String("skip", k, ":\n")
    return mma + "}"


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
        mma += String("@%pj bra skip", k, ";")
        mma += String(
            tcgen05_mma,
            " [$0], [$",
            7 + k,
            "], %rdb, $2, {$1, $1, $1, $1}, %ps;\n",
        )
        mma += String("skip", k, ":\n")
    return mma + "}"


@always_inline("nodebug")
fn bulk_mma[
    kind: UMMAKind,
    //,
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


@always_inline("nodebug")
fn bulk_mma[
    kind: UMMAKind,
    //,
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
    comptime assert num_k_mmas >= 1 and num_k_mmas <= 16
    comptime mma_string = build_mma_ts(
        String(kind),
        layout_b,
        operand_size=operand_size,
        num_k_mmas=num_k_mmas,
    )

    comptime constraints = "r,r,r,r,r,r,r" + ",r" * num_k_mmas
    comptime x = UInt32(4 * operand_size)
    # fmt: off
    comptime if num_k_mmas == 1:
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
fn llvm_opaque_tid() -> UInt32:
    return inlined_assembly["mov.u32 $0, %tid.x;", UInt32, constraints="=r"]()


@always_inline
fn intrin_ftz[intrin: String](a: Float32, b: Float32) -> Float32:
    return inlined_assembly[
        String(intrin, ".ftz.f32 $0, $1, $2;"),
        Float32,
        constraints="=f,f,f",
    ](a, b)


@always_inline
fn intrin[intrin: String](a: Float32, b: Float32, c: Float32) -> Float32:
    return inlined_assembly[
        String(intrin, ".f32 $0, $1, $2, $3;"),
        Float32,
        constraints="=f,f,f,f",
    ](a, b, c)


@always_inline
fn intrin_ftz_x2[
    intrin: String
](a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]) -> SIMD[
    DType.float32, 2
]:
    comptime s0 = """{
        .reg .b64 %ra;
        .reg .b64 %rb;
        .reg .b64 %rc;
        mov.b64 %ra, {$2, $3};
        mov.b64 %rb, {$4, $5};
        """
    comptime s1 = """.ftz.f32x2 %rc, %ra, %rb;
        mov.b64 {$0, $1}, %rc;
        }"""
    ret = inlined_assembly[
        String(s0, intrin, s1),
        _RegisterPackType[Float32, Float32],
        constraints="=f,=f,f,f,f,f",
    ](a[0], a[1], b[0], b[1])
    return {ret[0], ret[1]}


@always_inline
fn add_ftz(a: Float32, b: Float32) -> Float32:
    return intrin_ftz["add"](a, b)


@always_inline
fn sub_ftz(a: Float32, b: Float32) -> Float32:
    return intrin_ftz["sub"](a, b)


@always_inline
fn mul_ftz(a: Float32, b: Float32) -> Float32:
    return intrin_ftz["mul"](a, b)


@always_inline
fn max_ftz(a: Float32, b: Float32) -> Float32:
    return intrin_ftz["max"](a, b)


@always_inline
fn max_ftz(a: Float32, b: Float32, c: Float32) -> Float32:
    return intrin["max.ftz"](a, b, c)


@always_inline
fn add_ftz(
    a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]
) -> SIMD[DType.float32, 2]:
    return intrin_ftz_x2["add"](a, b)


@always_inline
fn sub_ftz(
    a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]
) -> SIMD[DType.float32, 2]:
    return intrin_ftz_x2["sub"](a, b)


@always_inline
fn mul_ftz(
    a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]
) -> SIMD[DType.float32, 2]:
    return intrin_ftz_x2["mul"](a, b)


@always_inline
fn add_ftz_rm(
    a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]
) -> SIMD[DType.float32, 2]:
    return intrin_ftz_x2["add.rm"](a, b)


@always_inline
fn fma_ftz(
    a: SIMD[DType.float32, 2],
    b: SIMD[DType.float32, 2],
    c: SIMD[DType.float32, 2],
) -> SIMD[DType.float32, 2]:
    ret = inlined_assembly[
        """{
        .reg .b64 %ra;
        .reg .b64 %rb;
        .reg .b64 %rc;
        .reg .b64 %rd;
        mov.b64 %ra, {$2, $3};
        mov.b64 %rb, {$4, $5};
        mov.b64 %rc, {$6, $7};
        fma.rn.ftz.f32x2 %rd, %ra, %rb, %rc;
        mov.b64 {$0, $1}, %rd;
        }""",
        _RegisterPackType[Float32, Float32],
        constraints="=f,=f,f,f,f,f,f,f",
    ](a[0], a[1], b[0], b[1], c[0], c[1])
    return {ret[0], ret[1]}


@always_inline
fn exp2_emulation[
    use_exp2_emulation: Bool = True
](x: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    comptime if use_exp2_emulation:
        comptime fp32_round_int = SIMD[DType.float32, 2]((1 << 23) + (1 << 22))
        clamped = max(x, -127)
        # We want to round down here, so that the fractional part is in [0, 1)
        rounded = add_ftz_rm(clamped, fp32_round_int)
        rounded_back = sub_ftz(rounded, fp32_round_int)
        frac = sub_ftz(clamped, rounded_back)
        # Tri Dao assumes x <= 127.0 and y <= 127.0
        frac_ex2 = fma_ftz(
            fma_ftz(
                fma_ftz(
                    0.077119089663028717041015625,
                    frac,
                    0.227564394474029541015625,
                ),
                frac,
                0.695146143436431884765625,
            ),
            frac,
            1.0,
        )
        # The integer floor of x & y are now in the last 8 bits of xy_rounded
        # We want the next 2 ops to round to nearest even. The rounding mode is important.
        return bitcast[DType.float32](
            bitcast[DType.int32](frac_ex2)
            + (bitcast[DType.int32](rounded) << 23)
        )
    else:
        return exp2(x)


@always_inline
fn elect_mma_arrive[
    cta_group: Int = 1
](
    mbar_ptr: UnsafePointer[address_space = AddressSpace.SHARED, ...],
    elect: Int32,
):
    """Arrive at the mbar pointer for the MMA instruction.

    Parameters:
        cta_group: Number of ctas used by MMA.

    Args:
        mbar_ptr: Pointer to the mbar.
        elect: `elect()`.
    """

    comptime assert cta_group in (1, 2), String(
        "Unsupported cta group: ", cta_group
    )

    comptime type = mbar_ptr.type
    comptime assert size_of[type]() == 8, "mbar_ptr must be 8 bytes"

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
    BN: Int, //, *, width: Int = 4
](
    x: LocalTensor[DType.float32, row_major[BN]()],
    out res: StaticTuple[Float32, width],
):
    comptime assert BN % (2 * width) == 0
    res = {}

    comptime for w in range(width):
        res[w] = max_ftz(
            rebind[Float32](x[3 * w]),
            rebind[Float32](x[3 * w + 1]),
            rebind[Float32](x[3 * w + 2]),
        )
    # max idx = 3 * (width-1) + 2 = 3*width - 1
    comptime remaining_iters = BN - 3 * width
    comptime num_iters = remaining_iters // (2 * width)
    comptime for i in range(num_iters):
        comptime col = i * 2 * width + 3 * width
        comptime for w in range(width):
            res[w] = max_ftz(
                res[w],
                rebind[Float32](x[col + 2 * w]),
                rebind[Float32](x[col + 2 * w + 1]),
            )

    comptime remainder_base = 3 * width + 2 * width * num_iters
    comptime end_iters = (BN - remainder_base) // 2
    comptime for w in range(end_iters):
        res[w] = max_ftz(
            res[w],
            rebind[Float32](x[remainder_base + 2 * w]),
            rebind[Float32](x[remainder_base + 2 * w + 1]),
        )
    comptime if (BN - remainder_base) % 2 == 1:
        res[end_iters] = max_ftz(res[end_iters], x[BN - 1])


@always_inline
fn maximum[
    BN: Int, //, *, width: Int = 4
](
    x: LocalTensor[DType.float32, row_major[BN]()],
    init: StaticTuple[Float32, width],
    out res: StaticTuple[Float32, width],
):
    comptime assert BN % (2 * width) == 0
    res = init

    # unroll (using SIMD) to break up dependency chain
    comptime for i in range(BN // (2 * width)):
        comptime for w in range(width):
            comptime j = i * 2 * width + 2 * w
            res[w] = max_ftz(
                res[w], rebind[Float32](x[j]), rebind[Float32](x[j + 1])
            )


@always_inline
fn maximum(x: StaticTuple[Float32, 4]) -> Float32:
    return max_ftz(max_ftz(x[0], x[1], x[2]), x[3])


@always_inline
fn maximum(x: StaticTuple[Float32, 4], init: Float32) -> Float32:
    return max_ftz(max_ftz(x[0], x[1], x[2]), x[3], init)


@always_inline
fn maximum(x: StaticTuple[Float32, 8]) -> Float32:
    var a = max_ftz(x[0], x[1], x[2])
    var b = max_ftz(x[3], x[4], x[5])
    var c = max_ftz(x[6], x[7])
    return max_ftz(a, b, c)


@always_inline
fn maximum(x: StaticTuple[Float32, 8], init: Float32) -> Float32:
    var a = max_ftz(init, x[0], x[1])
    var b = max_ftz(x[2], x[3], x[4])
    var c = max_ftz(x[5], x[6], x[7])
    return max_ftz(a, b, c)


@always_inline
fn sum[
    dtype: DType, BN: Int, //, *, width: Int = 8
](x: LocalTensor[dtype, row_major[BN]()]) -> SIMD[dtype, 2]:
    comptime assert BN % width == 0
    vx = x.vectorize[width]()
    acc = vx[0]

    # unroll (using SIMD) to break up dependency chain
    comptime for i in range(1, BN // width):
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
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ],
    batch_size_arg: Int,
    partition: PartitionType,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ],
) raises:
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match for FA3."
    comptime decoding: Bool = _is_decoding[MaxPromptLenType]()
    comptime assert (
        not decoding
    ), "this implementation does not support decoding"
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
    comptime BN = fa4_config.BN
    comptime num_threads = fa4_config.num_threads
    var q = rebind[UnsafePointer[Scalar[KVType.dtype], q_arg.origin]](q_arg)

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)
    var max_prompt_len: UInt32 = max_prompt_len_arg.as_uint32()
    var max_num_prompt_tiles: UInt32 = ceildiv(max_prompt_len, UInt32(BM))
    var block_x: UInt32 = max_num_prompt_tiles * partition.num_partitions()

    comptime RaggedStoreType = RaggedTMA3DTile[
        output_type,
        swizzle_mode,
        BM = BM // 2,
        BN = fa4_config.depth,
    ]

    var ragged_tma_store = RaggedStoreType.create(
        ctx,
        output.unsafe_ptr(),
        rows=num_rows_q,
        middle_dim=fa4_config.num_q_heads,
    )

    q_tma_op = q_tma[
        swizzle_mode,
        BM = BM // 2,
        depth = fa4_config.depth,
        q_num_heads = fa4_config.num_q_heads,
        group = fa4_config.group,
        decoding=False,
        num_qk_stages = fa4_config.num_qk_stages,
    ](ctx, q, num_rows_q)
    k_tma_op = k.create_tma_tile[
        fa4_config.swizzle_mode,
        BN = fa4_config.BN,
        depth = fa4_config.depth,
        BK = fa4_config.BK0,
    ](ctx)
    v_tma_op = v.create_tma_tile[
        fa4_config.swizzle_mode,
        BN = fa4_config.BN,
        depth = fa4_config.depth,
        BK = fa4_config.padded_depth,
    ](ctx)
    comptime assert BM == 256
    comptime SchedulerType = TransientScheduler[
        UInt32(BM), UInt32(fa4_config.num_q_heads)
    ]
    var scheduler: SchedulerType = SchedulerType()

    comptime if sink:
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
            config=fa4_config,
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
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = config.BM // 2,
        depth = config.depth,
        group = config.group,
        decoding=False,
        num_qk_stages = config.num_qk_stages,
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = config.BN,
        BK = config.BK0,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = config.BN,
        BK = config.padded_depth,
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
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ],
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    score_mod: ScoreModType,
    ctx: DeviceContext,
    num_rows_q: Int,
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        swizzle_mode,
        BM = config.BM // 2,
        BN = config.depth,
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
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = config.BM // 2,
        depth = config.depth,
        group = config.group,
        decoding=False,
        num_qk_stages = config.num_qk_stages,
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = config.BN,
        BK = config.BK0,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = config.BN,
        BK = config.padded_depth,
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
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        swizzle_mode,
        BM = config.BM // 2,
        BN = config.depth,
    ],
) raises:
    comptime if ragged:
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
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = config.BM // 2,
        depth = config.depth,
        group = config.group,
        decoding=False,
        num_qk_stages = config.num_qk_stages,
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = config.BN,
        BK = config.BK0,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN = config.BN,
        BK = config.padded_depth,
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
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        swizzle_mode,
        BM = config.BM // 2,
        BN = config.depth,
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
        max_seq_len.as_uint32(), UInt32(config.BM)
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
    ].kernel

    ctx.enqueue_function[kernel, kernel](
        q_tma_op,
        k_tma_op,
        v_tma_op,
        ragged_tma_store,
        kv_lut,
        scale,
        batch_size,
        num_keys_arg,
        pack,
        grid_dim=SchedulerType.grid_dim(batch_size, block_x),
        block_dim=(num_threads, 1, 1),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


struct StagedPipeline[num_kv_stages: Int, num_qk_stages: Int = 1](
    TrivialRegisterPassable
):
    """
    Unified pipeline for K, V, and KV tile barrier management.

    `num_kv_stages` refers to how many KV tile buffers we have for pipelining.
    `num_qk_stages` controls K loading staging for Q@K' MMA:
      - K can be loaded in num_qk_stages chunks, allowing MMA to start earlier
      - V always uses qk_stages=1 (complete tile required)

    Total stages = num_kv_stages * num_qk_stages.
    """

    comptime num_stages: Int = Self.num_kv_stages * Self.num_qk_stages

    # mbars are ordered in {producer, consumer} pairs
    var mbar: MBarType
    var state: PipelineState[Self.num_kv_stages]

    @always_inline
    fn __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    fn producer_mbar[qk_stage: Int = 0](self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.mbar + UInt32(Self.num_qk_stages) * idx + qk_stage

    @always_inline
    fn consumer_mbar[qk_stage: Int = 0](self, idx: UInt32) -> MBarType:
        comptime const_offset = qk_stage + Self.num_stages
        return self.mbar + UInt32(Self.num_qk_stages) * idx + const_offset

    @always_inline
    fn consumer_mbar[qk_stage: Int = 0](self) -> MBarType:
        return self.consumer_mbar[qk_stage](self.state.index())

    @always_inline("nodebug")
    fn producer_acquire[qk_stage: Int = Self.num_qk_stages - 1](self):
        """Wait until consumer has released the buffer for this stage."""
        self.consumer_mbar[qk_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_wait[qk_stage: Int = Self.num_qk_stages - 1](self):
        """Wait for producer to complete this stage."""
        self.producer_mbar[qk_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_release[
        qk_stage: Int = Self.num_qk_stages - 1
    ](mut self, e: Int32):
        """Release the buffer after consuming this stage."""
        elect_mma_arrive(self.consumer_mbar[qk_stage](), e)

        comptime if qk_stage == Self.num_qk_stages - 1:
            self.state.step()

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return UInt32(2 * Self.num_qk_stages * Self.num_kv_stages)


# Backward-compatible type aliases
comptime KPipeline = StagedPipeline
comptime VPipeline = StagedPipeline[_, 1]
comptime KVPipeline = StagedPipeline


struct TMADestination[dtype: DType, layout: Layout](TrivialRegisterPassable):
    var mbar: MBarType
    var smem: SharedMemLT[Self.dtype, Self.layout]

    @always_inline
    fn __init__(
        out self, mbar: MBarType, smem: SharedMemLT[Self.dtype, Self.layout]
    ):
        self.mbar = mbar
        self.smem = smem


struct TMAProducerPipeline[dtype: DType, config: FA4Config, is_k: Bool = True](
    TrivialRegisterPassable
):
    """Unified producer pipeline for K and V TMA loading.

    K loading (is_k=True): Can be staged (num_qk_stages chunks), uses k_major layout.
    V loading (is_k=False): Always complete (qk_stage=0), uses mn_major layout.
    """

    # Compute layout first using comptime, then use it in type
    comptime tile_layout: Layout = tile_layout_k_major[
        Self.dtype,
        Self.config.BN,
        Self.config.BK0,
        Self.config.swizzle_mode,
    ]() if Self.is_k else tile_layout_mn_major[
        Self.dtype,
        Self.config.padded_depth,
        Self.config.BK1,
        Self.config.swizzle_mode,
    ]()

    comptime TileType = SharedMemLT[Self.dtype, Self.tile_layout]
    comptime PairType = TMADestination[Self.dtype, Self.tile_layout]
    comptime elements: Int = Self.tile_layout.size()
    comptime elements_full: Int = Self.elements * Self.config.num_qk_stages if Self.is_k else Self.elements
    comptime tile_bytes: Int = Self.elements * size_of[Self.dtype]()
    # Backward-compatible aliases
    comptime bytes = Self.tile_bytes
    comptime SMemType = SharedMemPointer[Scalar[Self.dtype]]

    # K uses full staging, V uses qk_stages=1
    comptime num_qk_stages_effective: Int = Self.config.num_qk_stages if Self.is_k else 1

    var pipeline: StagedPipeline[
        Self.config.num_kv_stages, Self.num_qk_stages_effective
    ]
    var smem: Self.SMemType

    @always_inline
    fn __init__(out self, mbar: MBarType, smem: Self.SMemType):
        comptime if Self.is_k:
            comptime assert (
                Self.config.padded_depth % Self.config.num_qk_stages == 0
            ), "padded_depth must be divisible by num_qk_stages"
        self.pipeline = {mbar}
        self.smem = smem
        self.pipeline.state._phase = 1

    @always_inline
    fn __init__(
        out self,
        pipeline: StagedPipeline[
            Self.config.num_kv_stages, Self.num_qk_stages_effective
        ],
        smem: Self.SMemType,
    ):
        comptime if Self.is_k:
            comptime assert (
                Self.config.padded_depth % Self.config.num_qk_stages == 0
            ), "padded_depth must be divisible by num_qk_stages"
        self.pipeline = pipeline
        self.smem = smem
        self.pipeline.state._phase = 1

    @always_inline
    fn get_smem[*, qk_stage: Int = 0](self) -> Self.SMemType:
        """Get smem pointer for current stage."""

        comptime if Self.is_k:
            comptime stage_offset = qk_stage * Self.elements
            var dyn_offset: UInt32 = (
                UInt32(Self.elements_full) * self.pipeline.state.index()
            )
            return self.smem + stage_offset + dyn_offset
        else:
            var dyn_offset: UInt32 = (
                UInt32(Self.elements) * self.pipeline.state.index()
            )
            return self.smem + dyn_offset

    @always_inline
    fn get_tile[*, qk_stage: Int = 0](self) -> Self.PairType:
        """Get TMA destination for this stage."""
        p_mbar = self.pipeline.producer_mbar[qk_stage]()
        return {p_mbar, {self.get_smem[qk_stage=qk_stage]()}}

    @always_inline
    fn get_tile[*, qk_stage: Int = 0](self, e: Int32) -> Self.PairType:
        """Get TMA destination with optional expect_bytes."""
        p_mbar = self.pipeline.producer_mbar[qk_stage]()
        if e != 0:
            p_mbar[].expect_bytes(Int32(Self.tile_bytes))
        return {p_mbar, {self.get_smem[qk_stage=qk_stage]()}}

    @always_inline
    fn acquire[*, qk_stage: Int = 0](self):
        """Wait for consumer to release the buffer."""
        self.pipeline.producer_acquire[qk_stage]()

    @always_inline
    fn commit_step(mut self):
        """Step the pipeline. Commit is handled by tma_op.async_copy."""
        self.pipeline.state.step()

    # Backward-compatible K methods (for KProducerPipeline)
    comptime KPairType = Self.PairType  # Alias for backward compatibility

    @always_inline
    fn get_k_smem[*, qk_stage: Int](self) -> Self.SMemType:
        return self.get_smem[qk_stage=qk_stage]()

    @always_inline
    fn get_k[*, qk_stage: Int](self) -> Self.PairType:
        return self.get_tile[qk_stage=qk_stage]()

    @always_inline
    fn get_k[*, qk_stage: Int](self, e: Int32) -> Self.PairType:
        return self.get_tile[qk_stage=qk_stage](e)

    @always_inline
    fn acquire_k[*, qk_stage: Int](self):
        self.acquire[qk_stage=qk_stage]()

    @always_inline
    fn get_v_smem(self) -> Self.SMemType:
        return self.get_smem[qk_stage=0]()

    @always_inline
    fn get_v(self, e: Int32) -> Self.PairType:
        return self.get_tile[qk_stage=0](e)

    @always_inline
    fn acquire_v(self):
        self.acquire[qk_stage=0]()


# Backward-compatible type aliases
comptime KProducerPipeline = TMAProducerPipeline[_, _, True]
comptime VProducerPipeline = TMAProducerPipeline[_, _, False]


struct TMAConsumerPipeline[dtype: DType, config: FA4Config, is_k: Bool = True](
    TrivialRegisterPassable
):
    """Unified consumer pipeline for K and V TMA consumption.

    K consumption (is_k=True): Uses k_major layout, supports staged qk_stages.
    V consumption (is_k=False): Uses mn_major layout, always uses qk_stage=0.

    This follows the order of Tri Dao and Cutlass implementations
    (modulo any rotation of the ops through the iterations).

    We consume/produce in the following order:
        0. S0 <- Q0 @ Kn'
        1. O1 <- O1 + P1 @ V{n-1}
        2. S1 <- Q1 @ Kn'
        3. O0 <- O0 + P0 @ Vn

    Note that we have two MMA between calculating Si and consuming Pi,
    maximizing the overlap between MMAs and softmax calculation.
    """

    comptime full_kv_bytes = Self.config.BN * Self.config.padded_depth * size_of[
        Self.dtype
    ]()
    comptime staged_k_bytes = Self.config.BN * Self.config.BK0 * size_of[
        Self.dtype
    ]()

    # K uses full staging, V uses qk_stages=1
    comptime num_qk_stages_effective: Int = Self.config.num_qk_stages if Self.is_k else 1

    # Descriptor parameters differ by role
    comptime BMN: Int = Self.config.BN if Self.is_k else Self.config.padded_depth
    comptime BK: Int = Self.config.BK0 if Self.is_k else Self.config.BK1
    comptime is_k_major: Bool = Self.is_k

    var pipeline: StagedPipeline[
        Self.config.num_kv_stages, Self.num_qk_stages_effective
    ]
    var smem_desc: MMASmemDescriptorPair

    @always_inline
    fn __init__(
        out self,
        pipeline: StagedPipeline[
            Self.config.num_kv_stages, Self.num_qk_stages_effective
        ],
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipeline = pipeline
        self.smem_desc = smem_descriptor[
            BMN = Self.BMN,
            BK = Self.BK,
            swizzle_mode = Self.config.swizzle_mode,
            is_k_major = Self.is_k_major,
        ](smem)

    @always_inline
    fn __init__(
        out self,
        mbar: MBarType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        return Self(type_of(self.pipeline)(mbar), smem)

    @always_inline("nodebug")
    fn get(self) -> MMASmemDescriptorPair:
        """Get smem descriptor for current stage."""
        var dyn_offset: UInt32 = (
            UInt32(Self.full_kv_bytes) * self.pipeline.state.index()
        )
        return self.smem_desc + dyn_offset

    @always_inline("nodebug")
    fn wait[*, qk_stage: Int = 0](self):
        """Wait for tile from producer."""
        self.pipeline.consumer_wait[qk_stage]()

    @always_inline("nodebug")
    fn release[*, qk_stage: Int = 0](mut self, e: Int32):
        """Release buffer after consuming."""
        self.pipeline.consumer_release[qk_stage](e)

    # Backward-compatible K methods (for KConsumerPipeline)
    @always_inline("nodebug")
    fn get_k(self) -> MMASmemDescriptorPair:
        return self.get()

    @always_inline("nodebug")
    fn wait_k[*, qk_stage: Int = Self.config.num_qk_stages - 1](mut self):
        """Wait on K stage from the producer."""
        self.wait[qk_stage=qk_stage]()

    @always_inline("nodebug")
    fn release_k[
        *, qk_stage: Int = Self.config.num_qk_stages - 1
    ](mut self, e: Int32):
        """Release K buffer after consuming this stage."""
        self.release[qk_stage=qk_stage](e)

    # Backward-compatible V methods (for VConsumerPipeline)
    @always_inline("nodebug")
    fn get_v(self) -> MMASmemDescriptorPair:
        return self.get()

    @always_inline("nodebug")
    fn wait_v(self):
        """Wait for V tile."""
        self.wait[qk_stage=0]()

    @always_inline("nodebug")
    fn release_v(mut self, e: Int32):
        """Release V buffer after consuming."""
        self.release[qk_stage=0](e)


# Backward-compatible type aliases
comptime KConsumerPipeline = TMAConsumerPipeline[_, _, True]
comptime VConsumerPipeline = TMAConsumerPipeline[_, _, False]


struct RolePipeline[
    number_of_stages: Int,
    is_producer: Bool = True,
    producer_sub_stages: Int = 1,
    consumer_sub_stages: Int = 1,
](TrivialRegisterPassable):
    """
    Unified producer/consumer pipeline for barrier synchronization.

    Producer role: Starts with phase=1, uses acquire/commit methods.
    Consumer role: Starts with phase=0, uses wait/release methods.

    Sub-stages allow multiple barriers per stage:
    - Total producer barriers: num_stages * producer_sub_stages
    - Total consumer barriers: num_stages * consumer_sub_stages

    Synchronization behavior (example with num_stages=1):

    Producer:
    p0. consumer_mbar.wait(phase=1)  # 1 != 0: falls through
    p1. producer_mbar.commit()       # producer_mbar.phase=1
    p2. step()                       # phase = 0
    p3. consumer_mbar.wait(phase=0)  # 0 == 0: blocked until c1
    ...

    Consumer:
    c0. producer_mbar.wait(phase=0)  # 0 == 0: blocked until p1
    c1. consumer.release()           # consumer_mbar.phase=1
    c2. step()                       # phase = 1
    ...
    """

    comptime num_stages: Int = Self.number_of_stages

    var producer_mbar_base: MBarType
    var consumer_mbar_base: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    fn __init__(
        out self, producer_mbar_base: MBarType, consumer_mbar_base: MBarType
    ):
        self.producer_mbar_base = producer_mbar_base
        self.consumer_mbar_base = consumer_mbar_base
        self.state = {}

        comptime if Self.is_producer:
            # Producer starts with phase=1 so initial waits fall through
            self.state._phase = 1

    @always_inline
    fn producer_mbar[sub_stage_idx: Int = 0](self) -> MBarType:
        """Get producer mbar for current stage and optional sub-stage.

        Parameters:
            sub_stage_idx: Sub-stage index (0 to producer_sub_stages-1).
        """
        comptime assert (
            sub_stage_idx < Self.producer_sub_stages
        ), "sub_stage_idx out of range"
        return (
            self.producer_mbar_base
            + self.state.index() * UInt32(Self.producer_sub_stages)
            + sub_stage_idx
        )

    @always_inline
    fn consumer_mbar[sub_stage_idx: Int = 0](self) -> MBarType:
        """Get consumer mbar for current stage and optional sub-stage.

        Parameters:
            sub_stage_idx: Sub-stage index (0 to consumer_sub_stages-1).
        """
        comptime assert (
            sub_stage_idx < Self.consumer_sub_stages
        ), "sub_stage_idx out of range"
        return (
            self.consumer_mbar_base
            + self.state.index() * UInt32(Self.consumer_sub_stages)
            + sub_stage_idx
        )

    # Producer methods
    @always_inline("nodebug")
    fn acquire[sub_stage_idx: Int = 0](self):
        """Wait until consumer has released the buffer. Producer-only."""
        self.consumer_mbar[sub_stage_idx]()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn commit(mut self):
        """Commit production and step. Producer-only."""
        _ = self.producer_mbar()[].arrive()
        self.state.step()

    @always_inline("nodebug")
    fn commit_mma(self):
        """Commit via MMA arrive using elected thread. Producer-only."""
        mbar = self.producer_mbar()
        elect_mma_arrive(mbar, elect())

    @always_inline("nodebug")
    fn commit_mma(self, elect: Int32):
        """Commit via MMA arrive with explicit elect value. Producer-only."""
        mbar = self.producer_mbar()
        elect_mma_arrive(mbar, elect)

    # Consumer methods
    @always_inline("nodebug")
    fn wait(self):
        """Wait for producer to complete. Consumer-only."""
        self.producer_mbar()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn release[sub_stage_idx: Int = 0](mut self):
        """Release buffer at sub-stage and step. Consumer-only."""
        _ = self.consumer_mbar[sub_stage_idx]()[].arrive()
        self.state.step()

    @always_inline("nodebug")
    fn release_no_step[sub_stage_idx: Int = 0](self):
        """Release buffer without stepping. For multi-sub-stage release."""
        _ = self.consumer_mbar[sub_stage_idx]()[].arrive()

    # Shared method
    @always_inline("nodebug")
    fn step(mut self):
        self.state.step()


# Backward-compatible type aliases
comptime ProducerPipeline = RolePipeline[_, True, _, _]
comptime ConsumerPipeline = RolePipeline[_, False, _, _]


struct MBarPipeline[number_of_stages: Int](TrivialRegisterPassable):
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
        comptime for i in range(Self.number_of_stages):
            self.mbar[i].init(Int32(Int(num_producer)))

        comptime for i in range(Self.number_of_stages):
            self.mbar[i + Self.number_of_stages].init(Int32(Int(num_consumer)))

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return UInt32(2 * Self.number_of_stages)


@always_inline
fn apply_oob_mask[
    ScoreModType: ScoreModTrait,
    //,
    *,
    use_score_mod: Bool,
    mask_strategy: MaskStrategy,
    apply_log2e_after_mask: Bool,
](
    s_arg: SIMD[DType.float32, 2],
    score_mod: ScoreModType,
    *,
    prompt_idx: UInt32,
    q_head_idx: UInt32,
    kv_tile_start_row: Int32,
    max_seq_len: UInt32,
    num_keys: Int32,
    score_row: Int32,
    score_col: Int32,
) -> SIMD[DType.float32, 2]:
    s: SIMD[DType.float32, 2] = s_arg

    comptime if use_score_mod:
        s = mul_ftz(
            score_mod.score_mod(
                IndexList[4, element_type = DType.uint32](
                    Int(prompt_idx),
                    Int(q_head_idx),
                    Int(score_row),
                    Int(score_col),
                ),
                s,
                Int(max_seq_len),
            ),
            log2e,
        )
    elif apply_log2e_after_mask:
        s = mul_ftz(s, log2e)

    comptime if MaskStrategy.OUT_OF_BOUNDS in mask_strategy:
        s = (
            iota[DType.int32, 2](score_col)
            .lt(num_keys)
            .select(s, MASK_VALUE)
            # .select(s, min_or_neg_inf[DType.float32]())
        )

    return s


@always_inline
fn apply_mask[
    BN: Int,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    //,
    *,
    use_score_mod: Bool,
    mask_strategy: MaskStrategy,
    skip_scale: Bool = False,
](
    srow: LocalTensor[DType.float32, row_major[BN]()],
    mask: MaskType,
    score_mod: ScoreModType,
    scale_log2e: Float32,
    *,
    prompt_idx: UInt32,
    q_head_idx: UInt32,
    kv_tile_start_row: Int32,
    max_seq_len: UInt32,
    num_keys: Int32,
    score_row: Int32,
):
    comptime simd_size = 2
    comptime F32x2 = SIMD[DType.float32, simd_size]
    vs = srow.vectorize[simd_size]()

    comptime if (
        MaskStrategy.LOWER_TRIANGULAR in mask_strategy
        or MaskStrategy.UPPER_TRIANGULAR in mask_strategy
    ):
        comptime num_batches = BN // 32
        comptime assert (BN % 32) == 0

        # when score_row == kv_tile_start_row, 1 is valid
        var n_valid: Int32 = max(1 + score_row - kv_tile_start_row, 0)

        comptime for batch in range(num_batches):
            var mask_bits: UInt32 = 0xFFFF_FFFF

            comptime if MaskStrategy.LOWER_TRIANGULAR in mask_strategy:
                # Causal Mask
                # score_row >= kv_tile_start_row
                # 1 + score_row - kv_tile_start_row > 0
                # n_valid > 0
                mask_bits = (UInt32(1) << UInt32(n_valid)) - UInt32(
                    1
                ) if n_valid < 32 else mask_bits

            comptime if MaskStrategy.UPPER_TRIANGULAR in mask_strategy:
                # SlidingWindowCausalMask sliding window part
                # score_row - kv_tile_start_row < window_size
                # window_size + kv_tile_start_row - score_row > 0
                # window_size + 1 - (1 + score_row - kv_tile_start_row) > 0
                # window_size + 1 - n_valid > 0
                #
                # ex window_size = 1, score_row == kv_tile_start_row
                #    n_valid = 1
                # We should turn off `0`: first is on, and all the rest
                # ex window_size = 4, score_row == kv_tile_start_row + 5
                #    n_valid = 6
                # We should turn off `2`: first two off, all the rest on
                var mask_off_count: Int32 = (
                    n_valid - mask_strategy._upper_triangular_window_size
                )
                # we want mask_off_count `1`s
                mask_bits = (
                    (
                        mask_bits & (0xFFFF_FFFF << UInt32(mask_off_count))
                    ) if mask_off_count
                    < 32 else 0
                ) if mask_off_count > 0 else mask_bits

            comptime for n in range(32 // simd_size):
                comptime frag_col_simd = n + 32 * batch // simd_size
                comptime frag_col = frag_col_simd * simd_size
                var s: F32x2

                comptime if skip_scale:
                    s = rebind[F32x2](vs[frag_col_simd])
                else:
                    s = mul_ftz(rebind[F32x2](vs[frag_col_simd]), scale_log2e)

                comptime for i in range(simd_size):
                    comptime midx = n * simd_size + i
                    comptime flag: UInt32 = UInt32(1 << midx)
                    var in_bound: Bool = (mask_bits & flag) != UInt32(0)
                    var val: Float32 = s[i]
                    s[i] = val if in_bound else MASK_VALUE

                var score_col: Int32 = kv_tile_start_row + Int32(frag_col)
                vs[frag_col_simd] = rebind[vs.ElementType](
                    apply_oob_mask[
                        use_score_mod=use_score_mod,
                        mask_strategy=mask_strategy,
                        apply_log2e_after_mask = MaskType.apply_log2e_after_mask,
                    ](
                        s,
                        score_mod,
                        prompt_idx=prompt_idx,
                        q_head_idx=q_head_idx,
                        kv_tile_start_row=kv_tile_start_row,
                        max_seq_len=max_seq_len,
                        num_keys=num_keys,
                        score_row=score_row,
                        score_col=score_col,
                    )
                )
            n_valid = max(n_valid - 32, 0)

    else:
        comptime block_size = BN // simd_size

        comptime for n in range(block_size):
            # score_col = mask_frag_col + j * 8
            var s: F32x2

            comptime if skip_scale:
                s = rebind[F32x2](vs[n])
            else:
                s = mul_ftz(rebind[F32x2](vs[n]), scale_log2e)
            comptime frag_col = simd_size * n
            var score_col: Int32 = kv_tile_start_row + Int32(frag_col)

            comptime if MaskStrategy.COMPUTED in mask_strategy:
                s = mask.mask(
                    IndexList[4, element_type = DType.uint32](
                        Int(prompt_idx),
                        Int(q_head_idx),
                        Int(score_row),
                        Int(score_col),
                    ),
                    s,
                )

            vs[n] = rebind[vs.ElementType](
                apply_oob_mask[
                    use_score_mod=use_score_mod,
                    mask_strategy=mask_strategy,
                    apply_log2e_after_mask = MaskType.apply_log2e_after_mask,
                ](
                    s,
                    score_mod,
                    prompt_idx=prompt_idx,
                    q_head_idx=q_head_idx,
                    kv_tile_start_row=kv_tile_start_row,
                    max_seq_len=max_seq_len,
                    num_keys=num_keys,
                    score_row=score_row,
                    score_col=score_col,
                )
            )


struct FA4MiscMBars[
    *,
    num_qk_stages: Int = 1,
    num_pv_stages: Int = 1,
    num_kv_stages: Int = 2,
    separate_kv: Bool = True,
    use_order_barriers: Bool = True,
](TrivialRegisterPassable):
    """Manages all mbarrier resources for FA4.

    This struct consolidates all mbarrier management including:
    - S barriers (score MMA synchronization)
    - C barriers (correction synchronization)
    - Order barriers (softmax ordering)
    - Q1Sync barriers (Q tile synchronization)
    - K/V pipeline barriers
    - O pipeline barriers

    Parameters:
        num_qk_stages: Number of stages for Q@K' MMA (K loading can be staged).
        num_pv_stages: Number of stages for P@V MMA (P writing can be staged).
        num_kv_stages: Number of KV buffer stages for double/triple buffering.
        separate_kv: True for MHA (separate K/V barriers), False for MLA (unified KV).
        use_order_barriers: When True, allocate order barriers to prevent softmax
            warp group overlap. When False, order barriers are omitted.

    Memory layout (count=128 first, then count=1):
        [S0_cons] [S1_cons] [C0] [C1] [Order*] [O_cons] | [S0_prod] [S1_prod] [Q1Sync] [K] [V*] [O_prod]
        *Order barriers only present when use_order_barriers=True
        *V barriers only present when separate_kv=True
    """

    var mbar_base: MBarType

    # ---- Count=128 section (first in smem) ----
    # S consumer barriers: num_pv_stages per warp group
    comptime S0_consumer_offset = 0
    comptime S1_consumer_offset = Self.num_pv_stages
    # C barriers: 2 per warp group (producer + consumer, both count=128)
    comptime C0_offset = 2 * Self.num_pv_stages
    comptime C1_offset = Self.C0_offset + 2
    # Order barriers: 1 per warp group (count=128), conditional on use_order_barriers
    comptime num_order_barriers: Int = 2 if Self.use_order_barriers else 0
    comptime order_offset = Self.C1_offset + 2
    # O consumer barriers (count=128)
    comptime O_consumer_offset = Self.order_offset + Self.num_order_barriers

    # ---- Count=1 section ----
    # S producer barriers: 1 per warp group
    comptime S0_producer_offset = Self.O_consumer_offset + 2
    comptime S1_producer_offset = Self.S0_producer_offset + 1
    # Q1Sync barriers
    comptime Q1SyncIdx = Self.S1_producer_offset + 1
    # K pipeline barriers
    comptime K_offset = Self.Q1SyncIdx + Self.num_qk_stages
    comptime K_barriers: Int = 2 * Self.num_qk_stages * Self.num_kv_stages
    # V barriers only present when separate_kv=True (MHA uses separate K/V)
    comptime V_offset: Int = Self.K_offset + Self.K_barriers
    comptime V_barriers: Int = 2 * Self.num_kv_stages if Self.separate_kv else 0
    # O producer barriers (count=1)
    comptime O_producer_offset = Self.V_offset + Self.V_barriers

    # Total size includes all barriers
    comptime size = Self.O_producer_offset + 2
    comptime number_warpgroup_count = Self.S0_producer_offset

    @always_inline
    fn __init__(out self, mbar_base: MBarType):
        self.mbar_base = mbar_base

    @always_inline
    fn init(self, *, lane_idx: Int32):
        comptime if Self.size < WARP_SIZE:
            if lane_idx < Int32(Self.size):
                self.mbar_base[lane_idx].init(
                    Int32(
                        128 if lane_idx
                        < Int32(Self.number_warpgroup_count) else 1
                    )
                )
        elif Self.size == WARP_SIZE:
            self.mbar_base[lane_idx].init(
                Int32(
                    128 if lane_idx < Int32(Self.number_warpgroup_count) else 1
                )
            )
        else:
            comptime assert Self.number_warpgroup_count <= WARP_SIZE, String(
                "Number of count=128 barriers = ", Self.number_warpgroup_count
            )
            comptime assert (
                Self.size - Self.number_warpgroup_count <= WARP_SIZE
            ), String(
                "Number of count=1 barriers = ",
                Self.size - Self.number_warpgroup_count,
            )
            if lane_idx < Int32(Self.number_warpgroup_count):
                self.mbar_base[lane_idx].init(128)
            if lane_idx < Int32(Self.size - Self.number_warpgroup_count):
                self.mbar_base[
                    Int32(Self.number_warpgroup_count) + lane_idx
                ].init(1)

    # S pipeline type: 1 producer sub-stage, num_pv_stages consumer sub-stages
    comptime SPipelineProducer = RolePipeline[1, True, 1, Self.num_pv_stages]
    comptime SPipelineConsumer = RolePipeline[1, False, 1, Self.num_pv_stages]

    @always_inline
    fn producer_s0(self) -> Self.SPipelineProducer:
        """Get S producer for warp group 0."""
        return {
            self.mbar_base + Self.S0_producer_offset,
            self.mbar_base + Self.S0_consumer_offset,
        }

    @always_inline
    fn producer_s1(self) -> Self.SPipelineProducer:
        """Get S producer for warp group 1."""
        return {
            self.mbar_base + Self.S1_producer_offset,
            self.mbar_base + Self.S1_consumer_offset,
        }

    @always_inline
    fn consumer_s(self, wg_idx: UInt32) -> Self.SPipelineConsumer:
        """Get S consumer for given warp group."""
        return {
            self.mbar_base + Self.S0_producer_offset + wg_idx,
            self.mbar_base + UInt32(Self.num_pv_stages) * wg_idx,
        }

    @always_inline
    fn consumer_c0(self) -> ConsumerPipeline[1]:
        return {
            self.mbar_base + Self.C0_offset,
            self.mbar_base + Self.C0_offset + 1,
        }

    @always_inline
    fn consumer_c1(self) -> ConsumerPipeline[1]:
        return {
            self.mbar_base + Self.C1_offset,
            self.mbar_base + Self.C1_offset + 1,
        }

    @always_inline
    fn producer_c(self, wg_idx: UInt32) -> ProducerPipeline[1]:
        base = UInt32(Self.C0_offset) + 2 * wg_idx
        return {self.mbar_base + base, self.mbar_base + base + 1}

    @always_inline
    fn pipeline_order_wait(self, wg_idx: UInt32) -> MBarType:
        return self.mbar_base + Self.order_offset + wg_idx

    @always_inline
    fn pipeline_order_arrive(self, wg_idx: UInt32) -> MBarType:
        return self.mbar_base + (Self.order_offset + 1) - wg_idx

    @always_inline
    fn q1_wait_mbar(self) -> MBarType:
        return self.mbar_base + Self.Q1SyncIdx

    # K/V/O barrier accessors
    @always_inline("nodebug")
    fn get_k_mbars(self) -> MBarType:
        """Returns base pointer for K pipeline barriers."""
        return self.mbar_base + Self.K_offset

    @always_inline("nodebug")
    fn get_v_mbars(self) -> MBarType:
        """Returns base pointer for V pipeline barriers (MHA only)."""
        comptime assert (
            Self.separate_kv
        ), "Use get_kv_mbars for unified pipeline"
        return self.mbar_base + Self.V_offset

    @always_inline("nodebug")
    fn get_kv_mbars(self) -> MBarType:
        """Returns base pointer for unified KV pipeline barriers (MLA)."""
        return self.mbar_base + Self.K_offset

    # O pipeline convenience methods
    @always_inline("nodebug")
    fn producer_o0(self) -> ProducerPipeline[1]:
        """Get O producer for warp group 0."""
        return {
            self.mbar_base + Self.O_producer_offset,
            self.mbar_base + Self.O_consumer_offset,
        }

    @always_inline("nodebug")
    fn producer_o1(self) -> ProducerPipeline[1]:
        """Get O producer for warp group 1."""
        return {
            self.mbar_base + Self.O_producer_offset + 1,
            self.mbar_base + Self.O_consumer_offset + 1,
        }

    @always_inline("nodebug")
    fn consumer_o(self) -> ConsumerPipeline[2]:
        """Get O consumer pipeline."""
        return {
            self.mbar_base + Self.O_producer_offset,
            self.mbar_base + Self.O_consumer_offset,
        }

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return UInt32(Self.size)


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
](TrivialRegisterPassable):
    comptime qkv_type = Self.KVLUTType.dtype
    comptime accum_type = DType.float32
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
    comptime HalfBM = Self.BM // 2

    comptime num_qk_stages = Self.config.num_qk_stages
    comptime num_pv_stages = Self.config.num_pv_stages

    # Unified misc barriers type managing all barriers including K/V/O pipelines
    comptime MiscMBarsType = FA4MiscMBars[
        num_qk_stages = Self.num_qk_stages,
        num_pv_stages = Self.num_pv_stages,
        num_kv_stages = Self.config.num_kv_stages,
        separate_kv=True,
        use_order_barriers=EnableForcedOrdering,
    ]

    # First MMA is Q@K' (can be staged by num_qk_stages)
    # (BM x depth) @ (BN x depth)' -> (BM x BN)
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M = Self.MMA_M,  # generally 128
        MMA_N = Self.BN,
        BK = align_up(Self.depth, Self.config.MMA_K),  # BK in memory depth
        swizzle_a = Self.config.swizzle_mode,
        swizzle_b = Self.config.swizzle_mode,
        transpose_b=True,
        num_stages = Self.num_qk_stages,
    ]
    # Second MMA is P@V (V not staged, but P writing can be staged)
    # (BM x BN) @ (BN x depth) -> (BM x depth)
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M = Self.MMA_M,
        MMA_N = Self.config.padded_depth,
        BK = Self.BN,
        swizzle_b = Self.config.swizzle_mode,
        transpose_b=False,
        num_stages = Self.num_pv_stages,
    ]

    comptime swizzle_granularity = Self.config.swizzle_mode.bytes() // Self.qkv_dt_size
    comptime k_elements: UInt32 = UInt32(
        Self.swizzle_granularity * Self.config.BN
    )
    comptime qo_bytes: UInt32 = UInt32(Self.qkv_dt_size * Self.qo_elements)
    comptime k_bytes: UInt32 = UInt32(Self.qkv_dt_size) * Self.k_elements
    comptime MMA_K = 16
    comptime v_bytes_per_mma: UInt32 = UInt32(
        Self.qkv_dt_size * Self.MMA_K * Self.config.padded_depth
    )

    comptime PositionType = MHAPosition[
        Self.config.BM,
        Self.config.BN,
        Self.config.depth,
        Self.config.padded_depth,
        Self.config.num_q_heads,
        Self.config.group,
        _is_decoding[Self.MaxSeqLenType](),
    ]

    comptime q_offset: Int32 = 0
    comptime kv_offset: Int32 = Self.q_offset + Int32(
        Self.config.BM * Self.config.padded_depth
    )
    comptime correction_offset: Int32 = (
        Self.kv_offset
        + Int32(
            2
            * Self.config.num_kv_stages
            * Self.config.padded_depth
            * Self.config.BN
        )
    ) * Int32(size_of[Self.qkv_type]()) // Int32(size_of[DType.float32]())
    comptime mbar_offset = (
        Self.correction_offset + Int32(Self.config.BM)
    ) * Int32(size_of[DType.float32]()) // Int32(size_of[SharedMemBarrier]())

    @staticmethod
    @always_inline
    fn get_tmem_ptr(
        misc_mbars: Self.MiscMBarsType,
    ) -> SharedMemPointer[UInt32]:
        # tmem_ptr comes after all barriers (now unified in MiscMBarsType)
        return (misc_mbars.mbar_base + Self.MiscMBarsType.num_mbars()).bitcast[
            UInt32
        ]()

    @staticmethod
    @always_inline
    fn get_q_smem(
        misc_mbars: Self.MiscMBarsType,
    ) -> SharedMemPointer[Scalar[Self.qkv_type]]:
        return (misc_mbars.mbar_base - Self.mbar_offset).bitcast[
            Scalar[Self.qkv_type]
        ]() + Self.q_offset

    @staticmethod
    @always_inline
    fn get_kv_smem(
        misc_mbars: Self.MiscMBarsType,
    ) -> SharedMemPointer[Scalar[Self.qkv_type]]:
        return (misc_mbars.mbar_base - Self.mbar_offset).bitcast[
            Scalar[Self.qkv_type]
        ]() + Self.kv_offset

    @staticmethod
    @always_inline
    fn get_correction_smem(
        misc_mbars: Self.MiscMBarsType,
    ) -> SharedMemPointer[Float32]:
        return (misc_mbars.mbar_base - Self.mbar_offset).bitcast[
            Float32
        ]() + Self.correction_offset

    @staticmethod
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    fn kernel(
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            depth = Self.config.depth,
            group = Self.config.group,
            decoding=False,
            num_qk_stages = Self.config.num_qk_stages,
        ],
        k_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN = Self.config.BN,
            BK = Self.config.BK0,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN = Self.config.BN,
            BK = Self.config.padded_depth,
        ],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            BN = Self.config.depth,
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
        comptime assert Self.MMA_M == 64 or Self.MMA_M == 128
        comptime assert _is_decoding[Self.MaxSeqLenType]() == False
        comptime assert Self.config.supported(), (
            "depth = "
            + String(Self.config.depth)
            + "\nBN = "
            + String(Self.config.BN)
            + "\nnum_kv_stages = "
            + String(Self.config.num_kv_stages)
            + "\ntmem_used = "
            + String(Self.config.tmem_used)
            + "\nsmem_used = "
            + String(Self.config.smem_used)
        )
        comptime assert (
            not Self.SchedulerType.may_advance
        ), "Persistent kernels not yet supported with FA4"

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
        comptime assert (
            num_qo == 1 or num_qo == 2
        ), "Currently only support num_qo == 1 or 2"
        mbar_base = (
            external_memory[
                SharedMemBarrier,
                address_space = AddressSpace.SHARED,
                alignment=128,
                name="mha_dynamic_shared_memory",
            ]()
            + Self.mbar_offset
        )

        var misc_mbars: Self.MiscMBarsType = {mbar_base}

        # https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
        comptime num_reg_softmax = 192
        comptime num_reg_correction = 88
        comptime num_reg_other = 40

        comptime assert not Self.PartitionType.do_partition, (
            "Neither partitioning nor decoding are supported by the 2-q"
            " implementation."
        )

        var warp_idx = UInt32(warp.broadcast(warp_id()))
        if warp_idx == 0:
            # Initialize all barriers (S/C/order/Q1Sync/K/V/O) in one call
            misc_mbars.init(lane_idx=Int32(thread_idx.x))
        elif warp_idx == 1:
            tcgen05_alloc[Self.cta_group](
                Self.get_tmem_ptr(misc_mbars), Self.config.sm100_tmem_cols
            )
        elif warp_idx == 2:
            e = elect()
            if e != 0:
                q_tma_op.prefetch_descriptor()
            if e != 0:
                k_tma_op.prefetch_descriptor()
            if e != 0:
                v_tma_op.prefetch_descriptor()

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
                misc_mbars,
                pos.score_row,
                seq_info,
                mask,
                pos.num_keys,
                scale.cast[Self.accum_type](),
                score_mod,
                max_seq_len.as_uint32(),
                ragged_tma_store,
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
                misc_mbars,
                pos.score_row,
                pos.num_keys,
                mask,
            )
        else:
            if warp_idx == 13:  # produce
                warpgroup_reg_dealloc[num_reg_other]()
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
                    pos.score_row,
                    pos.num_keys,
                    seq_info,
                    max_seq_len,
                    mask,
                    q_tma_op,
                    k_tma_op,
                    v_tma_op,
                    kv_lut,
                )

            elif warp_idx == 12:  # Q @ K', P @ V
                warpgroup_reg_dealloc[num_reg_other]()
                var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                    batch_size, max_seq_len, valid_length, partition
                )

                if not seq_info.is_valid():
                    tcgen05_release_allocation_lock[Self.cta_group]()
                    tcgen05_dealloc[Self.cta_group](
                        Self.get_tmem_ptr(misc_mbars)[],
                        Self.config.sm100_tmem_cols,
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
                    misc_mbars,
                    pos.score_row,
                    pos.num_keys,
                    mask,
                )
            else:
                warpgroup_reg_dealloc[24]()

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
            Index[dtype = DType.int32](Self.BM, Self.BN),
        )

    @always_inline
    @staticmethod
    fn scale_write_output(
        local_row: UInt32,
        local_warp_idx: UInt32,
        warp_group_idx: UInt32,
        inv_row_sum: Scalar[Self.accum_type],
        o_smem_arg: SharedMemPointer[Scalar[Self.output_type]],
        o_tmem: TMemTile[Self.accum_type, Self.BM // 2, Self.padded_depth],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            BN = Self.config.depth,
        ],
        consumer_mbar: MBarType,
        num_output_rows: Int32,
        out_head_idx: UInt32,
        out_row_idx: UInt32,
    ):
        e = elect()
        if e != 0:
            ragged_tma_store.prefetch_descriptor()

        o = o_tmem.load_async_with_st_matrix_layout[
            num_threads=WARPGROUP_SIZE
        ]()
        comptime num_rows = o.layout[0].size()
        inv_row_sums = tt_stack_allocation[
            dtype = Self.accum_type, address_space = AddressSpace.LOCAL
        ](row_major[num_rows]())
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
        comptime for i in range(num_rows):
            # lane // 4, lane // 4 + 8, lane // 4 + 16, lane // 4 + 24
            inv_row_sums[i] = warp.shuffle_idx(
                inv_row_sum, lane_row + UInt32(8 * i)
            )

        comptime for i in range(num_rows):
            irs = o.element_type(
                rebind[Scalar[Self.accum_type]](inv_row_sums[i])
            )

            comptime for j in range(o.layout[1].size()):
                o[i, j] *= irs

        comptime swizzle = make_swizzle[
            Self.output_type, Self.config.swizzle_mode
        ]()

        comptime ST = STMatrixLayout[
            Self.BM // 2, Self.padded_depth, num_threads=WARPGROUP_SIZE
        ]

        comptime swizzle_granularity = Self.config.swizzle_mode.bytes() // size_of[
            Self.output_type
        ]()
        comptime iters = Self.padded_depth // swizzle_granularity

        comptime swizzle_block_size: UInt32 = UInt32(
            WARP_SIZE * swizzle_granularity
        )
        o_smem = o_smem_arg + local_warp_idx * swizzle_block_size

        comptime for i in range(2):
            comptime datapath_offset: UInt32 = UInt32(
                16 * i * swizzle_granularity
            )

            comptime for j in range(iters):
                comptime ofs = i * ST.frag_size + j * (ST.frag_size // iters)
                comptime reg_layout = row_major[1, ST.frag_size // iters]()
                var rows_of_o_frags = _LocalTT[Self.accum_type, reg_layout](
                    o.ptr + ofs, reg_layout
                )  # all the repeats across n and m

                comptime warp_smem_offset: UInt32 = datapath_offset + UInt32(
                    j * (Self.BM // 2) * swizzle_granularity
                )
                comptime smem_layout = row_major[16, swizzle_granularity]()
                var accum_smem_warp_tile = _SharedMemTT[
                    Self.output_type, smem_layout
                ](o_smem + warp_smem_offset, smem_layout)

                output_reg_to_smem_st_matrix[
                    BM=16,
                    swizzle=swizzle,
                    num_consumer=1,
                ](
                    lane,
                    local_warp_group_idx=0,
                    output_reg_tile=rows_of_o_frags,
                    accum_smem_tile=accum_smem_warp_tile,
                )
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))

        # # first thread of each warp_group
        if local_warp_idx == 0:
            if e != 0:
                fence_async_view_proxy()

            if e != 0:
                ragged_tma_store.async_copy_from(
                    o_smem,
                    ragged_idx=out_row_idx,
                    dynamic_dim=UInt32(num_output_rows),
                    middle_idx=out_head_idx,
                )
            if e != 0:
                cp_async_bulk_commit_group()
        cp_async_bulk_wait_group[0]()

    @staticmethod
    @always_inline
    fn softmax(
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        seq_info: SeqInfo,
        mask: Self.MaskType,
        num_keys: UInt32,
        scale: Scalar[Self.accum_type],
        score_mod: Self.ScoreModType,
        max_seq_len: UInt32,
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            BN = Self.config.depth,
        ],
        sink_weights: Self.SinkType,
    ):
        var tmem_addr: UInt32 = Self.get_tmem_ptr(mbars)[]
        var o_smem: SharedMemPointer[
            Scalar[Self.output_type]
        ] = Self.get_q_smem(mbars).bitcast[Scalar[Self.output_type]]()
        var o_prod_mbar: MBarType = (
            mbars.mbar_base + Self.MiscMBarsType.O_producer_offset
        )
        var o_cons_mbar: MBarType = (
            mbars.mbar_base + Self.MiscMBarsType.O_consumer_offset
        )
        var s_tmem: UInt32 = tmem_addr + UInt32(Self.config.TMEM_S0)

        # var tid = UInt32(thread_idx.x)
        var tid = llvm_opaque_tid()
        var row = tid % 128
        var warp_idx: UInt32 = warp.broadcast(tid // 32)
        var warp_group_idx: UInt32 = warp.broadcast(tid // 128)

        comptime if Self.config.split_m:
            # split-M: second S is (+16 rows) in st-matrix space
            s_tmem += (16 << 16) * warp_group_idx
        else:
            # 2-Q path: S1 is at +BN columns
            s_tmem += UInt32(Self.config.BN) * warp_group_idx

        p_tmem = s_tmem
        c_tmem = p_tmem + UInt32(Self.config.BN // 2)
        s_tile = Self.UMMA0Type.CType(s_tmem)
        p_tile = Self.UMMA1Type.AType(p_tmem)

        var pipeline_s = mbars.consumer_s(warp_group_idx)
        pipeline_c = mbars.producer_c(warp_group_idx)
        var order_phase: UInt32 = 1 - warp_group_idx

        comptime if EnableForcedOrdering:
            order_s_wait = mbars.pipeline_order_wait(warp_group_idx)
            order_s_arrive = mbars.pipeline_order_arrive(warp_group_idx)
        else:
            order_s_wait = MBarType()
            order_s_arrive = MBarType()

        var q_head_idx: UInt32 = seq_info.head_idx
        var scale_log2e: Scalar[Self.accum_type] = scale
        var correction_smem = Self.get_correction_smem(mbars) + tid

        comptime if not (
            Self.use_score_mod or Self.MaskType.apply_log2e_after_mask
        ):
            scale_log2e *= log2e

        # Fuse scale*log2e multiplication and row_max subtraction into a
        # single FMA in store_exp. Only valid on the default scaling path
        # where score_mod and apply_log2e_after_mask are both off.
        # Disabled when sink weights are used because the sink logit lives
        # in a different domain (scaled by log2e only, not scale*log2e).
        # To disable for NaN debugging, set use_fma = False.
        comptime use_fma = not (
            Self.use_score_mod
            or Self.MaskType.apply_log2e_after_mask
            or not Self.SinkType.is_null
        )

        @parameter
        @always_inline
        fn mask_row[
            BN: Int, //, mask_strategy: MaskStrategy
        ](s: LocalTensor[Self.accum_type, row_major[BN]()], kv_row: UInt32,):
            apply_mask[
                use_score_mod = Self.use_score_mod,
                mask_strategy=mask_strategy,
                skip_scale=use_fma,
            ](
                s,
                mask,
                score_mod,
                scale_log2e,
                prompt_idx=seq_info.prompt_idx,
                q_head_idx=q_head_idx,
                kv_tile_start_row=Int32(kv_row),
                max_seq_len=max_seq_len,
                num_keys=Int32(num_keys),
                score_row=Int32(score_row + tid),
            )

        # while waiting, offset output
        comptime splitBM = Self.BM // 2
        var num_output_rows = min(
            Int32(seq_info.seq_len)
            - Int32(seq_info.prompt_offset)
            - Int32(warp_group_idx) * Int32(splitBM),
            Int32(splitBM),
        )

        gmem_row = Self.PositionType.get_q_gmem_row[ragged = Self.ragged](
            seq_info, max_seq_len
        )
        s = tt_stack_allocation[
            dtype = Self.accum_type, address_space = AddressSpace.LOCAL
        ](row_major[Self.config.BN]())

        comptime max_unroll = 8

        @parameter
        @always_inline
        fn load_mask_max_impl[
            *, mask_strategy: MaskStrategy
        ](kv_row: UInt32) -> StaticTuple[Float32, max_unroll]:
            comptime if EnableForcedOrdering:
                order_s_wait[].wait(order_phase)
            pipeline_s.wait()
            tcgen05_fence_after()
            # break up into sets of 32
            # minimize wait time by using smallest first
            comptime BM = Self.config.BM // 2
            comptime batch_size = 32
            comptime has_remainder = (Self.config.BN % batch_size) != 0
            comptime first_cols = (
                Self.config.BN % batch_size
            ) if has_remainder else batch_size
            s0 = TMemTile[Self.accum_type, BM, first_cols](s_tmem).load_async()
            s1 = TMemTile[Self.accum_type, BM, batch_size](
                s_tmem + UInt32(first_cols)
            ).load_async()
            mask_row[mask_strategy=mask_strategy](s0, kv_row)
            vrow_max = maximum[width=max_unroll](s0)

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
            comptime for i in range(cols // (2 * batch_size)):
                comptime offset0 = first_cols + batch_size * (2 * i)
                comptime offset1 = first_cols + batch_size * (2 * i + 1)
                comptime offset2 = first_cols + batch_size * (2 * i + 2)

                comptime if offset1 >= Self.config.BN:
                    mask_row[mask_strategy=mask_strategy](
                        s1, kv_row + UInt32(offset0)
                    )
                    vrow_max = maximum(s1, vrow_max)
                    s.ptr.store(offset0, s1.ptr.load[width=batch_size]())
                else:
                    s2 = TMemTile[Self.accum_type, BM, batch_size](
                        s_tmem + UInt32(offset1)
                    ).load_async()
                    mask_row[mask_strategy=mask_strategy](
                        s1, kv_row + UInt32(offset0)
                    )
                    vrow_max = maximum(s1, vrow_max)
                    s.ptr.store(offset0, s1.ptr.load[width=batch_size]())

                    comptime if offset2 < Self.config.BN:
                        s1 = TMemTile[Self.accum_type, BM, batch_size](
                            s_tmem + UInt32(offset2)
                        ).load_async()
                    mask_row[mask_strategy=mask_strategy](
                        s2, kv_row + UInt32(offset1)
                    )
                    vrow_max = maximum(s2, vrow_max)
                    s.ptr.store(offset1, s2.ptr.load[width=batch_size]())

            return vrow_max

        @parameter
        @always_inline
        fn load_mask_max[
            *, mask_strategy: MaskStrategy
        ](kv_row: UInt32) -> Float32:
            return maximum(
                load_mask_max_impl[mask_strategy=mask_strategy](kv_row)
            )

        @parameter
        @always_inline
        fn load_mask_max[
            *, mask_strategy: MaskStrategy
        ](kv_row: UInt32, old_max: Float32) -> Float32:
            return maximum(
                load_mask_max_impl[mask_strategy=mask_strategy](kv_row), old_max
            )

        comptime f32x2 = SIMD[DType.float32, 2]

        @parameter
        @always_inline
        fn store_exp(row_max: Float32) -> f32x2:
            comptime exp_simd = 2
            comptime vs_len = Self.config.BN // exp_simd  # 128 // 2 = 64
            comptime assert (vs_len % Self.config.num_pv_stages) == 0
            comptime use_3_then_1_split = Self.UMMA1Type.use_3_then_1_split
            comptime batch_size = 32 if Self.config.num_pv_stages == 1 else vs_len // (
                4 if use_3_then_1_split else Self.config.num_pv_stages
            )
            comptime num_batch_iters = vs_len // batch_size
            comptime remainder = vs_len % batch_size
            comptime assert num_batch_iters > 0
            comptime BatchTileType = TMemTile[
                Self.qkv_type, Self.config.BM // 2, batch_size * exp_simd
            ]
            comptime RemainderTileType = TMemTile[
                Self.qkv_type, Self.config.BM // 2, remainder * exp_simd
            ]
            comptime assert (Self.config.BN % exp_simd) == 0

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
            # When use_fma, scores are unscaled; fuse scale+subtract
            # into fma_ftz(score, scale_log2e, -row_max*scale_log2e).
            var vrow_max: f32x2
            var vscale: f32x2
            var vneg_max_scaled: f32x2

            comptime if use_fma:
                vscale = f32x2(scale_log2e)
                vneg_max_scaled = f32x2(-row_max * scale_log2e)
                vrow_max = f32x2(0)  # unused
            else:
                vrow_max = f32x2(row_max)
                vscale = f32x2(0)  # unused
                vneg_max_scaled = f32x2(0)  # unused

            @parameter
            @always_inline
            fn score_to_logit(score: f32x2) -> f32x2:
                comptime if use_fma:
                    return fma_ftz(score, vscale, vneg_max_scaled)
                else:
                    return sub_ftz(score, vrow_max)

            var acc: f32x2 = exp2(score_to_logit(rebind[f32x2](vs[0])))
            vs[0] = rebind[vs.ElementType](acc)
            vsi = exp2(score_to_logit(rebind[f32x2](vs[1])))
            vs[1] = rebind[vs.ElementType](vsi)

            comptime if EnableEarlyAdd:
                acc = add_ftz(acc, vsi)
            comptime exp2_emulation_freq = 4

            comptime for i in range(2, 8):
                vs[i] = rebind[vs.ElementType](
                    score_to_logit(rebind[f32x2](vs[i]))
                )

            comptime for i in range(2, 8):
                comptime if EnableEarlyAdd or i % exp2_emulation_freq != 0:
                    vsi = exp2(rebind[f32x2](vs[i]))
                else:
                    vsi = exp2_emulation(rebind[f32x2](vs[i]))
                vs[i] = rebind[vs.ElementType](vsi)

                comptime if EnableEarlyAdd:
                    acc = add_ftz(acc, vsi)

            comptime for i in range(8, batch_size // 2):
                diff = score_to_logit(rebind[f32x2](vs[i]))

                comptime if EnableEarlyAdd or i % exp2_emulation_freq != 0:
                    vsi = exp2(diff)
                else:
                    vsi = exp2_emulation(diff)
                vs[i] = rebind[vs.ElementType](vsi)

                comptime if EnableEarlyAdd:
                    acc = add_ftz(acc, vsi)

            # at this point, we need 32 fewer fp32 registers but 16 more u32
            comptime for i in range(batch_size // 2, batch_size):
                diff = score_to_logit(rebind[f32x2](vs[i]))

                comptime if i % exp2_emulation_freq == 0:
                    vs[i] = rebind[vs.ElementType](exp2_emulation(diff))
                else:
                    vs[i] = rebind[vs.ElementType](exp2(diff))

            BatchTileType(p_tmem).store_async(
                LocalTensor[
                    Self.accum_type, row_major[batch_size * exp_simd]()
                ](s.ptr, row_major[batch_size * exp_simd]())
            )

            comptime for b in range(1, num_batch_iters):
                comptime offset = batch_size * b

                comptime if use_3_then_1_split:
                    comptime if 4 * b == 3 * num_batch_iters:
                        tcgen05_store_wait()
                        tcgen05_fence_before()
                        pipeline_s.release_no_step[0]()
                elif Self.config.num_pv_stages > 1:
                    comptime assert Self.config.num_pv_stages == num_batch_iters
                    tcgen05_store_wait()
                    tcgen05_fence_before()

                    comptime assert Self.config.num_pv_stages == num_batch_iters
                    pipeline_s.release_no_step[b - 1]()

                comptime for i in range(offset, offset + batch_size):
                    diff = score_to_logit(rebind[f32x2](vs[i]))

                    comptime if i % exp2_emulation_freq == 0:
                        vs[i] = rebind[vs.ElementType](exp2_emulation(diff))
                    else:
                        vs[i] = rebind[vs.ElementType](exp2(diff))

                comptime el_offset = offset * exp_simd
                comptime tmem_offset = (
                    el_offset * size_of[Self.qkv_type]()
                ) // size_of[Self.accum_type]()
                BatchTileType(p_tmem + UInt32(tmem_offset)).store_async(
                    LocalTensor[
                        Self.accum_type, row_major[batch_size * exp_simd]()
                    ](s.ptr + el_offset, row_major[batch_size * exp_simd]())
                )

            comptime if remainder > 0:
                comptime offset = batch_size * num_batch_iters

                comptime for i in range(offset, offset + remainder):
                    diff = score_to_logit(rebind[f32x2](vs[i]))

                    comptime if i % exp2_emulation_freq == 0:
                        vs[i] = rebind[vs.ElementType](exp2_emulation(diff))
                    else:
                        vs[i] = rebind[vs.ElementType](exp2(diff))

                comptime el_offset = offset * exp_simd
                comptime tmem_offset = (
                    el_offset * size_of[Self.qkv_type]()
                ) // size_of[Self.accum_type]()
                RemainderTileType(p_tmem + UInt32(tmem_offset)).store_async(
                    LocalTensor[
                        Self.accum_type, row_major[remainder * exp_simd]()
                    ](s.ptr + el_offset, row_major[remainder * exp_simd]())
                )

            tcgen05_store_wait()
            tcgen05_fence_before()
            pipeline_s.release[Self.config.num_pv_stages - 1]()

            comptime if EnableForcedOrdering:
                _ = order_s_arrive[].arrive()
                order_phase ^= 1
            pipeline_c.acquire()
            # now we can sum the remaining elements of `acc`
            comptime add_offset = batch_size // 2 if EnableEarlyAdd else 0
            var acc0: f32x2
            var acc1: f32x2
            var acc2: f32x2
            var acc3: f32x2

            comptime if EnableEarlyAdd:
                acc0 = acc
                acc1 = rebind[f32x2](vs[batch_size // 2])
                acc2 = rebind[f32x2](vs[batch_size // 2 + 1])
                acc3 = add_ftz(
                    rebind[f32x2](vs[batch_size // 2 + 2]),
                    rebind[f32x2](vs[batch_size // 2 + 3]),
                )
            else:
                acc0 = acc
                acc1 = rebind[f32x2](vs[1])
                acc2 = rebind[f32x2](vs[2])
                acc3 = rebind[f32x2](vs[3])

            comptime for i in range(add_offset + 4, vs_len, 4):
                acc0 = add_ftz(acc0, rebind[f32x2](vs[i]))
                acc1 = add_ftz(acc1, rebind[f32x2](vs[i + 1]))
                acc2 = add_ftz(acc2, rebind[f32x2](vs[i + 2]))
                acc3 = add_ftz(acc3, rebind[f32x2](vs[i + 3]))
            return add_ftz(add_ftz(acc0, acc1), add_ftz(acc2, acc3))

        var kv_row: UInt32 = mask.start_column[
            Self.BM, Self.BN, Self.page_size
        ](score_row)
        comptime mask_sets = Self.MaskType.nonfull_sets[Self.BM, Self.BN]()
        comptime mask_strategies = Self.MaskType.mask_strategies[
            Self.BM, Self.BN
        ]()
        comptime num_sets = len(mask_sets)

        var row_max: Float32
        var mask_iters: StaticTuple[UInt32, num_sets] = {}

        comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
            mask_ends = mask.masked_set_ends[
                BM = Self.BM, BN = Self.BN, page_size = Self.page_size
            ](score_row, num_keys)
            mask_iters[0] = mask_ends[0]

            comptime for i in range(1, num_sets):
                mask_iters[i] = mask_ends[i] - mask_ends[i - 1]

        comptime assert num_sets >= 1 and num_sets <= 3
        comptime assert (
            num_sets == 1 or mask_sets[0] != TileMaskStatus.UNKNOWN_MASK
        )

        comptime if num_sets == 1:
            row_max = load_mask_max[mask_strategy = mask_strategies[0]](kv_row)
            mask_iters[0] -= 1
        else:
            # find out which strategy to apply
            if mask_iters[0] > 0:
                row_max = load_mask_max[mask_strategy = mask_strategies[0]](
                    kv_row
                )
                mask_iters[0] -= 1
            else:
                comptime if num_sets == 2:
                    row_max = load_mask_max[mask_strategy = mask_strategies[1]](
                        kv_row
                    )
                    mask_iters[1] -= 1
                else:
                    if mask_iters[1] > 1:
                        row_max = load_mask_max[
                            mask_strategy = mask_strategies[1]
                        ](kv_row)
                        mask_iters[1] -= 1
                    else:
                        row_max = load_mask_max[
                            mask_strategy = mask_strategies[2]
                        ](kv_row)
                        mask_iters[2] -= 1
        var sink_weights_ptr = UnsafePointer[
            Scalar[Self.qkv_type], ImmutAnyOrigin
        ]()
        var sink_weight: Scalar[Self.accum_type]

        comptime if not Self.SinkType.is_null:
            sink_weights_ptr = rebind[
                UnsafePointer[Scalar[Self.qkv_type], ImmutAnyOrigin]
            ](sink_weights.value())
            var head_idx: UInt32 = seq_info.head_idx

            comptime if use_fma:
                sink_weight = sink_weights_ptr[head_idx].cast[Self.accum_type]()
            else:
                sink_weight = (
                    sink_weights_ptr[head_idx].cast[Self.accum_type]() * log2e
                )
            row_max = max(row_max, sink_weight)
        else:
            sink_weights_ptr = {}
            sink_weight = 0.0

        var row_sum: f32x2 = store_exp(row_max)

        var o_phase: UInt32 = 0  # initial wait is phase 0

        comptime if not Self.SinkType.is_null:
            comptime if use_fma:
                row_sum[0] += exp2((sink_weight - row_max) * scale_log2e)
            else:
                row_sum[0] += exp2(sink_weight - row_max)

        comptime rescale_threshold: Float32 = Float32(-8) if size_of[
            Self.qkv_type
        ]() >= 2 else Float32(0)

        comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
            comptime for i in range(num_sets):
                comptime mask_status = mask_sets[i]
                comptime mask_strategy = mask_strategies[i]
                var iters: UInt32

                iters = warp.broadcast(mask_iters[i])
                while iters != 0:
                    iters -= 1
                    kv_row += UInt32(Self.config.BN)
                    # calculate rowmax
                    old_max = row_max
                    var new_row_max: Float32 = load_mask_max[
                        mask_strategy=mask_strategy
                    ](kv_row, old_max)

                    diff = sub_ftz(old_max, new_row_max)

                    comptime if use_fma:
                        diff = mul_ftz(diff, scale_log2e)
                    var correction: Float32

                    comptime if rescale_threshold < 0:
                        # old_max - new_row_max < -8
                        # 8 < new_row_max - old_max
                        if _vote_nvidia_helper(diff < rescale_threshold) != 0:
                            row_max = new_row_max
                            correction = exp2(diff)
                        else:
                            correction = 1
                    else:
                        row_max = new_row_max
                        correction = exp2(diff)
                    correction_smem[] = correction
                    pipeline_c.commit()
                    # update s->p
                    local_rowsum = store_exp(row_max)
                    row_sum = fma_ftz(row_sum, f32x2(correction), local_rowsum)
                    o_phase ^= 1
        else:
            while True:
                kv_row += UInt32(Self.config.BN)
                if kv_row >= num_keys:
                    break
                mask_status = Self.mask_status(mask, score_row, kv_row)
                if mask_status == TileMaskStatus.FULL_MASK:
                    continue
                # calculate rowmax
                old_max = row_max
                var new_row_max: Scalar[Self.accum_type]
                if mask_status == TileMaskStatus.PARTIAL_MASK:
                    new_row_max = load_mask_max[
                        mask_strategy = MaskStrategy.COMPUTED
                        | MaskStrategy.OUT_OF_BOUNDS
                    ](kv_row, old_max)
                else:
                    new_row_max = load_mask_max[
                        mask_strategy = MaskStrategy.OUT_OF_BOUNDS
                    ](kv_row, old_max)

                diff = sub_ftz(old_max, new_row_max)

                comptime if use_fma:
                    diff = mul_ftz(diff, scale_log2e)
                var correction: Float32

                comptime if rescale_threshold < 0:
                    # old_max - new_row_max < -8
                    # 8 < new_row_max - old_max
                    if _vote_nvidia_helper(diff < rescale_threshold) != 0:
                        row_max = new_row_max
                        correction = exp2(diff)
                    else:
                        correction = 1
                else:
                    row_max = new_row_max
                    correction = exp2(diff)
                correction_smem[] = correction
                pipeline_c.commit()
                # update s->p
                local_rowsum = store_exp(row_max)
                row_sum = fma_ftz(row_sum, f32x2(correction), local_rowsum)
                o_phase ^= 1
        # Do the final correction and write
        inv_row_sum = recip(row_sum.reduce_add())
        o_tile = Self.UMMA1Type.CType(
            tmem_addr
            + UInt32(Self.config.TMEM_O0)
            + warp_group_idx * UInt32(Self.config.padded_depth)
        )
        # wait on the o_pipeline producer
        comptime assert size_of[Self.output_type]() == size_of[Self.qkv_type]()
        if num_output_rows > 0:
            o_prod_mbar[warp_group_idx].wait(o_phase)  # consumer wait
            tcgen05_fence_after()  # example 1
            # TODO: pass in a dedicated barrier that a q-writer can wait on in a persistent kernel?

            Self.scale_write_output(
                row,
                warp_idx & 3,
                warp_group_idx,
                inv_row_sum,
                o_smem
                + warp_group_idx * UInt32(Self.HalfBM * Self.padded_depth),
                o_tile,
                ragged_tma_store,
                o_cons_mbar + warp_group_idx,  # consumer arrive
                num_output_rows,
                q_head_idx,
                gmem_row + warp_group_idx * UInt32(Self.HalfBM),
            )
        named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
        if warp_idx == 0:
            tcgen05_release_allocation_lock[Self.cta_group]()
            tcgen05_dealloc[Self.cta_group](
                tmem_addr, Self.config.sm100_tmem_cols
            )

    @staticmethod
    @always_inline
    fn correction(
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
    ):
        comptime assert size_of[Self.accum_type]() == 4

        var tmem_addr: UInt32 = Self.get_tmem_ptr(mbars)[]
        o0_tmem = tmem_addr + UInt32(Self.config.TMEM_O0)
        o1_tmem = tmem_addr + UInt32(Self.config.TMEM_O1)
        var correction_smem_arg: SharedMemPointer[
            Scalar[Self.accum_type]
        ] = Self.get_correction_smem(mbars)

        pipeline_c0 = mbars.consumer_c0()
        pipeline_c1 = mbars.consumer_c1()
        pipeline_o = mbars.consumer_o()

        var iter_count: UInt32 = (
            mask.total_iters[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )

        comptime batch_size = 16 if Self.config.depth % 16 == 0 else 8
        comptime assert Self.config.depth % batch_size == 0
        # output is BM x depth
        comptime load_iters = Self.config.depth // (2 * batch_size)
        comptime load_remainder = Self.config.depth % (2 * batch_size)
        comptime assert load_iters > 1
        comptime assert (load_remainder == batch_size) or (load_remainder == 0)
        var correction_smem_0 = correction_smem_arg + UInt32(thread_idx.x) % 128
        var correction_smem_1 = correction_smem_0 + (Self.BM // 2)

        while iter_count != 0:
            iter_count -= 1

            comptime for i in range(2):
                # correct
                var c_scalar: Scalar[Self.accum_type]

                comptime if i == 0:
                    pipeline_c0.wait()
                    c_scalar = correction_smem_0[0]
                else:
                    pipeline_c1.wait()
                    c_scalar = correction_smem_1[0]

                change = _vote_nvidia_helper(c_scalar < 1.0) != 0
                pipeline_o.wait()
                if change:
                    # TODO: experiment with different batch sizes.
                    # The idea here is to both pipeline, and reduce peak register use.

                    var o_tmem: UInt32

                    comptime if i == 0:
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

                    comptime for b in range(load_iters):
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
                        ](o_tmem + UInt32(b1_offset))
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            pack=False,
                        ](o_tmem + UInt32(b0_offset0), o_b0 * c_scalar)

                        comptime if b0_offset1 + batch_size <= Self.config.depth:
                            o_b0 = tcgen05_ld[  # 0b0 start
                                datapaths=32,
                                bits=32,
                                repeat=batch_size,
                                dtype = Self.accum_type,
                                pack=False,
                                width=batch_size,
                            ](o_tmem + UInt32(b0_offset1))
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            pack=False,
                        ](o_tmem + UInt32(b1_offset), o_b1 * c_scalar)

                    comptime if load_remainder > 0:
                        comptime offset = 2 * batch_size * load_iters
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=load_remainder,
                            pack=False,
                        ](o_tmem + UInt32(offset), o_b0 * c_scalar)
                    tcgen05_store_wait()
                    tcgen05_fence_before()
                pipeline_o.release()

                comptime if i == 0:
                    pipeline_c0.release()
                else:
                    pipeline_c1.release()

    @staticmethod
    @always_inline
    fn load(
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        num_keys: UInt32,
        seq_info: SeqInfo,
        max_seq_len: Self.MaxSeqLenType,
        mask: Self.MaskType,
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            depth = Self.config.depth,
            group = Self.config.group,
            decoding=False,
            num_qk_stages = Self.config.num_qk_stages,
        ],
        k_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN = Self.config.BN,
            BK = Self.config.BK0,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN = Self.config.BN,
            BK = Self.config.padded_depth,
        ],
        kv_lut: Self.KVLUTType,
    ):
        comptime KPipeType = KProducerPipeline[
            Self.KVLUTType.dtype, Self.config
        ]
        comptime VPipeType = VProducerPipeline[
            Self.KVLUTType.dtype, Self.config
        ]

        # If two-qo, we produce qkv in a pattern of
        # q0 & k0, q1, v0, k1, v1, k2, v2...
        comptime SMemTensorLT[layout: Layout] = SharedMemLT[
            Self.KVLUTType.dtype, layout
        ]
        comptime QType = SMemTensorLT[type_of(q_tma_op).layout]
        comptime KType = SMemTensorLT[type_of(k_tma_op).layout]
        comptime VType = SMemTensorLT[type_of(v_tma_op).layout]

        var kv_head_idx: UInt32 = seq_info.head_idx // UInt32(Self.group)

        var q_smem: SharedMemPointer[
            Scalar[Self.KVLUTType.dtype]
        ] = Self.get_q_smem(mbars)
        comptime q_elements = Self.HalfBM * Self.config.BK0
        comptime assert q_elements == QType.layout.size()
        comptime q_bytes = size_of[Self.qkv_type]() * q_elements
        comptime qk_bytes = pipeline_k.bytes + q_bytes
        var k_smem = q_smem + Self.config.BM * Self.config.padded_depth
        var v_smem = (
            k_smem
            + (Self.config.BN * Self.config.padded_depth)
            * Self.config.num_kv_stages
        )
        var pipeline_k: KPipeType = {mbars.get_k_mbars(), k_smem}
        var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem}

        var mbark0: KPipeType.KPairType

        mbark0 = pipeline_k.get_k[qk_stage=0]()  # no wait
        var q_gmem_row: UInt32 = Self.PositionType.get_q_gmem_row[
            ragged = Self.ragged
        ](seq_info, max_seq_len)
        var q_head_idx: UInt32 = seq_info.head_idx
        e = elect()
        # copy q0
        if e != 0:
            # Q0
            mbark0.mbar[].expect_bytes(Int32(qk_bytes))
        # copy q0
        if e != 0:
            q_tma_op.async_copy[eviction_policy = CacheEviction.EVICT_FIRST](
                QType(q_smem),
                mbark0.mbar[],
                StaticTuple[UInt32, 3](0, q_head_idx, q_gmem_row),
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
        if e != 0:  # K0
            k_tma_op.async_copy(
                mbark0.smem,
                mbark0.mbar[],
                StaticTuple[UInt32, 3](0, kv_head_idx, kv_gmem_row),
            )

        comptime for qk_stage in range(1, Self.config.num_qk_stages):
            comptime d_idx = qk_stage * Self.config.BK0
            mbark = pipeline_k.get_k[qk_stage=qk_stage]()  # no wait
            if e != 0:
                mbark.mbar[].expect_bytes(Int32(qk_bytes))
            if e != 0:
                q_tma_op.async_copy[
                    eviction_policy = CacheEviction.EVICT_FIRST
                ](
                    QType(q_smem + q_elements * qk_stage),
                    mbark.mbar[],
                    StaticTuple[UInt32, 3](
                        UInt32(d_idx), q_head_idx, q_gmem_row
                    ),
                )
            if e != 0:
                k_tma_op.async_copy(
                    mbark.smem,
                    mbark.mbar[],
                    StaticTuple[UInt32, 3](
                        UInt32(d_idx), kv_head_idx, kv_gmem_row
                    ),
                )

        pipeline_k.commit_step()
        # Q1
        q_gmem_row += UInt32(Self.HalfBM)
        var q1_mbar = mbars.q1_wait_mbar()

        comptime for qk_stage in range(Self.config.num_qk_stages):
            comptime q_smem_offset = q_elements * (
                Self.config.num_qk_stages + qk_stage
            )
            comptime d_idx = qk_stage * Self.config.BK0
            if e != 0:
                q1_mbar[qk_stage].expect_bytes(Int32(q_bytes))
            if e != 0:
                q_tma_op.async_copy(
                    QType(q_smem + q_smem_offset),
                    q1_mbar[qk_stage],
                    StaticTuple[UInt32, 3](
                        UInt32(d_idx), q_head_idx, q_gmem_row
                    ),
                )
        # copy v0
        mbarv0 = pipeline_v.get_v(e)
        if e != 0:
            v_tma_op.async_copy(
                mbarv0.smem,
                mbarv0.mbar[],
                StaticTuple[UInt32, 3](0, kv_head_idx, kv_gmem_row),
            )
        pipeline_v.commit_step()
        comptime check_mask = mask.nonfull_sets[Self.BM, Self.BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK
        # kv producer loop
        while iter_count != 0:
            iter_count -= 1
            kv_row += UInt32(Self.config.BN)

            comptime if check_mask:
                if (
                    Self.mask_status(mask, score_row, kv_row)
                    == TileMaskStatus.FULL_MASK
                ):
                    continue
            kv_gmem_row = kv_lut.row_idx(seq_info.prompt_idx, kv_row)

            # produce k
            comptime for k_stage in range(Self.config.num_qk_stages):
                pipeline_k.acquire_k[qk_stage=k_stage]()
                mbarkn = pipeline_k.get_k[qk_stage=k_stage](e)
                comptime d_idx = k_stage * Self.config.BK0
                if e != 0:
                    k_tma_op.async_copy(
                        mbarkn.smem,
                        mbarkn.mbar[],
                        StaticTuple[UInt32, 3](
                            UInt32(d_idx), kv_head_idx, kv_gmem_row
                        ),
                    )
            pipeline_k.commit_step()

            pipeline_v.acquire_v()
            mbarvn = pipeline_v.get_v(e)
            if e != 0:
                v_tma_op.async_copy(
                    mbarvn.smem,
                    mbarvn.mbar[],
                    StaticTuple[UInt32, 3](0, kv_head_idx, kv_gmem_row),
                )
            pipeline_v.commit_step()

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
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
    ):
        var tmem_addr: UInt32 = Self.get_tmem_ptr(mbars)[]
        var q_smem: SharedMemPointer[
            Scalar[Self.KVLUTType.dtype]
        ] = Self.get_q_smem(mbars)

        s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        s1_tmem = tmem_addr + UInt32(Self.config.TMEM_S1)
        o0_tmem = tmem_addr + UInt32(Self.config.TMEM_O0)
        o1_tmem = tmem_addr + UInt32(Self.config.TMEM_O1)

        # S pipelines with sub-stages (1 producer, num_pv_stages consumers)
        var pipeline_s0 = mbars.producer_s0()
        var pipeline_s1 = mbars.producer_s1()
        # Keep consumer pointers for acquire operations (shared phase tracking)
        consumer_s0 = pipeline_s0.consumer_mbar_base
        consumer_s1 = pipeline_s1.consumer_mbar_base

        # O pipelines: o0 and o1 alternate with different initial phases
        var pipeline_o0 = mbars.producer_o0()
        var pipeline_o1 = mbars.producer_o1()
        # Keep consumer pointers for acquire operations (shared phase tracking)
        consumer_o0 = pipeline_o0.consumer_mbar_base
        consumer_o1 = pipeline_o1.consumer_mbar_base
        # Adjust pipeline_o0 phase to 0 (o0 waits with phases 0,1,0,1,...)
        pipeline_o0.step()

        comptime q0_size = Self.HalfBM * Self.config.padded_depth
        comptime q0_bytes = q0_size * size_of[Self.KVLUTType.dtype]()
        q0 = Self.descriptor_q(q_smem)
        q1 = q0 + UInt32(q0_bytes)
        kv_smem = q_smem + 2 * q0_size

        comptime q_sub_bytes = Self.HalfBM * Self.config.BK0 * size_of[
            Self.KVLUTType.dtype
        ]()

        var k_smem = q_smem + Self.config.BM * Self.config.padded_depth
        var v_smem = (
            k_smem
            + (Self.config.BN * Self.config.padded_depth)
            * Self.config.num_kv_stages
        )
        comptime KPipeType = KConsumerPipeline[
            Self.KVLUTType.dtype, Self.config
        ]
        comptime VPipeType = VConsumerPipeline[
            Self.KVLUTType.dtype, Self.config
        ]
        var pipeline_k: KPipeType = {mbars.get_k_mbars(), k_smem}
        var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem}

        # We peel the first iteration, as we want to wait on q1
        var iter_count: UInt32 = (
            mask.total_iters[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )

        # Q_0 @ K_0' (staged over num_qk_stages)
        k0 = pipeline_k.get_k()
        e = elect()

        comptime for qk_stage in range(Self.num_qk_stages):
            pipeline_k.wait_k[qk_stage=qk_stage]()  # [kv0]
            Self.UMMA0Type.mma[stage_idx=qk_stage](
                q0, k0, s0_tmem, elect=e, c_scale=0
            )
        pipeline_s0.commit_mma(e)

        # Q_1 @ K_0' (staged over num_qk_stages)
        var q1_mbar = mbars.q1_wait_mbar()

        comptime for qk_stage in range(Self.num_qk_stages):
            q1_mbar[qk_stage].wait()  # wait on Q1
            Self.UMMA0Type.mma[stage_idx=qk_stage](
                q1, k0, s1_tmem, elect=e, c_scale=0
            )
            pipeline_k.release_k[qk_stage=qk_stage](e)  # [kv0]->kv1
        pipeline_s1.commit_mma(e)

        vlatest = pipeline_v.get_v()  # [kv1]
        pipeline_v.wait_v()  # [kv1]

        # For the first V tile in the current KV stage buffer:
        # Use the SAME base pointer you used for K (no manual offset).
        comptime for pv_stage in range(Self.num_pv_stages):
            _ = consumer_s0[pv_stage].wait(0)

            Self.UMMA1Type.mma[stage_idx=pv_stage](
                s0_tmem, vlatest, o0_tmem, elect=e, c_scale=0
            )
        pipeline_o0.commit_mma(e)
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
            # Q_0 @ K_n' (staged over num_qk_stages)
            kn = pipeline_k.get_k()  # kv_{2n-1}->[kv_{2n}]

            comptime for qk_stage in range(Self.num_qk_stages):
                pipeline_k.wait_k[qk_stage=qk_stage]()  # kv_{2n-1}->[kv_{2n}]
                Self.UMMA0Type.mma[stage_idx=qk_stage](
                    q0, kn, s0_tmem, elect=e, c_scale=0
                )
            pipeline_s0.commit_mma(e)

            # O_1 + P_1 @ V_{n-1}
            _ = consumer_o1[].wait(phase_o)

            comptime for pv_stage in range(Self.num_pv_stages):
                _ = consumer_s1[pv_stage].wait(phase_s)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                )
            pipeline_o1.commit_mma(e)
            # pipeline_o.step()
            phase_o = phase_s
            c_scale = 1
            pipeline_v.release_v(e)  # [kv_{2n-1}]

            # Q_1 @ K_n' (staged over num_qk_stages)
            comptime for qk_stage in range(Self.num_qk_stages):
                Self.UMMA0Type.mma[stage_idx=qk_stage](
                    q1, kn, s1_tmem, elect=e, c_scale=0
                )
                pipeline_k.release_k[qk_stage=qk_stage](
                    e
                )  # [kv_{2n}]->kv_{2n+1}
            pipeline_s1.commit_mma(e)
            phase_s ^= 1

            # O_0 + P_0 @ V_n
            vlatest = pipeline_v.get_v()  # [kv_{2n+1}]
            pipeline_v.wait_v()  # [kv_{2n+1}]
            _ = consumer_o0[].wait(phase_o)

            comptime for pv_stage in range(Self.num_pv_stages):
                _ = consumer_s0[pv_stage].wait(phase_s)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1
                )
            pipeline_o0.commit_mma(e)

        _ = consumer_o1[].wait(phase_o)

        comptime for pv_stage in range(Self.num_pv_stages):
            _ = consumer_s1[pv_stage].wait(phase_s)
            Self.UMMA1Type.mma[stage_idx=pv_stage](
                s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
            )
        pipeline_o1.commit_mma(e)

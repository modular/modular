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
"""Shared SM100 attention primitives used by both MHA and MLA kernels.

This module contains generic SM100 (Blackwell) GPU primitives including:
- TMEM access helpers (TMemTile, STMatrixLayout)
- Pipeline synchronization (StagedPipeline, RolePipeline, etc.)
- FTZ arithmetic (add_ftz, sub_ftz, mul_ftz, etc.)
- Barrier helpers (FA4MiscMBars)
- MMA building blocks (bulk_mma, SM100TensorAccumulatorSS/TS)
- Masking utilities (apply_mask, apply_oob_mask)
"""

from math import ceildiv, exp2, align_up, iota
from math.constants import log2e
from sys import size_of
from sys._assembly import inlined_assembly
from bit import prev_power_of_two, pop_count
from gpu.globals import WARP_SIZE
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.memory import AddressSpace
from gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAInsDescriptor,
    UMMAKind,
    MMASmemDescriptorPair,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.barriers import (
    WarpGroupBarrier,
)
from gpu.compute.arch.tcgen05 import (
    tcgen05_ld,
    tcgen05_st,
    tcgen05_store_wait,
)
from layout.int_tuple import IntTuple
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout._layout import Layout as InternalLayout, row_major
from layout import TileTensor
from layout import stack_allocation as tt_stack_allocation
from layout.tma_async import PipelineState, SharedMemBarrier
from memory import bitcast
from nn.fa4_config import FA4Config
from nn.mha_mask import MHAMask, MASK_VALUE, MaskStrategy
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple
from linalg.arch.sm100.mma import smem_descriptor


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
        @parameter
        if m_mma == 0 and n_mma == 0:
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

        dst = type_of(dst).stack_allocation()
        self.load_st_matrix_chunk[
            num_threads=num_threads,
            start_repeat=0,
            num_repeats = st_mat_layout.repeat,
        ](dst)

    @always_inline
    fn load_st_matrix_chunk[
        *, num_threads: Int, start_repeat: Int, num_repeats: Int
    ](
        self,
        dst: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_type_size = Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        """Load a range of repeat columns from tmem into a pre-allocated
        tensor.

        Parameters:
            num_threads: Number of threads in the warp group.
            start_repeat: First repeat index to load (0-based).
            num_repeats: Number of repeats to load.

        Args:
            dst: Pre-allocated register tensor.
        """
        comptime st_mat_layout = STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_type_size = Self.dtype_size,
        ]()
        comptime load_dtype = DType.uint32
        var ptr = rebind[
            UnsafePointer[
                Scalar[load_dtype],
                MutAnyOrigin,
                address_space = AddressSpace.LOCAL,
            ]
        ](dst.ptr)

        @parameter
        @always_inline
        fn load_fn[pow_two: Int, local_offset: Int]():
            comptime assert pow_two + local_offset <= num_repeats
            comptime if pow_two > 0:
                comptime for m_mma in range(st_mat_layout.num_m_tiles):
                    comptime offsets = STMatrixOffsets[
                        Self.BM,
                        Self.BN,
                        num_threads=num_threads,
                        accum_type_size = Self.dtype_size,
                        curr_repeat=pow_two,
                        cumulative_repeat = start_repeat + local_offset,
                        m_mma=m_mma,
                    ]()
                    tmem = self.tmem_addr + UInt32(offsets.tmem_offset)
                    frag = tcgen05_ld[
                        datapaths=16,
                        bits = st_mat_layout.bits,
                        repeat=pow_two,
                        dtype=load_dtype,
                        pack=False,
                        width = offsets.local_frag_size_b32,
                    ](tmem)
                    ptr.store(offsets.ptr_offset, frag)

        comptime max_value = 64 if st_mat_layout.bits == 128 else 32
        break_into_powers_of_two[
            func=load_fn, N=num_repeats, max_value=max_value
        ]()

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
        @parameter
        if Self.num_stages == 1:
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

            @parameter
            if stage_idx == 0:
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
    comptime use_3_then_1_split: Bool = Self.num_stages == 2 and Self.num_k_blocks % 4 == 0
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
        @parameter
        if Self.num_stages == 1:
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

            var scale: UInt32

            @parameter
            if stage_idx == 0:
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
fn llvm_opaque_tid() -> UInt32:
    return inlined_assembly["mov.u32 $0, %tid.x;", UInt32, constraints="=r"]()


@always_inline
fn intrin_ftz[intrin: String](a: Float32, b: Float32) -> Float32:
    return inlined_assembly[
        String(intrin, ".ftz.f32 $0, $1, $2;"),
        Float32,
        constraints="=f,f,f",
        has_side_effect=False,
    ](a, b)


@always_inline
fn intrin[intrin: String](a: Float32, b: Float32, c: Float32) -> Float32:
    return inlined_assembly[
        String(intrin, ".f32 $0, $1, $2, $3;"),
        Float32,
        constraints="=f,f,f,f",
        has_side_effect=False,
    ](a, b, c)


@always_inline
fn intrin_ftz_x2[
    intrin: String
](a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]) -> SIMD[
    DType.float32, 2
]:
    return inlined_assembly[
        String(intrin, ".ftz.f32x2 $0, $1, $2;"),
        SIMD[DType.float32, 2],
        constraints="=l,l,l",
        has_side_effect=False,
    ](a, b)


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
    return inlined_assembly[
        "fma.rn.ftz.f32x2 $0, $1, $2, $3;",
        SIMD[DType.float32, 2],
        constraints="=l,l,l,l",
        has_side_effect=False,
    ](a, b, c)


@always_inline
fn exp2_emulation[
    use_exp2_emulation: Bool = True
](x: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    @parameter
    if use_exp2_emulation:
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
    res = {}

    @parameter
    for w in range(width):
        res[w] = max_ftz(
            rebind[Float32](x[3 * w]),
            rebind[Float32](x[3 * w + 1]),
            rebind[Float32](x[3 * w + 2]),
        )

    # max idx = 3 * (width-1) + 2 = 3*width - 1
    comptime remaining_iters = BN - 3 * width
    comptime num_iters = remaining_iters // (2 * width)

    @parameter
    for i in range(num_iters):
        comptime col = i * 2 * width + 3 * width

        @parameter
        for w in range(width):
            res[w] = max_ftz(
                res[w],
                rebind[Float32](x[col + 2 * w]),
                rebind[Float32](x[col + 2 * w + 1]),
            )

    comptime remainder_base = 3 * width + 2 * width * num_iters
    comptime end_iters = (BN - remainder_base) // 2

    @parameter
    for w in range(end_iters):
        res[w] = max_ftz(
            res[w],
            rebind[Float32](x[remainder_base + 2 * w]),
            rebind[Float32](x[remainder_base + 2 * w + 1]),
        )

    @parameter
    if (BN - remainder_base) % 2 == 1:
        res[end_iters] = max_ftz(res[end_iters], rebind[Float32](x[BN - 1]))


@always_inline
fn maximum[
    BN: Int, //, *, width: Int = 4
](
    x: LocalTensor[DType.float32, row_major[BN]()],
    init: StaticTuple[Float32, width],
    out res: StaticTuple[Float32, width],
):
    res = init

    # unroll (using SIMD) to break up dependency chain
    comptime num_iters = BN // (2 * width)

    @parameter
    for i in range(num_iters):

        @parameter
        for w in range(width):
            comptime j = i * 2 * width + 2 * w
            res[w] = max_ftz(
                res[w], rebind[Float32](x[j]), rebind[Float32](x[j + 1])
            )

    comptime remainder_base = 2 * width * num_iters
    comptime end_iters = (BN - remainder_base) // 2

    @parameter
    for w in range(end_iters):
        res[w] = max_ftz(
            res[w],
            rebind[Float32](x[remainder_base + 2 * w]),
            rebind[Float32](x[remainder_base + 2 * w + 1]),
        )

    @parameter
    if (BN - remainder_base) % 2 == 1:
        res[end_iters] = max_ftz(res[end_iters], rebind[Float32](x[BN - 1]))


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
    @parameter
    for i in range(1, BN // width):
        acc += vx[i]

    return acc.reduce_add[size_out=2]()
    # return rebind[SIMD[dtype,width]](acc)


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

        @parameter
        if qk_stage == Self.num_qk_stages - 1:
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

    @always_inline
    fn split_smem[
        first: Layout, second: Layout
    ](self) -> Tuple[
        SharedMemLT[Self.dtype, first], SharedMemLT[Self.dtype, second]
    ]:
        comptime first_size = first.size()
        return {
            SharedMemLT[Self.dtype, first](self.smem.ptr),
            SharedMemLT[Self.dtype, second](self.smem.ptr + first_size),
        }


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
        @parameter
        if Self.is_k:
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
        @parameter
        if Self.is_k:
            comptime assert (
                Self.config.padded_depth % Self.config.num_qk_stages == 0
            ), "padded_depth must be divisible by num_qk_stages"
        self.pipeline = pipeline
        self.smem = smem
        self.pipeline.state._phase = 1

    @always_inline
    fn get_smem[*, qk_stage: Int = 0](self) -> Self.SMemType:
        """Get smem pointer for current stage."""

        @parameter
        if Self.is_k:
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

        @parameter
        if Self.is_producer:
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
        @parameter
        for i in range(Self.number_of_stages):
            self.mbar[i].init(Int32(Int(num_producer)))

        @parameter
        for i in range(Self.number_of_stages):
            self.mbar[i + Self.number_of_stages].init(Int32(Int(num_consumer)))

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return UInt32(2 * Self.number_of_stages)


@always_inline
fn apply_oob_mask[
    *,
    mask_strategy: MaskStrategy,
    apply_log2e_after_mask: Bool,
](
    s_arg: SIMD[DType.float32, 2],
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

    @parameter
    if apply_log2e_after_mask:
        s = mul_ftz(s, log2e)

    @parameter
    if MaskStrategy.OUT_OF_BOUNDS in mask_strategy:
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
    //,
    *,
    mask_strategy: MaskStrategy,
    skip_scale: Bool = False,
](
    srow: LocalTensor[DType.float32, row_major[BN]()],
    mask: MaskType,
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

    @parameter
    if (
        MaskStrategy.LOWER_TRIANGULAR in mask_strategy
        or MaskStrategy.UPPER_TRIANGULAR in mask_strategy
    ):
        comptime num_batches = BN // 32
        comptime assert (BN % 32) == 0

        # when score_row == kv_tile_start_row, 1 is valid
        var n_valid: Int32 = max(1 + score_row - kv_tile_start_row, 0)

        @parameter
        for batch in range(num_batches):
            var mask_bits: UInt32 = 0xFFFF_FFFF

            @parameter
            if MaskStrategy.LOWER_TRIANGULAR in mask_strategy:
                # Causal Mask
                # score_row >= kv_tile_start_row
                # 1 + score_row - kv_tile_start_row > 0
                # n_valid > 0
                mask_bits = (UInt32(1) << UInt32(n_valid)) - UInt32(
                    1
                ) if n_valid < 32 else mask_bits

            @parameter
            if MaskStrategy.UPPER_TRIANGULAR in mask_strategy:
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

            @parameter
            for n in range(32 // simd_size):
                comptime frag_col_simd = n + 32 * batch // simd_size
                comptime frag_col = frag_col_simd * simd_size
                var s: F32x2

                @parameter
                if skip_scale:
                    s = rebind[F32x2](vs[frag_col_simd])
                else:
                    s = mul_ftz(rebind[F32x2](vs[frag_col_simd]), scale_log2e)

                @parameter
                for i in range(simd_size):
                    comptime midx = n * simd_size + i
                    comptime flag: UInt32 = UInt32(1 << midx)
                    var in_bound: Bool = (mask_bits & flag) != UInt32(0)
                    var val: Float32 = s[i]
                    s[i] = val if in_bound else MASK_VALUE

                var score_col: Int32 = kv_tile_start_row + Int32(frag_col)
                vs[frag_col_simd] = rebind[vs.ElementType](
                    apply_oob_mask[
                        mask_strategy=mask_strategy,
                        apply_log2e_after_mask = MaskType.apply_log2e_after_mask,
                    ](
                        s,
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

        @parameter
        for n in range(block_size):
            # score_col = mask_frag_col + j * 8
            var s: F32x2

            @parameter
            if skip_scale:
                s = rebind[F32x2](vs[n])
            else:
                s = mul_ftz(rebind[F32x2](vs[n]), scale_log2e)
            comptime frag_col = simd_size * n
            var score_col: Int32 = kv_tile_start_row + Int32(frag_col)

            @parameter
            if MaskStrategy.COMPUTED in mask_strategy:
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
                    mask_strategy=mask_strategy,
                    apply_log2e_after_mask = MaskType.apply_log2e_after_mask,
                ](
                    s,
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
        [S0_cons] [S1_cons] [C0] [C1] [Order*] | [S0_prod] [S1_prod] [Q1Sync] [K] [V*] [O_prod]
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
    # ---- Count=1 section ----
    # S producer barriers: 1 per warp group
    comptime S0_producer_offset = Self.order_offset + Self.num_order_barriers
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

    @staticmethod
    fn _init_count(lane_idx: Int32) -> Int32:
        """Return the mbarrier thread count for the given barrier index.

        S0_consumer[0] and S1_consumer[0] get count=256 (combined softmax +
        correction), other count=128 barriers keep 128, and count=1 barriers
        keep 1.
        """
        if lane_idx == Int32(Self.S0_consumer_offset) or lane_idx == Int32(
            Self.S1_consumer_offset
        ):
            return 256
        if lane_idx < Int32(Self.number_warpgroup_count):
            return 128
        return 1

    @always_inline
    fn init(self, *, lane_idx: Int32):
        @parameter
        if Self.size < WARP_SIZE:
            if lane_idx < Int32(Self.size):
                self.mbar_base[lane_idx].init(Self._init_count(lane_idx))
        elif Self.size == WARP_SIZE:
            self.mbar_base[lane_idx].init(Self._init_count(lane_idx))
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
                self.mbar_base[lane_idx].init(Self._init_count(lane_idx))
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

    @always_inline("nodebug")
    fn combined_p_o_consumer(self, wg_idx: UInt32) -> MBarType:
        """Combined P+O consumer barrier for given warp group.

        Arrived at by BOTH softmax (P ready) and correction (O rescaled).
        Returns S_consumer[0] for wg_idx=0 or wg_idx=1.
        """
        return self.mbar_base + UInt32(Self.num_pv_stages) * wg_idx

    # O pipeline convenience methods
    @always_inline("nodebug")
    fn consumer_o(self) -> RolePipeline[2, False, 1, Self.num_pv_stages]:
        """Get O consumer pipeline.

        Wait side: O_producer barriers (stride 1, indexed by stage).
        Release side: combined S+O barriers (S_consumer[0] per wg,
        stride num_pv_stages).
        """
        return {
            self.mbar_base + Self.O_producer_offset,
            self.mbar_base,
        }

    @always_inline("nodebug")
    fn producer_o0(self) -> ProducerPipeline[1]:
        """Get O producer for warp group 0."""
        return {
            self.mbar_base + Self.O_producer_offset,
            self.combined_p_o_consumer(0),
        }

    @always_inline("nodebug")
    fn producer_o1(self) -> ProducerPipeline[1]:
        """Get O producer for warp group 1."""
        return {
            self.mbar_base + Self.O_producer_offset + 1,
            self.combined_p_o_consumer(1),
        }

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return UInt32(Self.size)

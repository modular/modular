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
- MMA building blocks (bulk_mma, SM100TensorAccumulator)
- Masking utilities (apply_mask, apply_oob_mask)
"""

from std.math import ceildiv, exp2, align_up, iota
from std.math.constants import log2e
from std.sys import size_of
from std.sys._assembly import inlined_assembly
from std.sys.intrinsics import llvm_intrinsic
from std.bit import prev_power_of_two, pop_count
from std.gpu.globals import WARP_SIZE
from std.gpu.primitives.warp import broadcast
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.memory import AddressSpace
from std.gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAInsDescriptor,
    UMMAKind,
    MMASmemDescriptorPair,
)
from std.gpu.compute.arch.tcgen05 import tcgen05_ld, tcgen05_st
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tile_layout import (
    Layout as InternalLayout,
    row_major as tt_row_major,
)
from layout.tma_async import PipelineState, SharedMemBarrier
from std.memory import bitcast
from std.utils.type_functions import ConditionalType
from nn.attention.gpu.nvidia.sm100.attention import FA4Config

# `elect` is defined in the shared NVIDIA module so SM90 and SM100 can both use
# it without a cross-architecture import. Re-exported here for the many SM100
# callers (and tests) that import it from `attention_utils`.
from nn.attention.gpu.nvidia.common import elect
from nn.attention.mha_mask import MHAMask, MASK_VALUE, MaskStrategy
from nn.attention.mha_operand import (
    MHAOperand,
    PagedRowIndices,
    kv_sub_tile_rows,
    kv_num_sub_tiles,
    kv_tma_fold_chunks,
)
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from linalg.arch.sm100.mma import smem_descriptor


# IEEE-754 FP32 exponent bias.  Clamping `exp2` inputs at
# `-FP32_EXP_BIAS` keeps the result within FP32's representable range
# (the smallest normal positive float32 is `2^-126`, and going much
# more negative just underflows to zero).
comptime FP32_EXP_BIAS = 127


# TileTensor-based aliases for storage (native types)
comptime LocalTensor[
    dtype: DType,
    layout: InternalLayout,
] = TileTensor[
    dtype,
    InternalLayout[
        shape_types=layout.shape_types,
        stride_types=layout.stride_types,
    ],
    MutUntrackedOrigin,
    address_space=AddressSpace.LOCAL,
]
comptime SharedMemTensor[dtype: DType, layout: InternalLayout] = TileTensor[
    dtype,
    InternalLayout[
        shape_types=layout.shape_types,
        stride_types=layout.stride_types,
    ],
    MutUntrackedOrigin,
    address_space=AddressSpace.SHARED,
]

# Legacy LayoutTensor aliases for TMA/MMA API boundaries
comptime LocalLT[
    dtype: DType, layout: Layout, element_layout: Layout = Layout(1, 1)
] = LayoutTensor[
    dtype,
    layout,
    MutAnyOrigin,
    address_space=AddressSpace.LOCAL,
    element_layout=element_layout,
]
comptime SharedMemPointer[type: AnyType] = UnsafePointer[
    type, MutAnyOrigin, address_space=AddressSpace.SHARED
]
comptime MBarType = SharedMemPointer[SharedMemBarrier]


# PagedRowIndices, kv_sub_tile_rows, kv_num_sub_tiles are now defined in
# nn.attention.mha_operand and re-exported via the import above.


def extract_power_of_two(N: Int, i: Int) -> Int:
    pt = prev_power_of_two(N)
    rem = N
    for _ in range(i):
        rem -= pt
        pt = prev_power_of_two(rem)
    return pt


def cumulative_power_of_two(N: Int, i: Int) -> Int:
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
def break_into_powers_of_two[
    origins: OriginSet,
    //,
    func: def[pow_two: Int, offset: Int]() capturing[origins] -> None,
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
    accum_dtype_size: Int,
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
    comptime bits = Self.bits_per_byte * Self.frag_simdwidth * Self.thread_cols * Self.accum_dtype_size

    @always_inline
    def __init__(out self):
        pass


struct STMatrixOffsets[
    BM: Int,
    BN: Int,
    *,
    num_threads: Int,
    accum_dtype_size: Int,
    curr_repeat: Int,
    cumulative_repeat: Int,
    m_mma: Int,
](TrivialRegisterPassable):
    comptime STLayout = STMatrixLayout[
        Self.BM,
        Self.BN,
        num_threads=Self.num_threads,
        accum_dtype_size=Self.accum_dtype_size,
    ]

    comptime tmem_col_offset = Self.cumulative_repeat * Self.STLayout.frag_simdwidth * Self.STLayout.thread_cols
    comptime tmem_row_offset = 16 * Self.m_mma
    comptime tmem_offset = (Self.tmem_row_offset << 16) + Self.tmem_col_offset
    comptime b32_per_repeat = Self.STLayout.elements_per_repeat * Self.accum_dtype_size // 4
    comptime local_frag_size_b32 = Self.curr_repeat * Self.b32_per_repeat
    comptime ptr_offset = Self.b32_per_repeat * (
        Self.STLayout.repeat * Self.m_mma + Self.cumulative_repeat
    )

    @always_inline
    def __init__(out self):
        pass


@always_inline
def o_store_tma_blocks_per_op[
    output_type: DType,
    output_swizzle_mode: TensorMapSwizzle,
    ov_depth: Int,
    group: Int,
    depth_splits: Int,
]() -> Int:
    """Box size (swizzle-granularity blocks per batched O-store TMA).

    The O store splits the contiguous `ov_depth` into swizzle-granularity blocks
    `K = output_swizzle_mode.bytes() // size_of[output_type]` and a single batched
    TMA copies `ceil(n_blocks / depth_splits)` of them (vs `n_blocks` per-block
    copies). `depth_splits` is the number of contiguous depth ranges the issuers
    divide the store into:
      - MHA (`depth_splits == 2`): the descriptor is shared between the 1Q combine
        (2 warpgroups, 1 TMA each over its half) and the single-issuer scale_write
        (2 pipelined TMAs over its two halves), so the box is the half-depth
        `ceil(n_blocks / 2)`.
      - depth512 (`depth_splits == 1`): single issuer, no combine, so the box is
        the full depth `n_blocks` (one TMA).
    The box size is independent of `group`: `RaggedTMA3DTile` folds the
    `(middle_dim, rows)` selectors into one dim, so even fused GQA (`group > 1`)
    fits the *blocks* dimension within the 5D TMA limit (rank-5 batched store) and
    uses the same `ceil(n_blocks / depth_splits)` box.
    Returns 0 (per-block path) only when `output_swizzle_mode != SWIZZLE_NONE`:
    the blocked-smem / identity-layout invariant the batched box relies on holds
    only for SWIZZLE_NONE (e.g. an FP8-QKV MLA variant with a SWIZZLE_128B bf16
    output store stays per-block).
    `group` is retained in the signature for call-site compatibility but no longer
    gates the result. This is the single source of truth for `tma_blocks_per_op`
    across the O-store descriptor type/creation sites.
    """
    comptime if output_swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        return 0
    comptime K = output_swizzle_mode.bytes() // size_of[output_type]()
    comptime n_blocks = align_up(ov_depth, K) // K
    return ceildiv(n_blocks, depth_splits)


@always_inline
def scale_pack_o_row[
    n: Int, //, output_type: DType, w: Int, start: Int = 0
](o_vals: InlineArray[Scalar[DType.float32], n], inv_row_sum: Float32) -> SIMD[
    DType.uint32, w // 2
]:
    """Scale the `w` f32 O lanes `o_vals[start : start + w]` by `inv_row_sum`,
    cast to the 2-byte `output_type`, and pack into `w // 2` u32 lanes (the
    row-major 16 B SWIZZLE_NONE store register).

    `o_vals` is a `tcgen05_ld` result; `start`/`w` window it so one wide TMEM
    load can feed several stores (depth512 loads 16 lanes, stores two 8-lane
    blocks). Compute stays in f32x2 (64-bit) chunks because LLVM scalarizes
    wider SIMD here; only the packed u32 store register is built wide. Shared by
    the SM100 O-store writeback helpers (`fa4_scale_write_output`,
    `depth512_scale_write_output`).
    """
    comptime assert size_of[output_type]() == 2
    var packed = SIMD[DType.uint32, w // 2]()
    comptime for c in range(w // 2):
        var pair = (
            SIMD[DType.float32, 2](
                o_vals[start + 2 * c], o_vals[start + 2 * c + 1]
            )
            * inv_row_sum
        ).cast[output_type]()
        packed[c] = bitcast[DType.uint32, 1](pair)
    return packed


@always_inline
def combine_pack_o_row[
    n: Int, //, output_type: DType
](
    own: InlineArray[Scalar[DType.float32], n],
    peer: InlineArray[Scalar[DType.float32], n],
    scale_own: Float32,
    scale_peer: Float32,
) -> SIMD[DType.uint32, n // 2]:
    """LSE-combine `own * scale_own + peer * scale_peer` over `n` f32 O lanes,
    cast to the 2-byte `output_type`, and pack into `n // 2` u32 lanes.

    `own`/`peer` are `tcgen05_ld` results. f32x2 compute / wide-only store, as
    in `scale_pack_o_row`; the fused `peer.fma(scale_peer, own * scale_own)`
    form matches the per-element combine it replaces. Used by
    `fa4_lse_combine_write`.
    """
    comptime assert size_of[output_type]() == 2
    var packed = SIMD[DType.uint32, n // 2]()
    comptime for c in range(n // 2):
        var own_c = SIMD[DType.float32, 2](own[2 * c], own[2 * c + 1])
        var peer_c = SIMD[DType.float32, 2](peer[2 * c], peer[2 * c + 1])
        var comb = peer_c.fma(
            SIMD[DType.float32, 2](scale_peer), own_c * scale_own
        ).cast[output_type]()
        packed[c] = bitcast[DType.uint32, 1](comb)
    return packed


@always_inline
def _tmem_offset(dtype_size: Int, *, MMA_N: Int, m_mma: Int, n_mma: Int) -> Int:
    row = 16 * m_mma
    col = (MMA_N * n_mma * dtype_size) // 4
    return (row << 16) + col


@always_inline
def _tmem_offset[dtype: DType, *, MMA_N: Int, m_mma: Int, n_mma: Int]() -> Int:
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
    def __init__(out self, tmem_addr: UInt32):
        self.tmem_addr = tmem_addr

    @always_inline
    def __getitem__(self, i: UInt32) -> Self:
        return {self.tmem_addr + i * UInt32(Self.BN)}

    @always_inline
    def offset[m_mma: Int, n_mma: Int](self) -> UInt32:
        comptime if m_mma == 0 and n_mma == 0:
            return self.tmem_addr
        else:
            comptime linear = _tmem_offset[
                Self.dtype, MMA_N=Self.BN, m_mma=m_mma, n_mma=n_mma
            ]()

            return self.tmem_addr + UInt32(linear)

    @staticmethod
    @always_inline
    def allocate_register_tile[
        *, num_threads: Int
    ](
        out res: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        res = type_of(res).stack_allocation()

    @always_inline
    def store_async[
        *, num_threads: Int
    ](
        self,
        src: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        comptime assert Self.dtype_size <= 4
        ptr = src.ptr.bitcast[UInt32]()
        comptime st_mat_layout = STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
        ]
        comptime assert st_mat_layout.bits == 128 or st_mat_layout.bits == 256

        @parameter
        @always_inline
        def store_fn[pow_two: Int, offset: Int]():
            # pow_two is current repeat, offset total so far
            comptime if pow_two > 0:
                comptime for m_mma in range(st_mat_layout.num_m_tiles):
                    comptime offsets = STMatrixOffsets[
                        Self.BM,
                        Self.BN,
                        num_threads=num_threads,
                        accum_dtype_size=Self.dtype_size,
                        curr_repeat=pow_two,
                        cumulative_repeat=offset,
                        m_mma=m_mma,
                    ]()
                    tmem = self.tmem_addr + UInt32(offsets.tmem_offset)
                    var frag = InlineArray[
                        Scalar[DType.uint32], offsets.local_frag_size_b32
                    ](uninitialized=True)

                    comptime for _i in range(offsets.local_frag_size_b32):
                        frag[_i] = ptr.load(offsets.ptr_offset + _i)
                    # 16 x 256b results in repeated 8x4 matrix of <1,2> vector pattern
                    tcgen05_st[
                        datapaths=16,  # first dimension of the shape
                        bits=st_mat_layout.bits,  # second dimension of the shape
                        repeat=pow_two,
                        pack=False,
                    ](tmem, frag)

        comptime max_value = 64 if st_mat_layout.bits == 128 else 32
        break_into_powers_of_two[
            func=store_fn, N=st_mat_layout.repeat, max_value=max_value
        ]()

    @always_inline
    def load_async_with_st_matrix_layout[
        *, num_threads: Int
    ](
        self,
        out dst: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        comptime assert (
            Self.dtype_size <= 4
        ), "Loading for st matrix requires elements to be <= 4 bytes."
        comptime st_mat_layout = STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
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
            num_repeats=st_mat_layout.repeat,
        ](dst)

    @always_inline
    def load_st_matrix_chunk[
        *, num_threads: Int, start_repeat: Int, num_repeats: Int
    ](
        self,
        dst: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
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
            accum_dtype_size=Self.dtype_size,
        ]()
        comptime load_dtype = DType.uint32
        var ptr = rebind[
            UnsafePointer[
                Scalar[load_dtype],
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ]
        ](dst.ptr)

        @parameter
        @always_inline
        def load_fn[pow_two: Int, local_offset: Int]():
            comptime assert pow_two + local_offset <= num_repeats
            comptime if pow_two > 0:
                comptime for m_mma in range(st_mat_layout.num_m_tiles):
                    comptime offsets = STMatrixOffsets[
                        Self.BM,
                        Self.BN,
                        num_threads=num_threads,
                        accum_dtype_size=Self.dtype_size,
                        curr_repeat=pow_two,
                        cumulative_repeat=start_repeat + local_offset,
                        m_mma=m_mma,
                    ]()
                    tmem = self.tmem_addr + UInt32(offsets.tmem_offset)
                    frag = tcgen05_ld[
                        datapaths=16,
                        bits=st_mat_layout.bits,
                        repeat=pow_two,
                        dtype=load_dtype,
                        pack=False,
                        width=offsets.local_frag_size_b32,
                    ](tmem)

                    comptime for _i in range(offsets.local_frag_size_b32):
                        ptr.store(offsets.ptr_offset + _i, frag[_i])

        comptime max_value = 64 if st_mat_layout.bits == 128 else 32
        break_into_powers_of_two[
            func=load_fn, N=num_repeats, max_value=max_value
        ]()

    @always_inline
    def load_async(
        self,
        out dst: InlineArray[Scalar[Self.dtype], Self.BN],
    ):
        dst = InlineArray[Scalar[Self.dtype], Self.BN](uninitialized=True)
        # The uint32 bitcast path below assumes dtype_size == 4.
        # Sub-32-bit types (bf16, f16) pack multiple elements per uint32
        # and would need unpacking logic not yet implemented.
        comptime assert (
            Self.dtype_size == 4
        ), "load_async only supports 32-bit dtypes"
        comptime repeat = Self.dtype_size * Self.BN // 4
        comptime dtype = Self.dtype if Self.dtype_size == 4 else DType.uint32

        @parameter
        @always_inline
        def load_fn[pow_two: Int, offset: Int]():
            comptime if pow_two > 0:
                comptime if dtype == Self.dtype:
                    frag0 = tcgen05_ld[
                        datapaths=32,  # first dimension of the shape
                        bits=32,  # second dimension of the shape
                        repeat=pow_two,
                        dtype=Self.dtype,
                        pack=False,
                        width=pow_two,
                    ](self.tmem_addr + UInt32(offset))

                    comptime for _i in range(pow_two):
                        dst[offset + _i] = frag0[_i]
                else:
                    frag1 = tcgen05_ld[
                        datapaths=32,  # first dimension of the shape
                        bits=32,  # second dimension of the shape
                        repeat=pow_two,
                        dtype=DType.uint32,
                        pack=False,
                        width=pow_two,
                    ](self.tmem_addr + UInt32(offset))

                    comptime for _i in range(pow_two):
                        dst[offset + _i] = bitcast[Self.dtype](frag1[_i])

        break_into_powers_of_two[func=load_fn, N=repeat, max_value=128]()

    @always_inline
    def store_async[
        src_type: DType
    ](self, src: LocalTensor[src_type, row_major[Self.BN]()]):
        @parameter
        @always_inline
        def store_fn[pow_two: Int, offset: Int]():
            comptime if pow_two > 0:
                comptime frag_width = pow_two * Self.dtype_size // 4
                var frag = InlineArray[Scalar[DType.uint32], frag_width](
                    uninitialized=True
                )

                comptime if src_type == Self.dtype:
                    comptime for _i in range(frag_width):
                        frag[_i] = src.ptr.bitcast[UInt32]().load(offset + _i)
                elif pow_two > 1:
                    comptime size_ratio = size_of[src_type]() // Self.dtype_size
                    comptime cast_width = min(
                        4 if size_ratio
                        >= 4 else (2 if size_ratio >= 2 else pow_two),
                        pow_two,
                    )
                    comptime u32_per_cast = cast_width * Self.dtype_size // 4
                    comptime num_casts = pow_two // cast_width

                    comptime if u32_per_cast >= 1:
                        comptime for _i in range(num_casts):
                            var src_vec = src.raw_load[width=cast_width](
                                offset + _i * cast_width
                            )
                            var dst_vec = src_vec.cast[Self.dtype]()
                            var packed_chunk = bitcast[
                                DType.uint32, u32_per_cast
                            ](dst_vec)
                            comptime for _j in range(u32_per_cast):
                                frag[_i * u32_per_cast + _j] = packed_chunk[_j]
                    else:
                        var packed = bitcast[DType.uint32, frag_width](
                            src.raw_load[width=pow_two](offset).cast[
                                Self.dtype
                            ]()
                        )
                        comptime for _i in range(frag_width):
                            frag[_i] = packed[_i]
                else:
                    frag[0] = bitcast[DType.uint32](src[0].cast[Self.dtype]())

                tcgen05_st[
                    datapaths=32,  # first dimension of the shape
                    bits=32,  # second dimension of the shape
                    repeat=pow_two * Self.dtype_size // 4,
                    pack=False,
                ](self.tmem_addr + UInt32(offset * Self.dtype_size // 4), frag)

        break_into_powers_of_two[func=store_fn, N=Self.BN, max_value=128]()

    @always_inline
    def store_async[
        src_type: DType,
        src_len: Int,
        src_offset: Int = 0,
    ](self, src: InlineArray[Scalar[src_type], src_len]):
        @parameter
        @always_inline
        def store_fn[pow_two: Int, offset: Int]():
            comptime if pow_two > 0:
                comptime frag_width = pow_two * Self.dtype_size // 4
                var frag = InlineArray[Scalar[DType.uint32], frag_width](
                    uninitialized=True
                )

                comptime if src_type == Self.dtype:
                    comptime for _i in range(frag_width):
                        frag[_i] = bitcast[DType.uint32](
                            src[src_offset + offset + _i]
                        )
                else:
                    comptime sub_elements = 4 // Self.dtype_size
                    comptime size_ratio = size_of[src_type]() // Self.dtype_size
                    comptime cast_width = 4 if size_ratio >= 4 else (
                        2 if size_ratio >= 2 else 1
                    )

                    comptime for _i in range(frag_width):
                        var x: SIMD[Self.dtype, sub_elements] = {}
                        comptime if cast_width >= 2:
                            comptime for _j in range(
                                0, sub_elements, cast_width
                            ):
                                var src_vec: SIMD[src_type, cast_width] = {}
                                comptime for _k in range(cast_width):
                                    comptime idx = (
                                        src_offset
                                        + offset
                                        + _i * sub_elements
                                        + _j
                                        + _k
                                    )
                                    src_vec[_k] = src[idx]
                                var dst_vec = src_vec.cast[Self.dtype]()
                                comptime for _k in range(cast_width):
                                    x[_j + _k] = dst_vec[_k]
                        else:
                            comptime for _j in range(sub_elements):
                                comptime idx = (
                                    src_offset + offset + _i * sub_elements + _j
                                )
                                x[_j] = src[idx].cast[Self.dtype]()
                        frag[_i] = bitcast[DType.uint32, 1](x)
                tcgen05_st[
                    datapaths=32,
                    bits=32,
                    repeat=pow_two * Self.dtype_size // 4,
                    pack=False,
                ](self.tmem_addr + UInt32(offset * Self.dtype_size // 4), frag)

        break_into_powers_of_two[func=store_fn, N=Self.BN, max_value=128]()


struct SM100TensorAccumulator[
    operand_type: DType,
    accum_dtype: DType,
    MMA_M: Int,
    MMA_N: Int,
    BK: Int,
    *,
    a_tmem: Bool,
    mma_kind: UMMAKind = UMMAKind.KIND_F16,
    swizzle_a: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    cta_group: Int = 1,
    num_stages: Int = 1,
    b_page_dense: Bool = False,
](TrivialRegisterPassable):
    # This performs C = A @ B
    # where A is BM x BK and B is BN x BK if k major, else BK x BN.
    # `BK` is broken into `num_stages` and pipelined.
    #
    # The A operand is either an SMEM tile referenced through an
    # `MMASmemDescriptorPair` (`a_tmem=False`, the "SS" contraction, e.g.
    # Q@K' producing the unweighted score input of the `softmax`) or a
    # TMEM tile referenced through a raw TMEM address (`a_tmem=True`, the
    # "TS" contraction, e.g. P@V). B is always an SMEM descriptor.
    # `swizzle_a` is meaningful only for `a_tmem=False`.
    # The benefit of setting `stages > 1` is that this can hide latency.
    #
    # When `cta_group == 1 and MMA_M <= 64`, MMAs are issued as
    # `tcgen05.mma.ws` (warp-specialized), which uses the packed-TMEM
    # (1x4 / Layout G) datapath: the hardware subpartition folds
    # `m_pack = 128 // MMA_M` row-groups onto the same physical TMEM
    # columns, so the accumulator occupies only `MMA_N / m_pack` physical
    # columns and all 128 datapath lanes stay busy. For MMA_M == 64 the
    # non-ws form is also legal in hardware but uses only half the
    # datapaths, so this struct ALWAYS chooses ws there. The A (P) operand
    # in TMEM (`a_tmem=True`) follows the same packed convention -- its
    # producer must write it accordingly -- and consumers must read the
    # accumulator with the packed layout: all `m_pack` warps issue
    # tcgen05_ld/st against the SAME TMEM column address (no per-warp
    # column offsets) -- see sm100/CLAUDE.md.
    comptime use_ws = Self.cta_group == 1 and Self.MMA_M <= 64
    comptime tcgen05_mma_type = "tcgen05.mma.ws.cta_group::1."
    comptime operand_t = Self.operand_type
    comptime operand_size = size_of[Self.operand_t]()
    comptime accum_t = Self.accum_dtype
    comptime MMA_K = 16 if Self.operand_type.is_half_float() else 32
    # The TS quadrant requires BK % MMA_K == 0 (P columns are produced in
    # whole MMA_K blocks); the SS quadrant counts a ragged tail block.
    comptime num_k_mmas = (Self.BK // Self.MMA_K) if Self.a_tmem else ceildiv(
        Self.BK, Self.MMA_K
    )
    comptime swizzle_granularity = (
        Self.swizzle_b.bytes() if Self.a_tmem else max(
            Self.swizzle_a.bytes(), Self.swizzle_b.bytes()
        )
    ) // size_of[Self.operand_t]()
    # TMEM A (P) is written at exactly BK columns, so no swizzle padding
    # applies; SMEM A/B tiles are padded to the swizzle granularity.
    comptime padded_BK = Self.BK if Self.a_tmem else align_up(
        Self.BK, Self.swizzle_granularity
    )
    comptime num_k_blocks = Self.padded_BK // Self.MMA_K
    comptime use_3_then_1_split: Bool = Self.a_tmem and Self.num_stages == 2 and Self.num_k_blocks % 4 == 0
    comptime num_k_blocks_per_stage = Self.num_k_blocks // (
        4 if Self.use_3_then_1_split else Self.num_stages
    )

    # With cta_group > 1, each CTA's SMEM holds MMA_M/cta_group rows (A)
    # and MMA_N/cta_group columns (B).  The K-offset arithmetic in
    # `_build_mma` (SS path) uses these layouts, so BMN must match per-CTA
    # dimensions to keep addresses within each CTA's SMEM tile.
    #
    # For k_major A the outer-K stride is BMN * swizzle_width; halving BMN
    # halves that stride so K offsets stay in the per-CTA buffer.
    # For k_major B (transpose_b) the K stride doesn't depend on BMN,
    # but using per-CTA BMN is harmless and keeps the rule uniform.
    #
    # The TS quadrant historically builds b_layout with the full MMA_N
    # (no cta_group division); preserved as-is.
    comptime a_bmn: Int = align_up(Self.MMA_M // Self.cta_group, 8)
    comptime a_layout = tile_layout_k_major[
        Self.operand_t, Self.a_bmn, Self.padded_BK, Self.swizzle_a
    ]()
    comptime b_bmn: Int = Self.MMA_N if Self.a_tmem else (
        Self.MMA_N // Self.cta_group
    )
    comptime b_layout = tile_layout_k_major[
        Self.operand_t,
        Self.b_bmn,
        Self.padded_BK,
        Self.swizzle_b,
        page_dense=Self.b_page_dense,
    ]() if Self.transpose_b else tile_layout_mn_major[
        Self.operand_t,
        Self.b_bmn,
        Self.padded_BK,
        Self.swizzle_b,
        page_dense=Self.b_page_dense,
    ]()

    comptime idesc = UMMAInsDescriptor[Self.mma_kind].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype=DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=Self.transpose_b,
    ]()

    comptime AType = ConditionalType[
        Trait=TrivialRegisterPassable,
        If=Self.a_tmem,
        Then=TMemTile[Self.operand_type, Self.MMA_M, Self.BK],
        Else=MMASmemDescriptorPair,
    ]
    # The runtime argument type of `a` in `mma`/`mma_maybe_partial_k`:
    # a raw TMEM address for the TS quadrant, an SMEM descriptor pair
    # for the SS quadrant.
    comptime AInput = ConditionalType[
        Trait=TrivialRegisterPassable,
        If=Self.a_tmem,
        Then=UInt32,
        Else=MMASmemDescriptorPair,
    ]
    comptime BType = MMASmemDescriptorPair
    comptime CType = TMemTile[Self.accum_t, Self.MMA_M, Self.MMA_N]

    @staticmethod
    @always_inline("nodebug")
    def mma[
        *, stage_idx: Int = 0
    ](
        a: Self.AInput,
        b: Self.BType,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert (not Self.use_ws) or Self.MMA_M in (
            32,
            64,
        ), "ws path requires MMA_M in (32, 64)"

        comptime if Self.num_stages == 1:
            # Original single-stage behavior
            comptime if Self.a_tmem:
                var a_ = rebind[UInt32](a)
                comptime if Self.use_ws:
                    bulk_mma_ws_ts[
                        Self.mma_kind,
                        Self.operand_t,
                        b_BMN=Self.MMA_N,
                        b_BK=Self.padded_BK,
                        b_swizzle=Self.swizzle_b,
                        b_is_k_major=Self.transpose_b,
                        num_k_mmas=Self.num_k_mmas,
                        operand_size=Self.operand_size,
                        tcgen05_mma_type=Self.tcgen05_mma_type,
                        mma_k=Self.MMA_K,
                        b_page_dense=Self.b_page_dense,
                    ](Self.idesc, a_, b, c, c_scale, elect)
                else:
                    bulk_mma[
                        Self.b_layout,
                        mma_k=Self.MMA_K,
                        num_k_mmas=Self.num_k_mmas,
                        operand_size=Self.operand_size,
                        cta_group=Self.cta_group,
                    ](Self.idesc, a_, b, c, c_scale, elect)
            else:
                var a_ = rebind[MMASmemDescriptorPair](a)
                comptime if Self.use_ws:
                    bulk_mma_ws[
                        Self.mma_kind,
                        Self.operand_t,
                        Self.operand_t,
                        a_BMN=Self.a_bmn,
                        a_BK=Self.padded_BK,
                        a_swizzle=Self.swizzle_a,
                        a_is_k_major=True,
                        b_BMN=Self.b_bmn,
                        b_BK=Self.padded_BK,
                        b_swizzle=Self.swizzle_b,
                        b_is_k_major=Self.transpose_b,
                        num_k_mmas=Self.num_k_mmas,
                        operand_size=Self.operand_size,
                        tcgen05_mma_type=Self.tcgen05_mma_type,
                        mma_k=Self.MMA_K,
                        b_page_dense=Self.b_page_dense,
                    ](Self.idesc, a_, b, c, c_scale, elect)
                else:
                    bulk_mma[
                        Self.a_layout,
                        Self.b_layout,
                        num_k_mmas=Self.num_k_mmas,
                        mma_k=Self.MMA_K,
                        operand_size=Self.operand_size,
                        cta_group=Self.cta_group,
                    ](Self.idesc, a_, b, c, c_scale, elect)
        else:
            comptime start = 3 * stage_idx if Self.use_3_then_1_split else stage_idx
            comptime end = stage_idx + 3 if Self.use_3_then_1_split else stage_idx + 1
            comptime k_batch_start = Self.num_k_blocks_per_stage * start
            comptime k_batch_end = min(
                Self.num_k_blocks_per_stage * end, Self.num_k_mmas
            )
            comptime k_offset = k_batch_start * Self.MMA_K
            # Offset both A and B operands by k_offset.
            # B (smem) offset: move by k_offset rows of the descriptor.
            comptime b_byte_offset = (
                Self.b_layout(IntTuple(0, k_offset)) * Self.operand_size
            )
            var scale: UInt32

            comptime if stage_idx == 0:
                scale = c_scale
            else:
                scale = 1
            comptime if Self.a_tmem:
                # A (tmem) offset: P is MMA_M x BK, so the column offset is
                # k_offset * dtype_size / 4 (in tmem units).
                comptime a_tmem_offset = (k_offset * Self.operand_size) // 4
                var a_ = rebind[UInt32](a) + UInt32(a_tmem_offset)
                comptime if Self.use_ws:
                    bulk_mma_ws_ts[
                        Self.mma_kind,
                        Self.operand_t,
                        b_BMN=Self.MMA_N,
                        b_BK=Self.padded_BK,
                        b_swizzle=Self.swizzle_b,
                        b_is_k_major=Self.transpose_b,
                        num_k_mmas=k_batch_end - k_batch_start,
                        operand_size=Self.operand_size,
                        tcgen05_mma_type=Self.tcgen05_mma_type,
                        mma_k=Self.MMA_K,
                        b_page_dense=Self.b_page_dense,
                    ](
                        Self.idesc,
                        a_,
                        b + UInt32(b_byte_offset),
                        c,
                        scale,
                        elect,
                    )
                else:
                    bulk_mma[
                        Self.b_layout,
                        mma_k=Self.MMA_K,
                        num_k_mmas=k_batch_end - k_batch_start,
                        operand_size=Self.operand_size,
                        cta_group=Self.cta_group,
                    ](
                        Self.idesc,
                        a_,
                        b + UInt32(b_byte_offset),
                        c,
                        scale,
                        elect,
                    )
            else:
                # A (smem) offset: move by k_offset rows of the descriptor.
                comptime a_byte_offset = (
                    Self.a_layout(IntTuple(0, k_offset)) * Self.operand_size
                )
                var a_ = rebind[MMASmemDescriptorPair](a) + UInt32(
                    a_byte_offset
                )
                comptime if Self.use_ws:
                    bulk_mma_ws[
                        Self.mma_kind,
                        Self.operand_t,
                        Self.operand_t,
                        a_BMN=Self.a_bmn,
                        a_BK=Self.padded_BK,
                        a_swizzle=Self.swizzle_a,
                        a_is_k_major=True,
                        b_BMN=Self.b_bmn,
                        b_BK=Self.padded_BK,
                        b_swizzle=Self.swizzle_b,
                        b_is_k_major=Self.transpose_b,
                        num_k_mmas=k_batch_end - k_batch_start,
                        operand_size=Self.operand_size,
                        tcgen05_mma_type=Self.tcgen05_mma_type,
                        mma_k=Self.MMA_K,
                        b_page_dense=Self.b_page_dense,
                    ](
                        Self.idesc,
                        a_,
                        b + UInt32(b_byte_offset),
                        c,
                        scale,
                        elect,
                    )
                else:
                    bulk_mma[
                        Self.a_layout,
                        Self.b_layout,
                        num_k_mmas=k_batch_end - k_batch_start,
                        mma_k=Self.MMA_K,
                        operand_size=Self.operand_size,
                        cta_group=Self.cta_group,
                    ](
                        Self.idesc,
                        a_,
                        b + UInt32(b_byte_offset),
                        c,
                        scale,
                        elect,
                    )

    @staticmethod
    @always_inline("nodebug")
    def mma_maybe_partial_k[
        *, stage_idx: Int = 0
    ](
        a: Self.AInput,
        b: Self.BType,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
        valid_k_mmas: UInt32,
    ):
        # Contraction for the last KV tile, where only `valid_k_mmas`
        # MMA_K-blocks hold real data (for TS, the loaded V pages; requires
        # page_size % MMA_K == 0 so the loaded boundary is MMA_K-aligned --
        # enforced by FA4Config.supported()). Skipping the unloaded tail
        # blocks is bit-identical to the full contraction (for TS their P is
        # exactly 0 after masking) AND avoids reading uninitialized SMEM
        # (the `0 * NaN = NaN` bug). For a full last tile, calling with
        # `valid_k_mmas = num_k_mmas` degenerates to the plain contraction
        # (every `@!%pv` validity guard is never-true).

        # comptime k-block range owned by this stage -- mirror `mma`'s stage
        # split as top-scope ternaries (NOT a `comptime if` block, whose
        # branch scope would hide ks_start/ks_end).
        comptime _multi = Self.num_stages != 1
        comptime _start = 3 * stage_idx if Self.use_3_then_1_split else stage_idx
        comptime _end = (
            stage_idx + 3
        ) if Self.use_3_then_1_split else stage_idx + 1
        comptime ks_start = (
            Self.num_k_blocks_per_stage * _start
        ) if _multi else 0
        comptime ks_end = min(
            Self.num_k_blocks_per_stage * _end, Self.num_k_mmas
        ) if _multi else Self.num_k_mmas

        # Issue this stage's k-blocks as one fused inline-asm sequence. The
        # partial primitives predicate each block `jj` on a SEPARATE,
        # warp-uniform validity guard (`@!%pv`, run iff `jj < valid_k_mmas`)
        # while passing `elect` through UNMODIFIED, so the elect codegen
        # matches the full-tile path (no BSYNC.RECONVERGENT). `a`/`b` are the
        # stage-0 bases (unlike `mma`'s multi-stage path, no offsets are
        # applied here); the builders apply absolute per-block offsets, so no
        # stage-offset swizzle-linearity is assumed. `c_scale` initializes
        # the accumulator on stage 0's first block (jj 0; always valid since
        # valid_k_mmas >= 1); every later block accumulates.
        comptime assert (not Self.use_ws) or Self.MMA_M in (
            32,
            64,
        ), "ws path requires MMA_M in (32, 64)"

        comptime if Self.a_tmem:
            var a_ = rebind[UInt32](a)
            comptime if Self.use_ws:
                bulk_mma_ws_ts_partial[
                    Self.mma_kind,
                    Self.operand_t,
                    b_BMN=Self.MMA_N,
                    b_BK=Self.padded_BK,
                    b_swizzle=Self.swizzle_b,
                    b_is_k_major=Self.transpose_b,
                    num_k_mmas=ks_end - ks_start,
                    operand_size=Self.operand_size,
                    tcgen05_mma_type=Self.tcgen05_mma_type,
                    mma_k=Self.MMA_K,
                    k_start=ks_start,
                    b_page_dense=Self.b_page_dense,
                ](
                    Self.idesc,
                    a_,
                    b,
                    c,
                    c_scale,
                    elect,
                    valid_k_mmas,
                )
            else:
                bulk_mma_partial[
                    Self.b_layout,
                    mma_k=Self.MMA_K,
                    num_k_mmas=ks_end - ks_start,
                    operand_size=Self.operand_size,
                    k_start=ks_start,
                    cta_group=Self.cta_group,
                ](
                    Self.idesc,
                    a_,
                    b,
                    c,
                    c_scale,
                    elect,
                    valid_k_mmas,
                )
        else:
            var a_ = rebind[MMASmemDescriptorPair](a)
            comptime if Self.use_ws:
                bulk_mma_ws_partial[
                    Self.mma_kind,
                    Self.operand_t,
                    Self.operand_t,
                    a_BMN=Self.a_bmn,
                    a_BK=Self.padded_BK,
                    a_swizzle=Self.swizzle_a,
                    a_is_k_major=True,
                    b_BMN=Self.b_bmn,
                    b_BK=Self.padded_BK,
                    b_swizzle=Self.swizzle_b,
                    b_is_k_major=Self.transpose_b,
                    num_k_mmas=ks_end - ks_start,
                    operand_size=Self.operand_size,
                    tcgen05_mma_type=Self.tcgen05_mma_type,
                    mma_k=Self.MMA_K,
                    k_start=ks_start,
                    b_page_dense=Self.b_page_dense,
                ](
                    Self.idesc,
                    a_,
                    b,
                    c,
                    c_scale,
                    elect,
                    valid_k_mmas,
                )
            else:
                bulk_mma_ss_partial[
                    Self.a_layout,
                    Self.b_layout,
                    num_k_mmas=ks_end - ks_start,
                    mma_k=Self.MMA_K,
                    operand_size=Self.operand_size,
                    k_start=ks_start,
                    cta_group=Self.cta_group,
                ](
                    Self.idesc,
                    a_,
                    b,
                    c,
                    c_scale,
                    elect,
                    valid_k_mmas,
                )


def _build_mma[
    *, a_tmem: Bool, ws: Bool, partial: Bool
](
    kind: String,
    layout_a: Layout,
    layout_b: Layout,
    *,
    operand_size: Int,
    mma_k: Int,
    num_k_mmas: Int,
    k_start: Int = 0,
    cta_group: Int = 1,
    tcgen05_mma_type: String = "",
) -> String:
    # Unified PTX builder for the tcgen05 MMA contraction, parameterized over the
    # three axes that previously spawned eight near-duplicate builders:
    #   * `a_tmem`  -- A operand source: TS (TMEM address, `[$7]`/`[%rab]`) vs
    #                  SS (SMEM descriptor pair in `%rda` from `$7`/`$8`).
    #   * `ws`      -- warp-specialized datapath: `tcgen05_mma_type` instruction
    #                  with NO zero-column mask, vs non-ws
    #                  `tcgen05.mma.cta_group::N.` with the `{$1,...}` mask.
    #   * `partial` -- partial-K tail: each block carries a SEPARATE warp-uniform
    #                  validity guard `%pv = (valid_k_mmas <= jj)`.
    #
    # `layout_a` is consulted only for SS (`a_tmem=False`); TS computes the A
    # column stride directly. `cta_group` matters only for non-ws (`ws=False`);
    # `tcgen05_mma_type` only for ws; `k_start` only for partial.
    #
    # PREDICATION (the one rule that protects elect codegen): the form depends
    # ONLY on `partial`, never on `ws`.
    #   * full    -> single-instruction predication `@!%pj <instr>`. Keeping the
    #                MMA a straight-line predicated instruction is what lets the
    #                compiler recognize the single-lane `elect` and avoid emitting
    #                a `BSYNC.RECONVERGENT` into the SASS.
    #   * partial -> `@%pj bra skip{k}` + a SEPARATE `@!%pv` guard on the MMA. Two
    #                guards are needed (elect AND validity) and PTX allows one
    #                predicate per instruction, so elect uses the branch form
    #                while validity rides `%pv`. `%pv` is warp-uniform, so it never
    #                diverges and needs no reconvergence; `%pj` stays a pure
    #                function of the unmodified `elect`, preserving the codegen.
    # Blocks use ABSOLUTE k-index `jj = k_start + k` (for full, `k_start=0`).
    #
    # Plain `if` (not `comptime if`) is used throughout: the whole function is
    # comptime-evaluated, and plain `if` is function-scoped (Python-like) so
    # bindings like `operands` survive past the branch -- a `comptime if` branch
    # scope would hide them.
    # Pre-reserve so `mma` is heap-backed from the start: the comptime
    # interpreter cannot memcpy into a String's inline (SSO) buffer, so
    # appending a small fragment to a still-small string fails to interpret
    # ("can't get dst memory"). A heap-backed destination interprets fine.
    var mma = String(capacity=64)
    mma += "{\n"
    if not a_tmem:
        mma += ".reg .b64 %rda;\n"
    mma += ".reg .b64 %rdb;\n"
    mma += ".reg .s32 %ra;\n"
    if a_tmem:
        mma += ".reg .b32 %rab;\n"
    mma += ".reg .s32 %rb;\n"
    mma += ".reg .pred %pj;\n"
    mma += ".reg .pred %ps;\n"
    if partial:
        mma += ".reg .pred %pv;\n"
    mma += "setp.eq.s32 %pj, $6, 0;\n"

    # Instruction mnemonic (no predicate prefix; that is applied per-block below).
    instr = tcgen05_mma_type + kind if ws else (
        "tcgen05.mma.cta_group::" + String(cta_group) + "." + kind
    )
    # Non-ws zero-column mask operand; absent for ws.
    mask = (
        "{$1, $1, $1, $1}" if cta_group
        == 1 else "{$1, $1, $1, $1, $1, $1, $1, $1}"
    )
    # TMEM A column stride per k-mma (TS only).
    a_stride = mma_k * operand_size // 4
    # Operand slot holding the warp-uniform `valid_k_mmas` (partial only): A
    # consumes `$7,$8` for SS but only `$7` for TS, so the next free slot differs.
    valid_op = 8 if a_tmem else 9

    for k in range(num_k_mmas):
        jj = k_start + k
        # Warp-uniform validity guard: true once an absolute k-index lands past
        # the loaded region. Block jj == 0 is always loaded (valid_k_mmas >= 1).
        if partial and jj != 0:
            mma += String("setp.le.u32 %pv, $", valid_op, ", ", jj, ";\n")

        # A/B descriptor setup + the enable-input-d (`%ps`) scale predicate.
        if jj == 0:
            if not a_tmem:
                mma += "mov.b64 %rda, {$7, $8};\n"
            mma += "mov.b64 %rdb, {$4, $5};\n"
            # Absolute first block initializes the accumulator from c_scale ($3).
            mma += "setp.ne.b32 %ps, $3, 0;\n"
        else:
            b_offset = (layout_b(IntTuple(0, mma_k * jj)) * operand_size) >> 4
            if a_tmem:
                a_offset = a_stride * jj
                mma += String("add.s32 %ra, $7, ", a_offset, ";\n")
                mma += "mov.b32 %rab, %ra;\n"
                mma += String("add.s32 %rb, $4, ", b_offset, ";\n")
                mma += "mov.b64 %rdb, {%rb, $5};\n"
            elif partial:
                # SS-partial interleaving: A descriptor, then B descriptor.
                a_offset = (
                    layout_a(IntTuple(0, mma_k * jj)) * operand_size
                ) >> 4
                mma += String("add.s32 %ra, $7, ", a_offset, ";\n")
                mma += "mov.b64 %rda, {%ra, $8};\n"
                mma += String("add.s32 %rb, $4, ", b_offset, ";\n")
                mma += "mov.b64 %rdb, {%rb, $5};\n"
            else:
                # SS-full interleaving: both `add`s first, then both `mov`s.
                a_offset = (
                    layout_a(IntTuple(0, mma_k * jj)) * operand_size
                ) >> 4
                mma += String("add.s32 %ra, $7, ", a_offset, ";\n")
                mma += String("add.s32 %rb, $4, ", b_offset, ";\n")
                mma += "mov.b64 %rda, {%ra, $8};\n"
                mma += "mov.b64 %rdb, {%rb, $5};\n"
            # First accumulate block (of the whole tile, or of a later stage)
            # pins %ps = 1; it then stays set for every subsequent block.
            if k == 0 or jj == 1:
                mma += "setp.ne.b32 %ps, 1, 0;\n"

        # Result + operand list.
        if a_tmem:
            a_op = "$7" if jj == 0 else "%rab"
            operands = String(" [$0], [", a_op, "], %rdb, $2, ")
        else:
            operands = String(" [$0], %rda, %rdb, $2, ")
        if not ws:
            operands += mask + ", "
        operands += "%ps;\n"

        # Predication (form depends ONLY on `partial`; see header).
        if partial:
            mma += String("@%pj bra skip", k, ";\n")
            if jj != 0:
                mma += "@!%pv "
            mma += instr + operands
            mma += String("skip", k, ":\n")
        else:
            mma += "@!%pj " + instr + operands
    return mma + "}"


@always_inline("nodebug")
def bulk_mma[
    kind: UMMAKind,
    //,
    layout_a: Layout,
    layout_b: Layout,
    *,
    num_k_mmas: Int,
    mma_k: Int,
    operand_size: Int,
    cta_group: Int = 1,
](
    idesc: UMMAInsDescriptor[kind],
    a: MMASmemDescriptorPair,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
):
    # Full-tile SS (both operands SMEM descriptors), non-ws contraction.
    comptime assert cta_group in (1, 2)
    comptime mma_string = _build_mma[a_tmem=False, ws=False, partial=False](
        String(kind),
        layout_a,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        cta_group=cta_group,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a.lo, a.hi
    )


@always_inline("nodebug")
def bulk_mma[
    kind: UMMAKind,
    //,
    layout_b: Layout,
    *,
    mma_k: Int,
    num_k_mmas: Int,
    operand_size: Int,
    cta_group: Int = 1,
](
    idesc: UMMAInsDescriptor[kind],
    a: UInt32,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
):
    # Full-tile TS (A in TMEM, B an SMEM descriptor), non-ws contraction.
    # `_build_mma` ignores `layout_a` for TS, so `layout_b` fills that slot.
    comptime assert num_k_mmas >= 1 and num_k_mmas <= 16
    comptime assert cta_group in (1, 2)
    comptime mma_string = _build_mma[a_tmem=True, ws=False, partial=False](
        String(kind),
        layout_b,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        cta_group=cta_group,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a
    )


@always_inline("nodebug")
def bulk_mma_partial[
    kind: UMMAKind,
    //,
    layout_b: Layout,
    *,
    mma_k: Int,
    num_k_mmas: Int,
    operand_size: Int,
    k_start: Int = 0,
    cta_group: Int = 1,
](
    idesc: UMMAInsDescriptor[kind],
    a: UInt32,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
    valid_k_mmas: UInt32,
):
    # P@V contraction for a partially-loaded last KV tile (TS, non-ws). Issues
    # this stage's `num_k_mmas` k-blocks (absolute indices
    # `k_start ..< k_start + num_k_mmas`) as a SINGLE fused inline-asm sequence.
    # Each block's MMA carries a warp-uniform validity guard derived from
    # `valid_k_mmas` (the count of loaded MMA_K blocks) kept entirely SEPARATE
    # from the `elect` predicate -- so the elect codegen is identical to the
    # full-tile `bulk_mma` (no BSYNC.RECONVERGENT). `a`/`b` are the un-offset
    # (stage-0) bases; the builder applies absolute per-block offsets.
    comptime assert num_k_mmas >= 1 and num_k_mmas <= 16
    comptime assert cta_group in (1, 2)
    comptime mma_string = _build_mma[a_tmem=True, ws=False, partial=True](
        String(kind),
        layout_b,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        k_start=k_start,
        cta_group=cta_group,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a, valid_k_mmas
    )


@always_inline("nodebug")
def bulk_mma_ss_partial[
    kind: UMMAKind,
    //,
    layout_a: Layout,
    layout_b: Layout,
    *,
    num_k_mmas: Int,
    mma_k: Int,
    operand_size: Int,
    k_start: Int = 0,
    cta_group: Int = 1,
](
    idesc: UMMAInsDescriptor[kind],
    a: MMASmemDescriptorPair,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
    valid_k_mmas: UInt32,
):
    # Contraction over a partially-loaded last KV tile, SS (non-ws) variant:
    # both A and B come from SMEM descriptors. Each block's MMA carries a
    # warp-uniform validity guard derived from `valid_k_mmas` kept entirely
    # SEPARATE from `elect` (no BSYNC.RECONVERGENT). `a`/`b` are the un-offset
    # (stage-0) bases; the builder applies absolute per-block offsets.
    comptime assert num_k_mmas >= 1
    comptime assert cta_group in (1, 2)
    comptime mma_string = _build_mma[a_tmem=False, ws=False, partial=True](
        String(kind),
        layout_a,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        k_start=k_start,
        cta_group=cta_group,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem),
        0,
        idesc,
        c_scale,
        b.lo,
        b.hi,
        elect,
        a.lo,
        a.hi,
        valid_k_mmas,
    )


# ------------------------------------------------------------------------------
# SM100 warp-specialized (.ws) MMA building blocks
# ------------------------------------------------------------------------------


@always_inline
def bulk_mma_ws[
    kind: UMMAKind,
    a_dtype: DType,
    b_dtype: DType,
    *,
    a_BMN: Int,
    a_BK: Int,
    a_swizzle: TensorMapSwizzle,
    a_is_k_major: Bool,
    b_BMN: Int,
    b_BK: Int,
    b_swizzle: TensorMapSwizzle,
    b_is_k_major: Bool,
    num_k_mmas: Int,
    operand_size: Int,
    tcgen05_mma_type: String,
    mma_k: Int = 16,
    b_page_dense: Bool = False,
](
    idesc: UMMAInsDescriptor[kind],
    a: MMASmemDescriptorPair,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
):
    # Full-tile SS, warp-specialized. The tile layouts are computed from the
    # dtype/tile params (`_build_mma` takes `Layout` directly). `b_page_dense`
    # selects the row-major page-fold layout for the B operand (K / Q@K' is
    # k-major; the advance crosses a depth chunk by `_CM_NUM_ROWS*gran` instead
    # of `BN*gran`, derived from this layout).
    comptime layout_a = tile_layout_k_major[
        a_dtype, a_BMN, a_BK, a_swizzle
    ]() if a_is_k_major else tile_layout_mn_major[
        a_dtype, a_BMN, a_BK, a_swizzle
    ]()
    comptime layout_b = tile_layout_k_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]() if b_is_k_major else tile_layout_mn_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]()
    comptime mma_string = _build_mma[a_tmem=False, ws=True, partial=False](
        String(kind),
        layout_a,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        tcgen05_mma_type=tcgen05_mma_type,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a.lo, a.hi
    )


# ---- TS (TMEM-SMEM) .ws MMA building blocks ----


@always_inline
def bulk_mma_ws_ts[
    kind: UMMAKind,
    b_dtype: DType,
    *,
    b_BMN: Int,
    b_BK: Int,
    b_swizzle: TensorMapSwizzle,
    b_is_k_major: Bool,
    num_k_mmas: Int,
    operand_size: Int,
    tcgen05_mma_type: String,
    mma_k: Int = 16,
    b_page_dense: Bool = False,
](
    idesc: UMMAInsDescriptor[kind],
    a: UInt32,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
):
    # Full-tile TS, warp-specialized. `a` is a single TMEM base ($7); `_build_mma`
    # computes each k-tile's column offset in-PTX (`add.s32 %ra, $7, k*stride`),
    # so the old per-tile operand ladder is gone.
    comptime assert num_k_mmas >= 1 and num_k_mmas <= 16
    comptime layout_b = tile_layout_k_major[
        b_dtype, b_BMN, b_BK, b_swizzle
    ]() if b_is_k_major else tile_layout_mn_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]()
    comptime mma_string = _build_mma[a_tmem=True, ws=True, partial=False](
        String(kind),
        layout_b,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        tcgen05_mma_type=tcgen05_mma_type,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a
    )


# ---- partial-K (.ws) MMA building blocks ----


@always_inline
def bulk_mma_ws_partial[
    kind: UMMAKind,
    a_dtype: DType,
    b_dtype: DType,
    *,
    a_BMN: Int,
    a_BK: Int,
    a_swizzle: TensorMapSwizzle,
    a_is_k_major: Bool,
    b_BMN: Int,
    b_BK: Int,
    b_swizzle: TensorMapSwizzle,
    b_is_k_major: Bool,
    num_k_mmas: Int,
    operand_size: Int,
    tcgen05_mma_type: String,
    mma_k: Int = 16,
    k_start: Int = 0,
    b_page_dense: Bool = False,
](
    idesc: UMMAInsDescriptor[kind],
    a: MMASmemDescriptorPair,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
    valid_k_mmas: UInt32,
):
    # P@V contraction for a partially-loaded last KV tile, SS warp-specialized:
    # both A and B come from SMEM descriptors. Each block's MMA carries a
    # warp-uniform validity guard derived from `valid_k_mmas` kept entirely
    # SEPARATE from `elect` (no BSYNC.RECONVERGENT). `a`/`b` are the un-offset
    # (stage-0) bases; the builder applies absolute per-block offsets.
    comptime layout_a = tile_layout_k_major[
        a_dtype, a_BMN, a_BK, a_swizzle
    ]() if a_is_k_major else tile_layout_mn_major[
        a_dtype, a_BMN, a_BK, a_swizzle
    ]()
    comptime layout_b = tile_layout_k_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]() if b_is_k_major else tile_layout_mn_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]()
    comptime mma_string = _build_mma[a_tmem=False, ws=True, partial=True](
        String(kind),
        layout_a,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        k_start=k_start,
        tcgen05_mma_type=tcgen05_mma_type,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem),
        0,
        idesc,
        c_scale,
        b.lo,
        b.hi,
        elect,
        a.lo,
        a.hi,
        valid_k_mmas,
    )


@always_inline
def bulk_mma_ws_ts_partial[
    kind: UMMAKind,
    b_dtype: DType,
    *,
    b_BMN: Int,
    b_BK: Int,
    b_swizzle: TensorMapSwizzle,
    b_is_k_major: Bool,
    num_k_mmas: Int,
    operand_size: Int,
    tcgen05_mma_type: String,
    mma_k: Int = 16,
    k_start: Int = 0,
    b_page_dense: Bool = False,
](
    idesc: UMMAInsDescriptor[kind],
    a: UInt32,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
    valid_k_mmas: UInt32,
):
    # P@V contraction for a partially-loaded last KV tile, TS warp-specialized.
    # `a` is the un-offset (stage-0) TMEM base ($7); `_build_mma` computes each
    # block's ABSOLUTE column offset (`a_stride * (k_start + k)`) in-PTX, so the
    # old per-tile operand ladder is gone. `valid_k_mmas` rides $8 (A uses only
    # $7) and gates each block via a `%pv` guard kept SEPARATE from `elect` (no
    # BSYNC.RECONVERGENT).
    comptime assert num_k_mmas >= 1 and num_k_mmas <= 16
    comptime layout_b = tile_layout_k_major[
        b_dtype, b_BMN, b_BK, b_swizzle
    ]() if b_is_k_major else tile_layout_mn_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]()
    comptime mma_string = _build_mma[a_tmem=True, ws=True, partial=True](
        String(kind),
        layout_b,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        k_start=k_start,
        tcgen05_mma_type=tcgen05_mma_type,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a, valid_k_mmas
    )


@always_inline
def llvm_opaque_tid() -> UInt32:
    return llvm_intrinsic[
        "llvm.nvvm.read.ptx.sreg.tid.x", UInt32, has_side_effect=True
    ]()


@always_inline
def intrin_ftz[intrin: String](a: Float32, b: Float32) -> Float32:
    return inlined_assembly[
        String(intrin, ".ftz.f32 $0, $1, $2;"),
        Float32,
        constraints="=f,f,f",
        has_side_effect=False,
    ](a, b)


@always_inline
def intrin[intrin: String](a: Float32, b: Float32, c: Float32) -> Float32:
    return inlined_assembly[
        String(intrin, ".f32 $0, $1, $2, $3;"),
        Float32,
        constraints="=f,f,f,f",
        has_side_effect=False,
    ](a, b, c)


@always_inline
def intrin_ftz_x2[
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
def add_ftz(a: Float32, b: Float32) -> Float32:
    return intrin_ftz["add"](a, b)


@always_inline
def sub_ftz(a: Float32, b: Float32) -> Float32:
    return intrin_ftz["sub"](a, b)


@always_inline
def mul_ftz(a: Float32, b: Float32) -> Float32:
    return intrin_ftz["mul"](a, b)


@always_inline
def max_ftz(a: Float32, b: Float32) -> Float32:
    return intrin_ftz["max"](a, b)


@always_inline
def max_ftz(a: Float32, b: Float32, c: Float32) -> Float32:
    return intrin["max.ftz"](a, b, c)


@always_inline
def add_ftz(
    a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]
) -> SIMD[DType.float32, 2]:
    return intrin_ftz_x2["add"](a, b)


@always_inline
def sub_ftz(
    a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]
) -> SIMD[DType.float32, 2]:
    return intrin_ftz_x2["sub"](a, b)


@always_inline
def mul_ftz(
    a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]
) -> SIMD[DType.float32, 2]:
    return intrin_ftz_x2["mul"](a, b)


@always_inline
def add_ftz_rm(
    a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]
) -> SIMD[DType.float32, 2]:
    return intrin_ftz_x2["add.rm"](a, b)


@always_inline
def fma_ftz(
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
def exp2_emulation[
    use_exp2_emulation: Bool = True
](x: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    comptime if use_exp2_emulation:
        comptime fp32_round_int = SIMD[DType.float32, 2]((1 << 23) + (1 << 22))
        clamped = max(x, -FP32_EXP_BIAS)
        # We want to round down here, so that the fractional part is in [0, 1)
        rounded = add_ftz_rm(clamped, fp32_round_int)
        rounded_back = sub_ftz(rounded, fp32_round_int)
        frac = sub_ftz(clamped, rounded_back)
        # Degree-3 polynomial approximation of `2^x` on `x ∈ [0, 1)`.
        # Coefficients lifted from Tri Dao's FlashAttention-3
        # `exp2_emulated` (Dao-AILab/flash-attention, `flash_fwd_kernel*`)
        # — fit by minimax over the unit interval.
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
def elect_mma_arrive[
    cta_group: Int = 1
](
    mbar_ptr: UnsafePointer[address_space=AddressSpace.SHARED, ...],
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
        .reg .pred %p;
        setp.eq.s32  %p, $1, 0;
        @!%p tcgen05.commit.cta_group::"""
        + String(cta_group)
        + """.mbarrier::arrive::one.shared::cluster.b64 [$0];
        }""",
        NoneType,
        constraints="r, r",
    ](Int32(Int(mbar_ptr)), elect)


@always_inline
def expect_bytes_pred(
    mbar_ptr: UnsafePointer[address_space=AddressSpace.SHARED, ...],
    bytes: Int32,
    pred: Int32,
):
    """Issue `mbarrier.arrive.expect_tx.shared::cta.b64` predicated on
    `pred != 0`.

    Equivalent to:

        if pred != 0:
            mbar_ptr[].expect_bytes(bytes)

    but folds the runtime branch into a single PTX `@%p` predicate
    on the `mbarrier.arrive.expect_tx` instruction — no Mojo-level
    `if`, no SASS branch, no warp divergence.

    Args:
        mbar_ptr: Pointer to the shared-memory mbarrier (8-byte slot).
        bytes: Expected transaction byte count for this barrier.
        pred: Runtime predicate (typically the result of `elect()` or
            an `elect()`-derived `elect_mask`); the PTX instruction is
            skipped when this is 0.
    """

    comptime type = mbar_ptr.type
    comptime assert size_of[type]() == 8, "mbar_ptr must be 8 bytes"

    inlined_assembly[
        """{
        .reg .pred %p;
        .reg .b64 %state;
        setp.ne.s32 %p, $2, 0;
        @%p mbarrier.arrive.expect_tx.shared::cta.b64 %state, [$0], $1;
        }""",
        NoneType,
        constraints="r,r,r",
    ](Int32(Int(mbar_ptr)), bytes, pred)


@always_inline
def maximum[
    BN: Int, //, *, width: Int = 4
](
    x: InlineArray[Scalar[DType.float32], BN],
    out res: StaticTuple[Float32, width],
):
    res = {}

    comptime for w in range(width):
        res[w] = max_ftz(
            x[3 * w],
            x[3 * w + 1],
            x[3 * w + 2],
        )

    # max idx = 3 * (width-1) + 2 = 3*width - 1
    comptime remaining_iters = BN - 3 * width
    comptime num_iters = remaining_iters // (2 * width)

    comptime for i in range(num_iters):
        comptime col = i * 2 * width + 3 * width

        comptime for w in range(width):
            res[w] = max_ftz(
                res[w],
                x[col + 2 * w],
                x[col + 2 * w + 1],
            )

    comptime remainder_base = 3 * width + 2 * width * num_iters
    comptime end_iters = (BN - remainder_base) // 2

    comptime for w in range(end_iters):
        res[w] = max_ftz(
            res[w],
            x[remainder_base + 2 * w],
            x[remainder_base + 2 * w + 1],
        )

    comptime if (BN - remainder_base) % 2 == 1:
        res[end_iters] = max_ftz(res[end_iters], x[BN - 1])


@always_inline
def maximum[
    BN: Int, //, *, width: Int = 4
](
    x: InlineArray[Scalar[DType.float32], BN],
    init: StaticTuple[Float32, width],
    out res: StaticTuple[Float32, width],
):
    res = init

    # unroll (using SIMD) to break up dependency chain
    comptime num_iters = BN // (2 * width)

    comptime for i in range(num_iters):
        comptime for w in range(width):
            comptime j = i * 2 * width + 2 * w
            res[w] = max_ftz(res[w], x[j], x[j + 1])

    comptime remainder_base = 2 * width * num_iters
    comptime end_iters = (BN - remainder_base) // 2

    comptime for w in range(end_iters):
        res[w] = max_ftz(
            res[w],
            x[remainder_base + 2 * w],
            x[remainder_base + 2 * w + 1],
        )

    comptime if (BN - remainder_base) % 2 == 1:
        res[end_iters] = max_ftz(res[end_iters], x[BN - 1])


@always_inline
def maximum(x: StaticTuple[Float32, 4]) -> Float32:
    return max_ftz(max_ftz(x[0], x[1], x[2]), x[3])


@always_inline
def maximum(x: StaticTuple[Float32, 4], init: Float32) -> Float32:
    return max_ftz(max_ftz(x[0], x[1], x[2]), x[3], init)


@always_inline
def maximum(x: StaticTuple[Float32, 8]) -> Float32:
    var a = max_ftz(x[0], x[1], x[2])
    var b = max_ftz(x[3], x[4], x[5])
    var c = max_ftz(x[6], x[7])
    return max_ftz(a, b, c)


@always_inline
def maximum(x: StaticTuple[Float32, 8], init: Float32) -> Float32:
    var a = max_ftz(init, x[0], x[1])
    var b = max_ftz(x[2], x[3], x[4])
    var c = max_ftz(x[5], x[6], x[7])
    return max_ftz(a, b, c)


@always_inline
def sum[
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
    @__allow_legacy_any_origin_fields
    var mbar: MBarType
    var state: PipelineState[Self.num_kv_stages]

    @always_inline
    def __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    def producer_mbar[qk_stage: Int = 0](self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.mbar + UInt32(Self.num_qk_stages) * idx + qk_stage

    @always_inline
    def consumer_mbar[qk_stage: Int = 0](self, idx: UInt32) -> MBarType:
        comptime const_offset = qk_stage + Self.num_stages
        return self.mbar + UInt32(Self.num_qk_stages) * idx + const_offset

    @always_inline
    def consumer_mbar[qk_stage: Int = 0](self) -> MBarType:
        return self.consumer_mbar[qk_stage](self.state.index())

    @always_inline("nodebug")
    def producer_acquire[qk_stage: Int = Self.num_qk_stages - 1](self):
        """Wait until consumer has released the buffer for this stage."""
        self.consumer_mbar[qk_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    def consumer_wait[qk_stage: Int = Self.num_qk_stages - 1](self):
        """Wait for producer to complete this stage."""
        self.producer_mbar[qk_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    def consumer_release[
        qk_stage: Int = Self.num_qk_stages - 1
    ](mut self, e: Int32):
        """Release the buffer after consuming this stage."""
        elect_mma_arrive(self.consumer_mbar[qk_stage](), e)

        comptime if qk_stage == Self.num_qk_stages - 1:
            self.state.step()

    @always_inline("nodebug")
    def consumer_release_at(self, idx: UInt32, e: Int32):
        """Release a specific stage without stepping the pipeline state.

        Used for deferred V release in fused KV mode: V_{n-1} must be
        released while holding K_n, which is at a different pipeline index.
        """
        comptime qk_stage = Self.num_qk_stages - 1
        comptime const_offset = qk_stage + Self.num_stages
        var mbar = self.mbar + UInt32(Self.num_qk_stages) * idx + const_offset
        elect_mma_arrive(mbar, e)

    @staticmethod
    @always_inline
    def num_mbars() -> UInt32:
        return UInt32(2 * Self.num_qk_stages * Self.num_kv_stages)


# Backward-compatible type aliases
comptime KPipeline = StagedPipeline
comptime VPipeline = StagedPipeline[_, 1]
comptime KVPipeline = StagedPipeline


struct TMADestination[dtype: DType, smem_elems: Int](TrivialRegisterPassable):
    """Pairs a shared memory TileTensor with a barrier for TMA operations.

    The stored TileTensor uses a flat `row_major[smem_elems]()` layout —
    TMA only uses `.ptr`.
    """

    comptime SmemType = TileTensor[
        Self.dtype,
        type_of(tt_row_major[Self.smem_elems]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    @__allow_legacy_any_origin_fields
    var mbar: MBarType

    @__allow_legacy_any_origin_fields
    var smem: Self.SmemType

    @always_inline
    def __init__(
        out self,
        mbar: MBarType,
        smem: Self.SmemType,
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

    # Compute layout first using comptime, then use it in type.
    # For pair-CTA: K uses k_rows_per_cta (BN/2), V uses v_cols_per_cta (ov/2).
    comptime tile_layout: Layout = tile_layout_k_major[
        Self.dtype,
        Self.config.k_rows_per_cta(),
        Self.config.BK0,
        Self.config.swizzle_mode,
    ]() if Self.is_k else tile_layout_mn_major[
        Self.dtype,
        Self.config.v_cols_per_cta(),
        Self.config.BK1,
        Self.config.swizzle_mode,
    ]()

    comptime PairType = TMADestination[Self.dtype, Self.tile_layout.size()]
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

    @__allow_legacy_any_origin_fields
    var smem: Self.SMemType

    @always_inline
    def __init__(out self, mbar: MBarType, smem: Self.SMemType):
        comptime if Self.is_k:
            comptime assert (
                Self.config.padded_qk_depth % Self.config.num_qk_stages == 0
            ), "padded_qk_depth must be divisible by num_qk_stages"
        self.pipeline = {mbar}
        self.smem = smem
        self.pipeline.state._phase = 1

    @always_inline
    def __init__(
        out self,
        pipeline: StagedPipeline[
            Self.config.num_kv_stages, Self.num_qk_stages_effective
        ],
        smem: Self.SMemType,
    ):
        comptime if Self.is_k:
            comptime assert (
                Self.config.padded_qk_depth % Self.config.num_qk_stages == 0
            ), "padded_qk_depth must be divisible by num_qk_stages"
        self.pipeline = pipeline
        self.smem = smem
        self.pipeline.state._phase = 1

    @always_inline
    def get_smem[*, qk_stage: Int = 0](self) -> Self.SMemType:
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
    def get_tile[*, qk_stage: Int = 0](self) -> Self.PairType:
        """Get TMA destination for this stage."""
        p_mbar = self.pipeline.producer_mbar[qk_stage]()
        var smem = Self.PairType.SmemType(
            self.get_smem[qk_stage=qk_stage](),
            tt_row_major[Self.PairType.smem_elems](),
        )
        return {p_mbar, smem}

    @always_inline
    def get_tile[*, qk_stage: Int = 0](self, e: Int32) -> Self.PairType:
        """Get TMA destination with optional expect_bytes."""
        p_mbar = self.pipeline.producer_mbar[qk_stage]()
        if e != 0:
            p_mbar[].expect_bytes(Int32(Self.tile_bytes))
        var smem = Self.PairType.SmemType(
            self.get_smem[qk_stage=qk_stage](),
            tt_row_major[Self.PairType.smem_elems](),
        )
        return {p_mbar, smem}

    @always_inline
    def acquire[*, qk_stage: Int = 0](self):
        """Wait for consumer to release the buffer."""
        self.pipeline.producer_acquire[qk_stage]()

    @always_inline
    def commit_step(mut self):
        """Step the pipeline. Commit is handled by tma_op.async_copy."""
        self.pipeline.state.step()

    # Backward-compatible K methods (for KProducerPipeline)
    comptime KPairType = Self.PairType  # Alias for backward compatibility

    @always_inline
    def get_k_smem[*, qk_stage: Int](self) -> Self.SMemType:
        return self.get_smem[qk_stage=qk_stage]()

    @always_inline
    def get_k[*, qk_stage: Int](self) -> Self.PairType:
        return self.get_tile[qk_stage=qk_stage]()

    @always_inline
    def get_k[*, qk_stage: Int](self, e: Int32) -> Self.PairType:
        return self.get_tile[qk_stage=qk_stage](e)

    @always_inline
    def acquire_k[*, qk_stage: Int](self):
        self.acquire[qk_stage=qk_stage]()

    @always_inline
    def get_v_smem(self) -> Self.SMemType:
        return self.get_smem[qk_stage=0]()

    @always_inline
    def get_v(self, e: Int32) -> Self.PairType:
        return self.get_tile[qk_stage=0](e)

    @always_inline
    def acquire_v(self):
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

    comptime full_kv_bytes = (
        Self.config.k_rows_per_cta()
        * Self.config.padded_ov_depth
        * size_of[Self.dtype]()
        + Self.config.k_rows_per_cta()
        * Self.config.rope_depth()
        * Self.config.rope_dtype_size
    ) if Self.is_k else (
        Self.config.BN * Self.config.v_cols_per_cta() * size_of[Self.dtype]()
    )
    comptime staged_k_bytes = Self.config.k_rows_per_cta() * Self.config.BK0 * size_of[
        Self.dtype
    ]()

    # K uses full staging, V uses qk_stages=1
    comptime num_qk_stages_effective: Int = Self.config.num_qk_stages if Self.is_k else 1

    # Descriptor parameters differ by role
    comptime BMN: Int = Self.config.k_rows_per_cta() if Self.is_k else Self.config.v_cols_per_cta()
    comptime BK: Int = Self.config.BK0 if Self.is_k else Self.config.BK1
    comptime is_k_major: Bool = Self.is_k
    # Page-dense (row-major) layout: K (Q@K', k-major) gated by k_row_major(),
    # V (P@V, mn-major) by v_row_major(). `is_k_major=Self.is_k` (below) routes
    # the flag to the matching `tile_layout_*` branch in `smem_descriptor`.
    comptime page_dense: Bool = (
        Self.config.k_row_major() if Self.is_k else Self.config.v_row_major()
    )

    var pipeline: StagedPipeline[
        Self.config.num_kv_stages, Self.num_qk_stages_effective
    ]
    var smem_desc: MMASmemDescriptorPair

    @always_inline
    def __init__(
        out self,
        pipeline: StagedPipeline[
            Self.config.num_kv_stages, Self.num_qk_stages_effective
        ],
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipeline = pipeline
        self.smem_desc = smem_descriptor[
            BMN=Self.BMN,
            BK=Self.BK,
            swizzle_mode=Self.config.swizzle_mode,
            is_k_major=Self.is_k_major,
            page_dense=Self.page_dense,
        ](smem)

    @always_inline
    def __init__(
        out self,
        mbar: MBarType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        return Self(type_of(self.pipeline)(mbar), smem)

    @always_inline("nodebug")
    def get(self) -> MMASmemDescriptorPair:
        """Get smem descriptor for current stage."""
        var dyn_offset: UInt32 = (
            UInt32(Self.full_kv_bytes) * self.pipeline.state.index()
        )
        return self.smem_desc + dyn_offset

    @always_inline("nodebug")
    def wait[*, qk_stage: Int = 0](self):
        """Wait for tile from producer."""
        self.pipeline.consumer_wait[qk_stage]()

    @always_inline("nodebug")
    def release[*, qk_stage: Int = 0](mut self, e: Int32):
        """Release buffer after consuming."""
        self.pipeline.consumer_release[qk_stage](e)

    # Backward-compatible K methods (for KConsumerPipeline)
    @always_inline("nodebug")
    def get_k(self) -> MMASmemDescriptorPair:
        return self.get()

    @always_inline("nodebug")
    def wait_k[*, qk_stage: Int = Self.config.num_qk_stages - 1](mut self):
        """Wait on K stage from the producer."""
        self.wait[qk_stage=qk_stage]()

    @always_inline("nodebug")
    def release_k[
        *, qk_stage: Int = Self.config.num_qk_stages - 1
    ](mut self, e: Int32):
        """Release K buffer after consuming this stage."""
        self.release[qk_stage=qk_stage](e)

    # Backward-compatible V methods (for VConsumerPipeline)
    @always_inline("nodebug")
    def get_v(self) -> MMASmemDescriptorPair:
        return self.get()

    @always_inline("nodebug")
    def wait_v(self):
        """Wait for V tile."""
        self.wait[qk_stage=0]()

    @always_inline("nodebug")
    def release_v(mut self, e: Int32):
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
    cta_group: Int = 1,
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

    @__allow_legacy_any_origin_fields
    var producer_mbar_base: MBarType

    @__allow_legacy_any_origin_fields
    var consumer_mbar_base: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    def __init__(
        out self, producer_mbar_base: MBarType, consumer_mbar_base: MBarType
    ):
        self.producer_mbar_base = producer_mbar_base
        self.consumer_mbar_base = consumer_mbar_base
        self.state = {}

        comptime if Self.is_producer:
            # Producer starts with phase=1 so initial waits fall through
            self.state._phase = 1

    @always_inline
    def producer_mbar[sub_stage_idx: Int = 0](self) -> MBarType:
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
    def consumer_mbar[sub_stage_idx: Int = 0](self) -> MBarType:
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
    def acquire[sub_stage_idx: Int = 0](self):
        """Wait until consumer has released the buffer. Producer-only."""
        self.consumer_mbar[sub_stage_idx]()[].wait(self.state.phase())

    @always_inline("nodebug")
    def commit(mut self):
        """Commit production and step. Producer-only."""
        _ = self.producer_mbar()[].arrive()
        self.state.step()

    @always_inline("nodebug")
    def commit_mma(self):
        """Commit via MMA arrive using elected thread. Producer-only."""
        mbar = self.producer_mbar()
        elect_mma_arrive[cta_group=Self.cta_group](mbar, elect())

    @always_inline("nodebug")
    def commit_mma(self, elect: Int32):
        """Commit via MMA arrive with explicit elect value. Producer-only."""
        mbar = self.producer_mbar()
        elect_mma_arrive[cta_group=Self.cta_group](mbar, elect)

    # Consumer methods
    @always_inline("nodebug")
    def wait(self):
        """Wait for producer to complete. Consumer-only."""
        self.producer_mbar()[].wait(self.state.phase())

    @always_inline("nodebug")
    def release[sub_stage_idx: Int = 0](mut self):
        """Release buffer at sub-stage and step. Consumer-only."""
        _ = self.consumer_mbar[sub_stage_idx]()[].arrive()
        self.state.step()

    @always_inline("nodebug")
    def release_no_step[sub_stage_idx: Int = 0](self):
        """Release buffer without stepping. For multi-sub-stage release."""
        _ = self.consumer_mbar[sub_stage_idx]()[].arrive()

    # Shared method
    @always_inline("nodebug")
    def step(mut self):
        self.state.step()


# Backward-compatible type aliases
comptime ProducerPipeline = RolePipeline[_, True, _, _, _]
comptime ConsumerPipeline = RolePipeline[_, False, _, _, _]


struct MBarPipeline[number_of_stages: Int](TrivialRegisterPassable):
    comptime num_stages: Int = Self.number_of_stages

    # mbars are ordered in {producer, consumer} pairs
    @__allow_legacy_any_origin_fields
    var mbar: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    def __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    def init[*, num_producer: UInt32 = 1, num_consumer: UInt32 = 1](self):
        comptime for i in range(Self.number_of_stages):
            self.mbar[i].init(Int32(Int(num_producer)))

        comptime for i in range(Self.number_of_stages):
            self.mbar[i + Self.number_of_stages].init(Int32(Int(num_consumer)))

    @staticmethod
    @always_inline
    def num_mbars() -> UInt32:
        return UInt32(2 * Self.number_of_stages)


@always_inline
def apply_oob_mask[
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

    comptime if apply_log2e_after_mask:
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
def apply_mask[
    BN: Int,
    MaskType: MHAMask,
    //,
    *,
    mask_strategy: MaskStrategy,
    skip_scale: Bool = False,
](
    mut srow: InlineArray[Scalar[DType.float32], BN],
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

    comptime if MaskStrategy.BITMASK in mask_strategy or (
        MaskStrategy.OUT_OF_BOUNDS in mask_strategy
        and MaskStrategy.COMPUTED not in mask_strategy
    ):
        # Mask-driven bitmask path: each 32-col batch is masked by a 32-bit
        # visibility pattern. Either the mask provides it via `mask_bits()`
        # (`BITMASK`), or the kernel hardcodes "clip at num_keys" by setting
        # `OUT_OF_BOUNDS` alone — used by softmax warps that fast-path the
        # runtime-`NO_MASK` case.
        comptime num_batches = BN // 32
        comptime assert (BN % 32) == 0

        comptime for batch in range(num_batches):
            var col_start: Int32 = kv_tile_start_row + Int32(32 * batch)
            var mask_bits: UInt32

            comptime if MaskStrategy.BITMASK in mask_strategy:
                mask_bits = mask.mask_bits(
                    prompt_idx, score_row, col_start, num_keys
                )
            else:
                # OUT_OF_BOUNDS alone: low `n_valid_oob` bits set, where
                # n_valid_oob = max(num_keys - col_start, 0).
                var n_valid_oob: Int32 = max(num_keys - col_start, 0)
                mask_bits = (
                    (UInt32(1) << UInt32(n_valid_oob)) - UInt32(1)
                ) if n_valid_oob < 32 else UInt32(0xFFFF_FFFF)

            comptime for n in range(32 // simd_size):
                comptime frag_col_simd = n + 32 * batch // simd_size
                comptime frag_col = frag_col_simd * simd_size
                var s: F32x2

                comptime if skip_scale:
                    s = F32x2(srow[frag_col], srow[frag_col + 1])
                else:
                    s = mul_ftz(
                        F32x2(srow[frag_col], srow[frag_col + 1]), scale_log2e
                    )

                comptime for i in range(simd_size):
                    comptime midx = n * simd_size + i
                    comptime flag: UInt32 = UInt32(1 << midx)
                    var in_bound: Bool = (mask_bits & flag) != UInt32(0)
                    var val: Float32 = s[i]
                    s[i] = val if in_bound else MASK_VALUE

                comptime if MaskType.apply_log2e_after_mask:
                    s = mul_ftz(s, log2e)

                srow[frag_col] = s[0]
                srow[frag_col + 1] = s[1]

    else:
        comptime block_size = BN // simd_size

        comptime for n in range(block_size):
            # score_col = mask_frag_col + j * 8
            comptime frag_col = simd_size * n
            var s: F32x2

            comptime if skip_scale:
                s = F32x2(srow[frag_col], srow[frag_col + 1])
            else:
                s = mul_ftz(
                    F32x2(srow[frag_col], srow[frag_col + 1]), scale_log2e
                )
            var score_col: Int32 = kv_tile_start_row + Int32(frag_col)

            comptime if MaskStrategy.COMPUTED in mask_strategy:
                s = mask.mask(
                    IndexList[4, element_type=DType.uint32](
                        Int(prompt_idx),
                        Int(q_head_idx),
                        Int(score_row),
                        Int(score_col),
                    ),
                    s,
                )

            var result = apply_oob_mask[
                mask_strategy=mask_strategy,
                apply_log2e_after_mask=MaskType.apply_log2e_after_mask,
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
            srow[frag_col] = result[0]
            srow[frag_col + 1] = result[1]


@always_inline
def peel_mask[
    num_sets: Int,
    //,
    mask_strategies: StaticTuple[MaskStrategy, num_sets],
    load_fn: def[mask_strategy: MaskStrategy](UInt32) capturing -> Float32,
](mut mask_iters: StaticTuple[UInt32, num_sets], kv_row: UInt32,) -> Float32:
    """Determine which mask strategy applies to the peeled first iteration.

    Walks through mask sets to find the first with remaining iterations,
    calls load_fn with the corresponding strategy, and decrements the counter.
    Prevents UInt32 underflow when early sets are empty (e.g.
    SlidingWindowCausalMask with num_sets=3 and small sequences).
    """
    comptime assert num_sets in (1, 2, 3)
    comptime if num_sets == 1:
        mask_iters[0] -= 1
        return load_fn[mask_strategies[0]](kv_row)
    elif num_sets == 2:
        if mask_iters[0] > 0:
            mask_iters[0] -= 1
            return load_fn[mask_strategies[0]](kv_row)
        else:
            mask_iters[1] -= 1
            return load_fn[mask_strategies[1]](kv_row)
    else:
        if mask_iters[0] > 0:
            mask_iters[0] -= 1
            return load_fn[mask_strategies[0]](kv_row)
        elif mask_iters[1] > 0:
            mask_iters[1] -= 1
            return load_fn[mask_strategies[1]](kv_row)
        else:
            mask_iters[2] -= 1
            return load_fn[mask_strategies[2]](kv_row)


struct FA4MiscMBars[
    *,
    num_qk_stages: Int = 1,
    num_pv_stages: Int = 1,
    num_kv_stages: Int = 2,
    use_order_barriers: Bool = True,
    use_fused_kv: Bool = False,
    pair_cta: Bool = False,
    num_qo: Int = 2,
](TrivialRegisterPassable):
    """Manages all mbarrier resources for FA4.

    This struct consolidates all mbarrier management including:
    - S barriers (score MMA synchronization)
    - C barriers (correction synchronization)
    - Order barriers (softmax ordering)
    - Q1Sync barriers (Q tile synchronization)
    - K/V pipeline barriers (separate K and V)
    - O pipeline barriers

    Parameters:
        num_qk_stages: Number of stages for Q@K' MMA (K loading can be staged).
        num_pv_stages: Number of stages for P@V MMA (P writing can be staged).
        num_kv_stages: Number of KV buffer stages for double/triple buffering.
        use_order_barriers: When True, allocate order barriers to prevent softmax
            warp group overlap. When False, order barriers are omitted.
        use_fused_kv: Whether the K and V share the same pipeline, or separate.
        pair_cta: Whether to use 1-cta or 2-cta implementation.
        num_qo: Number of Q tiles per CTA. When 1, the `Q1Sync` slot is
            collapsed and `K_offset` shifts down by `num_qk_stages`. Must
            be 2 for any caller of `q1_wait_mbar()`.

    Memory layout (count=128 first, then count=1):
        [S0_cons] [S1_cons] [C0] [C1] [Order*] | [S0_prod] [S1_prod] [Q1Sync**] [K] [V] [O_prod]
        *Order barriers only present when use_order_barriers=True
        **Q1Sync barriers only present when num_qo == 2
    """

    @__allow_legacy_any_origin_fields
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
    # Q1Sync barriers (collapsed when num_qo == 1; q1_wait_mbar() is
    # then unsafe to call — see the comptime assert in q1_wait_mbar().)
    comptime Q1SyncIdx = Self.S1_producer_offset + 1
    comptime Q1Sync_count: Int = Self.num_qk_stages if Self.num_qo == 2 else 0
    # K pipeline barriers
    comptime K_offset = Self.Q1SyncIdx + Self.Q1Sync_count
    comptime K_barriers: Int = 2 * Self.num_qk_stages * Self.num_kv_stages
    # V pipeline barriers (separate from K, only in split mode)
    comptime V_offset: Int = Self.K_offset + Self.K_barriers
    comptime V_barriers: Int = 0 if Self.use_fused_kv else 2 * Self.num_kv_stages
    # O producer barriers (count=1)
    comptime O_producer_offset = Self.V_offset + Self.V_barriers

    # Total size includes all barriers
    comptime size = Self.O_producer_offset + 2
    comptime number_warpgroup_count = Self.S0_producer_offset

    @always_inline
    def __init__(out self, mbar_base: MBarType):
        self.mbar_base = mbar_base

    @staticmethod
    def _init_count(lane_idx: Int32) -> Int32:
        """Return the mbarrier thread count for the given barrier index.

        S0_consumer[0] and S1_consumer[0] get count=256 (combined softmax +
        correction), other S_consumer barriers get count=128 (softmax only).
        In pair-CTA mode, S_consumer counts double (both CTAs arrive).
        C and Order barriers are CTA-local and always count=128.
        """
        comptime cta_mult: Int = 2 if Self.pair_cta else 1
        # S_consumer[0] = combined P+O: softmax + correction from each CTA.
        if lane_idx == Int32(Self.S0_consumer_offset) or lane_idx == Int32(
            Self.S1_consumer_offset
        ):
            return Int32(256 * cta_mult)
        # Other S_consumer barriers: softmax only, scaled by cta_mult.
        if lane_idx < Int32(Self.C0_offset):
            return Int32(128 * cta_mult)
        # C and Order barriers: CTA-local, always 128.
        if lane_idx < Int32(Self.number_warpgroup_count):
            return 128
        return 1

    @always_inline
    def init(self, *, lane_idx: Int32):
        comptime if Self.size < WARP_SIZE:
            if lane_idx < Int32(Self.size):
                self.mbar_base[lane_idx].init(Self._init_count(lane_idx))
        elif Self.size == WARP_SIZE:
            self.mbar_base[lane_idx].init(Self._init_count(lane_idx))
        else:
            comptime assert Self.size <= 2 * WARP_SIZE, String(
                "Total barrier count = ",
                Self.size,
                " exceeds 2 * WARP_SIZE = ",
                2 * WARP_SIZE,
            )
            # Wave 1: first 32 barriers (all lanes participate).
            self.mbar_base[lane_idx].init(Self._init_count(lane_idx))
            # Wave 2: remaining barriers past index 32.
            if lane_idx < Int32(Self.size - WARP_SIZE):
                self.mbar_base[Int32(WARP_SIZE) + lane_idx].init(1)

    # S pipeline type: 1 producer sub-stage, num_pv_stages consumer sub-stages
    comptime SPipelineProducer = RolePipeline[1, True, 1, Self.num_pv_stages]
    comptime SPipelineConsumer = RolePipeline[1, False, 1, Self.num_pv_stages]

    @always_inline
    def producer_s0(self) -> Self.SPipelineProducer:
        """Get S producer for warp group 0."""
        return {
            self.mbar_base + Self.S0_producer_offset,
            self.mbar_base + Self.S0_consumer_offset,
        }

    @always_inline
    def producer_s1(self) -> Self.SPipelineProducer:
        """Get S producer for warp group 1."""
        return {
            self.mbar_base + Self.S1_producer_offset,
            self.mbar_base + Self.S1_consumer_offset,
        }

    @always_inline
    def consumer_s(self, wg_idx: UInt32) -> Self.SPipelineConsumer:
        """Get S consumer for given warp group."""
        return {
            self.mbar_base + Self.S0_producer_offset + wg_idx,
            self.mbar_base + UInt32(Self.num_pv_stages) * wg_idx,
        }

    @always_inline
    def consumer_c0(self) -> ConsumerPipeline[1]:
        return {
            self.mbar_base + Self.C0_offset,
            self.mbar_base + Self.C0_offset + 1,
        }

    @always_inline
    def consumer_c1(self) -> ConsumerPipeline[1]:
        return {
            self.mbar_base + Self.C1_offset,
            self.mbar_base + Self.C1_offset + 1,
        }

    @always_inline
    def producer_c(self, wg_idx: UInt32) -> ProducerPipeline[1]:
        base = UInt32(Self.C0_offset) + 2 * wg_idx
        return {self.mbar_base + base, self.mbar_base + base + 1}

    @always_inline
    def pipeline_order_wait(self, wg_idx: UInt32) -> MBarType:
        return self.mbar_base + Self.order_offset + wg_idx

    @always_inline
    def pipeline_order_arrive(self, wg_idx: UInt32) -> MBarType:
        return self.mbar_base + (Self.order_offset + 1) - wg_idx

    @always_inline
    def q1_wait_mbar(self) -> MBarType:
        comptime assert Self.num_qo == 2, (
            "q1_wait_mbar() requires num_qo == 2; the Q1Sync slot is"
            " collapsed when num_qo == 1."
        )
        return self.mbar_base + Self.Q1SyncIdx

    # K/V/O barrier accessors
    @always_inline("nodebug")
    def get_k_mbars(self) -> MBarType:
        """Returns base pointer for K pipeline barriers."""
        return self.mbar_base + Self.K_offset

    @always_inline("nodebug")
    def get_v_mbars(self) -> MBarType:
        """Returns base pointer for V pipeline barriers.
        In fused mode, returns the same as get_k_mbars (shared pipeline).
        """
        comptime if Self.use_fused_kv:
            return self.mbar_base + Self.K_offset
        else:
            return self.mbar_base + Self.V_offset

    @always_inline("nodebug")
    def combined_p_o_consumer(self, wg_idx: UInt32) -> MBarType:
        """Combined P+O consumer barrier for given warp group.

        Arrived at by BOTH softmax (P ready) and correction (O rescaled).
        Returns S_consumer[0] for wg_idx=0 or wg_idx=1.
        """
        return self.mbar_base + UInt32(Self.num_pv_stages) * wg_idx

    # O pipeline convenience methods
    @always_inline("nodebug")
    def consumer_o(self) -> RolePipeline[2, False, 1, Self.num_pv_stages]:
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
    def producer_o0(self) -> ProducerPipeline[1]:
        """Get O producer for warp group 0."""
        return {
            self.mbar_base + Self.O_producer_offset,
            self.combined_p_o_consumer(0),
        }

    @always_inline("nodebug")
    def producer_o1(self) -> ProducerPipeline[1]:
        """Get O producer for warp group 1."""
        return {
            self.mbar_base + Self.O_producer_offset + 1,
            self.combined_p_o_consumer(1),
        }

    @staticmethod
    @always_inline
    def num_mbars() -> UInt32:
        return UInt32(Self.size)

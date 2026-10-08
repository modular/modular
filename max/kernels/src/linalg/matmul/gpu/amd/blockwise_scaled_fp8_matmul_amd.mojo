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
"""Tiled MFMA blockwise-scaled FP8 dense matmul for AMD gfx950 (CDNA4).

Target: AMD MI350/MI355 (gfx950), MFMA `16x16x128` FP8 (E4M3 OCP), or
`16x16x32` when the K scale block is 64 wide.

Computes `C[m, n] = sum_k A[m, k] * B[n, k] * a_scale[k//K_SCALE, m]
* b_scale[n//N_SCALE, k//K_SCALE]` for blockwise-scaled FP8 with granularity
`(m, n, k) = (1, 128, 128)` or `(1, 64, 64)` and `transpose_b=True` (B is
`[N, K]`, K-major). This mirrors `naive_blockwise_scaled_fp8_matmul_kernel`
(`linalg/fp8_quantization.mojo`) exactly, but tiled.

The one hard idea is fp32 per-K-scale-block scale promotion. A single
epilogue cannot apply the scales because K spans many scale blocks, each
with its own scale. So the K loop steps by `BK == 128` (one DRAM->LDS slab
holding `BK // K_SCALE` scale blocks), MFMA-accumulates each block into a
temporary fp32 fragment `blk_acc` (reset per block), then folds
`promoted += blk_acc * a_scale * b_scale` into a persistent fp32
accumulator before the next block. An MFMA cannot straddle two scale
blocks, so a 128-wide block is one `16x16x128` MFMA and a 64-wide block is
two `16x16x32` MFMAs (`blockwise_fp8_mma_k`); the load pipeline is the same
either way.

Data movement (DRAM->regs->LDS->fragments) reuses the same proven
`RegTileLoader` / `RegTileWriterLDS` / `MmaOp` primitives as
`AMDMatmul` (`amd_matmul.mojo`). The C-fragment lane->(m, n) map used
by the scale promotion and the output store matches `AMDMatmul.run`'s
epilogue: for a 16x16 MFMA each lane owns one row (`lane % 16`) and
`c_frag_size` contiguous columns starting at
`(lane // 16) * c_frag_size`. Because a lane's whole fragment shares one
row `m`, `a_scale` is a single scalar per (lane, m_mma); the 4 columns
sit in one N scale block, so `b_scale` is a single scalar per
(lane, n_mma).

The per-output-tile body lives in `blockwise_scaled_fp8_matmul_amd_tile`
so the grouped (MoE) variant in `blockwise_scaled_fp8_grouped_matmul_amd`
can reuse the identical slab-promotion body and lane->(m, n) scale map --
the correctness-critical part is authored once, not copied.
"""

from std.math import ceildiv
from std.sys import simd_width_of, size_of

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier

from layout import Idx, TensorEngine, TensorLayout, TileTensor
from layout.swizzle import Swizzle
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation

from std.utils import StaticTuple
from std.utils.index import Index
from std.utils.numerics import get_accum_type

from .matmul_mma import MmaOp
from .amd_4wave_split_k_matmul import (
    SplitKWorkspace,
    _split_k_reduce_kernel,
)
from structured_kernels.amd_tile_io import RegTileLoader, RegTileWriterLDS


def blockwise_fp8_mma_k[K_SCALE: Int]() -> Int:
    """Returns the FP8 MFMA K depth for a K scale granularity on gfx950.

    An MFMA cannot straddle two K scale blocks, so a 128-wide block takes the
    full-rate `16x16x128` instruction and a 64-wide block takes `16x16x32`,
    the widest FP8 MFMA that keeps the 16x16 output fragment the scale
    promotion is written for.

    Parameters:
        K_SCALE: K scale granularity (64 or 128).

    Returns:
        128 when `K_SCALE` is a multiple of 128, else 32.
    """
    return 128 if K_SCALE % 128 == 0 else 32


@always_inline
def blockwise_scaled_fp8_matmul_amd_tile[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    accum_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    a_scale_layout: TensorLayout,
    b_scale_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    a_scale_engine: TensorEngine,
    b_scale_engine: TensorEngine,
    *,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    N_SCALE: Int,
    K_SCALE: Int,
    num_splits: Int = 1,
](
    c: TileTensor[mut=True, c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
    a_scales: TileTensor[
        a_scales_type, a_scale_layout, ImmutAnyOrigin, Engine=a_scale_engine
    ],
    b_scales: TileTensor[
        b_scales_type, b_scale_layout, ImmutAnyOrigin, Engine=b_scale_engine
    ],
    m_block: Int,
    n_block: Int,
    a_scale_col: Int,
    M: Int,
    k_split_id: Int = 0,
):
    """Computes one `BM x BN` output tile of a blockwise-scaled FP8 matmul.

    Shared by the dense and grouped (MoE) launchers. The dense kernel
    calls it once per `(m_block, n_block)` grid tile with `a_scale_col=0`;
    the grouped kernel calls it per expert with `a_scale_col=a_start_row`,
    so `a_scales` can span all tokens `[K // K_SCALE, total_tokens]` and
    still be read at the row this expert's tile owns.

    With `num_splits > 1` this computes one K-band of an inter-block
    split-K launch instead of the whole K range: only the slabs in
    `[k_split_id * SLABS_PER_SPLIT, (k_split_id + 1) * SLABS_PER_SPLIT)`
    are accumulated, and the result is stored as this split's fp32
    partial at row offset `k_split_id * M` of the stacked
    `(num_splits * M, N)` workspace the dense launcher passes as `c`. A
    separate reduce kernel sums the partials. `num_splits == 1` (the
    default; also what the grouped and batched callers use) leaves the
    body byte-identical to the unsplit kernel.

    Parameters:
        c_type: Output element type.
        a_type: A input element type (`float8_e4m3fn`).
        b_type: B input element type (`float8_e4m3fn`).
        a_scales_type: A scales element type (`float32`).
        b_scales_type: B scales element type (`float32`).
        accum_type: Accumulation element type (`float32`).
        c_layout: Layout of `c`.
        a_layout: Layout of `a`.
        b_layout: Layout of `b`.
        a_scale_layout: Layout of `a_scales`.
        b_scale_layout: Layout of `b_scales`.
        c_engine: Tensor engine of `c`.
        a_engine: Tensor engine of `a`.
        b_engine: Tensor engine of `b`.
        a_scale_engine: Tensor engine of `a_scales`.
        b_scale_engine: Tensor engine of `b_scales`.
        BM: Block tile rows (M).
        BN: Block tile cols (N).
        BK: Block tile K (a whole number of K scale blocks).
        WM: Warp tile rows (M).
        WN: Warp tile cols (N).
        MMA_M: MFMA M (16).
        MMA_N: MFMA N (16).
        MMA_K: MFMA K (128 or 32; must divide `K_SCALE`).
        N_SCALE: N scale granularity (64 or 128).
        K_SCALE: K scale granularity (64 or 128).
        num_splits: Inter-block split-K factor (1 = no split).

    Args:
        c: Output tile `[M, N]` — for `num_splits > 1` the stacked
            `(num_splits * M, N)` float32 workspace.
        a: A input `[M, K]`, K-major.
        b: B input `[N, K]`, K-major (`transpose_b`).
        a_scales: A per-block scales `[K // K_SCALE, cols]`, column indexed
            by `a_scale_col + m`.
        b_scales: B per-block scales `[N // N_SCALE, K // K_SCALE]`.
        m_block: Output tile row index (M / `BM`).
        n_block: Output tile col index (N / `BN`).
        a_scale_col: Base column into `a_scales` for this tile's rows.
        M: Number of valid rows in `c` / `a` (dynamic; OOB masked).
        k_split_id: K-band index of this CTA (`num_splits > 1` only).
    """
    comptime assert accum_type == .float32
    comptime assert (
        MMA_M == 16 and MMA_N == 16
    ), "scale-promotion fragment map assumes MFMA 16x16"
    comptime assert (
        BK % K_SCALE == 0
    ), "a BK slab must hold whole K scale blocks"
    comptime assert (
        K_SCALE % MMA_K == 0
    ), "an MFMA cannot straddle two K scale blocks"
    comptime assert (
        N_SCALE % MMA_N == 0
    ), "a lane's output columns must share one N scale block"

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N
    comptime c_frag_size = MMA_M * MMA_N // WARP_SIZE
    comptime num_c_regs = num_m_mmas * num_n_mmas * c_frag_size
    comptime num_threads = num_warps_m * num_warps_n * WARP_SIZE

    # DRAM->reg->LDS tiling keys off the host SIMD width (like `AMDMatmul`),
    # so the loader and `copy_blocked` agree for any k-tile divisor.
    comptime simd_width = simd_width_of[a_type]()
    comptime assert K_SCALE % simd_width == 0
    comptime assert (BM * BK) % num_threads == 0
    comptime assert (BN * BK) % num_threads == 0
    comptime load_thread_cols = BK // simd_width
    comptime load_thread_rows = num_threads // load_thread_cols
    comptime a_reg_elems = BM * BK // num_threads
    comptime b_reg_elems = BN * BK // num_threads

    # One k-tile per K scale block: each `mma_op.mma[k]` contracts one scale's
    # columns (promotion runs per k-tile); the A/B K permutation stays within it.
    comptime k_tile_size = K_SCALE
    comptime k_group_size = K_SCALE // MMA_K
    comptime num_k_tiles = BK // k_tile_size

    comptime N = type_of(c).static_shape[1]
    comptime K = type_of(a).static_shape[1]
    comptime assert N > 0 and K > 0, "N and K must be static"
    # K need not be a multiple of BK: the straddling k-tile's tail is zeroed
    # below, k-tiles past K are skipped, scales use `ceildiv(K, K_SCALE)` blocks.
    comptime NUM_SLABS = ceildiv(K, BK)
    comptime TAIL_K_TILES = ceildiv(K - (NUM_SLABS - 1) * BK, k_tile_size)

    # Split-K banding: split `k_split_id` owns its whole BK slabs; scales index
    # by absolute `k_blk`, so no re-basing. `num_splits == 1` keeps full range.
    comptime SLABS_PER_SPLIT = NUM_SLABS // num_splits
    comptime assert (
        SLABS_PER_SPLIT * num_splits == NUM_SLABS
    ), "num_splits must evenly divide the K slab count"

    # Warp index decomposition (matches AMDMatmul's divmod order).
    var warp_mn, warp_n = divmod(Int(warp_id()), num_warps_n)
    var warp_m = warp_mn % num_warps_m

    var a_gmem = a.bitcast[a_type]()
    var b_gmem = b.bitcast[a_type]()

    var a_smem = stack_allocation[a_type, address_space=.SHARED](
        row_major[BM, BK]()
    )
    var b_smem = stack_allocation[a_type, address_space=.SHARED](
        row_major[BN, BK]()
    )

    var a_load_reg = stack_allocation[a_type, address_space=.LOCAL](
        row_major[1, a_reg_elems]()
    )
    var b_load_reg = stack_allocation[a_type, address_space=.LOCAL](
        row_major[1, b_reg_elems]()
    )

    var a_blockrow = a_gmem.tile[BM, K](m_block, 0)
    var b_blockrow = b_gmem.tile[BN, K](n_block, 0)

    comptime load_layout = row_major[load_thread_rows, load_thread_cols]()
    var a_loader = RegTileLoader[a_type, load_layout](
        a_blockrow, bounds_from=a_gmem
    )
    var b_loader = RegTileLoader[a_type, load_layout](
        b_blockrow, bounds_from=b_gmem
    )

    # XOR-16 LDS swizzle (`Swizzle(3,4,4)`): XORs bits [8,11) into [4,7), so
    # it is an IDENTITY at every shipped config (all store/`load_frag`
    # indices are < 256) -- LDS is plain row-major here despite the name.
    # Gated to K % BK == 0 (tail path unswizzled).
    comptime swizzle = Optional[Swizzle](
        Swizzle(3, 4, 4)
    ) if K % BK == 0 else Optional[Swizzle]()

    var mma_op = MmaOp[
        out_type=accum_type,
        in_type=a_type,
        shape=Index(MMA_M, MMA_N, MMA_K),
        k_group_size=k_group_size,
        num_k_tiles=num_k_tiles,
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas,
        swizzle=swizzle,
    ]()

    # Persistent fp32 accumulator that scale-promotion folds into.
    var promoted = stack_allocation[accum_type, address_space=.LOCAL](
        row_major[num_m_mmas, num_n_mmas * c_frag_size]()
    )
    comptime for i in range(num_c_regs):
        promoted.raw_store(i, Scalar[accum_type](0))

    # Per-lane output fragment coordinates (constant across slabs).
    var lane = Int(lane_id())
    var lane_group, lane_row = divmod(lane, MMA_M)
    var warp_tile_m = m_block * BM + warp_m * WM
    var warp_tile_n = n_block * BN + warp_n * WN

    var blk_acc = mma_op.accum_tile()

    for kb in range(NUM_SLABS):
        # Split-K band filter: keep only this split's slabs (compiled out for
        # `num_splits == 1`). `kb`/`k_split_id` are CTA-uniform, so no barrier divergence.
        comptime if num_splits > 1:
            if (
                kb < k_split_id * SLABS_PER_SPLIT
                or kb >= (k_split_id + 1) * SLABS_PER_SPLIT
            ):
                continue
        var a_block = a_blockrow.tile[BM, BK](0, kb)
        var b_block = b_blockrow.tile[BN, BK](0, kb)
        a_loader.load(a_load_reg, a_block.vectorize[1, simd_width]())
        b_loader.load(b_load_reg, b_block.vectorize[1, simd_width]())

        RegTileWriterLDS[
            load_layout, swizzle=swizzle, num_threads=num_threads
        ].copy_blocked[k_tile_size](a_smem, a_load_reg)
        RegTileWriterLDS[
            load_layout, swizzle=swizzle, num_threads=num_threads
        ].copy_blocked[k_tile_size](b_smem, b_load_reg)
        barrier()

        # Partial final K-slab: buffer OOB clamp only zeros past the last in-range
        # row, so for M>1 zero the straddling k-tile's tail columns (block-uniform; off when K % BK == 0).
        comptime if K % BK != 0:
            comptime k_valid = K % BK
            comptime kt_partial = k_valid // k_tile_size
            comptime c_valid = k_valid % k_tile_size
            comptime if c_valid != 0:
                if kb == NUM_SLABS - 1:
                    comptime n_oob = k_tile_size - c_valid
                    var a_tail = TileTensor(
                        a_smem.ptr + kt_partial * BM * k_tile_size,
                        row_major[BM, k_tile_size](),
                    )
                    var b_tail = TileTensor(
                        b_smem.ptr + kt_partial * BN * k_tile_size,
                        row_major[BN, k_tile_size](),
                    )
                    var tid = Int(warp_id()) * WARP_SIZE + Int(lane_id())
                    for i in range(tid, BM * n_oob, num_threads):
                        var row, col = divmod(i, n_oob)
                        a_tail.store_linear(
                            Index(row, c_valid + col),
                            Scalar[a_type](0),
                        )
                    for i in range(tid, BN * n_oob, num_threads):
                        var row, col = divmod(i, n_oob)
                        b_tail.store_linear(
                            Index(row, c_valid + col),
                            Scalar[a_type](0),
                        )
                    barrier()

        comptime for k in range(num_k_tiles):
            var a_blk = TileTensor(
                a_smem.ptr + k * BM * k_tile_size,
                row_major[BM, k_tile_size](),
            )
            var b_blk = TileTensor(
                b_smem.ptr + k * BN * k_tile_size,
                row_major[BN, k_tile_size](),
            )
            mma_op.load_frag[k](
                a_blk.tile[WM, k_tile_size](warp_m, 0),
                b_blk.tile[WN, k_tile_size](warp_n, 0),
            )
        barrier()

        comptime for k in range(num_k_tiles):
            # A k-tile entirely past K (final slab) has no scale block; skip it.
            # `k < TAIL_K_TILES` is comptime, so it folds away for full k-tiles.
            if k < TAIL_K_TILES or kb < NUM_SLABS - 1:
                var k_blk = kb * num_k_tiles + k

                comptime for i in range(num_c_regs):
                    blk_acc.raw_store(i, Scalar[accum_type](0))
                mma_op.mma[k]()

                # fp32 per-block promotion: `a_scale[k_blk, m]` one scalar per row,
                # `b_scale[nb, k_blk]` one per n-block (c_frag_size columns share `nb`).
                comptime for m_mma in range(num_m_mmas):
                    var m = warp_tile_m + m_mma * MMA_M + lane_row
                    var m_idx = min(m, M - 1)
                    var a_s = rebind[Scalar[a_scales_type]](
                        a_scales.load_linear(Index(k_blk, a_scale_col + m_idx))
                    ).cast[accum_type]()

                    comptime for n_mma in range(num_n_mmas):
                        var n0 = (
                            warp_tile_n
                            + n_mma * MMA_N
                            + lane_group * c_frag_size
                        )
                        var nb = min(n0, N - 1) // N_SCALE
                        var b_s = rebind[Scalar[b_scales_type]](
                            b_scales.load_linear(Index(nb, k_blk))
                        ).cast[accum_type]()
                        var scale = a_s * b_s

                        comptime off = (
                            m_mma * num_n_mmas * c_frag_size
                            + n_mma * c_frag_size
                        )
                        var blk_v = blk_acc.raw_load[width=c_frag_size](off)
                        var prom_v = promoted.raw_load[width=c_frag_size](off)
                        promoted.raw_store[width=c_frag_size](
                            off, prom_v + blk_v * scale
                        )

    # Output store with per-element OOB masking. Split-K rows live at
    # `k_split_id * M` in the stacked (num_splits*M, N) workspace `c`; `num_splits == 1` keeps it byte-identical.
    var c_row_base = 0
    comptime if num_splits > 1:
        c_row_base = k_split_id * M
    comptime for m_mma in range(num_m_mmas):
        var m = warp_tile_m + m_mma * MMA_M + lane_row
        if m < M:
            comptime for n_mma in range(num_n_mmas):
                var n0 = warp_tile_n + n_mma * MMA_N + lane_group * c_frag_size
                comptime off = (
                    m_mma * num_n_mmas * c_frag_size + n_mma * c_frag_size
                )
                var v = promoted.raw_load[width=c_frag_size](off).cast[c_type]()
                comptime for e in range(c_frag_size):
                    var col = n0 + e
                    if col < N:
                        c.store_linear(Index(c_row_base + m, col), v[e])


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((BM // WM) * (BN // WN) * WARP_SIZE)
    )
)
@__name(
    t"blockwise_scaled_fp8_matmul_amd_BM{BM}_BN{BN}_BK{BK}_WM{WM}_WN{WN}_SK{num_splits}"
)
def blockwise_scaled_fp8_matmul_amd_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    accum_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    a_scale_layout: TensorLayout,
    b_scale_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    a_scale_engine: TensorEngine,
    b_scale_engine: TensorEngine,
    *,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    N_SCALE: Int,
    K_SCALE: Int,
    num_splits: Int = 1,
](
    c: TileTensor[mut=True, c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
    a_scales: TileTensor[
        a_scales_type, a_scale_layout, ImmutAnyOrigin, Engine=a_scale_engine
    ],
    b_scales: TileTensor[
        b_scales_type, b_scale_layout, ImmutAnyOrigin, Engine=b_scale_engine
    ],
):
    """Computes a `BM x BN` output tile of a blockwise-scaled FP8 matmul.

    Thin wrapper over `blockwise_scaled_fp8_matmul_amd_tile`: `block_idx.y`
    is the M-tile, `block_idx.x` the N-tile, `a_scales` spans exactly this
    matmul's `M` rows so the column base is `0`.

    Parameters:
        c_type: Output element type.
        a_type: A input element type (must be `float8_e4m3fn`).
        b_type: B input element type (must be `float8_e4m3fn`).
        a_scales_type: A scales element type (`float32`).
        b_scales_type: B scales element type (`float32`).
        accum_type: Accumulation element type (`float32`).
        c_layout: Layout of `c`.
        a_layout: Layout of `a`.
        b_layout: Layout of `b`.
        a_scale_layout: Layout of `a_scales`.
        b_scale_layout: Layout of `b_scales`.
        c_engine: Tensor engine of `c`.
        a_engine: Tensor engine of `a`.
        b_engine: Tensor engine of `b`.
        a_scale_engine: Tensor engine of `a_scales`.
        b_scale_engine: Tensor engine of `b_scales`.
        BM: Block tile rows (M).
        BN: Block tile cols (N).
        BK: Block tile K (a whole number of K scale blocks).
        WM: Warp tile rows (M).
        WN: Warp tile cols (N).
        MMA_M: MFMA M (16).
        MMA_N: MFMA N (16).
        MMA_K: MFMA K (128 or 32; must divide `K_SCALE`).
        N_SCALE: N scale granularity (64 or 128).
        K_SCALE: K scale granularity (64 or 128).
        num_splits: Inter-block split-K factor (1 = no split).

    Args:
        c: Output tile `[M, N]` — for `num_splits > 1` the stacked
            `(num_splits * M, N)` float32 workspace.
        a: A input `[M, K]`, K-major.
        b: B input `[N, K]`, K-major (`transpose_b`).
        a_scales: A per-block scales `[K // K_SCALE, M]`.
        b_scales: B per-block scales `[N // N_SCALE, K // K_SCALE]`.
    """
    # K-band id from grid.z, split mode only: `num_splits == 1` must not read
    # block_idx.z (this kernel may run as a device fn under a different grid.z).
    var split_id = Int(block_idx.z) if num_splits > 1 else 0
    # `c` is the stacked (num_splits*M, N) workspace when splitting, so the
    # true row count comes from `a`.
    var M = Int(a.dim[0]())
    blockwise_scaled_fp8_matmul_amd_tile[
        c_type,
        a_type,
        b_type,
        a_scales_type,
        b_scales_type,
        accum_type,
        c_layout,
        a_layout,
        b_layout,
        a_scale_layout,
        b_scale_layout,
        c_engine,
        a_engine,
        b_engine,
        a_scale_engine,
        b_scale_engine,
        BM=BM,
        BN=BN,
        BK=BK,
        WM=WM,
        WN=WN,
        MMA_M=MMA_M,
        MMA_N=MMA_N,
        MMA_K=MMA_K,
        N_SCALE=N_SCALE,
        K_SCALE=K_SCALE,
        num_splits=num_splits,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        Int(block_idx.y),
        Int(block_idx.x),
        0,
        M,
        split_id,
    )


def _pick_split_k_num_splits[
    K: Int, BK: Int, base_ctas: Int, cta_cap: Int
]() -> Int:
    """Comptime split-K factor for the small-M decode regime.

    Picks the largest `num_splits` that keeps the split legal and the total
    CTA count `base_ctas * num_splits` at or under `cta_cap`. Cap here, not
    with `min(pick(), cap)`: that can name a factor that does not divide K.
    Legality (mirrors `blockwise_scaled_fp8_matmul_amd_tile`'s split assert):
      * `K % BK == 0` (asserted) and `num_splits` divides `K // BK`;
      * each split owns at least 2 BK slabs (`K // num_splits >= 2*BK`):
        with 1 slab per split the separate reduce launch dominates the
        tiny per-split matmul, which regresses small-K shapes.
    Profitability: the split must at least quadruple the CTA count
    (`num_splits >= 4`). The deterministic reduce launch adds ~2us of
    fixed per-call cost; a 2x CTA multiplier does not pay for it when the
    plain launch already sits near the launch-latency floor — measured on
    kv_b (N=3584, K=512: 4 slabs, s=2): 5.8us -> 7.8us, a regression.
    Returns 1 when no split qualifies (the caller keeps the plain launch).

    Parameters:
        K: Static K dimension.
        BK: Block tile K (a whole number of K scale blocks).
        base_ctas: CTAs the plain launch already runs at this M bucket —
            `ceildiv(N, BN) * ceildiv(M, BM)`.
        cta_cap: CTA budget, ~2 waves (all CUs + one for latency hiding).
    """
    comptime assert K % BK == 0, "split-K requires K % BK == 0"
    comptime total_slabs = K // BK
    var best = 1
    comptime for s in range(4, total_slabs + 1):
        comptime if (
            total_slabs % s == 0
            and total_slabs // s >= 2
            and base_ctas * s <= cta_cap
        ):
            best = s
    return best


def _launch_blockwise_split_k[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    N_SCALE: Int,
    K_SCALE: Int,
    num_splits: Int,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[mut=False, a_scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[mut=False, b_scales_type, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Single-launch split-K blockwise-FP8 matmul + reduce.

    Mirrors `amd_4wave_split_k_matmul` / `_launch_block_scaled_split_k`:
    each of the `grid.z = num_splits` K-bands accumulates its fp32 partial
    (scales are already folded per K-scale-block, so a band's partial is
    exact) into its own `[M, N]` region of a stacked `(num_splits * M, N)`
    float32 workspace, then `_split_k_reduce_kernel` — enqueued on the same
    stream, so naturally serialized — sums the partials and casts to `c`'s
    dtype. No atomics: the reduce is deterministic and beats RMW contention
    at these sizes.
    """
    comptime N = type_of(c).static_shape[1]
    var M = Int(a.dim[0]())
    var elems_per_split = M * N

    var workspace = SplitKWorkspace[num_splits](ctx, elems_per_split)
    # Row-major (num_splits*M, N) f32 workspace: split `s`'s (m, n) at
    # `scratch[s*M*N + m*N + n]`, the flat layout `_split_k_reduce_kernel` reduces.
    var ws_tile = TileTensor(
        workspace.scratch.unsafe_ptr(),
        row_major((Int(num_splits * M), Idx[N])),
    )

    comptime kernel = blockwise_scaled_fp8_matmul_amd_kernel[
        DType.float32,
        a_type,
        b_type,
        a_scales_type,
        b_scales_type,
        DType.float32,
        type_of(ws_tile).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_scales).LayoutType,
        type_of(b_scales).LayoutType,
        type_of(ws_tile).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        type_of(a_scales).Engine,
        type_of(b_scales).Engine,
        BM=BM,
        BN=BN,
        BK=BK,
        WM=WM,
        WN=WN,
        MMA_M=MMA_M,
        MMA_N=MMA_N,
        MMA_K=MMA_K,
        N_SCALE=N_SCALE,
        K_SCALE=K_SCALE,
        num_splits=num_splits,
    ]

    comptime num_threads = (BM // WM) * (BN // WN) * WARP_SIZE
    ctx.enqueue_function[kernel](
        ws_tile,
        a,
        b,
        a_scales,
        b_scales,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), num_splits),
        block_dim=(num_threads, 1, 1),
    )

    # Reduce + cast on the same stream (serialized after the matmul): sums the
    # `num_splits` f32 partials at flat `tid` and casts to `c`'s dtype.
    comptime block_dim_x: Int = 256
    var total_elems = M * N
    var num_blocks = ceildiv(total_elems, block_dim_x)
    comptime reduce_kernel = _split_k_reduce_kernel[num_splits, c_type]
    ctx.enqueue_function[reduce_kernel](
        workspace.scratch.unsafe_ptr(),
        c.ptr,
        Int32(total_elems),
        Int32(elems_per_split),
        Int32(N),
        grid_dim=(num_blocks,),
        block_dim=(block_dim_x,),
    )
    # Keep the workspace alive until both kernels are enqueued.
    _ = workspace^


def blockwise_scaled_fp8_matmul_amd[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    *,
    transpose_b: Bool,
    BM: Int = 64,
    BN: Int = 64,
    BK: Int = 128,
    WM: Int = 64,
    WN: Int = 64,
    N_SCALE: Int = 128,
    K_SCALE: Int = 128,
    accum_type: DType = get_accum_type[c_type](),
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[mut=False, a_scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[mut=False, b_scales_type, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Enqueues the tiled blockwise-scaled FP8 dense matmul for gfx950.

    Parameters:
        c_type: Output element type.
        a_type: A input element type (`float8_e4m3fn`).
        b_type: B input element type (`float8_e4m3fn`).
        a_scales_type: A scales element type (`float32`).
        b_scales_type: B scales element type (`float32`).
        transpose_b: Must be `True` (B is `[N, K]`).
        BM: Block tile rows (M). Default 64 (1 warp/CTA at WM=64): smaller
            CTAs pack more of them per CU, so a K-slab's `barrier()` stall
            in one CTA overlaps another CTA's independent MFMA/LDS work
            instead of idling the whole CU. Measured 23-48% faster than
            128 across the GLM dense/grouped production shapes.
        BN: Block tile cols (N). Same reasoning as `BM`.
        BK: Block tile K (a whole number of K scale blocks).
        WM: Warp tile rows (M).
        WN: Warp tile cols (N).
        N_SCALE: N scale granularity (64 or 128).
        K_SCALE: K scale granularity (64 or 128).
        accum_type: Accumulation element type (`float32`).

    Args:
        c: Output `[M, N]`.
        a: A input `[M, K]`, K-major.
        b: B input `[N, K]`, K-major (`transpose_b`).
        a_scales: A per-block scales `[K // K_SCALE, M]`.
        b_scales: B per-block scales `[N // N_SCALE, K // K_SCALE]`.
        ctx: Device context.

    Decode regime (small M): the launch geometry is
    `(ceildiv(N, BN), ceildiv(M, BM))`, which at M <= 64 is one or a few
    M-tiles — only `ceildiv(N, BN)` CTAs against ~2 waves of CUs. This
    launcher then transparently splits K across `grid.z` (fp32 partials +
    deterministic reduce; see `_launch_blockwise_split_k`) with a
    comptime-picked `num_splits` that lifts the CTA count toward
    `sm_count * 2`. No tile-shape change can fix this: CTAs cannot be
    manufactured in the M direction at M <= 64, only in K.
    """
    comptime assert transpose_b, "transpose_b must be True"
    comptime assert (
        a_type == .float8_e4m3fn and b_type == .float8_e4m3fn
    ), "tiled blockwise FP8 matmul supports only float8_e4m3fn inputs"
    comptime assert (
        a_scales_type == .float32 and b_scales_type == .float32
    ), "tiled blockwise FP8 matmul supports only float32 scales"
    comptime assert accum_type == .float32

    comptime MMA_M = 16
    comptime MMA_N = 16
    comptime MMA_K = blockwise_fp8_mma_k[K_SCALE]()

    var M = Int(c.dim[0]())
    comptime N = type_of(c).static_shape[1]
    comptime K = type_of(a).static_shape[1]

    if M == 0 or N == 0 or K == 0:
        return

    # === Decode-regime split-K (small M) ===
    # At small M the problem is launch-starved, not bandwidth-bound: the
    # plain geometry runs `ceildiv(N, BN) * ceildiv(M, BM)` CTAs on ~256 CUs,
    # far short of filling them. Two comptime split factors cover the M
    # buckets the gate below admits:
    #   * `_sk_splits_1m`: M fits one M-tile (M <= BM) — the CTAs-per-split
    #     estimate is exactly `ceildiv(N, BN)`.
    #   * `_sk_splits_mt`: BM < M <= SK_MAX_M — fold the bucket's worst-case
    #     M-tile count into the estimate so the split does not overshoot
    #     the CTA cap with 2-4 M-tiles in flight.
    # Both return 1 (no split) when the shape already fills the GPU or no
    # legal factor exists, so the plain launch below is unchanged there.
    # The cap is `sm_count * 2` — every CU plus a second wave for latency
    # hiding.
    comptime _gpu = ctx.default_device_info
    comptime SK_CTA_WAVES = 2
    comptime _sk_cta_cap = _gpu.sm_count * SK_CTA_WAVES
    # Decode band only: above it the f32 workspace traffic outweighs the
    # parallelism win, so larger M keeps the single-launch path.
    comptime SK_MAX_M = 64
    comptime SK_MAX_WORKSPACE_BYTES = 128 * 1024 * 1024
    comptime _sk_splits_1m = _pick_split_k_num_splits[
        K, BK, ceildiv(N, BN), _sk_cta_cap
    ]() if K % BK == 0 else 1
    comptime _sk_splits_mt = _pick_split_k_num_splits[
        K, BK, ceildiv(N, BN) * ceildiv(SK_MAX_M, BM), _sk_cta_cap
    ]() if K % BK == 0 else 1

    comptime if _sk_splits_1m > 1:
        comptime _ws_max_m = SK_MAX_WORKSPACE_BYTES // (
            _sk_splits_1m * N * size_of[DType.float32]()
        )
        if M <= min(BM, _ws_max_m):
            _launch_blockwise_split_k[
                c_type,
                a_type,
                b_type,
                a_scales_type,
                b_scales_type,
                BM=BM,
                BN=BN,
                BK=BK,
                WM=WM,
                WN=WN,
                MMA_M=MMA_M,
                MMA_N=MMA_N,
                MMA_K=MMA_K,
                N_SCALE=N_SCALE,
                K_SCALE=K_SCALE,
                num_splits=_sk_splits_1m,
            ](c, a, b, a_scales, b_scales, ctx)
            return
    comptime if _sk_splits_mt > 1:
        comptime _ws_max_m = SK_MAX_WORKSPACE_BYTES // (
            _sk_splits_mt * N * size_of[DType.float32]()
        )
        if M <= min(SK_MAX_M, _ws_max_m):
            _launch_blockwise_split_k[
                c_type,
                a_type,
                b_type,
                a_scales_type,
                b_scales_type,
                BM=BM,
                BN=BN,
                BK=BK,
                WM=WM,
                WN=WN,
                MMA_M=MMA_M,
                MMA_N=MMA_N,
                MMA_K=MMA_K,
                N_SCALE=N_SCALE,
                K_SCALE=K_SCALE,
                num_splits=_sk_splits_mt,
            ](c, a, b, a_scales, b_scales, ctx)
            return

    comptime kernel = blockwise_scaled_fp8_matmul_amd_kernel[
        c_type,
        a_type,
        b_type,
        a_scales_type,
        b_scales_type,
        accum_type,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_scales).LayoutType,
        type_of(b_scales).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        type_of(a_scales).Engine,
        type_of(b_scales).Engine,
        BM=BM,
        BN=BN,
        BK=BK,
        WM=WM,
        WN=WN,
        MMA_M=MMA_M,
        MMA_N=MMA_N,
        MMA_K=MMA_K,
        N_SCALE=N_SCALE,
        K_SCALE=K_SCALE,
    ]

    comptime num_threads = (BM // WM) * (BN // WN) * WARP_SIZE
    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        a_scales,
        b_scales,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
        block_dim=(num_threads, 1, 1),
    )

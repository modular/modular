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
"""Blockwise-scaled FP8 dense matmul for NVIDIA parts without `tcgen05`.

`blockwise_scaled_fp8_with_epilogue` sends sm_100 (B100/B200/B300) to the
`tcgen05` structured kernel and AMD gfx950 to the MFMA tiled kernel; every
other NVIDIA part -- GB10 (sm_121a) today -- fell through to the naive kernel,
which is one thread per output element with scalar loads. At M=1 that streams
the weights at 9 GB/s on a 250 GB/s part.

This file is the missing middle: a tiled, bandwidth-first kernel with the same
mathematics as the naive one.

Math
----
For each 128-wide k block `kb` the kernel forms the *full-precision fp32 block
sum* `sum_k a[m,k] * b[n,k]` over that block and folds the two block scales in
once:

    acc[m] += block_sum(kb) * (a_scale[kb, m] * b_scale[n // 128, kb])

The scales are never folded into the fp8 values, so the result is the same
arithmetic the naive kernel performs, only summed in a different order (a
`_CHUNK`-lane SIMD reduction instead of a scalar chain).

Shape of the work
-----------------
One CTA owns one (`BN`-column n block, m block of `_BM` rows) of the output,
times one split of the K range. It walks K in 128-wide slabs, staging each `b`
slab as fp8 in shared memory and each `a` slab as fp32 (every thread reads all
`_BM` rows of it, so a broadcast-shared copy is the cheap form).

Bandwidth, not math, is the constraint: at M=1 the kernel reads the weight
matrix once and does 8 FMAs per weight byte. Three things are arranged so the
global loads never stall the CTA:

* every thread stages the `b` slab, not only the `BN` threads that own an
  output column -- two warps of staging became eight, which is where the loads
  in flight per CTA come from,
* the staging is one slab deep: the next slab's global reads are issued into
  registers before the current slab's MACs, so they are in flight while the
  CTA multiplies, and a two-buffer shared ring lets them be written out
  without waiting for a second barrier,
* the split-K factor is chosen at launch: while the grid has fewer CTAs than
  the device has SMs, the K range is split 2 or 4 ways and the fp32 partial
  sums are added by a second launch. Below one CTA per SM the throughput of
  this kernel is linear in CTA count, and N=5120 at M=1 with 128-wide tiles
  is only 40 CTAs on GB10's 48 SMs.

Global and shared transactions are 8 bytes (`_CHUNK` fp8 values) and the
shared `b` rows are padded by 8 bytes to spread the banks; there is no
`ldmatrix` swizzle on this tile. Nothing here dequantizes the weights: `b` is
read as fp8 and converted in registers.

Decode
------
At `M <= 4` neither tile above is the right shape: the 8-row tile multiplies
seven clamped rows for every real one and pays a shared-memory round trip and
a barrier per slab. `blockwise_scaled_fp8_gemv_nvidia_kernel` streams the
weights straight into registers instead -- 16-byte loads, 8 output columns per
CTA, 4 warps splitting K -- and reaches 86-90% of GB10's 250 GB/s wall on the
Qwen3.8-27B-FP8 projections, against 59-68% for the 8-row tile.

Prefill
-------
The 8-row tile is a small-batch tile. `_BM = 8` rows means an M-token multiply
walks the weight matrix `M / 8` times, and each MAC carries a shared-memory
load, so it is bandwidth optimal only while M is small: at M=4088 it re-reads
28.75 GB 511 times.

`blockwise_scaled_fp8_matmul_nvidia_tc_kernel` is the prefill tile: 128 rows by
128 columns, arithmetic on the tensor cores (`mma.sync.m16n8k32` for fp8), one
thread per 64-element warp-tile row band, 64 fp32 accumulators in registers.
The block scales are folded with a running ratio instead of a second
accumulator set -- see that kernel's docstring for the arithmetic.
"""

from std.math import ceildiv
from std.sys import size_of
from std.utils import StaticTuple

from layout import Coord, TensorLayout, TileTensor, row_major
from layout.swizzle import Swizzle, make_ldmatrix_swizzle
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape
from layout.tile_tensor import stack_allocation
from max.algorithm import elementwise
from max.gpu import WARP_SIZE, block_idx, lane_id, warp_id
from max.gpu.host import DeviceContext, GridBlockLaunchConfig
from max.gpu.host.info import GPUInfo, _is_sm12x_gpu
from max.gpu.primitives import warp
from max.gpu.sync import barrier

from ...utils import ElementwiseEpilogueFn, no_epilogue_fn

comptime _BM = 8
"""Output rows per CTA. The m grid dimension re-reads `b`, so this is the row
count served by one pass over the weights."""

comptime _BK = 128
"""K depth of one staged slab: one `a_scale`/`b_scale` block."""

comptime _NUM_THREADS = 256
"""Threads per CTA. Bandwidth sets this, not math: the whole CTA stages the
`b` slab and the first `BN` threads are the ones that own an output column for
all `_BM` rows."""

comptime _CHUNK = 8
"""FP8 values per global and shared transaction (8 bytes)."""

comptime _B_ROW = _BK + 8
"""Padded shared row stride of a `b` slab, in bytes: 34 banks, not 32."""

comptime _RING = 2
"""Shared `b`/`a` slab buffers. At step `st` the CTA multiplies buffer
`st % 2` and publishes the prefetched slab into `(st + 1) % 2`, whose previous
occupant was multiplied at `st - 1` and retired by that step's barrier."""

comptime _K_CHUNKS = _BK // _CHUNK
"""Chunks per shared row."""

comptime _MAX_SPLIT = 4
"""Largest split-K factor the launcher will pick."""

comptime _TC_BM = 128
"""Output rows per CTA on the tensor-core path. This is the number that
removes the prefill wall: the weight slab a CTA stages serves 128 rows, not
8."""

comptime _TC_BN = 128
"""Output columns per CTA on the tensor-core path, and therefore one whole
`b_scale` block: `b_scale[n0 // 128, kb]` is a single scalar per CTA."""

comptime _TC_BK = 128
"""K depth of one staged slab on the tensor-core path. A slab is exactly one
blockwise-scale block, so the K loop is `ceildiv(K, 128)` iterations."""

comptime _TC_NUM_THREADS = 256
"""Threads per CTA on the tensor-core path: 8 warps in a 2 (m) x 4 (n)
grid over the 128x128 tile."""

comptime _TC_WM = 64
"""Rows of the block tile one warp owns."""

comptime _TC_WN = 32
"""Columns of the block tile one warp owns."""

comptime _TC_VEC = 16
"""FP8 values per staged transaction: one 16-byte vector, which is also the
granularity of the `ldmatrix` swizzle below."""

comptime _TC_MIN_M = 16
"""Smallest `M` the tensor-core path takes. Below it the 8-row SIMT tile owns
fewer wasted rows than the 128-row tile does: at M=16 the tensor-core tile
computes eight times the needed rows, which is still cheaper than two passes
over the weights, but at M=1 it would be pure waste."""


comptime _GV_VEC = 16
"""FP8 values per weight load on the decode GEMV: one 16-byte vector."""

comptime _GV_COLS = 8
"""Output columns one CTA owns on the decode GEMV. Each lane's `a` vector is
loaded once and multiplied into all of them, and `_GV_COLS` divides 128, so a
CTA's columns share one `b_scale` block."""

comptime _GV_UNROLL = 1
"""K steps one GEMV warp issues before it multiplies: the loads of all of them
are in flight together, `_GV_COLS * _GV_UNROLL` 16-byte vectors per lane."""

comptime _GV_WARPS = 4
"""Warps per CTA on the decode GEMV. They split the K range of the CTA's
columns between them, so the number of warps streaming the weights is
`N / _GV_COLS * _GV_WARPS` whatever the shape."""

comptime _GV_MAX_M = 4
"""Largest `M` the decode GEMV takes. Every row re-reads nothing -- the weights
stream once whatever `M` is -- but each row adds `_GV_COLS` accumulators per
lane, and past a few rows the 8-row tile's shared `a` slab is the cheaper
form."""


def _nvidia_blockwise_tiled_ok[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    transpose_b: Bool,
    scales_granularity_mnk: Coord[Int, Int, Int],
    device_info: GPUInfo,
]() -> Bool:
    """Gate for the tiled blockwise-FP8 dense matmul on NVIDIA parts without
    tcgen05.

    Extracted so a test can assert the gate directly instead of only
    exercising it through `blockwise_scaled_fp8_with_epilogue`, which passes
    even when this predicate regresses to False (it falls back to the naive
    kernel with identical numerics).

    Parameters:
        c_type: Output dtype.
        a_type: dtype of `a`.
        b_type: dtype of `b`.
        a_scales_type: dtype of `a`'s scales.
        b_scales_type: dtype of `b`'s scales.
        transpose_b: Whether `b` is K-major.
        scales_granularity_mnk: Blockwise scale granularity `(m, n, k)`.
        device_info: The device's `GPUInfo`.

    Returns:
        True when the tiled kernel covers this call.
    """
    return comptime (
        # `_is_sm12x_gpu` compares only the compute capability, and AMD parts
        # report 12.x too (gfx1250, gfx1201, gfx1200), so the vendor is checked
        # first: this kernel issues NVIDIA `mma.sync`.
        device_info.api == "cuda"
        and _is_sm12x_gpu(device_info)
        and transpose_b
        and a_type == .float8_e4m3fn
        and b_type == .float8_e4m3fn
        and a_scales_type == .float32
        and b_scales_type == .float32
        and (c_type == .bfloat16 or c_type == .float32)
        and scales_granularity_mnk[0] == 1
        and scales_granularity_mnk[1] == 128
        and scales_granularity_mnk[2] == 128
        # K need not be a multiple of 128 and N need not be a multiple of 128:
        # the boundary path zero-masks both, and scales index by real `kb`.
    )


@__name(
    t"blockwise_scaled_fp8_matmul_nvidia_{dst_type}_{a_type}_{b_type}_bn{BN}_partials{has_partials}"
)
def blockwise_scaled_fp8_matmul_nvidia_kernel[
    BN: Int,
    has_partials: Bool,
    dst_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    dst_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    a_scale_layout: TensorLayout,
    b_scale_layout: TensorLayout,
](
    dst: TileTensor[mut=True, dst_type, dst_layout, MutAnyOrigin],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin],
    a_scales: TileTensor[a_scales_type, a_scale_layout, ImmutAnyOrigin],
    b_scales: TileTensor[b_scales_type, b_scale_layout, ImmutAnyOrigin],
    k_blocks_per_split: Int32,
):
    """Multiplies one `_BM`-row by `BN`-column output block of the
    blockwise-scaled FP8 matmul, over one split of the K range.

    Grid `(ceildiv(N, BN), ceildiv(M, _BM), num_splits)`: block `x` owns output
    columns `[BNx, BNx + BN)`, block `y` owns output rows `[8y, 8y + 8)`, block
    `z` owns `k_blocks_per_split` of the K blocks. The CTA walks its K blocks
    in 128-wide slabs, keeping the full-precision fp32 block sum of each slab
    and folding `a_scale * b_scale` in once per slab.

    Args:
        dst: Rank-2 output tensor -- `c` when `has_partials` is False, the
            `(num_splits * M)` x `N` fp32 partial-sum buffer otherwise.
        a: Rank-2 FP8 input matrix in K-major format.
        b: Rank-2 FP8 weight matrix in K-major format (N x K).
        a_scales: Rank-2 per-block scales for `a`, `[ceildiv(K, 128), M]`.
        b_scales: Rank-2 per-block scales for `b`,
            `[ceildiv(N, 128), ceildiv(K, 128)]`.
        k_blocks_per_split: K blocks handed to one split along `z`.
    """
    comptime accum_type = DType.float32
    comptime BM = _BM
    comptime BK = _BK
    comptime NT = _NUM_THREADS
    comptime CH = _CHUNK
    comptime BCH = BN * _K_CHUNKS // NT

    comptime assert (
        BN * _K_CHUNKS
    ) % NT == 0 and NT % _K_CHUNKS == 0, (
        "the b staging map must cover whole chunks and whole rows"
    )
    comptime assert BN <= NT, "the MAC threads are the first BN threads"

    var n0 = Int(block_idx.x) * BN
    # The b scales are blocked every 128 columns, so the block index comes
    # from the column, not the CTA: a 64-wide tile puts two CTAs in one block.
    var n_blk = n0 // 128
    var m0 = Int(block_idx.y) * BM
    var tid = Int(warp_id()) * WARP_SIZE + Int(lane_id())

    var M = Int(a.dim[0]())
    var N = Int(dst.dim[1]())
    var K = Int(a.dim[1]())
    var n_blocks = ceildiv(K, BK)

    # This thread's output column, and its slots in the staging maps.
    var n_local = tid
    var a_row = tid // _K_CHUNKS
    var a_col = (tid % _K_CHUNKS) * CH

    # This split's slice of the K blocks.
    var per_split = Int(k_blocks_per_split)
    var kb_begin = min(Int(block_idx.z) * per_split, n_blocks)
    var kb_end = min(n_blocks, kb_begin + per_split)
    var steps = max(0, kb_end - kb_begin)

    # Shared staging. `b` is padded per row; `a` is staged as fp32 (broadcast).
    var b_sh = stack_allocation[a_type, address_space=.SHARED, alignment=16](
        row_major[_RING * BN * _B_ROW]()
    )
    var a_sh = stack_allocation[
        accum_type, address_space=.SHARED, alignment=32
    ](row_major[_RING * BM * BK]())

    var acc = StaticTuple[Scalar[accum_type], BM](fill=0.0)
    var blk = StaticTuple[SIMD[accum_type, CH], BM](
        fill=SIMD[accum_type, CH](0.0)
    )

    # A `width=CH` global read is only known to be aligned when the row stride
    # is: every row of `a` and `b` is `K` bytes long. `K % CH != 0` keeps the
    # element-granular path, which is correct for any K.
    var k_align_ok = (K % CH) == 0

    # `st == -1` only stages slab 0. Every later step multiplies the slab the
    # previous step staged while the next one is in flight: the loads are
    # issued into `loaded` (independent, so they all go out together), the
    # MACs of the staged slab run while they are in memory, and only then are
    # they written to the ring. One barrier retires a slab.
    for st in range(-1, steps):
        var has_next = (st + 1) < steps
        var loaded = StaticTuple[SIMD[a_type, CH], BCH](
            fill=SIMD[a_type, CH](Scalar[a_type](0))
        )
        var a_loaded = SIMD[a_type, CH](Scalar[a_type](0))

        # 1. Issue every global read of the next slab.
        if has_next:
            var k0 = (kb_begin + st + 1) * BK
            var b_fast = k_align_ok and (n0 + BN <= N) and (k0 + BK <= K)
            comptime for i in range(BCH):
                var ci = tid + i * NT
                var row = ci // _K_CHUNKS
                var col = (ci % _K_CHUNKS) * CH
                if b_fast:
                    loaded[i] = rebind[SIMD[a_type, CH]](
                        b.load[width=CH, alignment=CH](
                            Coord(n0 + row, k0 + col)
                        )
                    )
                else:
                    # Boundary CTA: zero-fill whatever falls outside the
                    # operand so a partial n block and a K tail keep the block
                    # sum exact.
                    var v = SIMD[a_type, CH](Scalar[a_type](0))
                    if n0 + row < N:
                        for e in range(CH):
                            if k0 + col + e < K:
                                v[e] = rebind[Scalar[a_type]](
                                    b.load(Coord(n0 + row, k0 + col + e))
                                )
                    loaded[i] = v

            # The `a` slab: one chunk per thread, rows past `M` zeroed. Only
            # the first `BM * _K_CHUNKS` threads own a chunk of it.
            if a_row < BM:
                var a_fast = k_align_ok and (k0 + BK <= K) and (m0 + a_row < M)
                if a_fast:
                    a_loaded = a.load[width=CH, alignment=CH](
                        Coord(m0 + a_row, k0 + a_col)
                    )
                else:
                    if m0 + a_row < M:
                        for e in range(CH):
                            if k0 + a_col + e < K:
                                a_loaded[e] = a.load(
                                    Coord(m0 + a_row, k0 + a_col + e)
                                )

        # 2. Multiply the staged slab. `b` is read once per slab per thread
        # from shared and multiplied into all `_BM` rows from registers.
        if st >= 0 and tid < BN:
            comptime for m in range(BM):
                blk[m] = SIMD[accum_type, CH](0.0)

            var buf_b = (st % _RING) * BN * _B_ROW + n_local * _B_ROW
            var buf_a = (st % _RING) * BM * BK
            comptime for j in range(_K_CHUNKS):
                var bv = b_sh.raw_load[width=CH, alignment=8](
                    buf_b + j * CH
                ).cast[accum_type]()
                comptime for m in range(BM):
                    blk[m] += (
                        a_sh.raw_load[width=CH, alignment=32](
                            buf_a + m * BK + j * CH
                        )
                        * bv
                    )

            # One fp32 fold per 128-wide block: the block sum carries the full
            # precision of the products and the scales are applied once.
            var kb = kb_begin + st
            var b_scale = rebind[Scalar[b_scales_type]](
                b_scales.load(Coord(n_blk, kb))
            ).cast[accum_type]()
            comptime for m in range(BM):
                var m_idx = min(m0 + m, M - 1)
                var a_scale = rebind[Scalar[a_scales_type]](
                    a_scales.load(Coord(kb, m_idx))
                ).cast[accum_type]()
                acc[m] += blk[m].reduce_add() * a_scale * b_scale

        # 3. Publish the next slab into the ring buffer this step's MAC just
        # vacated.
        if has_next:
            var buf = ((st + 1) % _RING) * BN * _B_ROW
            comptime for i in range(BCH):
                var ci = tid + i * NT
                var row = ci // _K_CHUNKS
                var col = (ci % _K_CHUNKS) * CH
                b_sh.raw_store[width=CH, alignment=8](
                    buf + row * _B_ROW + col,
                    rebind[SIMD[a_type, CH]](loaded[i]),
                )
            if a_row < BM:
                a_sh.raw_store[width=CH, alignment=32](
                    ((st + 1) % _RING) * BM * BK + a_row * BK + a_col,
                    a_loaded.cast[accum_type](),
                )

        barrier()

    var rows_valid = min(BM, M - m0)
    var row_offset = 0
    comptime if has_partials:
        row_offset = Int(block_idx.z) * M
    comptime for m in range(BM):
        if tid < BN and m < rows_valid and n0 + n_local < N:
            dst.store(
                Coord(row_offset + m0 + m, n0 + n_local),
                acc[m].cast[dst_type](),
            )


@__name(
    t"blockwise_scaled_fp8_gemv_nvidia_{dst_type}_{a_type}_{b_type}_rows{ROWS}"
)
def blockwise_scaled_fp8_gemv_nvidia_kernel[
    ROWS: Int,
    dst_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    dst_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    a_scale_layout: TensorLayout,
    b_scale_layout: TensorLayout,
](
    dst: TileTensor[mut=True, dst_type, dst_layout, MutAnyOrigin],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin],
    a_scales: TileTensor[a_scales_type, a_scale_layout, ImmutAnyOrigin],
    b_scales: TileTensor[b_scales_type, b_scale_layout, ImmutAnyOrigin],
):
    """Multiplies up to `ROWS` rows of `a` into `_GV_COLS` columns per CTA.

    The decode tile. Grid `ceildiv(N, _GV_COLS)`: block `x` owns output
    columns `[n0, n0 + _GV_COLS)` for every row, and its `_GV_WARPS` warps
    split K between them. Warp `w` takes the 512-wide K steps
    `w, w + _GV_WARPS, ...`, lane `l` reading the 16-byte vector at
    `k = 16 l + 512 step` of each weight row, so a warp step is one 512-byte
    coalesced read per row. The weights go from global memory to registers
    with no shared staging, and the loads of `_GV_UNROLL` steps are issued
    before any of them is multiplied. Shared memory only carries the
    `_GV_WARPS` warp sums of each output to the final add.

    A 16-byte vector never straddles a 128-wide scale block, so each lane
    folds `a_scale * b_scale` into its 16-product sum and the CTA adds the
    lanes at the end. That is the naive kernel's arithmetic summed in a
    different order.

    Requires `K % 16 == 0`, which keeps every vector inside a row and
    aligned; the launcher sends any other K to the 8-row tile.

    Args:
        dst: Rank-2 output tensor, `M` x `N` with `M <= ROWS`.
        a: Rank-2 FP8 input matrix in K-major format.
        b: Rank-2 FP8 weight matrix in K-major format (N x K).
        a_scales: Rank-2 per-block scales for `a`, `[ceildiv(K, 128), M]`.
        b_scales: Rank-2 per-block scales for `b`,
            `[ceildiv(N, 128), ceildiv(K, 128)]`.
    """
    comptime accum_type = DType.float32
    comptime VEC = _GV_VEC
    comptime COLS = _GV_COLS
    comptime U = _GV_UNROLL
    comptime W = _GV_WARPS
    comptime STEP = WARP_SIZE * VEC
    comptime assert 128 % COLS == 0, "a CTA's columns share one b_scale block"

    var M = Int(a.dim[0]())
    var N = Int(dst.dim[1]())
    var K = Int(a.dim[1]())

    var lane = Int(lane_id())
    var wid = Int(warp_id())
    var n0 = Int(block_idx.x) * COLS
    var n_blk = n0 // 128

    var acc = StaticTuple[Scalar[accum_type], ROWS * COLS](fill=0.0)

    for k_base in range(wid * STEP + lane * VEC, K, W * U * STEP):
        # 1. Issue every load of `U` steps. Out-of-range vectors and columns
        # stay zero, so they add nothing.
        var bv = StaticTuple[SIMD[b_type, VEC], U * COLS](
            fill=SIMD[b_type, VEC](Scalar[b_type](0))
        )
        var av = StaticTuple[SIMD[a_type, VEC], U * ROWS](
            fill=SIMD[a_type, VEC](Scalar[a_type](0))
        )
        comptime for u in range(U):
            var k0 = k_base + u * W * STEP
            if k0 < K:
                comptime for c in range(COLS):
                    if n0 + c < N:
                        bv[u * COLS + c] = rebind[SIMD[b_type, VEC]](
                            b.load[width=VEC, alignment=VEC](Coord(n0 + c, k0))
                        )
                comptime for r in range(ROWS):
                    if r < M:
                        av[u * ROWS + r] = rebind[SIMD[a_type, VEC]](
                            a.load[width=VEC, alignment=VEC](Coord(r, k0))
                        )

        # 2. Multiply them, one scale fold per 16-product sum.
        comptime for u in range(U):
            var k0 = k_base + u * W * STEP
            if k0 < K:
                var kb = k0 // _BK
                var b_s = rebind[Scalar[b_scales_type]](
                    b_scales.load(Coord(n_blk, kb))
                ).cast[accum_type]()
                comptime for r in range(ROWS):
                    if r < M:
                        var s = (
                            rebind[Scalar[a_scales_type]](
                                a_scales.load(Coord(kb, r))
                            ).cast[accum_type]()
                            * b_s
                        )
                        var a_f = av[u * ROWS + r].cast[accum_type]()
                        comptime for c in range(COLS):
                            acc[r * COLS + c] += (
                                a_f * bv[u * COLS + c].cast[accum_type]()
                            ).reduce_add() * s

    var sums = stack_allocation[
        accum_type, address_space=.SHARED, alignment=16
    ](row_major[W * ROWS * COLS]())
    comptime for i in range(ROWS * COLS):
        var part = warp.sum(acc[i])
        if lane == 0:
            sums.raw_store(wid * ROWS * COLS + i, part)
    barrier()

    if wid == 0:
        for i in range(lane, ROWS * COLS, WARP_SIZE):
            var total = Scalar[accum_type](0.0)
            comptime for w in range(W):
                total += sums.raw_load(w * ROWS * COLS + i)
            var r = i // COLS
            var c = i % COLS
            if r < M and n0 + c < N:
                dst.store(Coord(r, n0 + c), total.cast[dst_type]())


@__name(t"blockwise_scaled_fp8_matmul_nvidia_tc_{dst_type}_{a_type}")
def blockwise_scaled_fp8_matmul_nvidia_tc_kernel[
    dst_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    dst_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    a_scale_layout: TensorLayout,
    b_scale_layout: TensorLayout,
](
    dst: TileTensor[mut=True, dst_type, dst_layout, MutAnyOrigin],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin],
    a_scales: TileTensor[a_scales_type, a_scale_layout, ImmutAnyOrigin],
    b_scales: TileTensor[b_scales_type, b_scale_layout, ImmutAnyOrigin],
):
    """Multiplies one `_TC_BM`-row by `_TC_BN`-column output block of the
    blockwise-scaled FP8 matmul, on the tensor cores.

    Grid `(ceildiv(M, 128), ceildiv(N, 128))`: block `x` owns 128 output rows
    and block `y` owns 128 output columns. The `x` axis is the M tiles on
    purpose. The CTAs a launch schedules together then walk the SAME 128-column
    weight panel, so the panel is read from DRAM once per resident CTA wave and
    served to the rest from L2; put the N tiles on `x` and every CTA pulls its
    own column block through the memory system instead.

    The tile walks K in 128-wide slabs, one slab per K loop iteration, staging
    both operands as fp8 in shared memory and issuing four `mma.sync` steps per
    slab per warp. The block scales are folded with a running ratio:

        acc = 0; s_ref = 0
        for t in slabs:
            if s[t] == 0:
                stage the row of `a` as zeros  # G_t = 0, acc untouched
            else:
                acc *= s_ref / s[t]            # 0 before the first nonzero
                s_ref = s[t]
            mma slab t into acc                # raw fp8 products, unscaled
        C = acc * s_ref

    where `s[t] = a_scale[t, m] * b_scale[n // 128, t]`. After slab `t`,
    `acc = (sum_{u<=t} G_u s_u) / s_ref` for the raw block sums `G_u`, so the
    final multiply by `s_ref` lands on the exact sum. A slab whose scale is
    zero contributes exactly zero, as it does in the naive kernel: its `a`
    rows are never loaded, so the MMA adds nothing to those rows, and `s_ref`
    keeps the last nonzero scale so the slabs before it are not erased. The
    alternative -- a second accumulator set -- is 64 more registers per
    thread, which does not fit beside these.

    `b_scale` is one scalar for the whole CTA because `_TC_BN = 128` covers
    exactly one 128-column scale block, so the ratio is per output row only.

    The running form assumes a row's nonzero slab scales stay within fp32's
    normal range of each other: `acc` carries the partial sum divided by the
    latest scale, so a slab scale many decades below its predecessors scales
    `acc` up by that ratio. Dynamic activation scales are `amax / 448` per
    128-wide group, so reaching fp32 overflow takes a group whose largest
    magnitude is ~1e-30 after groups near 1e4; a denormal scale (amax below
    ~1e-36) is outside the assumption, and the naive kernel, which folds each
    slab's scale in directly, has no such limit.

    Args:
        dst: Rank-2 output tensor.
        a: Rank-2 FP8 input matrix in K-major format.
        b: Rank-2 FP8 weight matrix in K-major format (N x K).
        a_scales: Rank-2 per-block scales for `a`, `[ceildiv(K, 128), M]`.
        b_scales: Rank-2 per-block scales for `b`,
            `[ceildiv(N, 128), ceildiv(K, 128)]`.
    """
    comptime accum_type = DType.float32
    comptime BM = _TC_BM
    comptime BN = _TC_BN
    comptime BK = _TC_BK
    comptime NT = _TC_NUM_THREADS
    comptime WM = _TC_WM
    comptime WN = _TC_WN
    comptime VEC = _TC_VEC
    comptime VPR = BK // VEC
    comptime VPT = BM * VPR // NT
    comptime num_warps_n = BN // WN
    comptime num_warps_m = BM // WM

    comptime assert (
        VPT * NT == BM * VPR
    ), "the staging map must cover whole 16-byte vectors of the slab"
    comptime assert (
        num_warps_m * num_warps_n * WARP_SIZE == NT
    ), "one warp per 64x32 warp tile of the 128x128 block tile"

    comptime mma_shape = get_mma_shape[a_type, accum_type]()
    comptime MMA_M = Int(mma_shape[0])
    comptime MMA_N = Int(mma_shape[1])
    comptime MMA_K = Int(mma_shape[2])
    comptime num_k_mmas = BK // MMA_K
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime a_frag_size = frag_size[0]
    comptime b_frag_size = frag_size[1]
    comptime c_frag_size = frag_size[2]
    # Two accumulator rows per thread per m subtile: `lane // 4` and 8 rows
    # below it, which is where `_c_fragment_coord` puts registers 0-1 and 2-3.
    comptime thread_rows = 2 * num_m_mmas

    var M = Int(a.dim[0]())
    var N = Int(dst.dim[1]())
    var K = Int(a.dim[1]())

    var m0 = Int(block_idx.x) * BM
    var n0 = Int(block_idx.y) * BN
    # The b scales are blocked every 128 columns and `BN` is one block, so the
    # block index is the CTA's own column coordinate.
    var n_blk = n0 // 128

    var lane = Int(lane_id())
    var warp = Int(warp_id())
    var tid = warp * WARP_SIZE + lane
    var warp_y = warp // num_warps_n
    var warp_x = warp % num_warps_n

    var slabs = ceildiv(K, BK)

    # One slab deep. The staging is synchronous, so a second buffer would only
    # spill registers: this kernel's job is the arithmetic, and the pipeline
    # can come once the numbers are right.
    var a_sh = stack_allocation[a_type, address_space=.SHARED, alignment=16](
        row_major[BM, BK]()
    )
    var b_sh = stack_allocation[b_type, address_space=.SHARED, alignment=16](
        row_major[BN, BK]()
    )

    var a_reg = stack_allocation[a_type, address_space=.LOCAL](
        row_major[num_m_mmas, a_frag_size]()
    )
    var b_reg = stack_allocation[b_type, address_space=.LOCAL](
        row_major[num_n_mmas, b_frag_size]()
    )
    var c_reg = stack_allocation[accum_type, address_space=.LOCAL](
        row_major[num_m_mmas * num_n_mmas, c_frag_size]()
    )
    var c_zero = SIMD[accum_type, c_frag_size](Scalar[accum_type](0))
    comptime for i in range(num_m_mmas * num_n_mmas):
        c_reg.store[width=c_frag_size, alignment=16](Coord(i, 0), c_zero)

    var mma_op = TensorCore[accum_type, a_type, mma_shape, True]()
    # The staged tiles are swizzled with the pattern `TensorCore.load_a` and
    # `load_b` derive from their row stride (`BK`), so both sides address the
    # same 16-byte vectors.
    comptime swizzle_a = Optional[Swizzle](make_ldmatrix_swizzle[a_type, BK]())

    # `s[t]` for this thread's output rows, one entry per accumulator row.
    var prev_s = StaticTuple[Scalar[accum_type], thread_rows](fill=0.0)

    for slab in range(slabs):
        var k0 = slab * BK
        var b_s = rebind[Scalar[b_scales_type]](
            b_scales.load(Coord(n_blk, slab))
        ).cast[accum_type]()

        # Stage the slab: every thread copies `VPT` 16-byte vectors of each
        # operand. Anything the operands do not cover is zero-filled -- a
        # partial n block, a partial m block and a K tail all have to keep the
        # block sum exact -- and so is an `a` row whose scale `s` is zero for
        # this slab, which is how that slab contributes nothing to the row.
        # `vec ^ (row % 8)` is the ldmatrix swizzle: the 16-byte unit index
        # within the row is xored with the row index.
        comptime for i in range(VPT):
            var ci = tid + i * NT
            var srow = ci // VPR
            var vec = ci % VPR
            var scol = vec * VEC
            var swz = (vec ^ (srow % 8)) * VEC
            # A 16-byte global read is only known to be aligned when the row
            # stride is, and every row of `a` and `b` is `K` bytes long.
            var in_row = k0 + scol + VEC <= K

            var aval = SIMD[a_type, VEC](Scalar[a_type](0))
            var s_row = (
                rebind[Scalar[a_scales_type]](
                    a_scales.load(Coord(slab, min(m0 + srow, M - 1)))
                ).cast[accum_type]()
                * b_s
            )
            if m0 + srow < M and s_row != 0:
                if (K % VEC == 0) and in_row:
                    aval = a.load[width=VEC, alignment=VEC](
                        Coord(m0 + srow, k0 + scol)
                    )
                else:
                    for e in range(VEC):
                        if k0 + scol + e < K:
                            aval[e] = rebind[Scalar[a_type]](
                                a.load(Coord(m0 + srow, k0 + scol + e))
                            )
            a_sh.store[width=VEC, alignment=VEC](Coord(srow, swz), aval)

            var bval = SIMD[b_type, VEC](Scalar[b_type](0))
            if n0 + srow < N:
                if (K % VEC == 0) and in_row:
                    bval = b.load[width=VEC, alignment=VEC](
                        Coord(n0 + srow, k0 + scol)
                    )
                else:
                    for e in range(VEC):
                        if k0 + scol + e < K:
                            bval[e] = rebind[Scalar[b_type]](
                                b.load(Coord(n0 + srow, k0 + scol + e))
                            )
            b_sh.store[width=VEC, alignment=VEC](Coord(srow, swz), bval)

        barrier()

        # Rescale the accumulators from the last nonzero scale to this slab's
        # before adding to them. A zero scale leaves them alone: the staging
        # above zeroed that row of `a`, so the slab adds nothing, and keeping
        # `prev_s` at the last nonzero scale keeps the earlier slabs intact.
        # Until the first nonzero scale `prev_s` is 0 and so are the
        # accumulators, so the zero ratio that produces is harmless.
        comptime for j in range(thread_rows):
            var grow = (
                m0 + warp_y * WM + (j // 2) * MMA_M + lane // 4 + (j % 2) * 8
            )
            var s_cur = (
                rebind[Scalar[a_scales_type]](
                    a_scales.load(Coord(slab, min(grow, M - 1)))
                ).cast[accum_type]()
                * b_s
            )
            if s_cur != 0:
                if slab > 0:
                    var ratio = prev_s[j] / s_cur
                    comptime for n_sub in range(num_n_mmas):
                        var crow = n_sub * num_m_mmas + j // 2
                        var ccol = (j % 2) * 2
                        c_reg.store[width=2, alignment=8](
                            Coord(crow, ccol),
                            c_reg.load[width=2, alignment=8](Coord(crow, ccol))
                            * ratio,
                        )
                prev_s[j] = s_cur

        var a_warp = a_sh.tile[WM, BK](Coord(warp_y, 0))
        var b_warp = b_sh.tile[WN, BK](Coord(warp_x, 0))
        comptime for ks in range(num_k_mmas):
            mma_op.load_a[swizzle_a](a_warp, a_reg, ks)
            mma_op.load_b(
                b_warp.bitcast[a_type](), b_reg.bitcast[a_type](), ks, warp_x
            )
            mma_op.mma(
                a_reg.vectorize[1, a_frag_size](),
                b_reg.vectorize[1, b_frag_size]().bitcast[a_type](),
                c_reg.vectorize[1, c_frag_size](),
            )
        barrier()

    # `acc` holds `sum_u G_u s_u / s_ref`; fold `s_ref` in.
    comptime for j in range(thread_rows):
        comptime for n_sub in range(num_n_mmas):
            var crow = n_sub * num_m_mmas + j // 2
            var ccol = (j % 2) * 2
            c_reg.store[width=2, alignment=8](
                Coord(crow, ccol),
                c_reg.load[width=2, alignment=8](Coord(crow, ccol)) * prev_s[j],
            )

    # Row and column of fragment element `r` of a 16x8 subtile, per
    # `TensorCore._c_fragment_coord`: adjacent columns pair up, then the row
    # advances by 8.
    comptime for m_sub in range(num_m_mmas):
        comptime for n_sub in range(num_n_mmas):
            var crow = n_sub * num_m_mmas + m_sub
            comptime for r in range(c_frag_size):
                var grow = (
                    m0 + warp_y * WM + m_sub * MMA_M + lane // 4 + (r // 2) * 8
                )
                var gcol = (
                    n0 + warp_x * WN + n_sub * MMA_N + (lane % 4) * 2 + r % 2
                )
                if grow < M and gcol < N:
                    dst.store(
                        Coord(grow, gcol),
                        c_reg.load(Coord(crow, r)).cast[dst_type](),
                    )


@__name(t"blockwise_scaled_fp8_nvidia_partials_reduce_{c_type}")
def blockwise_scaled_fp8_nvidia_reduce_kernel[
    c_type: DType,
    c_layout: TensorLayout,
    p_layout: TensorLayout,
](
    c: TileTensor[mut=True, c_type, c_layout, MutAnyOrigin],
    partials: TileTensor[.float32, p_layout, ImmutAnyOrigin],
    num_splits: Int32,
):
    """Adds the split-K partial sums and writes them out at `c_type`.

    Grid `(ceildiv(N, 256), M)`: one thread per output element. The partials
    are already in output units (each split folded its own block scales), so
    this is a plain sum.

    Args:
        c: Rank-2 output tensor.
        partials: `(num_splits * M)` x `N` fp32 partial sums.
        num_splits: Number of partial-sum slices to add.
    """
    var n_idx = (
        Int(block_idx.x) * _NUM_THREADS
        + Int(warp_id()) * WARP_SIZE
        + Int(lane_id())
    )
    var m_idx = Int(block_idx.y)
    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    if n_idx < N and m_idx < M:
        var total = Float32(0.0)
        for s in range(Int(num_splits)):
            total += partials.load(Coord(s * M + m_idx, n_idx))
        c.store(Coord(m_idx, n_idx), total.cast[c_type]())


def _nvidia_blockwise_tiles(
    M: Int,
    N: Int,
    k_blocks: Int,
    sm_count: Int,
    bn_override: Int,
    split_override: Int,
) -> Tuple[Int, Int]:
    """Picks the tile width and split-K factor a launch runs with.

    Both are properties of the 8-row SIMT tile and are what `_TC_MIN_M` and
    `_nvidia_blockwise_tc_ok` read to route a prefill shape to the tensor-core
    tile, so they are one function: the route and the tile cannot disagree.

    Args:
        M: Number of output rows.
        N: Number of output columns.
        k_blocks: Number of 128-wide K blocks.
        sm_count: SM count of the device the launch runs on (48 on GB10, 170
            on an RTX 5090), read from the device at run time: the per-target
            `GPUInfo.sm_count` is one number for every part of an arch.
        bn_override: Forced tile width, or 0 to choose one.
        split_override: Forced split-K factor, or 0 to choose one.

    Returns:
        The `(bn, split)` the launcher runs with.
    """
    var m_tiles = ceildiv(M, _BM)

    # 128 wide is the tile. A 64-wide CTA stages half the bytes per slab and
    # measures 45% slower on both ends of the shape range: 81 GB/s against 148
    # at N=5120 and 92 against 169 at N=17408, which is why the narrow tile is
    # reachable only through the test override.
    var bn = 128
    if bn_override != 0:
        bn = bn_override

    # Split-K only fills SMs. `N / 128` CTAs at M=1 is 40 on GB10's 48 SMs for
    # N=5120, so a split is what keeps every SM streaming; past one CTA per SM
    # it is neutral (148.4 against 148.6 GB/s at N=5120, split 1 against 4).
    var split = 1
    if split_override != 0:
        split = split_override
    else:
        var ctas = ceildiv(N, bn) * m_tiles
        while (
            ctas * split < sm_count
            and split * 2 <= _MAX_SPLIT
            and k_blocks >= 4 * split * 2
        ):
            split *= 2
    if split > k_blocks:
        split = 1
    return (bn, split)


def _nvidia_blockwise_tc_ok(M: Int, bn: Int, split: Int) -> Bool:
    """Whether the launcher hands this problem to the tensor-core kernel.

    Extracted so a test can assert the route a shape takes: a numeric match
    against the naive kernel cannot tell the two tiles apart, since both are
    correct.

    Args:
        M: Number of output rows.
        bn: Tile width `_nvidia_blockwise_tiles` chose.
        split: Split-K factor `_nvidia_blockwise_tiles` chose.

    Returns:
        True when prefill's 128-row tensor-core tile runs.
    """
    return M >= _TC_MIN_M and split == 1 and bn == _TC_BN


def _nvidia_blockwise_gemv_ok(
    M: Int, K: Int, bn_override: Int, split_override: Int
) -> Bool:
    """Whether the launcher hands this problem to the decode GEMV.

    Extracted so a test can assert the route a shape takes, as with
    `_nvidia_blockwise_tc_ok`.

    Args:
        M: Number of output rows.
        K: Reduction depth.
        bn_override: Forced tile width of the 8-row tile, or 0.
        split_override: Forced split-K factor of the 8-row tile, or 0.

    Returns:
        True when the decode GEMV runs. A forced tile width or split names the
        8-row tile, so either one keeps it.
    """
    return (
        M <= _GV_MAX_M
        and K % _GV_VEC == 0
        and bn_override == 0
        and split_override == 0
    )


def blockwise_scaled_fp8_matmul_nvidia[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    //,
    *,
    transpose_b: Bool = True,
    scales_granularity_mnk: Coord[Int, Int, Int] = Coord[Int, Int, Int](
        1, 128, 128
    ),
    has_epilogue_fn: Bool = False,
    bn_override: Int = 0,
    split_override: Int = 0,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[mut=False, a_scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[mut=False, b_scales_type, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
) raises:
    """Dispatches the tiled blockwise scaled FP8 matmul kernel on the GPU.

    Enqueues `blockwise_scaled_fp8_matmul_nvidia_kernel` with one CTA per
    (`BN`-column n block, 8-row m block, K split) of the `M` x `N` output.
    `BN` is 128 unless overridden, and the K range is split 2 or 4 ways while
    the grid has fewer CTAs than the device has SMs, which is where this
    kernel's bandwidth comes from at small M; a split writes fp32 partial sums
    and enqueues the reduction straight after.

    A prefill-shaped problem -- `M >= _TC_MIN_M` with no split, which is where
    the 8-row tile would start re-reading the weights for every 8 rows -- takes
    `blockwise_scaled_fp8_matmul_nvidia_tc_kernel` instead, one CTA per 128x128
    block of the output. A decode-shaped one -- `M <= _GV_MAX_M`, `K % 16 == 0`
    and no forced tile width or split -- takes
    `blockwise_scaled_fp8_gemv_nvidia_kernel`, one CTA per 8 output columns.

    Args:
        c: Rank-2 output accumulator tensor.
        a: Rank-2 FP8 input matrix in K-major format.
        b: Rank-2 FP8 weight matrix in K-major format (N x K).
        a_scales: Rank-2 per-block scales for `a` in M-major format.
        b_scales: Rank-2 per-block scales for `b` in K-major format.
        ctx: Device context used to enqueue the kernel.
        epilogue_fn: With `has_epilogue_fn`, applied to every output element
            by a second elementwise launch over `c`.
    """
    comptime assert a_type == b_type == .float8_e4m3fn, (
        "Only float8_e4m3fn is supported for input dtype for blockwise"
        " scaled fp8 matmul"
    )
    comptime assert (
        a_scales_type == b_scales_type == .float32
    ), "Only float32 is supported for blockwise scaled fp8 scales"
    comptime assert c_type == .bfloat16 or c_type == .float32, (
        "Only bfloat16 and float32 are supported for the output of the tiled"
        " blockwise scaled fp8 matmul"
    )
    comptime assert (
        transpose_b
    ), "The tiled NVIDIA blockwise scaled fp8 matmul requires a K-major B"
    comptime assert (
        bn_override == 0 or bn_override == 64 or bn_override == 128
    ), (
        "bn_override must be 0 (launcher's choice), 64 or 128: those are the"
        " SIMT tile widths this launcher instantiates"
    )
    comptime assert (
        scales_granularity_mnk[0] == 1
        and scales_granularity_mnk[1] == 128
        and scales_granularity_mnk[2] == 128
    ), (
        "The tiled NVIDIA blockwise scaled fp8 matmul is the (m, n, k) ="
        " (1, 128, 128) case"
    )

    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    var K = Int(a.dim[1]())

    if M == 0 or N == 0 or K == 0:
        return

    var k_blocks = ceildiv(K, _BK)
    var m_tiles = ceildiv(M, _BM)
    var bn, split = _nvidia_blockwise_tiles(
        M,
        N,
        k_blocks,
        ctx.device().multiprocessor_count(),
        bn_override,
        split_override,
    )
    var blocks_per_split = ceildiv(k_blocks, split)

    if M == 1 and _nvidia_blockwise_gemv_ok(M, K, bn_override, split_override):
        comptime kernel = blockwise_scaled_fp8_gemv_nvidia_kernel[
            1,
            c_type,
            a_type,
            b_type,
            a_scales_type,
            b_scales_type,
            c.LayoutType,
            a.LayoutType,
            b.LayoutType,
            a_scales.LayoutType,
            b_scales.LayoutType,
        ]
        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            a_scales,
            b_scales,
            execution_config=GridBlockLaunchConfig(
                grid_dim=(ceildiv(N, _GV_COLS), 1, 1),
                block_dim=(_GV_WARPS * WARP_SIZE, 1, 1),
            ),
        )
    elif _nvidia_blockwise_gemv_ok(M, K, bn_override, split_override):
        comptime kernel = blockwise_scaled_fp8_gemv_nvidia_kernel[
            _GV_MAX_M,
            c_type,
            a_type,
            b_type,
            a_scales_type,
            b_scales_type,
            c.LayoutType,
            a.LayoutType,
            b.LayoutType,
            a_scales.LayoutType,
            b_scales.LayoutType,
        ]
        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            a_scales,
            b_scales,
            execution_config=GridBlockLaunchConfig(
                grid_dim=(ceildiv(N, _GV_COLS), 1, 1),
                block_dim=(_GV_WARPS * WARP_SIZE, 1, 1),
            ),
        )
    elif _nvidia_blockwise_tc_ok(M, bn, split):
        comptime kernel = blockwise_scaled_fp8_matmul_nvidia_tc_kernel[
            c_type,
            a_type,
            b_type,
            a_scales_type,
            b_scales_type,
            c.LayoutType,
            a.LayoutType,
            b.LayoutType,
            a_scales.LayoutType,
            b_scales.LayoutType,
        ]
        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            a_scales,
            b_scales,
            execution_config=GridBlockLaunchConfig(
                grid_dim=(
                    ceildiv(M, _TC_BM),
                    ceildiv(N, _TC_BN),
                    1,
                ),
                block_dim=(_TC_NUM_THREADS, 1, 1),
            ),
        )
    elif split == 1:
        if bn == 64:
            comptime kernel = blockwise_scaled_fp8_matmul_nvidia_kernel[
                64,
                False,
                c_type,
                a_type,
                b_type,
                a_scales_type,
                b_scales_type,
                c.LayoutType,
                a.LayoutType,
                b.LayoutType,
                a_scales.LayoutType,
                b_scales.LayoutType,
            ]
            ctx.enqueue_function[kernel](
                c,
                a,
                b,
                a_scales,
                b_scales,
                Int32(blocks_per_split),
                execution_config=GridBlockLaunchConfig(
                    grid_dim=(ceildiv(N, 64), m_tiles, 1),
                    block_dim=(_NUM_THREADS, 1, 1),
                ),
            )
        else:
            comptime kernel = blockwise_scaled_fp8_matmul_nvidia_kernel[
                128,
                False,
                c_type,
                a_type,
                b_type,
                a_scales_type,
                b_scales_type,
                c.LayoutType,
                a.LayoutType,
                b.LayoutType,
                a_scales.LayoutType,
                b_scales.LayoutType,
            ]
            ctx.enqueue_function[kernel](
                c,
                a,
                b,
                a_scales,
                b_scales,
                Int32(blocks_per_split),
                execution_config=GridBlockLaunchConfig(
                    grid_dim=(ceildiv(N, 128), m_tiles, 1),
                    block_dim=(_NUM_THREADS, 1, 1),
                ),
            )
    else:
        var partials_device = ctx.enqueue_create_buffer[.float32](split * M * N)
        var partials = TileTensor(
            partials_device, row_major(Coord(split * M, N))
        )
        if bn == 64:
            comptime kernel = blockwise_scaled_fp8_matmul_nvidia_kernel[
                64,
                True,
                .float32,
                a_type,
                b_type,
                a_scales_type,
                b_scales_type,
                type_of(partials).LayoutType,
                a.LayoutType,
                b.LayoutType,
                a_scales.LayoutType,
                b_scales.LayoutType,
            ]
            ctx.enqueue_function[kernel](
                partials,
                a,
                b,
                a_scales,
                b_scales,
                Int32(blocks_per_split),
                execution_config=GridBlockLaunchConfig(
                    grid_dim=(ceildiv(N, 64), m_tiles, split),
                    block_dim=(_NUM_THREADS, 1, 1),
                ),
            )
        else:
            comptime kernel = blockwise_scaled_fp8_matmul_nvidia_kernel[
                128,
                True,
                .float32,
                a_type,
                b_type,
                a_scales_type,
                b_scales_type,
                type_of(partials).LayoutType,
                a.LayoutType,
                b.LayoutType,
                a_scales.LayoutType,
                b_scales.LayoutType,
            ]
            ctx.enqueue_function[kernel](
                partials,
                a,
                b,
                a_scales,
                b_scales,
                Int32(blocks_per_split),
                execution_config=GridBlockLaunchConfig(
                    grid_dim=(ceildiv(N, 128), m_tiles, split),
                    block_dim=(_NUM_THREADS, 1, 1),
                ),
            )
        comptime reduce_kernel = blockwise_scaled_fp8_nvidia_reduce_kernel[
            c_type, c.LayoutType, type_of(partials).LayoutType
        ]
        ctx.enqueue_function[reduce_kernel](
            c,
            partials,
            Int32(split),
            execution_config=GridBlockLaunchConfig(
                grid_dim=(ceildiv(N, _NUM_THREADS), M, 1),
                block_dim=(_NUM_THREADS, 1, 1),
            ),
        )
        # `partials` is a view; the buffer's last use would otherwise be its
        # construction, and a stream-ordered free could then run before the
        # two launches that read and write it.
        _ = partials_device^

    # `has_epilogue_fn`: the kernel wrote `c`, so the caller's epilogue is one
    # elementwise pass over it -- the same two-launch shape
    # `blockwise_scaled_fp8_with_epilogue` uses for the tcgen05 path.
    comptime if has_epilogue_fn:
        comptime simd_size = 32 // size_of[c_type]()

        def value_epilogue_wrapper[
            simd_width: Int, alignment: Int = 1
        ](idx: Coord) {var c, var epilogue_fn}:
            var c_val = c.load[simd_width](idx)
            epilogue_fn[c_type, simd_width, alignment=alignment](idx, c_val)

        elementwise[
            simd_size, _trace_description="blockwise_scaled_fp8_nvidia_epilogue"
        ](value_epilogue_wrapper, Coord(M, N), ctx)

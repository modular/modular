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
"""Apple M5 weight-only FP8 (W8A16) tiled matmul: bf16 A x fp8 B -> fp32.

Target: Apple M5 (`compute_capability == 5`) only.

`out = activation @ weight^T` where the FP8 weight stays `float8_e4m3fn [N, K]`
in DRAM (one byte per element -- NOT packed, NO block scales). This is the
tiled-MMA decode kernel that beats the warp-per-column GEMV's structural ceiling
at large-N / small-K (Mamba in_proj `[17504, 3136]`, MLP up `[12544, 3136]`),
where the GEMV is capped ~250 GB/s while a tiled matmul amortizes the per-output
cost across a threadgroup tile.

## Why this structure (NOT the FP4 `matmul2d` transpose_right / SMEM path)

FP4 (`matmul2d_fp4.mojo`) decodes through SMEM because its dequant (nibble unpack
+ E2M1 LUT + per-16 block scale) is expensive and its per-lane B gather is
N-scattered -- coalescing it through SMEM took it 10 -> 38 TF/s (KB
`apple-m5-gpu-perf-model`). FP8 has NO such decode: the widen is native. Crucially,
the Apple simdgroup MMA (`_mma_apple_transposable`, `mma_apple.mojo:213`) accepts
an FP8 B operand DIRECTLY (`float8_e4m3fn` is in its valid-float input set; KGEN
lowers the fp8 fragment to AIR's `<8 x i8>`), so a bf16 A x fp8 B -> fp32 MMA is a
single native instruction -- no manual widen, no SMEM staging.

So this kernel mirrors the WINNING dense bf16 structure (`AppleM5MatMul`), which
sustains ~400 GB/s at exactly this M=1 large-N/small-K decode regime: a
simdgroup-tiled GEMM with `MmaOpApple`, the B operand loaded DIRECTLY from DRAM
each K-strip (Apple has no async copy and SMEM staging DEGRADES matmul -- KB
`apple-m5-gpu-performance-considerations`), coalesced via the col-major `(K, N)`
weight view (K-contiguous fast axis). The ONLY difference from the dense bf16
path is `MmaOpApple`'s `b_type = float8_e4m3fn` (1 byte/weight vs 2) -- the
"weight-load seam" swap. `MmaOpApple` already carries a separate `b_type` and
loads A / B fragments independently, so this needs NO change to the shared MMA op
or the bf16 matmul.

The per-tensor scalar `weight_scale` folds in OUTSIDE the kernel (a post-matmul
multiply by the graph lowering), identically to the GEMV: a scalar factors out of
the fp32 sum, so the fold is EXACT.

Ragged M/N/K are handled by `MmaOpApple`'s bounded load (zero-fill OOB) + the
`< M`/`< N` store guard -- no tile-alignment requirement (unlike the FP4
interior-only `matmul2d`). The 2x2 (SG=32) simdgroup subtile is the M5
register-cliff optimum (KB `apple-m5-gpu-perf-model`); do NOT exceed 4
accumulators.
"""

from std.gpu import WARP_SIZE, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv

from layout import Idx, TileTensor
from layout.coord import Coord
from layout.tile_layout import Layout, TensorLayout

from linalg.arch.apple.mma import MmaOpApple
from linalg.matmul.gpu.apple.matmul2d_fp4 import _require_apple_m5
from linalg.utils import elementwise_epilogue_type


struct Matmul2dFp8[
    c_type: DType = DType.float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    block_m: Int = 64,
    block_n: Int = 64,
    block_k: Int = 16,
    sg_m: Int = 32,
    sg_n: Int = 32,
]:
    """W8A16 simdgroup-tiled GEMM: bf16 A x fp8 B -> fp32, direct DRAM B feed.

    Mirrors `AppleM5MatMul` (the ~400 GB/s dense bf16 decode path) with the sole
    change that the B (weight) operand is `float8_e4m3fn` via `MmaOpApple`'s
    `b_type`. A is bf16, C is `c_type`, accumulation fp32. No SMEM, no barrier:
    each K-strip's B fragments load straight from DRAM (coalesced K-contiguous via
    the col-major `(K, N)` view) and feed the native fp8-capable MMA.

    Parameters:
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32.
        elementwise_lambda_fn: Optional fused epilogue (AMD's `(row, col)`
            contract). Currently unused on the interior store; the launcher routes
            fused-epilogue shapes elsewhere.
        block_m: Threadgroup block rows `BM` (multiple of `SG_M`).
        block_n: Threadgroup block cols `BN` (multiple of `SG_N`).
        block_k: K-strip depth `BK` per accumulate step (multiple of 16).
        sg_m: Simdgroup subtile rows `SG_M` (multiple of 16).
        sg_n: Simdgroup subtile cols `SG_N` (multiple of 16). `sg_m/16 * sg_n/16`
            is the accumulator count; `2x2 == 4` is the M5 optimum (do NOT exceed).
    """

    comptime in_type = DType.bfloat16  # activation dtype (A + MMA input)
    comptime b_type = DType.float8_e4m3fn  # weight dtype (B, native fp8 MMA)
    comptime BM = Self.block_m
    comptime BN = Self.block_n
    comptime BK = Self.block_k
    comptime SG_M = Self.sg_m
    comptime SG_N = Self.sg_n
    comptime NUM_MMA_M = Self.SG_M // 16
    comptime NUM_MMA_N = Self.SG_N // 16
    comptime NUM_SG_M = Self.BM // Self.SG_M
    comptime NUM_SG_N = Self.BN // Self.SG_N
    comptime NUM_SG = Self.NUM_SG_M * Self.NUM_SG_N
    comptime THREADS_PER_BLOCK = Self.NUM_SG * WARP_SIZE
    # bf16 A x fp8 B -> fp32. `b_type` is the ONLY deviation from the dense bf16
    # `AppleM5MatMul.Mma`; transpose is supplied by the col-major (K, N) B view
    # (hw_transpose_b = b_col_major XOR transpose_b), matching the dense path.
    comptime Mma = MmaOpApple[
        DType.float32,
        Self.in_type,
        Self.NUM_MMA_M,
        Self.NUM_MMA_N,
        b_type=Self.b_type,
    ]

    @__name(t"apple_matmul2d_fp8_run_{Self.c_type}")
    @staticmethod
    def run[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        w_layout: TensorLayout,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin],
        weight: TileTensor[Self.b_type, w_layout, ImmutAnyOrigin],
        M: Int,
        N: Int,
        K: Int,
    ):
        """W8A16 kernel entry. C `(M, N)`, A `(M, K)` bf16, weight `(N, K)` fp8.

        Grid `(ceil(N/BN), ceil(M/BM))` threadgroups of `THREADS_PER_BLOCK`.
        Ragged M/N/K handled by the bounded MMA load + the `< M`/`< N` store
        guard (no tile-alignment requirement).
        """
        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime SG_M = Self.SG_M
        comptime SG_N = Self.SG_N

        var tile_m = Int(block_idx.y)
        var tile_n = Int(block_idx.x)

        var sg_id = Int(thread_idx.x) // WARP_SIZE
        var sg_m_idx = sg_id // Self.NUM_SG_N
        var sg_n_idx = sg_id % Self.NUM_SG_N

        var row_base = tile_m * BM + sg_m_idx * SG_M
        var col_base = tile_n * BN + sg_n_idx * SG_N

        # Fully-OOB simdgroups: no later threadgroup-uniform op (no SMEM/barrier),
        # so the early return is safe (matches `AppleM5MatMul._run_gemm_body`).
        if row_base >= M or col_base >= N:
            return

        var sg_row_idx = row_base // SG_M
        var sg_col_idx = col_base // SG_N

        # A `(M, K)` row-major; this simdgroup's `(SG_M, K)` slab (K hoisted out
        # of the K-loop, matching `DenseALoader`). B `(N, K)` fp8 viewed col-major
        # `(K, N)` (stride (1, K)) -> the K axis is the contiguous fast axis of the
        # B-fragment load (coalesced), and hw_transpose_b = True feeds it as the
        # right operand. This simdgroup's `(K, SG_N)` col slab.
        var a_ptr = a.ptr.unsafe_origin_cast[ImmutUntrackedOrigin]()
        var a_mat = TileTensor(a_ptr, Layout(Coord(M, K), Coord(K, Idx[1])))
        var a_slab = a_mat.tile(Coord(Idx[SG_M], K), Coord(Int(sg_row_idx), 0))

        var w_ptr = weight.ptr.unsafe_origin_cast[ImmutUntrackedOrigin]()
        var b_mat = TileTensor(w_ptr, Layout(Coord(K, N), Coord(Idx[1], K)))
        var b_slab = b_mat.tile(Coord(K, Idx[SG_N]), Coord(0, Int(sg_col_idx)))

        var mma_op = Self.Mma()
        var accum = Self.Mma.zero_accum()

        var sg_m_end = row_base + SG_M
        var sg_n_end = col_base + SG_N
        var is_edge_tile = (sg_m_end > M) or (sg_n_end > N)

        var k_full_strips = K // BK
        var has_k_tail = (K % BK) != 0

        if is_edge_tile:
            var valid_rows = max(1, min(SG_M, M - row_base))
            var valid_cols = max(1, min(SG_N, N - col_base))
            var k_total = k_full_strips + (1 if has_k_tail else 0)
            for ks in range(k_total):
                var k_valid = min(BK, K - ks * BK)
                var a_sub = a_slab.tile[SG_M, BK](0, ks)
                var b_sub = b_slab.tile[BK, SG_N](ks, 0)
                mma_op.mma[bounded=True](
                    accum,
                    a_sub,
                    b_sub,
                    a_valid_rows=valid_rows,
                    b_valid_cols=valid_cols,
                    k_valid=k_valid,
                )
        else:
            for ks in range(k_full_strips):
                var a_sub = a_slab.tile[SG_M, BK](0, ks)
                var b_sub = b_slab.tile[BK, SG_N](ks, 0)
                mma_op.mma[bounded=False](
                    accum,
                    a_sub,
                    b_sub,
                    a_valid_rows=SG_M,
                    b_valid_cols=SG_N,
                    k_valid=BK,
                )
            if has_k_tail:
                var k_tail = K - k_full_strips * BK
                var a_sub = a_slab.tile[SG_M, BK](0, k_full_strips)
                var b_sub = b_slab.tile[BK, SG_N](k_full_strips, 0)
                mma_op.mma[bounded=True](
                    accum,
                    a_sub,
                    b_sub,
                    a_valid_rows=SG_M,
                    b_valid_cols=SG_N,
                    k_valid=k_tail,
                )

        # Store this lane's accumulators. Each `SIMD[f32, 8]` fragment maps to rows
        # {rb, rb+8} x cols {cb..cb+3} (the `_apple_frag_layout`, owned at the MMA
        # layer; `MmaOpApple._store_fragment`). Guard `< M`/`< N` for partial
        # tiles; cast to `c_type` on store.
        comptime for mi in range(Self.NUM_MMA_M):
            comptime for ni in range(Self.NUM_MMA_N):
                var frag = accum[mi * Self.NUM_MMA_N + ni]
                var frow = row_base + mi * 16
                var fcol = col_base + ni * 16
                comptime for e in range(8):
                    var gm = frow + mma_op.rb + (e // 4) * 8
                    var gn = fcol + mma_op.cb + (e % 4)
                    if gm < M and gn < N:
                        c.store[width=1](
                            Coord(gm, gn),
                            SIMD[Self.c_type, 1](frag[e].cast[Self.c_type]()),
                        )


@always_inline
def enqueue_matmul2d_fp8[
    c_type: DType = DType.float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    block_m: Int = 64,
    block_n: Int = 64,
    block_k: Int = 16,
    sg_m: Int = 32,
    sg_n: Int = 32,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[DType.bfloat16, ...],
    weight: TileTensor[DType.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    """Enqueue the tiled W8A16 GEMM: `out = a @ W_fp8^T` (raw, unscaled).

    `a` is the bf16 activation `(M, K)`, `weight` the FP8-E4M3 weight `(N, K)`
    (`transpose_b`). C is `(M, N)`. bf16 A x fp8 B -> fp32 on the native Apple MMA,
    direct DRAM B feed (no SMEM). Any M/N/K (bounded MMA + guarded store). The
    per-tensor scalar `weight_scale` folds post-matmul (graph lowering).

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability == 5`).
    """
    _require_apple_m5(ctx)

    comptime assert (
        c_type == DType.float16
        or c_type == DType.bfloat16
        or c_type == DType.float32
    ), "enqueue_matmul2d_fp8: c_type must be one of {fp16, bf16, fp32}"

    comptime MM = Matmul2dFp8[
        c_type,
        elementwise_lambda_fn,
        block_m,
        block_n,
        block_k,
        sg_m,
        sg_n,
    ]

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(Int(a.dim[0]()) == m, "A shape (M, K) must match C's M")
    debug_assert(Int(weight.dim[0]()) == n, "weight must be (N, K)")
    debug_assert(Int(weight.dim[1]()) == k, "weight must be (N, K)")

    var grid_m = ceildiv(m, MM.BM)
    var grid_n = ceildiv(n, MM.BN)

    comptime kernel = MM.run[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(weight).LayoutType,
    ]
    ctx.enqueue_function[kernel](
        c,
        a.as_immut(),
        weight.as_immut(),
        m,
        n,
        k,
        grid_dim=(grid_n, grid_m),
        block_dim=(MM.THREADS_PER_BLOCK),
    )

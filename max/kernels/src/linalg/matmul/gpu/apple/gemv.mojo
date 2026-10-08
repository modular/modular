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
"""Apple M5 bf16/fp16 decode GEMV: `out = x @ W^T` for a few activation rows.

At batch-1 (and small-batch) decode a Linear is a matrix-vector product: the
weight `W[N, K]` is read once and there is no activation matrix to feed the
simdgroup MMA, so the kernel is bound by the weight read. This is the 16-bit
sibling of `fp8_gemv.mojo`: register-resident, no threadgroup memory, no
`barrier()`, no MMA.

Each warp owns `rows_per_warp` consecutive rows of `W` (output columns). Its
32 lanes stride down K in `tile_k`-element chunks, so adjacent lanes read
adjacent 16-byte runs of every row. Per chunk a lane issues `rows_per_warp`
weight loads and `tile_m` activation loads, and the K loop is unrolled
`unroll` chunks deep so each lane keeps `rows_per_warp * unroll` weight loads
in flight; the load count in flight is what reaches DRAM bandwidth when N is
small. Products accumulate in fp32 vectors that are reduced once, after the K
loop, by a `warp.sum` per output.

The weight is read once for all `tile_m` activation rows, so the kernel also
covers small-batch decode (`M <= tile_m`) at the cost of one extra fp32
accumulator set per row.
"""

from std.collections import Optional
from std.math import ceildiv, min

from max.gpu import WARP_SIZE, global_idx, lane_id
from max.gpu.host import DeviceContext
import max.gpu.primitives.warp as warp

from layout import Coord, TensorEngine, TensorLayout, TileTensor

from linalg.utils import elementwise_epilogue_type


@inline(.always)
def _accumulate_chunk[
    in_type: DType,
    a_layout: TensorLayout,
    w_layout: TensorLayout,
    a_engine: TensorEngine,
    w_engine: TensorEngine,
    tile_m: Int,
    rows_per_warp: Int,
    tile_k: Int,
](
    a: TileTensor[in_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    weight: TileTensor[in_type, w_layout, ImmutAnyOrigin, Engine=w_engine],
    k0: Int,
    w_rows: Array[Int, rows_per_warp],
    m_idx: Array[Int, tile_m],
    mut acc: Array[SIMD[.float32, tile_k], tile_m * rows_per_warp],
):
    var xv = Array[SIMD[.float32, tile_k], tile_m](
        fill=SIMD[.float32, tile_k](0)
    )
    comptime for mi in range(tile_m):
        xv[mi] = a.load[width=tile_k](Coord(m_idx[mi], k0)).cast[.float32]()
    comptime for r in range(rows_per_warp):
        var wv = weight.load[width=tile_k](Coord(w_rows[r], k0)).cast[
            .float32
        ]()
        comptime for mi in range(tile_m):
            acc[mi * rows_per_warp + r] += xv[mi] * wv


@__name(
    t"apple_gemv_{c_type}_{in_type}_m{tile_m}_r{rows_per_warp}_k{tile_k}_u{unroll}"
)
def apple_gemv_kernel[
    c_type: DType,
    in_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    w_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    w_engine: TensorEngine,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    tile_m: Int,
    rows_per_warp: Int,
    tile_k: Int,
    unroll: Int,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[in_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    weight: TileTensor[in_type, w_layout, ImmutAnyOrigin, Engine=w_engine],
    m_arg: Int32,
    n_arg: Int32,
    k_arg: Int32,
):
    """Computes `rows_per_warp` columns of `c[:m] = a[:m] @ weight^T` per warp.

    `c` is `[M, N]`, `a` is `[M, K]` and `weight` is `[N, K]`, all row-major,
    with `M <= tile_m`. K must be a multiple of `tile_k`, so every chunk load
    is aligned to its width. Accumulation is fp32.
    """
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)

    var n0 = (Int(global_idx.x) // WARP_SIZE) * rows_per_warp
    if n0 >= n:
        return
    var lid = Int(lane_id())

    # The last warp may own rows past N. It re-reads row N - 1 in their place
    # (so the K loop has no per-row branch) and skips their stores. Rows past
    # M are handled the same way.
    var w_rows = Array[Int, rows_per_warp](fill=0)
    comptime for r in range(rows_per_warp):
        w_rows[r] = min(n0 + r, n - 1)
    var m_idx = Array[Int, tile_m](fill=0)
    comptime for mi in range(tile_m):
        m_idx[mi] = min(mi, m - 1)

    var acc = Array[SIMD[.float32, tile_k], tile_m * rows_per_warp](
        fill=SIMD[.float32, tile_k](0)
    )

    comptime accumulate = _accumulate_chunk[
        in_type,
        a_layout,
        w_layout,
        a_engine,
        w_engine,
        tile_m,
        rows_per_warp,
        tile_k,
    ]
    var nchunk = k // tile_k
    var chunk = lid
    while chunk + (unroll - 1) * WARP_SIZE < nchunk:
        comptime for u in range(unroll):
            accumulate(
                a, weight, (chunk + u * WARP_SIZE) * tile_k, w_rows, m_idx, acc
            )
        chunk += unroll * WARP_SIZE
    while chunk < nchunk:
        accumulate(a, weight, chunk * tile_k, w_rows, m_idx, acc)
        chunk += WARP_SIZE

    comptime for mi in range(tile_m):
        comptime for r in range(rows_per_warp):
            var dot = warp.sum(acc[mi * rows_per_warp + r].reduce_add())
            var col = n0 + r
            if lid == r and mi < m and col < n:
                var y = dot.cast[c_type]()
                comptime if elementwise_lambda_fn:
                    comptime epilogue = elementwise_lambda_fn.value()
                    epilogue[c_type, 1]((mi, col), y)
                else:
                    c.store(Coord(mi, col), y)


@inline(.always)
def enqueue_apple_gemv_config[
    c_type: DType,
    in_type: DType,
    //,
    *,
    tile_m: Int,
    rows_per_warp: Int,
    tile_k: Int = 8,
    unroll: Int = 2,
    warps_per_block: Int = 8,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[in_type, ...],
    weight: TileTensor[in_type, ...],
    ctx: DeviceContext,
) raises:
    """Enqueues `apple_gemv_kernel` with an explicit launch configuration.

    Parameters:
        c_type: Output element type. Accumulation is fp32.
        in_type: Activation and weight element type (bf16 or fp16).
        tile_m: Largest activation row count the launch supports.
        rows_per_warp: Weight rows (output columns) per warp.
        tile_k: Elements per lane per K chunk.
        unroll: K chunks per loop iteration.
        warps_per_block: Warps per threadgroup.
        elementwise_lambda_fn: Optional epilogue applied to each output.

    Args:
        c: Output `[M, N]`.
        a: Activation `[M, K]` with `M <= tile_m` and `K % tile_k == 0`.
        weight: Weight `[N, K]`.
        ctx: Device context to enqueue on.
    """
    comptime assert in_type in (
        DType.bfloat16,
        DType.float16,
    ), "apple_gemv: in_type must be bf16 or fp16"
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())
    debug_assert(m <= tile_m, "apple_gemv: M exceeds tile_m")
    debug_assert(k % tile_k == 0, "apple_gemv: K must be a multiple of tile_k")

    comptime BLK = warps_per_block * WARP_SIZE
    var num_warps = ceildiv(n, rows_per_warp)
    comptime kernel = apple_gemv_kernel[
        c_type,
        in_type,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(weight).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(weight).Engine,
        elementwise_lambda_fn,
        tile_m,
        rows_per_warp,
        tile_k,
        unroll,
    ]
    ctx.enqueue_function[kernel](
        c,
        a.as_imm(),
        weight.as_imm(),
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=ceildiv(num_warps, warps_per_block),
        block_dim=BLK,
    )


comptime APPLE_GEMV_MAX_M = 8
"""Largest M `enqueue_apple_gemv` serves. At M = 16 the per-row fp32
accumulators spill and the kernel falls below the tiled matmul."""

comptime _APPLE_GEMV_MIN_K_UP_TO_M4 = 512
"""K below which a warp's lanes run out of chunks to keep in flight
(`WARP_SIZE * tile_k * unroll`) and the tiled matmul wins, for M <= 4."""

comptime _APPLE_GEMV_MIN_K_ABOVE_M4 = 1024
"""The same floor for M > 4, whose `tile_m = 8` launch carries twice the
accumulators at half the unroll."""


@inline(.always)
def apple_gemv_small_batch_supported(m: Int, k: Int) -> Bool:
    """Returns whether `_matmul_gpu` routes an `[m, k]` activation here.

    Measured on M5 Max with `bench_matmul.mojo` (`bench_gemv_apple.yaml`)
    against the tiled `AppleM5MatMul`: up to 3x faster at decode shapes with
    K >= 1024 and 2 <= M <= 8. At K = 512 it wins for M <= 4 (1.13-1.50x) and
    loses at M = 8 (0.89-0.96x); at K = 256 it loses from M = 4. At M == 1 it
    is 0.84-1.10x the split-K GEMV, which already streams weights above 20 MB
    at 535-580 GB/s, so M == 1 stays there.
    """
    var min_k = (
        _APPLE_GEMV_MIN_K_UP_TO_M4 if m <= 4 else _APPLE_GEMV_MIN_K_ABOVE_M4
    )
    return 2 <= m <= APPLE_GEMV_MAX_M and k >= min_k and k % 8 == 0


@inline(.always)
def enqueue_apple_gemv[
    c_type: DType,
    in_type: DType,
    //,
    *,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[in_type, ...],
    weight: TileTensor[in_type, ...],
    ctx: DeviceContext,
) raises:
    """Enqueues `c = a @ weight^T` for `1 <= M <= APPLE_GEMV_MAX_M`.

    Picks the smallest `tile_m` covering M, one weight row per warp, with the
    K unroll measured best for that `tile_m`.

    Parameters:
        c_type: Output element type. Accumulation is fp32.
        in_type: Activation and weight element type (bf16 or fp16).
        elementwise_lambda_fn: Optional epilogue applied to each output.

    Args:
        c: Output `[M, N]`.
        a: Activation `[M, K]` with `K % 8 == 0`.
        weight: Weight `[N, K]`.
        ctx: Device context to enqueue on.
    """
    var m = Int(c.dim[0]())
    debug_assert(1 <= m <= APPLE_GEMV_MAX_M, "apple_gemv: M must be in [1, 8]")
    if m <= 2:
        enqueue_apple_gemv_config[
            tile_m=2,
            rows_per_warp=1,
            unroll=4,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c, a, weight, ctx)
    elif m <= 4:
        enqueue_apple_gemv_config[
            tile_m=4,
            rows_per_warp=1,
            unroll=4,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c, a, weight, ctx)
    else:
        enqueue_apple_gemv_config[
            tile_m=8,
            rows_per_warp=1,
            unroll=2,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c, a, weight, ctx)

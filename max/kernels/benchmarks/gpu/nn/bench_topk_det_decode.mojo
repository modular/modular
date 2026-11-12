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
"""Per-launch time of the deterministic indexer top-k at decode shapes.

A captured decode graph freezes the indexer's row stride at the served
maximum length, so every decode and verify row is `N = 262152` wide with a
per-row live bound, bf16 scores and `K = 2048`, over fewer rows than SMs.
This times the kernels that serve the `ordered=False, deterministic=True`
contract there, launched back to back so a launch's cost is the stream's
throughput rather than one launch's host round trip, and checks that they
agree slot for slot:

- `ranked`: the one-CTA-per-row ranked streaming select
  (`_histsel_topk_bounded_kernel`), as the dispatch launched it before the
  cluster kernel existed.
- `cluster`: `topk_cluster_det` at the cluster size the dispatch picks, or
  `--cs`.
- `dispatch`: `persistent_topk_block_split[deterministic=True]`.

    bench_topk_det_decode --rows=1 --bound=10000 --dist=normal
    bench_topk_det_decode --sweep=True

`--sweep=True` runs rows 1, 4, 32, 128 against bounds 1536..262000 and both
distributions in one process.
"""

from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import B200
from internal_utils import arg_parse
from structured_kernels.trace_buf import NullTrace

from nn.attention.gpu.nvidia.topk_cluster_det import (
    topk_cluster_det,
    topk_cluster_det_cluster_size,
)
from nn.topk_bitonic import (
    RowBoundsPtr,
    _histsel_topk_bounded_kernel,
    _HSEL_RANK_BITS,
    _HSEL_SEL_CAP,
    _HSEL_SMEM_BYTES,
    _HSEL_TAIL_BITS,
    _PTOPK_BLOCK,
    persistent_topk_block_split,
)


@inline(.always)
def _mix64(x: UInt64) -> UInt64:
    """SplitMix64's finalizer."""
    var z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@inline(.always)
def _u01(i: UInt64) -> Float64:
    var h = _mix64((i + 1) * 0x9E3779B97F4A7C15)
    return Float64(h >> 11) * (1.0 / 9007199254740992.0)


def _sample(dist: String, r: Int, c: Int) -> Float32:
    """`normal`, or `q17`: 17 levels, the plateaus fp8 indexer scores
    collapse to."""
    var base = (UInt64(r) << 40) + (UInt64(c) << 3)
    if dist == "q17":
        return Float32(Int(_u01(base) * 17.0)) / 16.0
    var acc = Float64(0.0)
    for k in range(6):
        acc += _u01(base + UInt64(k))
    return Float32((acc - 3.0) * 1.4142135623730951)


def _time_us[
    F: def() raises -> None
](ctx: DeviceContext, launch: F, iters: Int) raises -> Float64:
    """Microseconds per launch over `iters` back-to-back launches."""
    for _ in range(50):
        launch()
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        launch()
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(iters) / 1.0e3


def _run(
    ctx: DeviceContext,
    rows: Int,
    N: Int,
    K: Int,
    bound: Int,
    dist: String,
    iters: Int,
    cs_arg: Int,
) raises:
    var scores_buf = ctx.enqueue_create_buffer[.bfloat16](rows * N)
    var ref_buf = ctx.enqueue_create_buffer[.int32](rows * K)
    var out_buf = ctx.enqueue_create_buffer[.int32](rows * K)
    var dsp_buf = ctx.enqueue_create_buffer[.int32](rows * K)
    var bounds_buf = ctx.enqueue_create_buffer[.int32](rows)
    with scores_buf.map_to_host() as h:
        for r in range(rows):
            for c in range(min(bound, N)):
                h[r * N + c] = _sample(dist, r, c).cast[.bfloat16]()
    with bounds_buf.map_to_host() as b:
        for r in range(rows):
            b[r] = Int32(bound)
    ctx.synchronize()

    var scores = rebind[ImmPointer[BFloat16, ImmutAnyOrigin]](
        scores_buf.unsafe_ptr()
    )
    var ref_idxs = rebind[MutPointer[Int32, MutAnyOrigin]](ref_buf.unsafe_ptr())
    var out_idxs = rebind[MutPointer[Int32, MutAnyOrigin]](out_buf.unsafe_ptr())
    var dsp_idxs = rebind[MutPointer[Int32, MutAnyOrigin]](dsp_buf.unsafe_ptr())
    var bounds = RowBoundsPtr(
        rebind[ImmPointer[Int32, ImmutAnyOrigin]](bounds_buf.unsafe_ptr())
    )
    var cs = cs_arg if cs_arg > 0 else topk_cluster_det_cluster_size[
        B200.sm_count
    ](rows)

    @inline(.always)
    def ranked() raises {imm}:
        ctx.enqueue_function[
            _histsel_topk_bounded_kernel[
                NullTrace,
                RB=RowBoundsPtr,
                prefetch=True,
                tail_bits=_HSEL_TAIL_BITS,
                rank_bits=_HSEL_RANK_BITS,
                rank_slots=True,
                sel_cap=_HSEL_SEL_CAP,
                in_dtype=.bfloat16,
            ]
        ](
            scores,
            ref_idxs,
            Int32(N),
            Int32(K),
            bounds,
            NullTrace(),
            grid_dim=rows,
            block_dim=_PTOPK_BLOCK,
            shared_mem_bytes=_HSEL_SMEM_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(_HSEL_SMEM_BYTES)
            ),
        )

    @inline(.always)
    def cluster() raises {imm}:
        topk_cluster_det(
            ctx,
            rebind[ImmPointer[BFloat16, ImmUntrackedOrigin]](scores),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](out_idxs),
            N,
            K,
            rows,
            bounds,
            cs,
        )

    @inline(.always)
    def dispatch() raises {imm}:
        persistent_topk_block_split[deterministic=True](
            ctx, scores, dsp_idxs, N, K, rows, Optional(bounds)
        )

    var t_ranked = _time_us(ctx, ranked, iters)
    var t_cluster = _time_us(ctx, cluster, iters)
    var t_dispatch = _time_us(ctx, dispatch, iters)

    var mismatches = 0
    with ref_buf.map_to_host() as rh:
        with out_buf.map_to_host() as oh:
            with dsp_buf.map_to_host() as dh:
                for i in range(rows * K):
                    if rh[i] != oh[i]:
                        mismatches += 1
                    if rh[i] != dh[i]:
                        mismatches += 1
    print(
        t"RESULT rows={rows} N={N} K={K} bound={bound} dist={dist} cs={cs}"
        t" ranked_us={t_ranked} cluster_us={t_cluster}"
        t" dispatch_us={t_dispatch} mismatches={mismatches}"
    )
    _ = scores_buf
    _ = ref_buf
    _ = out_buf
    _ = dsp_buf
    _ = bounds_buf


def main() raises:
    var rows = arg_parse("rows", 1)
    var N = arg_parse("N", 262152)
    var K = arg_parse("K", 2048)
    var bound = arg_parse("bound", 10000)
    var dist = arg_parse("dist", String("normal"))
    var iters = arg_parse("iters", 2000)
    var cs = arg_parse("cs", 0)
    var sweep = arg_parse("sweep", False)

    with DeviceContext() as ctx:
        if not sweep:
            _run(ctx, rows, N, K, bound, dist, iters, cs)
            return
        for r in [1, 4, 32, 128]:
            for b in [1536, 4000, 10000, 32000, 128000, 262000]:
                for d in ["normal", "q17"]:
                    _run(ctx, r, N, K, b, String(d), iters, 0)

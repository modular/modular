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
"""The two MAX GPU top-k kernels on the shapes DeepSelect's harness reports.

deepseek-ai/DeepSelect `tests/test.py --perf-only` times two scenarios and
prints effective bandwidth, counting input bytes plus every output written:

    indexer:  bf16 input, K=512, rows in {6,256,512,768,4096}, N in 1K..1M,
              indices only, unordered
    sampler:  fp32 input, K=512, N=129280, rows in {6,256,512,768,4096},
              sorted values and int64 indices

Two MAX kernels can serve those contracts. `topk_gpu` is what `mo.top_k`
lowers to: any dtype, always sorted, values and indices. It is a k-round
block max-extract, so K=512 is far from the K<=255 the sampler tunes for.
`persistent_topk_block_split` is the DSA indexer's own select: fp32 scores
in, unordered int32 indices out, which is exactly the indexer contract above
except for the input width, so on bf16 shapes it reads twice the bytes.

Each launch is timed on its own with CUDA events, after a 1 GB memset that
both evicts L2 and keeps the stream busy while the host enqueues the next
launch, so neither a cache-warm input nor launch latency lands in the
number. DeepSelect's harness reaches the same state with an 8 GB memset
before each profiled run.

    --dtype=bfloat16 (int32 indices, the indexer contract) or
    --dtype=float32  (int64 indices, the sampler contract); a sweep picks
    its own scenario's dtype unless one is given.
    --kernel=topk_gpu|bitonic|all  --sweep=indexer|sampler
    --rows=6 --N=16384 --K=512 --iters=20 --verify=True   (single shape)
"""

from std.math import ceildiv
from std.sys.info import size_of

from internal_utils import arg_parse
from layout import TensorLayout, TileTensor, row_major
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from nn.topk import topk_gpu
from nn.topk_bitonic import persistent_topk_block_split

comptime FILL_BLOCK = 256
comptime FILL_PER_THREAD = 8
comptime IH_TERMS = 6
comptime IH_SCALE: Float32 = 1.4142135
comptime FLUSH_BYTES = 1 << 30
comptime VERIFY_ROWS = 16
comptime SLOW_LAUNCH_NS = 200_000_000
comptime SLOW_ITERS = 5


@inline(.always)
def _mix64(x: UInt64) -> UInt64:
    var z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@inline(.always)
def _u01(x: UInt64) -> Float32:
    return Float32(Float64(_mix64(x) >> 11) * (1.0 / 9007199254740992.0))


@inline(.always)
def _normal(i: UInt64, seed: UInt64) -> Float32:
    # Irwin-Hall sum of six uniforms; each term has variance 1/2, hence the
    # scale. Hash-keyed so any thread can produce any element.
    var base = (i + seed * 0x9E3779B97F4A7C15) * UInt64(IH_TERMS)
    var acc = Float32(0)
    comptime for k in range(IH_TERMS):
        acc += _u01(base + UInt64(k))
    return (acc - Float32(IH_TERMS) / 2) * IH_SCALE


def _fill_normal_kernel[
    dtype: DType, LT: TensorLayout
](dst: TileTensor[dtype, LT, MutAnyOrigin], n: Int64, seed: UInt64):
    comptime assert dst.flat_rank == 1
    var base = global_idx.x * FILL_PER_THREAD
    for j in range(FILL_PER_THREAD):
        var i = base + j
        if i < Int(n):
            dst[i] = rebind[dst.ElementType](
                _normal(UInt64(i), seed).cast[dtype]()
            )


def _fill_normal[
    dtype: DType
](ctx: DeviceContext, buf: DeviceBuffer[dtype], n: Int, seed: UInt64) raises:
    var layout = row_major(n)
    var tt = TileTensor(buf, layout)
    comptime kernel = _fill_normal_kernel[dtype, type_of(layout)]
    ctx.enqueue_function[kernel](
        tt,
        Int64(n),
        seed,
        grid_dim=ceildiv(n, FILL_BLOCK * FILL_PER_THREAD),
        block_dim=FILL_BLOCK,
    )


def _measure[
    FuncType: def(DeviceContext) raises -> None
](
    ctx: DeviceContext,
    launch: FuncType,
    flush: DeviceBuffer[.uint8],
    iters: Int,
) raises -> Float64:
    """Mean microseconds per launch, each launch timed alone behind an L2
    flush."""
    for _ in range(3):
        launch(ctx)
    ctx.synchronize()
    var probe_ns = ctx.execution_time(launch, 1)
    var n = SLOW_ITERS if probe_ns > SLOW_LAUNCH_NS else iters
    var total_ns = 0
    for _ in range(n):
        ctx.enqueue_memset(flush, UInt8(0))
        total_ns += ctx.execution_time(launch, 1)
    return Float64(total_ns) / Float64(n) / 1000.0


def _verify[
    dtype: DType, idx_type: DType
](
    ctx: DeviceContext,
    in_buf: DeviceBuffer[dtype],
    idxs_buf: DeviceBuffer[idx_type],
    vals_buf: Optional[DeviceBuffer[dtype]],
    N: Int,
    K: Int,
    check_rows: Int,
) raises -> Bool:
    """Checks the first `check_rows` rows: indices in range and unique, values
    equal to the gathered input, and fewer than K inputs above the smallest
    selected value."""
    var in_view = in_buf.create_sub_buffer[dtype](0, check_rows * N)
    var idx_view = idxs_buf.create_sub_buffer[idx_type](0, check_rows * K)
    var in_host = ctx.enqueue_create_host_buffer[dtype](check_rows * N)
    var idx_host = ctx.enqueue_create_host_buffer[idx_type](check_rows * K)
    var vals_host = ctx.enqueue_create_host_buffer[dtype](check_rows * K)
    ctx.enqueue_copy(in_host, in_view)
    ctx.enqueue_copy(idx_host, idx_view)
    if vals_buf:
        var vals_view = vals_buf.value().create_sub_buffer[dtype](
            0, check_rows * K
        )
        ctx.enqueue_copy(vals_host, vals_view)
        ctx.synchronize()
        _ = vals_view^
    ctx.synchronize()

    var mark = List[Int](length=N, fill=-1)
    var ok = True
    for r in range(check_rows):
        var min_selected = Float32(1.0e30)
        for j in range(K):
            var idx = Int(idx_host[r * K + j])
            if idx < 0 or idx >= N:
                ok = False
                continue
            if mark[idx] == r:
                ok = False
            mark[idx] = r
            var x = in_host[r * N + idx].cast[.float32]()
            if vals_buf and vals_host[r * K + j].cast[.float32]() != x:
                ok = False
            if x < min_selected:
                min_selected = x
        var greater = 0
        for c in range(N):
            if in_host[r * N + c].cast[.float32]() > min_selected:
                greater += 1
        if greater >= K:
            ok = False
    _ = in_view^
    _ = idx_view^
    return ok


def _report(
    kernel: StaticString,
    dtype: DType,
    idx_type: DType,
    rows: Int,
    N: Int,
    K: Int,
    mean_us: Float64,
    num_bytes: Int,
    check: String,
):
    print(
        "RESULT kernel=",
        kernel,
        " dtype=",
        dtype,
        " idx=",
        idx_type,
        " rows=",
        rows,
        " N=",
        N,
        " K=",
        K,
        " mean_us=",
        mean_us,
        " tbs=",
        Float64(num_bytes) / mean_us / 1.0e6,
        " check=",
        check,
        sep="",
    )


def bench_topk_gpu[
    dtype: DType, idx_type: DType
](
    ctx: DeviceContext,
    flush: DeviceBuffer[.uint8],
    rows: Int,
    N: Int,
    K: Int,
    iters: Int,
    verify: Bool,
) raises:
    var in_buf = ctx.enqueue_create_buffer[dtype](rows * N)
    var vals_buf = ctx.enqueue_create_buffer[dtype](rows * K)
    var idxs_buf = ctx.enqueue_create_buffer[idx_type](rows * K)
    _fill_normal(ctx, in_buf, rows * N, 1)
    var in_tt = TileTensor(in_buf, row_major(rows, N))
    var vals_tt = TileTensor(vals_buf, row_major(rows, K))
    var idxs_tt = TileTensor(idxs_buf, row_major(rows, K))
    ctx.synchronize()

    @inline(.always)
    def launch(c: DeviceContext) raises {imm}:
        topk_gpu[sampling=False, largest=True](c, K, in_tt, vals_tt, idxs_tt)

    var mean_us = _measure(ctx, launch, flush, iters)
    var check = String("skipped")
    if verify:
        check = "ok" if _verify(
            ctx,
            in_buf,
            idxs_buf,
            Optional[DeviceBuffer[dtype]](vals_buf),
            N,
            K,
            min(rows, VERIFY_ROWS),
        ) else "FAIL"
    var num_bytes = rows * N * size_of[dtype]() + rows * K * (
        size_of[dtype]() + size_of[idx_type]()
    )
    _report("topk_gpu", dtype, idx_type, rows, N, K, mean_us, num_bytes, check)
    _ = in_buf^
    _ = vals_buf^
    _ = idxs_buf^


def bench_bitonic(
    ctx: DeviceContext,
    flush: DeviceBuffer[.uint8],
    rows: Int,
    N: Int,
    K: Int,
    iters: Int,
    verify: Bool,
) raises:
    var in_buf = ctx.enqueue_create_buffer[.float32](rows * N)
    var idxs_buf = ctx.enqueue_create_buffer[.int32](rows * K)
    _fill_normal(ctx, in_buf, rows * N, 1)
    ctx.synchronize()

    @inline(.always)
    def launch(c: DeviceContext) raises {imm}:
        persistent_topk_block_split[ordered=False, deterministic=False](
            c,
            rebind[ImmPointer[Float32, ImmutAnyOrigin]](in_buf.unsafe_ptr()),
            rebind[MutPointer[Int32, MutAnyOrigin]](idxs_buf.unsafe_ptr()),
            N,
            K,
            rows,
        )

    var mean_us = _measure(ctx, launch, flush, iters)
    var check = String("skipped")
    if verify:
        check = "ok" if _verify(
            ctx, in_buf, idxs_buf, None, N, K, min(rows, VERIFY_ROWS)
        ) else "FAIL"
    var num_bytes = rows * N * size_of[Float32]() + rows * K * size_of[Int32]()
    _report("bitonic", .float32, .int32, rows, N, K, mean_us, num_bytes, check)
    _ = in_buf^
    _ = idxs_buf^


def _run_shape[
    dtype: DType, idx_type: DType
](
    ctx: DeviceContext,
    flush: DeviceBuffer[.uint8],
    kernel: String,
    rows: Int,
    N: Int,
    K: Int,
    iters: Int,
    verify: Bool,
) raises:
    if kernel == "topk_gpu" or kernel == "all":
        bench_topk_gpu[dtype, idx_type](ctx, flush, rows, N, K, iters, verify)
    if kernel == "bitonic" or kernel == "all":
        bench_bitonic(ctx, flush, rows, N, K, iters, verify)


def _sweep[
    dtype: DType, idx_type: DType
](
    ctx: DeviceContext,
    flush: DeviceBuffer[.uint8],
    kernel: String,
    sweep: String,
    rows: Int,
    N: Int,
    K: Int,
    iters: Int,
    verify: Bool,
) raises:
    var rows_list = [6, 256, 512, 768, 4096]
    if sweep == "indexer":
        var n_list = [
            1024,
            4096,
            16384,
            65536,
            131072,
            262144,
            524288,
            1048576,
        ]
        for r in rows_list:
            for n in n_list:
                _run_shape[dtype, idx_type](
                    ctx, flush, kernel, r, n, K, iters, verify
                )
    elif sweep == "sampler":
        for r in rows_list:
            _run_shape[dtype, idx_type](
                ctx, flush, kernel, r, 129280, K, iters, verify
            )
    else:
        _run_shape[dtype, idx_type](
            ctx, flush, kernel, rows, N, K, iters, verify
        )


def main() raises:
    var kernel = arg_parse("kernel", String("all"))
    var sweep = arg_parse("sweep", String(""))
    var dtype = arg_parse("dtype", String(""))
    var rows = arg_parse("rows", 6)
    var N = arg_parse("N", 16384)
    var K = arg_parse("K", 512)
    var iters = arg_parse("iters", 20)
    var verify = arg_parse("verify", True)

    if kernel not in ["topk_gpu", "bitonic", "all"]:
        raise Error("unknown kernel: ", kernel)
    if sweep not in ["", "indexer", "sampler"]:
        raise Error("unknown sweep: ", sweep)
    if dtype == "":
        dtype = "float32" if sweep == "sampler" else "bfloat16"
    if dtype not in ["bfloat16", "float32"]:
        raise Error("unknown dtype: ", dtype)

    with DeviceContext() as ctx:
        var flush = ctx.enqueue_create_buffer[.uint8](FLUSH_BYTES)
        ctx.synchronize()
        if dtype == "float32":
            _sweep[.float32, .int64](
                ctx, flush, kernel, sweep, rows, N, K, iters, verify
            )
        else:
            _sweep[.bfloat16, .int32](
                ctx, flush, kernel, sweep, rows, N, K, iters, verify
            )
        _ = flush^

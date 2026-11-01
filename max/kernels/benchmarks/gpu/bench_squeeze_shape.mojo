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
"""Benchmarks the GPU `mo.squeeze_shape` / `mo.unsqueeze_shape` kernels.

Calls the `squeeze_shape_gpu` / `unsqueeze_shape_gpu` launchers that the
registered ops call. Every case is checked against a host reference before it
is timed.

Two timings per case, both from the bench framework:

- `*_gpu`: device time per launch, from `DeviceContext.execution_time`
  (GPU events around a batch of back-to-back launches).
- `*_e2e`: host wall time of one launch plus `ctx.synchronize()`.

With `--nsys` the harness skips the bench framework: after verifying, it
enqueues each case `NSYS_LAUNCHES` times in the order it prints, so a profile
of the run can be split into per-case blocks of kernel launches.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.sys import argv
from std.testing import assert_equal

from builtin_kernels.gather_scatter import (
    squeeze_shape_gpu,
    unsqueeze_shape_gpu,
)
from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext

comptime NSYS_LAUNCHES = 200
comptime shape_dtype = DType.int64


def _case_name[
    squeeze: Bool, rank: Int, num_indices: Int, indices_type: DType
]() -> String:
    var op = String("squeeze") if squeeze else String("unsqueeze")
    return String(t"{op}/rank={rank}/n={num_indices}/idx={indices_type}")


def _shape_indices(index_rank: Int, num_indices: Int) -> List[Int]:
    """Returns distinct, unsorted shape indices, every other one negative."""
    var step = max(index_rank // num_indices, 1)
    var dims = List[Int]()
    for k in range(num_indices):
        dims.append(index_rank - 1 - k * step)
    if num_indices >= 2:
        var first = dims[0]
        dims[0] = dims[1]
        dims[1] = first
    for k in range(0, num_indices, 2):
        dims[k] -= index_rank
    return dims^


def _reference(
    squeeze: Bool, shape: List[Int], indices: List[Int], index_rank: Int
) -> List[Int]:
    var listed = List[Bool](length=index_rank, fill=False)
    for i in indices:
        listed[i + index_rank if i < 0 else i] = True
    var out = List[Int]()
    if squeeze:
        for d in range(len(shape)):
            if not listed[d]:
                out.append(shape[d])
    else:
        var j = 0
        for d in range(index_rank):
            if listed[d]:
                out.append(1)
            else:
                out.append(shape[j])
                j += 1
    return out^


def run_case[
    squeeze: Bool, rank: Int, num_indices: Int, indices_type: DType
](mut m: Bench, ctx: DeviceContext, nsys: Bool) raises:
    comptime out_rank = rank - num_indices if squeeze else rank + num_indices
    comptime index_rank = rank if squeeze else out_rank
    var name = _case_name[squeeze, rank, num_indices, indices_type]()

    var shape = List[Int]()
    for d in range(rank):
        shape.append(10 + d)
    var indices = _shape_indices(index_rank, num_indices)
    var expected = _reference(squeeze, shape, indices, index_rank)

    var in_host = ctx.enqueue_create_host_buffer[shape_dtype](rank)
    var idx_host = ctx.enqueue_create_host_buffer[indices_type](num_indices)
    var out_host = ctx.enqueue_create_host_buffer[shape_dtype](out_rank)
    ctx.synchronize()
    for d in range(rank):
        in_host[d] = Scalar[shape_dtype](shape[d])
    for k in range(num_indices):
        idx_host[k] = Scalar[indices_type](indices[k])

    var in_dev = ctx.enqueue_create_buffer[shape_dtype](rank)
    var idx_dev = ctx.enqueue_create_buffer[indices_type](num_indices)
    var out_dev = ctx.enqueue_create_buffer[shape_dtype](out_rank)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    out_dev.enqueue_fill(-1)

    var input = in_dev.unsafe_ptr().as_unsafe_any_origin()
    var shape_indices = idx_dev.unsafe_ptr().as_unsafe_any_origin()
    var output = out_dev.unsafe_ptr().as_unsafe_any_origin()

    @inline(.always)
    def launch(ctx: DeviceContext) raises {imm}:
        comptime if squeeze:
            squeeze_shape_gpu(
                output, 1, input, 1, shape_indices, 1, rank, num_indices, ctx
            )
        else:
            unsqueeze_shape_gpu(
                output, 1, input, 1, shape_indices, 1, rank, num_indices, ctx
            )

    launch(ctx)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    for d in range(out_rank):
        assert_equal(
            Int(out_host[d]), expected[d], String(t"{name}: output[{d}]")
        )

    if nsys:
        print("NSYS_CASE", name, NSYS_LAUNCHES)
        for _ in range(NSYS_LAUNCHES):
            launch(ctx)
        ctx.synchronize()
        return

    @inline(.always)
    def bench_gpu(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch, ctx)

    @inline(.always)
    def bench_e2e(mut b: Bencher) raises {imm}:
        def launch_and_sync() raises {imm}:
            launch(ctx)
            ctx.synchronize()

        b.iter(launch_and_sync)

    m.bench_function(bench_gpu, BenchId(name, "gpu"))
    m.bench_function(bench_e2e, BenchId(name, "e2e"))


def run_op[
    squeeze: Bool, indices_type: DType
](mut m: Bench, ctx: DeviceContext, nsys: Bool) raises:
    run_case[squeeze, 2, 1, indices_type](m, ctx, nsys)
    run_case[squeeze, 4, 1, indices_type](m, ctx, nsys)
    run_case[squeeze, 4, 2, indices_type](m, ctx, nsys)
    run_case[squeeze, 8, 3, indices_type](m, ctx, nsys)
    comptime if not squeeze:
        run_case[squeeze, 8, 8, indices_type](m, ctx, nsys)
    run_case[squeeze, 16, 4, indices_type](m, ctx, nsys)
    run_case[squeeze, 64, 8, indices_type](m, ctx, nsys)


def main() raises:
    var nsys = False
    for arg in argv():
        if arg == "--nsys":
            nsys = True

    var m = Bench(
        BenchConfig(min_runtime_secs=0.2, max_runtime_secs=5.0, max_iters=2000)
    )
    with DeviceContext() as ctx:
        run_op[True, DType.int64](m, ctx, nsys)
        run_op[True, DType.int32](m, ctx, nsys)
        run_op[False, DType.int64](m, ctx, nsys)
        run_op[False, DType.int32](m, ctx, nsys)
    if not nsys:
        m.dump_report()

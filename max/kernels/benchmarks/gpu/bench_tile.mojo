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
"""Benchmarks the GPU `mo.tile` kernel (nn.tile, the registered body).

Every case is checked against a host reference before it is timed.

Two timings per case, both from the bench framework:

- `*_gpu`: device time per launch, from `DeviceContext.execution_time`
  (GPU events around a batch of back-to-back launches).
- `*_e2e`: host wall time of one launch plus `ctx.synchronize()`.

With `--nsys` the harness skips the bench framework: after verifying, it
enqueues each case `NSYS_LAUNCHES` times in the order it prints, so a
profile of the run can be split into per-case blocks of kernel launches.
"""

from layout import TileTensor, row_major
from nn.tile import tile
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.sys import argv
from std.testing import assert_equal
from std.utils import IndexList

from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext

comptime NSYS_LAUNCHES = 200
comptime REPEATS_TYPE = DType.int64


def _case_name[dtype: DType, rank: Int, *sr: Int]() -> String:
    """Case id like `tile/f32/r2_[2048,1024]x[1,4]` (shape then repeats)."""
    var name = String(t"tile/{dtype}_r{rank}_[")
    comptime for i in range(rank):
        comptime if i > 0:
            name += ","
        name += String(sr[i])
    name += "]x["
    comptime for i in range(rank):
        comptime if i > 0:
            name += ","
        name += String(sr[rank + i])
    name += "]"
    return name


def _numel[rank: Int, *sr: Int, tiled: Bool]() -> Int:
    """Element count of the input (`tiled=False`) or output (`True`)."""
    var n = 1
    comptime for i in range(rank):
        comptime if tiled:
            n *= sr[i] * sr[rank + i]
        else:
            n *= sr[i]
    return n


def run_case[
    dtype: DType, rank: Int, *sr: Int
](mut m: Bench, ctx: DeviceContext, nsys: Bool) raises:
    comptime in_numel = _numel[rank, *sr, tiled=False]()
    comptime out_numel = _numel[rank, *sr, tiled=True]()
    var name = _case_name[dtype, rank, *sr]()

    # Host input with a known pattern; the reference tiles it in the same
    # order the kernel must produce.
    var in_host = ctx.enqueue_create_host_buffer[dtype](in_numel)
    var out_host = ctx.enqueue_create_host_buffer[dtype](out_numel)
    ctx.synchronize()
    for i in range(in_numel):
        in_host[i] = Scalar[dtype](i % 997)

    var in_dev = ctx.enqueue_create_buffer[dtype](in_numel)
    var out_dev = ctx.enqueue_create_buffer[dtype](out_numel)
    ctx.enqueue_copy(in_dev, in_host)
    out_dev.enqueue_fill(0)

    # Repeats live in a host stack array (the kernel reads them on the
    # host and captures the values by value, exactly as the graph
    # compiler's execute path does).
    var repeats_stack = Array[Scalar[REPEATS_TYPE], rank](fill={})
    comptime for i in range(rank):
        repeats_stack[i] = Scalar[REPEATS_TYPE](sr[rank + i])

    var in_dims = IndexList[rank]()
    var out_dims = IndexList[rank]()
    comptime for i in range(rank):
        in_dims[i] = sr[i]
        out_dims[i] = sr[i] * sr[rank + i]

    @inline(.always)
    def launch(ctx: DeviceContext) raises {mut}:
        # Per-rank layout literals: `comptime if` elaborates only the
        # taken branch, so each branch can bind differently-shaped
        # TileTensors.
        comptime if rank == 1:
            var input = TileTensor(in_dev, row_major[sr[0]]())
            var output = TileTensor(out_dev, row_major[sr[0] * sr[1]]())
            var repeats = TileTensor(repeats_stack, row_major[rank]())
            tile[target="gpu"](
                input.make_dynamic[.int64](),
                repeats.make_dynamic[.int64](),
                output.make_dynamic[.int64](),
                ctx,
            )
        elif rank == 2:
            var input = TileTensor(in_dev, row_major[sr[0], sr[1]]())
            var output = TileTensor(
                out_dev,
                row_major[sr[0] * sr[2], sr[1] * sr[3]](),
            )
            var repeats = TileTensor(repeats_stack, row_major[rank]())
            tile[target="gpu"](
                input.make_dynamic[.int64](),
                repeats.make_dynamic[.int64](),
                output.make_dynamic[.int64](),
                ctx,
            )
        elif rank == 3:
            var input = TileTensor(in_dev, row_major[sr[0], sr[1], sr[2]]())
            var output = TileTensor(
                out_dev,
                row_major[sr[0] * sr[3], sr[1] * sr[4], sr[2] * sr[5]](),
            )
            var repeats = TileTensor(repeats_stack, row_major[rank]())
            tile[target="gpu"](
                input.make_dynamic[.int64](),
                repeats.make_dynamic[.int64](),
                output.make_dynamic[.int64](),
                ctx,
            )
        else:
            var input = TileTensor(
                in_dev, row_major[sr[0], sr[1], sr[2], sr[3]]()
            )
            var output = TileTensor(
                out_dev,
                row_major[
                    sr[0] * sr[4],
                    sr[1] * sr[5],
                    sr[2] * sr[6],
                    sr[3] * sr[7],
                ](),
            )
            var repeats = TileTensor(repeats_stack, row_major[rank]())
            tile[target="gpu"](
                input.make_dynamic[.int64](),
                repeats.make_dynamic[.int64](),
                output.make_dynamic[.int64](),
                ctx,
            )

    launch(ctx)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    # Reference: expected[i] = input[flat coord of i wrapped per axis].
    # The decode walks axes innermost-first with runtime divmod; this is a
    # host-side check, so plain arithmetic is fine.
    for i in range(out_numel):
        var flat = i
        var src = 0
        var src_stride = 1
        for ax in range(rank - 1, -1, -1):
            var coord = flat % out_dims[ax]
            flat = (flat - coord) // out_dims[ax]
            src += (coord % in_dims[ax]) * src_stride
            src_stride *= in_dims[ax]
        assert_equal(
            Int(out_host[i]),
            Int(Scalar[dtype](src % 997)),
            String(t"{name}: output[{i}]"),
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


def run_fp32(mut m: Bench, ctx: DeviceContext, nsys: Bool) raises:
    comptime dt = DType.float32
    # Large rank-2, all axes repeated: the classic expand-embeddings copy.
    run_case[dt, 2, 1024, 1024, 2, 2](m, ctx, nsys)
    # Large rank-2, last axis only: contiguous blocks per row.
    run_case[dt, 2, 4096, 1024, 1, 4](m, ctx, nsys)
    # Large rank-3, middle axis: strided source reads.
    run_case[dt, 3, 64, 128, 512, 1, 8, 1](m, ctx, nsys)
    # Medium rank-4: MoE-style gather blocks.
    run_case[dt, 4, 8, 128, 32, 64, 2, 1, 4, 1](m, ctx, nsys)
    # Small rank-1: launch-floor case.
    run_case[dt, 1, 1024, 3](m, ctx, nsys)
    # Tiny rank-2: pure launch overhead.
    run_case[dt, 2, 2, 3, 2, 2](m, ctx, nsys)


def run_bf16(mut m: Bench, ctx: DeviceContext, nsys: Bool) raises:
    comptime dt = DType.bfloat16
    run_case[dt, 2, 1024, 1024, 2, 2](m, ctx, nsys)
    run_case[dt, 2, 4096, 1024, 1, 4](m, ctx, nsys)


def run_int8(mut m: Bench, ctx: DeviceContext, nsys: Bool) raises:
    comptime dt = DType.int8
    run_case[dt, 2, 8192, 2048, 1, 2](m, ctx, nsys)
    run_case[dt, 3, 128, 128, 256, 1, 1, 8](m, ctx, nsys)


def run_int64(mut m: Bench, ctx: DeviceContext, nsys: Bool) raises:
    comptime dt = DType.int64
    run_case[dt, 2, 1024, 1024, 2, 2](m, ctx, nsys)


def main() raises:
    var nsys = False
    for arg in argv():
        if arg == "--nsys":
            nsys = True

    var m = Bench(
        BenchConfig(min_runtime_secs=0.2, max_runtime_secs=5.0, max_iters=2000)
    )
    with DeviceContext() as ctx:
        run_fp32(m, ctx, nsys)
        run_bf16(m, ctx, nsys)
        run_int8(m, ctx, nsys)
        run_int64(m, ctx, nsys)
    if not nsys:
        m.dump_report()

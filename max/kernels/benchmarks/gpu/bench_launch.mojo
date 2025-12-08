# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from benchmark import Bench, BenchConfig, Bencher, BenchId
from gpu.host import DeviceContext, Dim
from layout import *

comptime NUM_KERNELS_PER_ITERATION = 128
comptime NUM_WARMUP_ITERATIONS = 3
comptime NUM_ITERATIONS = 1000


fn empty_kernel():
    pass


fn empty_kernel_many_params[
    layout_1: Layout,
    layout_2: Layout,
    layout_3: Layout,
    layout_4: Layout,
    layout_5: Layout,
    layout_6: Layout,
    layout_7: Layout,
    layout_8: Layout,
    layout_9: Layout,
]():
    pass


fn small_kernel(ptr: UnsafePointer[UInt64, MutAnyOrigin]):
    var value = ptr[]


fn bench_empty_launch_caller(mut m: Bench, ctx: DeviceContext) raises:
    @parameter
    @always_inline
    fn bench_empty_launch(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function_checked[empty_kernel, empty_kernel](
                grid_dim=Dim(1), block_dim=Dim(1)
            )

        b.iter_custom[launch](ctx)

    m.bench_function[bench_empty_launch](BenchId("bench_empty_launch"))


fn bench_empty_launch_many_params_caller(
    mut m: Bench, ctx: DeviceContext
) raises:
    comptime func_alias = empty_kernel_many_params[
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
    ]

    @parameter
    @always_inline
    fn bench_empty_launch_many_params(mut b: Bencher) raises:
        @parameter
        fn launch() raises:
            ctx.enqueue_function_checked[func_alias, func_alias](
                grid_dim=Dim(1), block_dim=Dim(1)
            )

        b.iter[launch]()
        ctx.synchronize()

    m.bench_function[bench_empty_launch_many_params](
        BenchId("bench_empty_launch_many_params")
    )


fn bench_gpu_kernel_enqueue_caller(mut m: Bench, ctx: DeviceContext) raises:
    var size = 1
    var buf = ctx.create_buffer_sync[DType.uint64](size)

    # Warm up before benchmarking
    for _ in range(NUM_WARMUP_ITERATIONS):
        ctx.enqueue_function_checked[small_kernel, small_kernel](
            buf, grid_dim=Dim(1), block_dim=Dim(1)
        )

    # Benchmark Mojo function
    @parameter
    @always_inline
    fn bench_gpu_kernel_enqueue(mut b: Bencher) raises:
        @parameter
        fn launch() raises:
            for _ in range(NUM_KERNELS_PER_ITERATION):
                ctx.enqueue_function_checked[small_kernel, small_kernel](
                    buf, grid_dim=Dim(1), block_dim=Dim(1)
                )

        b.iter[launch]()
        ctx.synchronize()

    m.bench_function[bench_gpu_kernel_enqueue](
        BenchId("bench_gpu_kernel_enqueue")
    )


def main():
    with DeviceContext() as ctx:
        var m = Bench(
            BenchConfig(
                min_runtime_secs=0.0,
                max_runtime_secs=600.0,
                max_iters=NUM_ITERATIONS,
                max_batch_size=100,
            )
        )
        bench_empty_launch_caller(m, ctx)
        bench_empty_launch_many_params_caller(m, ctx)
        bench_gpu_kernel_enqueue_caller(m, ctx)
        m.dump_report()

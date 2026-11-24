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

from std.random import rand
from std.sys import argv

from std.benchmark import (
    Bench,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.bit import log2_floor
from max.gpu.host import DeviceBuffer, DeviceContext, default_accelerator
from kernels.matrix_multiplication import MatrixMultiplication
from kernels.tensor_core_mma import TensorCoreMMA
from kernels.top_k import TopK
from layout import TileTensor, row_major


def random_buffer[
    dtype: DType
](ctx: DeviceContext, size: Int) raises -> DeviceBuffer[dtype]:
    var buffer = ctx.enqueue_create_buffer[dtype](size)
    with buffer.map_to_host() as host_buffer:
        rand(host_buffer.as_span())
    return buffer


def top_k() raises:
    print("Running top-k benchmark...")
    comptime batch_size = 30_000
    comptime K = 32
    comptime els = batch_size * K
    comptime val_dtype = DType.float32
    comptime idx_dtype = DType.int32

    comptime layout = row_major[batch_size, K]()

    var cpu_ctx = DeviceContext.cpu()

    var in_vals_buf = random_buffer[val_dtype](cpu_ctx, els)
    var out_vals_buf = random_buffer[val_dtype](cpu_ctx, els)
    var out_idxs_buf = random_buffer[idx_dtype](cpu_ctx, els)
    var in_vals = TileTensor(in_vals_buf, layout)
    var out_vals = TileTensor(out_vals_buf, layout)
    var out_idxs = TileTensor(out_idxs_buf, layout)

    var b = Bench()
    var flops = ThroughputMeasure(BenchMetric.flops, els * log2_floor(K))
    var elements = ThroughputMeasure(BenchMetric.elements, els)
    var metrics: List = [flops, elements]

    def top_k_cpu() raises {imm}:
        TopK.run[K=K](out_vals, out_idxs, in_vals, cpu_ctx)

    b.bench_function(top_k_cpu, BenchId("top_k_custom", "cpu"), metrics)

    comptime if default_accelerator().is_nvidia_gpu():
        var gpu_ctx = DeviceContext()

        var out_vals_dev_buf = random_buffer[val_dtype](gpu_ctx, els)
        var out_idxs_dev_buf = random_buffer[idx_dtype](gpu_ctx, els)
        var in_vals_dev_buf = random_buffer[val_dtype](gpu_ctx, els)
        var out_vals_dev = TileTensor(out_vals_dev_buf, layout)
        var out_idxs_dev = TileTensor(out_idxs_dev_buf, layout)
        var in_vals_dev = TileTensor(in_vals_dev_buf, layout)

        def top_k_gpu() raises {imm}:
            TopK.run[K=K](out_vals_dev, out_idxs_dev, in_vals_dev, gpu_ctx)

        b.bench_function(top_k_gpu, BenchId("top_k_custom", "gpu"), metrics)
    b.config.verbose_metric_names = False
    print(b)


def matmul() raises:
    print("Running matmul benchmark...")
    comptime M = 1028
    comptime K = 1028
    comptime N = 1028

    comptime dtype = DType.float32

    comptime FLOPS = M * N * (2 * K - 1)

    var cpu_ctx = DeviceContext.cpu()

    var a_buf = random_buffer[dtype](cpu_ctx, M * K)
    var b_buf = random_buffer[dtype](cpu_ctx, K * N)
    var c_buf = random_buffer[dtype](cpu_ctx, M * N)
    var a = TileTensor(a_buf, row_major[M, K]())
    var b = TileTensor(b_buf, row_major[K, N]())
    var c = TileTensor(c_buf, row_major[M, N]())

    var bench = Bench()
    var flops = ThroughputMeasure(BenchMetric.flops, FLOPS)
    var elements = ThroughputMeasure(BenchMetric.elements, M * N)
    var metrics: List = [flops, elements]

    def matmul_cpu() raises {imm}:
        MatrixMultiplication["naive"].run(c, a, b, cpu_ctx)

    bench.bench_function(matmul_cpu, BenchId("cpu", "naive"), metrics)

    comptime target = default_accelerator()

    comptime if (
        target.is_amd_gpu() or target.is_apple_gpu() or target.is_nvidia_gpu()
    ):
        # TODO(MSTDL-3208): Should be DeviceContext[target]
        var gpu_ctx = DeviceContext()
        var a_dev_buf = random_buffer[dtype](gpu_ctx, M * K)
        var b_dev_buf = random_buffer[dtype](gpu_ctx, K * N)
        var c_dev_buf = random_buffer[dtype](gpu_ctx, M * N)
        var a_dev = TileTensor(a_dev_buf, row_major[M, K]())
        var b_dev = TileTensor(b_dev_buf, row_major[K, N]())
        var c_dev = TileTensor(c_dev_buf, row_major[M, N]())

        def bench_matmul_kernel[impl: StaticString]() raises {mut bench, imm}:
            def bench_gpu() raises {imm}:
                MatrixMultiplication[impl].run(c_dev, a_dev, b_dev, gpu_ctx)

            bench.bench_function(
                bench_gpu, BenchId("gpu", String(impl)), metrics
            )

        bench_matmul_kernel["naive"]()
        bench_matmul_kernel["coalescing"]()
        bench_matmul_kernel["tiled"]()
        bench_matmul_kernel["tiled_register"]()
        bench_matmul_kernel["block_tiled"]()
        bench_matmul_kernel["block_tiled_vectorized"]()

        comptime if not target.is_apple_gpu():
            bench_matmul_kernel["tensor_core"]()

    bench.config.verbose_metric_names = False
    print(bench)


def tensor_core_mma() raises:
    print("Running tensor core mma benchmark...")
    comptime M = 4096
    comptime N = 4096
    comptime K = 4096

    comptime dtype = DType.float16

    comptime FLOPS = M * N * (2 * K - 1)

    var bench = Bench()
    var flops = ThroughputMeasure(BenchMetric.flops, FLOPS)
    var elements = ThroughputMeasure(BenchMetric.elements, M * N)
    var metrics: List = [flops, elements]

    comptime perform_validation = False

    comptime if perform_validation:
        bench.config.max_iters = 1
        bench.config.max_batch_size = 1
        bench.config.num_repetitions = 1

    # TODO: Add NVIDIA GPU support
    comptime if default_accelerator().is_amd_gpu():
        var gpu_ctx = DeviceContext()
        var a_dev_buf = random_buffer[dtype](gpu_ctx, M * K)
        var b_dev_buf = random_buffer[dtype](gpu_ctx, K * N)
        var c_dev_buf = random_buffer[DType.float32](gpu_ctx, M * N)
        var a_dev = TileTensor(a_dev_buf, row_major[M, K]())
        var b_dev = TileTensor(b_dev_buf, row_major[K, N]())
        var c_dev = TileTensor(c_dev_buf, row_major[M, N]())

        def bench_matmul_kernel[impl: StaticString]() raises {mut bench, imm}:
            def bench_gpu() raises {imm}:
                TensorCoreMMA[impl].run[M=M, N=N, K=K](
                    c_dev,
                    a_dev,
                    b_dev,
                    perform_validation,
                    gpu_ctx,
                )

            bench.bench_function(
                bench_gpu, BenchId("gpu", String(impl)), metrics
            )

        bench_matmul_kernel["naive_tensor"]()
        bench_matmul_kernel["basic_shared_mem"]()
        bench_matmul_kernel["multi_block_tiled"]()
        bench_matmul_kernel["scheduler_hints"]()
        bench_matmul_kernel["double_buffer"]()
        bench_matmul_kernel["mma_tile_buffers"]()

    bench.config.verbose_metric_names = False
    print(bench)


def main() raises:
    var args = argv()
    if len(args) == 1:
        top_k()
        matmul()
    else:
        for arg in argv():
            if arg == "--top-k":
                top_k()
            if arg == "--matmul":
                matmul()
            if arg == "--tensor-core-mma":
                tensor_core_mma()

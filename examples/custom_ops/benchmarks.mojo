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

from kernels.top_k import TopK
from kernels.matrix_multiplication import MatrixMultiplication
from gpu.host import DeviceContext
from utils import IndexList
from max.driver import cpu
from max.tensor import (
    ManagedTensorSlice,
    InputTensor,
    OutputTensor,
    StaticTensorSpec,
)
from random import rand
from memory import UnsafePointer
from runtime.asyncrt import DeviceContextPtr
from benchmark import ThroughputMeasure, BenchId, BenchMetric, Bench, Bencher
from bit import log2_floor
from sys import sizeof, has_nvidia_gpu_accelerator
from memory import AddressSpace


def top_k():
    alias batch_size = 30_000
    alias K = 32
    alias els = batch_size * K
    alias rank = 2
    alias shape = IndexList[rank](batch_size, K)
    alias val_dtype = DType.float32
    alias idx_dtype = DType.int32

    # Slightly better performance compared to `create_unknown`. Using global
    # address space doesn't improve perf for GPU.
    alias val_spec = StaticTensorSpec[val_dtype, rank](
        shape=(batch_size, K),
        strides=(K, 1),
        alignment=sizeof[val_dtype](),
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
    )
    alias idx_spec = StaticTensorSpec[idx_dtype, rank](
        shape=(batch_size, K),
        strides=(K, 1),
        alignment=sizeof[idx_dtype](),
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
    )

    var in_vals = InputTensor[static_spec=val_spec].rand()
    var out_vals = OutputTensor[static_spec=val_spec].rand()
    var out_idxs = OutputTensor[static_spec=idx_spec].rand()

    var cpu_ctx = DeviceContext(api="cpu")

    @parameter
    @always_inline
    fn bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn run_bench() raises:
            TopK.execute[K=K, target="cpu"](
                out_vals, out_idxs, in_vals, cpu_ctx
            )

        b.iter[run_bench]()

    var flops = ThroughputMeasure(BenchMetric.flops, els * log2_floor(K))
    var elements = ThroughputMeasure(BenchMetric.elements, els)

    var b = Bench()
    b.bench_function[bench_cpu](BenchId("top_k_custom", "cpu"), flops, elements)

    @parameter
    if has_nvidia_gpu_accelerator():
        var gpu_ctx = DeviceContext()

        var in_vals_dev_buff = gpu_ctx.enqueue_create_buffer[val_dtype](els)
        var out_vals_dev_buff = gpu_ctx.enqueue_create_buffer[val_dtype](els)
        var out_idxs_dev_buff = gpu_ctx.enqueue_create_buffer[idx_dtype](els)

        gpu_ctx.enqueue_copy(in_vals_dev_buff, in_vals.unsafe_ptr())

        var out_vals_dev = OutputTensor[static_spec=val_spec](
            out_vals_dev_buff.unsafe_ptr(), shape
        )
        var out_idxs_dev = OutputTensor[static_spec=idx_spec](
            out_idxs_dev_buff.unsafe_ptr(), shape
        )
        var in_vals_dev = InputTensor[static_spec=val_spec](
            in_vals_dev_buff.unsafe_ptr(), shape
        )

        @parameter
        @always_inline
        fn bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            fn kernel_launch(gpu_ctx: DeviceContext) raises:
                TopK.execute[K=K, target="gpu"](
                    out_vals_dev, out_idxs_dev, in_vals_dev, gpu_ctx
                )

            b.iter_custom[kernel_launch](gpu_ctx)

        b.bench_function[bench_gpu](
            BenchId("top_k_custom", "gpu"), flops, elements
        )
        _ = in_vals_dev_buff
        _ = out_vals_dev_buff
        _ = out_idxs_dev_buff

    b.config.verbose_metric_names = False
    print(b)

    _ = in_vals
    _ = out_vals
    _ = out_idxs


def matmul():
    alias M = 1028
    alias K = 1028
    alias N = 1028
    alias FLOPS = M * N * (2 * K - 1)

    alias rank = 2
    alias a_shape = IndexList[rank](M, K)
    alias b_shape = IndexList[rank](K, N)
    alias c_shape = IndexList[rank](M, N)

    alias a_els = a_shape.flattened_length()
    alias b_els = b_shape.flattened_length()
    alias c_els = c_shape.flattened_length()

    alias dtype = DType.float32

    alias a_spec = StaticTensorSpec[dtype, rank](
        shape=(M, K),
        strides=(K, 1),
        alignment=sizeof[dtype](),
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
    )
    alias b_spec = StaticTensorSpec[dtype, rank](
        shape=(K, N),
        strides=(N, 1),
        alignment=sizeof[dtype](),
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
    )
    alias c_spec = StaticTensorSpec[dtype, rank](
        shape=(M, N),
        strides=(N, 1),
        alignment=sizeof[dtype](),
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
    )

    var a = InputTensor[static_spec=a_spec].rand()
    var b = InputTensor[static_spec=b_spec].rand()
    var c = OutputTensor[static_spec=c_spec].rand()

    var cpu_ctx = DeviceContext(api="cpu")
    var bench = Bench()
    var flops = ThroughputMeasure(BenchMetric.flops, FLOPS)
    var elements = ThroughputMeasure(BenchMetric.elements, M * N)

    @parameter
    @always_inline
    fn bench_cpu(mut bencher: Bencher) raises:
        @parameter
        @always_inline
        fn run_bench() raises:
            MatrixMultiplication["naive"].execute[target="cpu"](
                c, a, b, cpu_ctx
            )

        bencher.iter[run_bench]()

    bench.bench_function[bench_cpu](BenchId("cpu", "naive"), flops, elements)

    @parameter
    if has_nvidia_gpu_accelerator():
        var gpu_ctx = DeviceContext()
        var a_dev_buf = gpu_ctx.enqueue_create_buffer[dtype](a_els)
        var b_dev_buf = gpu_ctx.enqueue_create_buffer[dtype](b_els)
        var c_dev_buf = gpu_ctx.enqueue_create_buffer[dtype](c_els)

        gpu_ctx.enqueue_copy(a_dev_buf, a.unsafe_ptr())
        gpu_ctx.enqueue_copy(b_dev_buf, b.unsafe_ptr())

        var c_dev = InputTensor[static_spec=c_spec](
            c_dev_buf.unsafe_ptr(), c_shape
        )
        var a_dev = OutputTensor[static_spec=a_spec](
            a_dev_buf.unsafe_ptr(), a_shape
        )
        var b_dev = OutputTensor[static_spec=b_spec](
            b_dev_buf.unsafe_ptr(), b_shape
        )

        @parameter
        def bench_matmul_kernel[impl: StringLiteral]():
            @parameter
            @always_inline
            fn bench_gpu(mut bench: Bencher) raises:
                @parameter
                @always_inline
                fn kernel_launch(gpu_ctx: DeviceContext) raises:
                    MatrixMultiplication[impl].execute[target="gpu"](
                        c_dev, a_dev, b_dev, gpu_ctx
                    )

                bench.iter_custom[kernel_launch](gpu_ctx)

            bench.bench_function[bench_gpu](
                BenchId("gpu", impl), flops, elements
            )

        bench_matmul_kernel["naive"]()
        bench_matmul_kernel["coalescing"]()
        bench_matmul_kernel["tiled"]()
        bench_matmul_kernel["tiled_register"]()
        bench_matmul_kernel["block_tiled"]()
        bench_matmul_kernel["block_tiled_vectorized"]()
        # TODO add origin to `ManagedTensorSlice` to avoid this
        _ = a_dev_buf
        _ = b_dev_buf
        _ = c_dev_buf

    bench.config.verbose_metric_names = False
    print(bench)

    # TODO add origin to `ManagedTensorSlice` to avoid this
    _ = a
    _ = b
    _ = c


# TODO: arg parsing to select benchmarks
def main():
    top_k()
    matmul()

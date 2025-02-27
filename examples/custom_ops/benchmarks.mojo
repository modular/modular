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
from max.driver.device import cpu_device
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
from sys import sizeof, has_nvidia_gpu_accelerator, has_amd_gpu_accelerator
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

    var in_vals = UnsafePointer[Scalar[val_dtype]].alloc(els)
    var out_vals = UnsafePointer[Scalar[val_dtype]].alloc(els)
    var out_idxs = UnsafePointer[Scalar[idx_dtype]].alloc(els)

    rand(in_vals, els)

    var cpu_ctx_ptr = cpu_device().unsafe_ptr()

    @parameter
    @always_inline
    fn bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn run_bench() raises:
            TopK.execute[K=K, target="cpu"](
                out_vals=OutputTensor[static_spec=val_spec](out_vals, shape),
                out_idxs=OutputTensor[static_spec=idx_spec](out_idxs, shape),
                in_vals=InputTensor[static_spec=val_spec](in_vals, shape),
                ctx=cpu_ctx_ptr,
            )

        b.iter[run_bench]()

    var flops = ThroughputMeasure(BenchMetric.flops, els * log2_floor(K))
    var elements = ThroughputMeasure(BenchMetric.elements, els)

    var b = Bench()
    b.bench_function[bench_cpu](BenchId("top_k_custom", "cpu"), flops, elements)

    @parameter
    if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
        var gpu_ctx = DeviceContext()
        var dev_in_vals = gpu_ctx.enqueue_create_buffer[val_dtype](els)
        var dev_out_vals = gpu_ctx.enqueue_create_buffer[val_dtype](els)
        var dev_out_idxs = gpu_ctx.enqueue_create_buffer[idx_dtype](els)
        gpu_ctx.copy(dev_in_vals, in_vals)

        @parameter
        @always_inline
        fn bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            fn kernel_launch() raises:
                TopK.execute[K=K, target="gpu"](
                    out_vals=OutputTensor[static_spec=val_spec](
                        dev_out_vals.unsafe_ptr(), shape
                    ),
                    out_idxs=OutputTensor[static_spec=idx_spec](
                        dev_out_idxs.unsafe_ptr(), shape
                    ),
                    in_vals=InputTensor[static_spec=val_spec](
                        dev_in_vals.unsafe_ptr(), shape
                    ),
                    ctx=gpu_ctx,
                )

            b.iter[kernel_launch]()

        b.bench_function[bench_gpu](
            BenchId("top_k_custom", "gpu"), flops, elements
        )
        _ = gpu_ctx

    b.config.verbose_metric_names = False
    print(b)

    _ = cpu_ctx_ptr
    in_vals.free()
    out_vals.free()
    out_idxs.free()


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

    var a_ptr = UnsafePointer[Scalar[dtype]].alloc(a_els)
    var b_ptr = UnsafePointer[Scalar[dtype]].alloc(b_els)
    var c_ptr = UnsafePointer[Scalar[dtype]].alloc(c_els)

    var a = InputTensor[static_spec=a_spec](a_ptr, a_shape)
    var b = InputTensor[static_spec=b_spec](b_ptr, b_shape)
    var c = OutputTensor[static_spec=c_spec](c_ptr, c_shape)

    rand(a_ptr, a_els)
    rand(b_ptr, b_els)

    var cpu_ctx_ptr = cpu_device().unsafe_ptr()
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
                c,
                a,
                b,
                cpu_ctx_ptr,
            )

        bencher.iter[run_bench]()

    bench.bench_function[bench_cpu](BenchId("cpu", "naive"), flops, elements)

    @parameter
    if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
        var gpu_ctx = DeviceContext()
        var a_dev = gpu_ctx.enqueue_create_buffer[dtype](a_els)
        var b_dev = gpu_ctx.enqueue_create_buffer[dtype](b_els)
        var c_dev = gpu_ctx.enqueue_create_buffer[dtype](c_els)
        var c = InputTensor[static_spec=c_spec](c_dev.unsafe_ptr(), c_shape)
        var a = OutputTensor[static_spec=a_spec](a_dev.unsafe_ptr(), a_shape)
        var b = OutputTensor[static_spec=b_spec](b_dev.unsafe_ptr(), b_shape)
        gpu_ctx.copy(a_dev, a_ptr)
        gpu_ctx.copy(b_dev, b_ptr)

        @parameter
        def bench_matmul_kernel[impl: StringLiteral]():
            @parameter
            @always_inline
            fn bench_gpu(mut bench: Bencher) raises:
                @parameter
                @always_inline
                fn kernel_launch(gpu_ctx: DeviceContext) raises:
                    MatrixMultiplication[impl].execute[target="gpu"](
                        c,
                        a,
                        b,
                        gpu_ctx,
                    )

                var gpu_ctx = DeviceContext()
                bench.iter_custom[kernel_launch](gpu_ctx)
                _ = gpu_ctx

            bench.bench_function[bench_gpu](
                BenchId("gpu", impl), flops, elements
            )

        bench_matmul_kernel["naive"]()
        bench_matmul_kernel["coalescing"]()
        bench_matmul_kernel["tiled"]()
        bench_matmul_kernel["tiled_register"]()
        bench_matmul_kernel["block_tiled"]()
        bench_matmul_kernel["block_tiled_vectorized"]()
        _ = gpu_ctx
        _ = a_dev
        _ = b_dev
        _ = c_dev

    bench.config.verbose_metric_names = False
    print(bench)

    a_ptr.free()
    b_ptr.free()
    c_ptr.free()


# TODO: arg parsing to select benchmarks
def main():
    top_k()
    matmul()

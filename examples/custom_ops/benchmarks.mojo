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

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    BenchConfig,
)
from bit import log2_floor
from buffer.dimlist import DimList
from gpu.host import DeviceBuffer, DeviceContext
from kernels.matrix_multiplication import MatrixMultiplication
from kernels.tensor_core_mma import TensorCoreMMA
from kernels.causal_conv1d import CausalConv1Dgpu, CausalConv1Dcpu
from kernels.top_k import TopK
from math import iota
from random import rand
from sys import (
    argv,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
)
from gpu.host import DeviceContext, DeviceBuffer
from tensor_internal import (
    Input,
    IOSpec,
    ManagedTensorSlice,
    Output,
    StaticTensorSpec,
)


# Wrap a ManagedTensorSlice and DeviceBuffer as an owning Tensor
@fieldwise_init
struct Tensor[
    dtype: DType,
    rank: Int, //,
    io_spec: IOSpec,
    static_spec: StaticTensorSpec[dtype, rank],
](ImplicitlyCopyable, Movable):
    alias size = Int(static_spec.shape.product())

    var slice: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
    var buffer: DeviceBuffer[dtype]

    fn __init__(out self, ctx: DeviceContext) raises:
        self.buffer = ctx.enqueue_create_buffer[dtype](Self.size)

        self.slice = ManagedTensorSlice[
            io_spec=io_spec, static_spec=static_spec
        ](
            self.buffer.unsafe_ptr(),
            Self.static_spec.shape.into_index_list[rank](),
            Self.static_spec.strides.into_index_list[rank](),
        )

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self.buffer.unsafe_ptr()

    fn rand(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            rand(host_buffer.unsafe_ptr(), Self.size)
            return self

    fn iota(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            iota(host_buffer.unsafe_ptr(), Self.size)
            return self

    fn fill(self, value: Scalar[dtype]) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            var ptr = host_buffer.unsafe_ptr()
            for i in range(Self.size):
                ptr[i] = value
            return self

    fn custom_fill_a(self, M: Int, K: Int) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            var ptr = host_buffer.unsafe_ptr()
            for i in range(M):
                for j in range(K):
                    ptr[i * K + j] = i
            return self

    fn custom_fill_b(self, K: Int, N: Int) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            var ptr = host_buffer.unsafe_ptr()
            for i in range(K):
                for j in range(N):
                    ptr[i * N + j] = j
            return self


def top_k():
    print("Running top-k benchmark...")
    alias batch_size = 30_000
    alias K = 32
    alias els = batch_size * K
    alias rank = 2
    alias val_dtype = DType.float32
    alias idx_dtype = DType.int32

    alias shape = DimList(batch_size, K)
    alias val_spec = StaticTensorSpec[val_dtype, rank](shape)
    alias idx_spec = StaticTensorSpec[idx_dtype, rank](shape)

    var cpu_ctx = DeviceContext(api="cpu")

    var in_vals = Tensor[Input, val_spec](cpu_ctx).rand()
    var out_vals = Tensor[Output, val_spec](cpu_ctx).rand()
    var out_idxs = Tensor[Output, idx_spec](cpu_ctx).rand()

    var b = Bench()
    var flops = ThroughputMeasure(BenchMetric.flops, els * log2_floor(K))
    var elements = ThroughputMeasure(BenchMetric.elements, els)
    var metrics = List(flops, elements)

    @parameter
    def top_k_cpu():
        TopK.execute[K=K, target="cpu"](
            out_vals.slice, out_idxs.slice, in_vals.slice, cpu_ctx
        )

    b.bench_function[top_k_cpu](BenchId("top_k_custom", "cpu"), metrics)

    @parameter
    if has_nvidia_gpu_accelerator():
        var gpu_ctx = DeviceContext()

        var out_vals_dev = Tensor[Output, val_spec](gpu_ctx).rand()
        var out_idxs_dev = Tensor[Output, idx_spec](gpu_ctx).rand()
        var in_vals_dev = Tensor[Input, val_spec](gpu_ctx).rand()

        @parameter
        def top_k_gpu():
            TopK.execute[K=K, target="gpu"](
                out_vals_dev.slice,
                out_idxs_dev.slice,
                in_vals_dev.slice,
                gpu_ctx,
            )

        b.bench_function[top_k_gpu](BenchId("top_k_custom", "gpu"), metrics)
    b.config.verbose_metric_names = False
    print(b)


def matmul():
    print("Running matmul benchmark...")
    alias M = 1028
    alias K = 1028
    alias N = 1028

    alias rank = 2
    alias dtype = DType.float32

    alias FLOPS = M * N * (2 * K - 1)

    alias a_spec = StaticTensorSpec[dtype, rank](DimList(M, K))
    alias b_spec = StaticTensorSpec[dtype, rank](DimList(K, N))
    alias c_spec = StaticTensorSpec[dtype, rank](DimList(M, N))

    var cpu_ctx = DeviceContext(api="cpu")

    var a = Tensor[Input, a_spec](cpu_ctx).rand()
    var b = Tensor[Input, b_spec](cpu_ctx).rand()
    var c = Tensor[Output, c_spec](cpu_ctx).rand()

    var bench = Bench()
    var flops = ThroughputMeasure(BenchMetric.flops, FLOPS)
    var elements = ThroughputMeasure(BenchMetric.elements, M * N)
    var metrics = List(flops, elements)

    @parameter
    def matmul_cpu():
        MatrixMultiplication["naive"].execute[target="cpu"](
            c.slice, a.slice, b.slice, cpu_ctx
        )

    bench.bench_function[matmul_cpu](BenchId("cpu", "naive"), metrics)

    @parameter
    if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
        var gpu_ctx = DeviceContext()
        var a_dev = Tensor[Input, a_spec](gpu_ctx).rand()
        var b_dev = Tensor[Input, b_spec](gpu_ctx).rand()
        var c_dev = Tensor[Output, c_spec](gpu_ctx).rand()

        @parameter
        def bench_matmul_kernel[impl: StaticString]():
            @parameter
            def bench_gpu():
                MatrixMultiplication[impl].execute[target="gpu"](
                    c_dev.slice, a_dev.slice, b_dev.slice, gpu_ctx
                )

            bench.bench_function[bench_gpu](
                BenchId("gpu", String(impl)), metrics
            )

        bench_matmul_kernel["naive"]()
        bench_matmul_kernel["coalescing"]()
        bench_matmul_kernel["tiled"]()
        bench_matmul_kernel["tiled_register"]()
        bench_matmul_kernel["block_tiled"]()
        bench_matmul_kernel["block_tiled_vectorized"]()
        bench_matmul_kernel["tensor_core"]()

    bench.config.verbose_metric_names = False
    print(bench)


def tensor_core_mma():
    print("Running tensor core mma benchmark...")
    alias M = 4096
    alias N = 4096
    alias K = 4096

    alias rank = 2
    alias dtype = DType.float16

    alias FLOPS = M * N * (2 * K - 1)

    alias a_spec = StaticTensorSpec[dtype, rank](DimList(M, K))
    alias b_spec = StaticTensorSpec[dtype, rank](DimList(K, N))
    alias c_spec = StaticTensorSpec[DType.float32, rank](DimList(M, N))

    var cpu_ctx = DeviceContext(api="cpu")

    var a = Tensor[Input, a_spec](cpu_ctx).rand()
    var b = Tensor[Input, b_spec](cpu_ctx).rand()
    var c = Tensor[Output, c_spec](cpu_ctx).rand()

    var bench = Bench()
    var flops = ThroughputMeasure(BenchMetric.flops, FLOPS)
    var elements = ThroughputMeasure(BenchMetric.elements, M * N)
    var metrics = List(flops, elements)

    alias perform_validation = False

    @parameter
    if perform_validation:
        bench.config.max_iters = 1
        bench.config.max_batch_size = 1
        bench.config.num_repetitions = 1

    # TODO: Add NVIDIA GPU support
    @parameter
    if has_amd_gpu_accelerator():
        var gpu_ctx = DeviceContext()
        var a_dev = Tensor[Input, a_spec](gpu_ctx).rand()
        var b_dev = Tensor[Input, b_spec](gpu_ctx).rand()
        var c_dev = Tensor[Output, c_spec](gpu_ctx).rand()

        @parameter
        def bench_matmul_kernel[impl: StaticString]():
            @parameter
            def bench_gpu():
                TensorCoreMMA[impl].execute[target="gpu", M=M, N=N, K=K](
                    c_dev.slice,
                    a_dev.slice,
                    b_dev.slice,
                    perform_validation,
                    gpu_ctx,
                )

            bench.bench_function[bench_gpu](
                BenchId("gpu", String(impl)), metrics
            )

        bench_matmul_kernel["naive_tensor"]()
        bench_matmul_kernel["basic_shared_mem"]()
        bench_matmul_kernel["multi_block_tiled"]()
        bench_matmul_kernel["scheduler_hints"]()
        bench_matmul_kernel["double_buffer"]()
        bench_matmul_kernel["mma_tile_buffers"]()

    bench.config.verbose_metric_names = False
    print(bench)


def run_conv1d[impl: StaticString]():
    alias nBatches = 128
    alias nChannels = 8
    alias sequenceLength = 1024 * 128
    alias kWidth = 4
    alias kNThreads = 128
    alias kNElts = 4

    alias dtype = DType.bfloat16

    alias x_spec = StaticTensorSpec[dtype, 3](
        DimList(nBatches, nChannels, sequenceLength)
    )
    alias w_spec = StaticTensorSpec[dtype, 2](DimList(nChannels, kWidth))
    alias b_spec = StaticTensorSpec[dtype, 1](DimList(nChannels))
    alias xx2Dshape = StaticTensorSpec[dtype, 2](
        DimList(nBatches * nChannels, sequenceLength)
    )
    runOnCPU = False

    var bench = Bench(BenchConfig(max_iters=10))
    var flops = ThroughputMeasure(
        BenchMetric.flops,
        nBatches * nChannels * sequenceLength * kNElts * kWidth * 2,
    )
    var elements = ThroughputMeasure(
        BenchMetric.elements, nBatches * nChannels * sequenceLength
    )
    if runOnCPU:
        var cpu_ctx = DeviceContext(api="cpu")

        var x = Tensor[
            Input,
            StaticTensorSpec[dtype, 3](
                DimList(nBatches, nChannels, sequenceLength)
            ),
        ](cpu_ctx).rand()
        var w = Tensor[Input, w_spec](cpu_ctx).rand()
        var b = Tensor[Input, b_spec](cpu_ctx).rand()
        var o = Tensor[Output, x_spec](cpu_ctx).rand()
        var xx2Do = Tensor[Output, xx2Dshape](cpu_ctx).rand()

        @parameter
        @always_inline
        fn bench_cpu(mut bencher: Bencher) raises:
            @parameter
            @always_inline
            fn run_bench() raises:
                CausalConv1Dcpu.execute[
                    kNThreads, kNElts, kWidth, target="cpu"
                ](o.slice, xx2Do.slice, x.slice, w.slice, b.slice, cpu_ctx)

            bencher.iter[run_bench]()

        bench.bench_function[bench_cpu](
            BenchId("cpu", "naive"), flops, elements
        )

    @parameter
    if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
        var gpu_ctx = DeviceContext()
        var x_dev = Tensor[Input, x_spec](gpu_ctx).rand()
        var b_dev = Tensor[Input, b_spec](gpu_ctx).rand()
        var w_dev = Tensor[Input, w_spec](gpu_ctx).rand()
        var o_dev = Tensor[Output, x_spec](gpu_ctx).rand()
        var xx2Do = Tensor[Output, xx2Dshape](gpu_ctx).rand()

        @parameter
        def bench_conv1d_kernel[impl: StaticString]():
            @parameter
            @always_inline
            fn bench_gpu(mut bench: Bencher) raises:
                @parameter
                @always_inline
                fn kernel_launch(gpu_ctx: DeviceContext) raises:
                    CausalConv1Dgpu.execute[
                        kNThreads, kNElts, kWidth, target="gpu"
                    ](
                        o_dev.slice,
                        xx2Do.slice,
                        x_dev.slice,
                        w_dev.slice,
                        b_dev.slice,
                        gpu_ctx,
                    )

                bench.iter_custom[kernel_launch](gpu_ctx)

            bench.bench_function[bench_gpu](
                BenchId(impl, String(impl)), flops, elements
            )

        bench_conv1d_kernel[impl]()
        print(bench)


def main():
    var args = argv()
    if len(args) == 1:
        top_k()
        matmul()
    else:
        for arg in argv():
            if arg == "--conv1d":
                run_conv1d["conv1d"]()
            if arg == "--top-k":
                top_k()
            if arg == "--matmul":
                matmul()
            if arg == "--tensor-core-mma":
                tensor_core_mma()

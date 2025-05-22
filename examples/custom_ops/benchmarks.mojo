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

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from bit import log2_floor
from buffer.dimlist import DimList
from gpu.host import DeviceBuffer, DeviceContext
from kernels.matrix_multiplication import MatrixMultiplication
from kernels.top_k import TopK
from math import iota
from memory import AddressSpace, UnsafePointer
from random import rand
from sys import has_nvidia_gpu_accelerator, sizeof
from tensor_internal import (
    Input,
    InputTensor,
    IOSpec,
    ManagedTensorSlice,
    MutableInput,
    Output,
    OutputTensor,
    StaticTensorSpec,
)
from utils import IndexList


# Wrap a ManagedTensorSlice and DeviceBuffer as an owning Tensor
@value
struct Tensor[
    dtype: DType,
    rank: Int, //,
    io_spec: IOSpec,
    static_spec: StaticTensorSpec[dtype, rank],
]:
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


def top_k():
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
    alias M = 1028
    alias K = 1028
    alias N = 1028

    alias rank = 2
    alias dtype = DType.float32

    alias FLOPS = M * N * (2 * K - 1)

    alias a_spec = StaticTensorSpec[dtype, rank]((M, K))
    alias b_spec = StaticTensorSpec[dtype, rank]((K, N))
    alias c_spec = StaticTensorSpec[dtype, rank]((M, N))

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
    if has_nvidia_gpu_accelerator():
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


# TODO: arg parsing to select benchmarks
def main():
    top_k()
    matmul()

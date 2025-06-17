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

from sys import env_get_bool, env_get_dtype, env_get_int, env_get_string

from algorithm.functional import stencil, stencil_gpu
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from testing import assert_almost_equal

from utils import IndexList
from utils.numerics import min_or_neg_inf


fn assert_allclose[
    dtype: DType, rank: Int, shape: DimList
](
    h_output_ref: NDBuffer[dtype, rank, _, shape],
    h_output_gpu: NDBuffer[dtype, rank, _, shape],
) raises:
    var shape_ = h_output_ref.get_shape()
    for i in range(shape_.flattened_length()):
        assert_almost_equal(h_output_ref.data[i], h_output_gpu.data[i])


fn bench_stencil_avg_pool[
    dtype: DType,
    batch_size: Int,
    input_height: Int,
    input_width: Int,
    pool_window_h: Int,
    pool_window_w: Int,
    num_channels: Int,
](ctx: DeviceContext, mut m: Bench) raises:
    alias rank = 4
    alias dilation = 1
    alias stencil_rank = 2
    alias simd_width = 1

    alias input_shape = DimList(1, input_height, input_width, num_channels)
    alias output_height = input_height - pool_window_h + 1
    alias output_width = input_width - pool_window_w + 1
    alias output_shape = DimList(1, output_height, output_width, num_channels)

    # Create host buffers
    var h_input_ptr = UnsafePointer[Scalar[dtype]].alloc(
        Int(input_shape.product())
    )
    var h_input = NDBuffer[dtype, rank, _, input_shape](h_input_ptr)
    var h_output_ptr = UnsafePointer[Scalar[dtype]].alloc(
        Int(output_shape.product())
    )
    var h_output = NDBuffer[dtype, rank, _, output_shape](h_output_ptr)
    var h_output_ref_ptr = UnsafePointer[Scalar[dtype]].alloc(
        Int(output_shape.product())
    )
    var h_output_ref = NDBuffer[dtype, rank, _, output_shape](h_output_ref_ptr)

    # Initialize input data
    for i in range(h_input.num_elements()):
        h_input.data[i] = i + 1
    h_output_ref.fill(0)
    h_output.fill(0)

    # Create device buffers
    var d_input_buf = ctx.enqueue_create_buffer[dtype](
        Int(input_shape.product())
    )
    var d_input = NDBuffer[dtype, rank](d_input_buf.unsafe_ptr(), input_shape)
    var d_output_buf = ctx.enqueue_create_buffer[dtype](
        Int(output_shape.product())
    )
    var d_output = NDBuffer[dtype, rank](
        d_output_buf.unsafe_ptr(), output_shape
    )

    # Copy to device
    ctx.enqueue_copy(d_input_buf, h_input.data)
    ctx.enqueue_copy(d_output_buf, h_output.data)

    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank, **_]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    @always_inline
    @parameter
    fn avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    @parameter
    fn avg_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank, **_],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return dilation

    # GPU Implementation benchmark
    @always_inline
    @__copy_capture(d_input)
    @parameter
    fn load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, **_]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load[width=simd_width](point)
        )

    @always_inline
    @__copy_capture(d_output)
    @parameter
    fn avg_pool_compute_finalize_gpu[
        simd_width: Int
    ](point: IndexList[rank, **_], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        d_output.store(point, res)

    @parameter
    @always_inline
    fn bench_gpu(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            alias stencil_axis = IndexList[stencil_rank](1, 2)
            stencil_gpu[
                rank,
                stencil_rank,
                stencil_axis,
                simd_width,
                dtype,
                map_fn[stencil_rank],
                dilation_fn,
                load_fn_gpu,
                avg_pool_compute_init,
                avg_pool_compute,
                avg_pool_compute_finalize_gpu,
            ](ctx, d_output.get_shape(), d_input.get_shape())

        b.iter_custom[kernel_launch](ctx)

    # CPU Implementation benchmark
    @always_inline
    @__copy_capture(h_input)
    @parameter
    fn load_fn_cpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, **_]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load[width=simd_width](point)
        )

    @always_inline
    @__copy_capture(h_output_ref)
    @parameter
    fn avg_pool_compute_finalize_cpu[
        simd_width: Int
    ](point: IndexList[rank, **_], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        h_output_ref.store(point, res)

    @parameter
    @always_inline
    fn bench_cpu(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch():
            alias stencil_axis = IndexList[stencil_rank](1, 2)
            stencil[
                rank,
                stencil_rank,
                stencil_axis,
                simd_width,
                dtype,
                map_fn[stencil_rank],
                dilation_fn,
                load_fn_cpu,
                avg_pool_compute_init,
                avg_pool_compute,
                avg_pool_compute_finalize_cpu,
            ](h_output_ref.get_shape(), h_input.get_shape())

        b.iter[kernel_launch]()

    # Calculate FLOPs for throughput measurement
    fn compute_flops() -> Int:
        return (
            input_height * input_width * pool_window_h * pool_window_w * 2
        )  # One add, one divide per window element

    # Ensure correctness
    assert_allclose(h_output_ref, h_output)

    # Run benchmarks
    var bench_name = String(
        "stencil_avg_pool_",
        batch_size,
        "x",
        input_height,
        "x",
        input_width,
        "x",
        num_channels,
    )
    m.bench_function[bench_gpu](
        BenchId(bench_name + "_gpu"),
        ThroughputMeasure(BenchMetric.flops, compute_flops()),
    )

    m.bench_function[bench_cpu](
        BenchId(bench_name + "_cpu"),
        ThroughputMeasure(BenchMetric.flops, compute_flops()),
    )

    # Ensure correctness
    ctx.enqueue_copy(h_output.data, d_output_buf)
    ctx.synchronize()
    assert_allclose(h_output_ref, h_output)

    _ = d_input_buf^
    _ = d_output_buf^
    h_input_ptr.free()
    h_output_ptr.free()
    h_output_ref_ptr.free()


fn bench_stencil_max_pool[
    dtype: DType,
    batch_size: Int,
    input_height: Int,
    input_width: Int,
    pool_window_h: Int,
    pool_window_w: Int,
    num_channels: Int,
](ctx: DeviceContext, mut m: Bench) raises:
    alias rank = 4
    alias dilation = 1
    alias stencil_rank = 2
    alias simd_width = 1

    alias input_shape = DimList(1, input_height, input_width, num_channels)
    alias output_height = input_height - pool_window_h + 1
    alias output_width = input_width - pool_window_w + 1
    alias output_shape = DimList(1, output_height, output_width, num_channels)

    # Create host buffers
    var h_input_ptr = UnsafePointer[Scalar[dtype]].alloc(
        Int(input_shape.product())
    )
    var h_input = NDBuffer[dtype, rank, _, input_shape](h_input_ptr)
    var h_output_ptr = UnsafePointer[Scalar[dtype]].alloc(
        Int(output_shape.product())
    )
    var h_output = NDBuffer[dtype, rank, _, output_shape](h_output_ptr)
    var h_output_ref_ptr = UnsafePointer[Scalar[dtype]].alloc(
        Int(output_shape.product())
    )
    var h_output_ref = NDBuffer[dtype, rank, _, output_shape](h_output_ref_ptr)

    # Initialize input data
    for i in range(h_input.num_elements()):
        h_input.data[i] = i + 1
    h_output_ref.fill(0)
    h_output.fill(0)

    # Create device buffers
    var d_input_buf = ctx.enqueue_create_buffer[dtype](
        Int(input_shape.product())
    )
    var d_input = NDBuffer[dtype, rank](d_input_buf.unsafe_ptr(), input_shape)
    var d_output_buf = ctx.enqueue_create_buffer[dtype](
        Int(output_shape.product())
    )
    var d_output = NDBuffer[dtype, rank](
        d_output_buf.unsafe_ptr(), output_shape
    )

    # Copy to device
    ctx.enqueue_copy(d_input_buf, h_input.data)
    ctx.enqueue_copy(d_output_buf, h_output.data)

    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank, **_]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    @always_inline
    @parameter
    fn max_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    @always_inline
    @parameter
    fn max_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank, **_],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return max(val, result)

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return dilation

    # GPU Implementation benchmark
    @always_inline
    @__copy_capture(d_input)
    @parameter
    fn load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, **_]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load[width=simd_width](point)
        )

    @always_inline
    @__copy_capture(d_output)
    @parameter
    fn max_pool_compute_finalize_gpu[
        simd_width: Int
    ](point: IndexList[rank, **_], val: SIMD[dtype, simd_width]):
        d_output.store(point, val)

    @parameter
    @always_inline
    fn bench_gpu(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            alias stencil_axis = IndexList[stencil_rank](1, 2)
            stencil_gpu[
                rank,
                stencil_rank,
                stencil_axis,
                simd_width,
                dtype,
                map_fn[stencil_rank],
                dilation_fn,
                load_fn_gpu,
                max_pool_compute_init,
                max_pool_compute,
                max_pool_compute_finalize_gpu,
            ](ctx, d_output.get_shape(), d_input.get_shape())

        b.iter_custom[kernel_launch](ctx)

    # CPU Implementation benchmark
    @always_inline
    @__copy_capture(h_input)
    @parameter
    fn load_fn_cpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, **_]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load[width=simd_width](point)
        )

    @always_inline
    @__copy_capture(h_output_ref)
    @parameter
    fn max_pool_compute_finalize_cpu[
        simd_width: Int
    ](point: IndexList[rank, **_], val: SIMD[dtype, simd_width]):
        h_output_ref.store(point, val)

    @parameter
    @always_inline
    fn bench_cpu(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch():
            alias stencil_axis = IndexList[stencil_rank](1, 2)
            stencil[
                rank,
                stencil_rank,
                stencil_axis,
                simd_width,
                dtype,
                map_fn[stencil_rank],
                dilation_fn,
                load_fn_cpu,
                max_pool_compute_init,
                max_pool_compute,
                max_pool_compute_finalize_cpu,
            ](h_output_ref.get_shape(), h_input.get_shape())

        b.iter[kernel_launch]()

    # Calculate FLOPs for throughput measurement
    fn compute_flops() -> Int:
        return (
            input_height * input_width * pool_window_h * pool_window_w
        )  # One comparison per window element

    # Run benchmarks
    var bench_name = String(
        "stencil_max_pool_",
        batch_size,
        "x",
        input_height,
        "x",
        input_width,
        "x",
        num_channels,
    )
    m.bench_function[bench_gpu](
        BenchId(bench_name + "_gpu"),
        ThroughputMeasure(BenchMetric.flops, compute_flops()),
    )

    m.bench_function[bench_cpu](
        BenchId(bench_name + "_cpu"),
        ThroughputMeasure(BenchMetric.flops, compute_flops()),
    )

    # Ensure correctness
    ctx.enqueue_copy(h_output.data, d_output_buf)
    ctx.synchronize()
    assert_allclose(h_output_ref, h_output)

    _ = d_input_buf^
    _ = d_output_buf^
    h_input_ptr.free()
    h_output_ptr.free()
    h_output_ref_ptr.free()


fn bench_stencil_avg_pool_padded[
    dtype: DType,
    batch_size: Int,
    input_height: Int,
    input_width: Int,
    pool_window_h: Int,
    pool_window_w: Int,
    pad_h: Int,
    pad_w: Int,
](ctx: DeviceContext, mut m: Bench) raises:
    alias rank = 4
    alias stencil_rank = 2
    alias simd_width = 1
    alias dilation = 1

    alias input_shape = DimList(1, input_height, input_width, 1)
    alias output_height = input_height - pool_window_h + pad_h * 2 + 1
    alias output_width = input_width - pool_window_w + pad_w * 2 + 1
    alias output_shape = DimList(1, output_height, output_width, 1)

    # Create host buffers
    var h_input_ptr = UnsafePointer[Scalar[dtype]].alloc(
        Int(input_shape.product())
    )
    var h_input = NDBuffer[dtype, rank, _, input_shape](h_input_ptr)
    var h_output_ptr = UnsafePointer[Scalar[dtype]].alloc(
        Int(output_shape.product())
    )
    var h_output = NDBuffer[dtype, rank, _, output_shape](h_output_ptr)
    var h_output_ref_ptr = UnsafePointer[Scalar[dtype]].alloc(
        Int(output_shape.product())
    )
    var h_output_ref = NDBuffer[dtype, rank, _, output_shape](h_output_ref_ptr)

    # Initialize input data
    for i in range(h_input.num_elements()):
        h_input.data[i] = i + 1
    h_output_ref.fill(0)
    h_output.fill(0)

    # Create device buffers
    var d_input_buf = ctx.enqueue_create_buffer[dtype](
        Int(input_shape.product())
    )
    var d_input = NDBuffer[dtype, rank](d_input_buf.unsafe_ptr(), input_shape)
    var d_output_buf = ctx.enqueue_create_buffer[dtype](
        Int(output_shape.product())
    )
    var d_output = NDBuffer[dtype, rank](
        d_output_buf.unsafe_ptr(), output_shape
    )

    # Copy to device
    ctx.enqueue_copy(d_input_buf, h_input.data)
    ctx.enqueue_copy(d_output_buf, h_output.data)

    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank, **_]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](
            point[0] - pad_h, point[1] - pad_w
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h - pad_h, point[1] + pool_window_w - pad_w
        )
        return lower_bound, upper_bound

    @always_inline
    @parameter
    fn avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    @parameter
    fn avg_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank, **_],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return dilation

    # GPU Implementation benchmark
    @always_inline
    @__copy_capture(d_input)
    @parameter
    fn load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, **_]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load[width=simd_width](point)
        )

    @always_inline
    @__copy_capture(d_output)
    @parameter
    fn avg_pool_compute_finalize_gpu[
        simd_width: Int
    ](point: IndexList[rank, **_], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        d_output.store(point, res)

    @parameter
    @always_inline
    fn bench_gpu(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            alias stencil_axis = IndexList[stencil_rank](1, 2)
            stencil_gpu[
                rank,
                stencil_rank,
                stencil_axis,
                simd_width,
                dtype,
                map_fn[stencil_rank],
                dilation_fn,
                load_fn_gpu,
                avg_pool_compute_init,
                avg_pool_compute,
                avg_pool_compute_finalize_gpu,
            ](ctx, d_output.get_shape(), d_input.get_shape())

        b.iter_custom[kernel_launch](ctx)

    # CPU Implementation benchmark
    @always_inline
    @__copy_capture(h_input)
    @parameter
    fn load_fn_cpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, **_]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load[width=simd_width](point)
        )

    @always_inline
    @__copy_capture(h_output_ref)
    @parameter
    fn avg_pool_compute_finalize_cpu[
        simd_width: Int
    ](point: IndexList[rank, **_], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        h_output_ref.store(point, res)

    @parameter
    @always_inline
    fn bench_cpu(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch():
            alias stencil_axis = IndexList[stencil_rank](1, 2)
            stencil[
                rank,
                stencil_rank,
                stencil_axis,
                simd_width,
                dtype,
                map_fn[stencil_rank],
                dilation_fn,
                load_fn_cpu,
                avg_pool_compute_init,
                avg_pool_compute,
                avg_pool_compute_finalize_cpu,
            ](h_output_ref.get_shape(), h_input.get_shape())

        b.iter[kernel_launch]()

    # Calculate FLOPs for throughput measurement
    fn compute_flops() -> Int:
        return (
            input_height * input_width * pool_window_h * pool_window_w * 2
        )  # One add, one divide per window element

    # Ensure correctness
    assert_allclose(h_output_ref, h_output)

    # Run benchmarks
    var bench_name = String(
        "stencil_avg_pool_padded_",
        batch_size,
        "x",
        input_height,
        "x",
        input_width,
        "_pad",
        pad_h,
        "x",
        pad_w,
    )

    m.bench_function[bench_gpu](
        BenchId(bench_name + "_gpu"),
        ThroughputMeasure(BenchMetric.flops, compute_flops()),
    )

    m.bench_function[bench_cpu](
        BenchId(bench_name + "_cpu"),
        ThroughputMeasure(BenchMetric.flops, compute_flops()),
    )

    # Ensure correctness
    ctx.enqueue_copy(h_output.data, d_output_buf)
    ctx.synchronize()
    assert_allclose(h_output_ref, h_output)

    _ = d_input_buf^
    _ = d_output_buf^
    h_input_ptr.free()
    h_output_ptr.free()
    h_output_ref_ptr.free()


fn main() raises:
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()
    alias batch_size = env_get_int["batch_size", 128]()
    alias input_height = env_get_int["input_height", 1024]()
    alias input_width = env_get_int["input_width", 1024]()
    alias num_channels = env_get_int["num_channels", 3]()
    alias pool_window_h = env_get_int["pool_window_h", 3]()
    alias pool_window_w = env_get_int["pool_window_w", 3]()

    alias pad_h = env_get_int["pad_h", 0]()
    alias pad_w = env_get_int["pad_w", 0]()
    alias method = env_get_string["method", "max_pool"]()

    var m = Bench()
    with DeviceContext() as ctx:

        @parameter
        if method == "avg_pool":
            bench_stencil_avg_pool[
                dtype,
                batch_size,
                input_height,
                input_width,
                pool_window_h,
                pool_window_w,
                num_channels,
            ](ctx, m)
        elif method == "max_pool":
            bench_stencil_max_pool[
                dtype,
                batch_size,
                input_height,
                input_width,
                pool_window_h,
                pool_window_w,
                num_channels,
            ](ctx, m)
        elif method == "avg_pool_padded":
            bench_stencil_avg_pool_padded[
                dtype,
                batch_size,
                input_height,
                input_width,
                pool_window_h,
                pool_window_w,
                pad_h,
                pad_w,
            ](ctx, m)

    m.dump_report()

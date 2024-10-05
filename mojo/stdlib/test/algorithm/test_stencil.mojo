# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Issue #23536
# RUN: %mojo-no-debug %s | FileCheck %s

from algorithm.functional import stencil
from buffer import NDBuffer
from buffer.dimlist import DimList

from utils import IndexList
from utils.numerics import min_or_neg_inf

alias _map_fn_type = fn[rank: Int] (IndexList[rank]) capturing -> (
    IndexList[rank],
    IndexList[rank],
)
alias load_fn_type = fn[dtype: DType, rank: Int, simd_width: Int] (
    IndexList[rank]
) capturing -> SIMD[dtype, simd_width]


fn fill_buffer[
    dtype: DType, rank: Int, shape: DimList
](buf: NDBuffer[dtype, rank, shape]):
    var s: Int = 1
    for i in range(buf.get_rank()):
        s *= buf.dim(i)

    for j in range(s):
        buf.flatten()[j] = Scalar[dtype](j) + 1


# TODO: Refactor tests
# CHECK-LABEL: test_stencil_avg_pool
fn test_stencil_avg_pool():
    print("== test_stencil_avg_pool")
    alias rank = 4
    alias stencil_rank = 2
    alias dtype = DType.float32
    alias simd_with = 1

    alias input_width = 5
    alias input_height = 5

    alias stride = 1
    alias pool_window_h = 3
    alias pool_window_w = 3
    alias dilation = 1

    alias input_shape = DimList(1, input_height, input_width, 1)

    alias output_heigh = input_height - pool_window_h + 1
    alias output_width = input_width - pool_window_w + 1

    alias output_shape = DimList(1, output_heigh, output_width, 1)

    var pad_value = 0

    var input = NDBuffer[dtype, rank, input_shape].stack_allocation()
    var output = NDBuffer[dtype, rank, output_shape].stack_allocation()

    fill_buffer(input)
    output.fill(0)

    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    @always_inline
    @__copy_capture(input)
    @parameter
    fn load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    fn avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    @parameter
    fn avg_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return 1

    @always_inline
    @__copy_capture(output)
    @parameter
    fn avg_pool_compute_finalize[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        output.store(point, res)

    alias stencil_axis = IndexList[stencil_rank](1, 2)
    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_with,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize,
    ](output.get_shape(), input.get_shape())

    # CHECK: 7.0    8.0     9.0
    # CHECK: 12.0    13.0    14.0
    # CHECK: 17.0    18.0    19.0
    for i in range(0, output_heigh):
        for j in range(0, output_width):
            print(output[0, i, j, 0], "\t", end="")
        print("")


# CHECK-LABEL: test_stencil_avg_pool_padded
fn test_stencil_avg_pool_padded():
    print("== test_stencil_avg_pool_padded")
    alias rank = 4
    alias stencil_rank = 2
    alias dtype = DType.float32
    alias simd_with = 1

    alias input_width = 5
    alias input_height = 5

    alias stride = 1
    alias pool_window_h = 5
    alias pool_window_w = 5
    alias dilation = 1
    alias pad_h = 2
    alias pad_w = 2

    alias input_shape = DimList(1, input_height, input_width, 1)

    alias output_heigh = input_height - pool_window_h + pad_h * 2 + 1
    alias output_width = input_width - pool_window_w + pad_w * 2 + 1

    alias output_shape = DimList(1, output_heigh, output_width, 1)

    var pad_value = 0

    var input = NDBuffer[dtype, rank, input_shape].stack_allocation()
    var output = NDBuffer[dtype, rank, output_shape].stack_allocation()

    fill_buffer(input)
    output.fill(0)

    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
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
    @__copy_capture(input)
    @parameter
    fn load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    fn avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    @parameter
    fn avg_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    @__copy_capture(output)
    @parameter
    fn avg_pool_compute_finalize[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        output.store(point, res)

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return 1

    alias stencil_axis = IndexList[stencil_rank](1, 2)
    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_with,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize,
    ](output.get_shape(), input.get_shape())

    # CHECK: 2.5199999809265137      3.5999999046325684      4.8000001907348633      4.0799999237060547      3.2400000095367432
    # CHECK: 4.559999942779541       6.4000000953674316      8.3999996185302734      7.0399999618530273      5.5199999809265137
    # CHECK: 7.1999998092651367      10.0    13.0    10.800000190734863      8.3999996185302734
    # CHECK: 6.9600000381469727      9.6000003814697266      12.399999618530273      10.239999771118164      7.9200000762939453
    # CHECK: 6.119999885559082       8.3999996185302734      10.800000190734863      8.880000114440918       6.8400001525878906
    for i in range(0, output_heigh):
        for j in range(0, output_width):
            print(output[0, i, j, 0], "\t", end="")
        print("")


# CHECK-LABEL: test_stencil_avg_pool_stride_2
fn test_stencil_avg_pool_stride_2():
    print("== test_stencil_avg_pool_stride_2")
    alias rank = 4
    alias stencil_rank = 2
    alias dtype = DType.float32
    alias simd_with = 1

    alias input_width = 7
    alias input_height = 7

    alias stride = 2
    alias pool_window_h = 3
    alias pool_window_w = 3
    alias dilation = 1

    alias input_shape = DimList(1, input_height, input_width, 1)

    alias output_heigh = (input_height - pool_window_h) // stride + 1
    alias output_width = (input_width - pool_window_w) // stride + 1

    alias output_shape = DimList(1, output_heigh, output_width, 1)

    var pad_value = 0

    var input = NDBuffer[dtype, rank, input_shape].stack_allocation()
    var output = NDBuffer[dtype, rank, output_shape].stack_allocation()

    fill_buffer(input)
    output.fill(0)

    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            (point[0] * stride + pool_window_h),
            (point[1] * stride + pool_window_w),
        )
        return lower_bound, upper_bound

    @always_inline
    @__copy_capture(input)
    @parameter
    fn load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    fn avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    @parameter
    fn avg_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    @__copy_capture(output)
    @parameter
    fn avg_pool_compute_finalize[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        output.store(point, res)

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return 1

    alias stencil_axis = IndexList[stencil_rank](1, 2)
    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_with,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize,
    ](output.get_shape(), input.get_shape())

    # CHECK: 9.0     11.0    13.0
    # CHECK: 23.0    25.0    27.0
    # CHECK: 37.0    39.0    41.0
    for i in range(0, output_heigh):
        for j in range(0, output_width):
            print(output[0, i, j, 0], "\t", end="")
        print("")


# CHECK-LABEL: test_stencil_max_pool_dilation_2
fn test_stencil_max_pool_dilation_2():
    print("== test_stencil_max_pool_dilation_2")
    alias rank = 4
    alias stencil_rank = 2
    alias dtype = DType.float32
    alias simd_with = 1

    alias input_width = 7
    alias input_height = 7

    alias stride = 1
    alias pool_window_h = 3
    alias pool_window_w = 3
    alias dilation = 2

    alias input_shape = DimList(1, input_height, input_width, 1)

    alias output_heigh = (
        input_height - pool_window_h - (pool_window_h - 1) * (dilation - 1)
    ) // stride + 1
    alias output_width = (
        input_width - pool_window_w - (pool_window_w - 1) * (dilation - 1)
    ) // stride + 1

    alias output_shape = DimList(1, output_heigh, output_width, 1)

    var pad_value = 0

    var input = NDBuffer[dtype, rank, input_shape].stack_allocation()
    var output = NDBuffer[dtype, rank, output_shape].stack_allocation()

    fill_buffer(input)
    output.fill(0)

    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            (point[0] * stride + pool_window_h * dilation),
            (point[1] * stride + pool_window_w * dilation),
        )
        return lower_bound, upper_bound

    @always_inline
    @__copy_capture(input)
    @parameter
    fn load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    fn max_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    @always_inline
    @parameter
    fn max_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return max(val, result)

    @always_inline
    @__copy_capture(output)
    @parameter
    fn max_pool_compute_finalize[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        output.store(point, val)

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return dilation

    alias stencil_axis = IndexList[stencil_rank](1, 2)
    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_with,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn,
        max_pool_compute_init,
        max_pool_compute,
        max_pool_compute_finalize,
    ](output.get_shape(), input.get_shape())

    # CHECK: 33.0    34.0    35.0
    # CHECK: 40.0    41.0    42.0
    # CHECK: 47.0    48.0    49.0
    for i in range(0, output_heigh):
        for j in range(0, output_width):
            print(output[0, i, j, 0], "\t", end="")
        print("")


fn main():
    test_stencil_avg_pool()
    test_stencil_avg_pool_padded()
    test_stencil_avg_pool_stride_2()
    test_stencil_max_pool_dilation_2()

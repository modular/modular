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

from algorithm._gpu.reduction import reduce_launch
from gpu.host import DeviceContext
from testing import assert_equal, TestSuite

from utils import IndexList, StaticTuple

comptime num_reductions = 2


fn fused_reduce_inner_test[
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    rank: Int,
    dtype: DType,
](
    shape: IndexList[rank],
    init: StaticTuple[Scalar[dtype], num_reductions],
    expected_vals0: List[Float32],
    expected_vals1: List[Float32],
    ctx: DeviceContext,
    offset: Int = 1,
    axis: Int = rank - 1,
) raises:
    var out_shape = shape
    out_shape[axis] = 1

    var in_size = shape.flattened_length()
    var out_size = out_shape.flattened_length()

    assert_equal(
        len(expected_vals0),
        out_size,
        "expected vals must match output shape",
    )
    assert_equal(
        len(expected_vals1),
        out_size,
        "expected vals must match output shape",
    )

    var vec_device = ctx.enqueue_create_buffer[dtype](in_size)
    with vec_device.map_to_host() as vec_host:
        for i in range(in_size):
            vec_host[i] = Scalar[dtype](i // shape[axis] + offset)

    var res_device0 = ctx.enqueue_create_buffer[dtype](out_size)
    var res_device1 = ctx.enqueue_create_buffer[dtype](out_size)
    var input_buf_device = Span[Scalar[dtype]](
        ptr=vec_device.unsafe_ptr(), length=shape.flattened_length()
    )
    var output_buf_device0 = Span[Scalar[dtype]](
        ptr=res_device0.unsafe_ptr(), length=out_shape.flattened_length()
    )
    var output_buf_device1 = Span[Scalar[dtype]](
        ptr=res_device1.unsafe_ptr(), length=out_shape.flattened_length()
    )

    @__copy_capture(input_buf_device, shape)
    @parameter
    fn input_fn[
        dtype: DType,
        width: Int,
        _rank: Int,
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var c = rebind[IndexList[rank]](coords)
        var linear_idx = 0
        var stride = 1

        comptime for i in reversed(range(rank)):
            linear_idx += c[i] * stride
            stride *= shape[i]
        return rebind[SIMD[dtype, width]](
            input_buf_device.unsafe_ptr().load[width=width](linear_idx)
        )

    @__copy_capture(output_buf_device0, output_buf_device1, out_shape)
    @parameter
    fn output_fn[
        _dtype: DType, width: Int, _rank: Int
    ](
        coords: IndexList[_rank],
        val: StaticTuple[SIMD[_dtype, width], num_reductions],
    ):
        var c = rebind[IndexList[rank]](coords)
        var linear_idx = 0
        var stride = 1

        comptime for i in reversed(range(rank)):
            linear_idx += c[i] * stride
            stride *= out_shape[i]
        output_buf_device0.unsafe_ptr().store[width=width](
            linear_idx, rebind[SIMD[dtype, width]](val[0])
        )
        output_buf_device1.unsafe_ptr().store[width=width](
            linear_idx, rebind[SIMD[dtype, width]](val[1])
        )

    reduce_launch[num_reductions, input_fn, output_fn, reduce_fn, rank, dtype](
        shape, axis, init, ctx
    )

    with res_device0.map_to_host() as res_host0:
        for i in range(out_shape.flattened_length()):
            assert_equal(
                String(res_host0[i].cast[DType.float32]()),
                String(expected_vals0[i]),
            )

    with res_device1.map_to_host() as res_host1:
        for i in range(out_shape.flattened_length()):
            assert_equal(
                String(res_host1[i].cast[DType.float32]()),
                String(expected_vals1[i]),
            )

    _ = vec_device
    _ = res_device0
    _ = res_device1


fn reduce_inner_test[
    reduce_fn: fn[dtype: DType, width: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    rank: Int,
    dtype: DType,
    expected_vals_type: DType,
](
    shape: IndexList[rank],
    init: Scalar[dtype],
    expected_vals: List[Scalar[expected_vals_type]],
    ctx: DeviceContext,
    offset: Int = 1,
    axis: Int = rank - 1,
) raises:
    comptime num_reductions = 1

    var out_shape = shape
    out_shape[axis] = 1

    var in_size = shape.flattened_length()
    var out_size = shape.flattened_length() // shape[axis]
    assert_equal(
        len(expected_vals), out_size, "expected vals must match output shape"
    )

    var vec_device = ctx.enqueue_create_buffer[dtype](in_size)

    with vec_device.map_to_host() as vec_host:
        for i in range(in_size):
            vec_host[i] = Scalar[dtype](i // shape[axis] + offset)

    var res_device = ctx.enqueue_create_buffer[dtype](out_size)
    var input_buf_device = Span[Scalar[dtype]](
        ptr=vec_device.unsafe_ptr(), length=shape.flattened_length()
    )
    var output_buf_device = Span[Scalar[dtype]](
        ptr=res_device.unsafe_ptr(), length=out_shape.flattened_length()
    )

    @always_inline
    @parameter
    fn reduce_wrapper[
        dtype: DType, width: Int, reduction_idx: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert reduction_idx < num_reductions, "invalid reduction idx"

        return reduce_fn[dtype, width](lhs, rhs)

    @__copy_capture(input_buf_device, shape)
    @parameter
    fn input_fn[
        dtype: DType,
        width: Int,
        _rank: Int,
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var c = rebind[IndexList[rank]](coords)
        var linear_idx = 0
        var stride = 1

        comptime for i in reversed(range(rank)):
            linear_idx += c[i] * stride
            stride *= shape[i]
        return rebind[SIMD[dtype, width]](
            input_buf_device.unsafe_ptr().load[width=width](linear_idx)
        )

    @__copy_capture(output_buf_device, out_shape)
    @parameter
    fn output_fn[
        _dtype: DType, width: Int, _rank: Int
    ](
        coords: IndexList[_rank],
        val: StaticTuple[SIMD[_dtype, width], num_reductions],
    ):
        var c = rebind[IndexList[rank]](coords)
        var linear_idx = 0
        var stride = 1

        comptime for i in reversed(range(rank)):
            linear_idx += c[i] * stride
            stride *= out_shape[i]
        output_buf_device.unsafe_ptr().store[width=width](
            linear_idx, rebind[SIMD[dtype, width]](val[0])
        )

    reduce_launch[
        num_reductions, input_fn, output_fn, reduce_wrapper, rank, dtype
    ](shape, axis, StaticTuple[_, num_reductions](init), ctx)

    with res_device.map_to_host() as res_host:
        for i in range(out_shape.flattened_length()):
            assert_equal(String(res_host[i]), String(expected_vals[i]))

    _ = vec_device
    _ = res_device


def test_reduce():
    @parameter
    fn reduce_add[
        dtype: DType,
        width: Int,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return x + y

    @parameter
    fn reduce_max[
        dtype: DType,
        width: Int,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return max(x, y)

    @parameter
    fn fused_reduce_add_max[
        dtype: DType,
        width: Int,
        reduction_idx: Int,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert reduction_idx < 2, "reduction idx OOB"

        comptime func = reduce_max if reduction_idx == 0 else reduce_add
        return func(x, y)

    with DeviceContext() as ctx:
        reduce_inner_test[reduce_add](
            IndexList[3](2, 3, 257),
            Float32(0),
            [Float32(257.0), 514.0, 771.0, 1028.0, 1285.0, 1542.0],
            ctx,
        )

        reduce_inner_test[reduce_add](
            IndexList[2](5, 257),
            Float32(0),
            [Float32(257.0), 514.0, 771.0, 1028.0, 1285.0],
            ctx,
        )

        reduce_inner_test[reduce_add](
            IndexList[4](2, 2, 2, 1029),
            Float32(0),
            [
                Float32(1029.0),
                2058.0,
                3087.0,
                4116.0,
                5145.0,
                6174.0,
                7203.0,
                8232.0,
            ],
            ctx,
        )

        reduce_inner_test[reduce_add](
            IndexList[3](5, 3, 2),
            Float32(0),
            [
                Float32(15.0),
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
            ],
            ctx,
            axis=0,
        )

        reduce_inner_test[reduce_add](
            IndexList[3](5, 3, 2),
            Float32(0),
            [
                Float32(4.0),
                5.0,
                10.0,
                11.0,
                16.0,
                17.0,
                22.0,
                23.0,
                28.0,
                29.0,
            ],
            ctx,
            axis=1,
        )

        reduce_inner_test[reduce_max](
            IndexList[2](5, 3),
            Float32.MIN,
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            ctx,
        )

        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.float32](
            IndexList[2](5, 3),
            StaticTuple[Float32, 2](Float32.MIN, 0.0),
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            [Float32(3.0), 6.0, 9.0, 12.0, 15.0],
            ctx,
        )

        # bf16 tests
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            BFloat16.MIN,
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            ctx,
        )

        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.bfloat16](
            IndexList[2](5, 3),
            StaticTuple[BFloat16, 2](BFloat16.MIN, 0.0),
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            [Float32(3.0), 6.0, 9.0, 12.0, 15.0],
            ctx,
        )

        # fp16 tests
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            Float16.MIN,
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            ctx,
        )

        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.float16](
            IndexList[2](5, 3),
            StaticTuple[Float16, 2](Float16.MIN, 0.0),
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            [Float32(3.0), 6.0, 9.0, 12.0, 15.0],
            ctx,
        )

        # int64 tests
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            Int64.MIN,
            [Int64(1), 2, 3, 4, 5],
            ctx,
        )
        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.int64](
            IndexList[2](5, 3),
            StaticTuple[Int64, 2](Int64.MIN, 0),
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            [Float32(3.0), 6.0, 9.0, 12.0, 15.0],
            ctx,
        )
        # Add offset to ensure upper and lower 32 bits of element are non-zero
        var offset: Int = 0xDEADBEEF
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            Int64.MIN,
            [
                Int64(offset),
                Int64(offset + 1),
                Int64(offset + 2),
                Int64(offset + 3),
                Int64(offset + 4),
            ],
            ctx,
            offset=offset,
        )
        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.int64](
            IndexList[2](5, 3),
            StaticTuple[Int64, 2](Int64.MIN, 0),
            [
                Float32(offset),
                Float32(Float64(offset) + 1.0),
                Float32(Float64(offset) + 2.0),
                Float32(Float64(offset) + 3.0),
                Float32(Float64(offset) + 4.0),
            ],
            [
                Float32(Float64(offset) * 3 + 3.0),
                Float32(Float64(offset) * 3 + 6.0),
                Float32(Float64(offset) * 3 + 9.0),
                Float32(Float64(offset) * 3 + 12.0),
                Float32(Float64(offset) * 3 + 15.0),
            ],
            ctx,
            offset=offset,
        )

        # bool tests
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            Scalar[DType.bool].MIN,
            [Scalar[DType.bool](True), False, True, False, True],
            ctx,
        )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

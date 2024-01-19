# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full -I %S/.. %s | FileCheck %s

from math import max

from GatherScatter import scatter_elements
from runtime.llcl import Runtime
from tensor import Tensor, TensorShape
from test_utils import linear_fill


fn test_case[
    type: DType,
](
    input_shape: TensorShape,
    indices_shape: TensorShape,
    axis: Int,
    data_vals: VariadicList[SIMD[type, 1]],
    indices_vals: VariadicList[Int32],
    updates_vals: VariadicList[SIMD[type, 1]],
    output_ref_vals: VariadicList[SIMD[type, 1]],
) raises:
    @always_inline
    @parameter
    fn use_update[
        _type: DType, width: Int
    ](input_val: SIMD[_type, width], update_val: SIMD[_type, width]) -> SIMD[
        _type, width
    ]:
        return update_val

    test_case[type, use_update](
        input_shape,
        indices_shape,
        axis,
        data_vals,
        indices_vals,
        updates_vals,
        output_ref_vals,
    )


fn test_case[
    type: DType,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
](
    input_shape: TensorShape,
    indices_shape: TensorShape,
    axis: Int,
    data_vals: VariadicList[SIMD[type, 1]],
    indices_vals: VariadicList[Int32],
    updates_vals: VariadicList[SIMD[type, 1]],
    output_ref_vals: VariadicList[SIMD[type, 1]],
) raises:
    var data = Tensor[type](input_shape)
    linear_fill(data, data_vals)
    var indices = Tensor[DType.int32](indices_shape)
    linear_fill(indices, indices_vals)
    var updates = Tensor[type](indices_shape)
    linear_fill(updates, updates_vals)
    let output = Tensor[type](input_shape)

    scatter_elements[reduce_fn](
        data._to_ndbuffer[2](),
        indices._to_ndbuffer[2](),
        updates._to_ndbuffer[2](),
        axis,
        output._to_ndbuffer[2](),
    )

    _ = data
    _ = indices
    _ = updates

    var output_ref = Tensor[type](input_shape)
    linear_fill(output_ref, output_ref_vals)

    for i in range(output.num_elements()):
        if output_ref._to_buffer()[i] != output._to_buffer()[i]:
            print_no_newline("FAIL: mismatch at idx ")
            print(i)


fn main() raises:
    fn test_scatter_ax0() raises:
        print("== test_scatter_ax0")
        let data = VariadicList[Float32](0, 0, 0, 0, 0, 0, 0, 0, 0)
        let indices = VariadicList[Int32](1, 0, 2, 0, 2, 1)
        let updates = VariadicList[Float32](1.0, 1.1, 1.2, 2.0, 2.1, 2.2)
        let output_ref = VariadicList[Float32](
            2.0, 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2
        )
        test_case[DType.float32](
            TensorShape(3, 3),
            TensorShape(2, 3),
            0,
            data,
            indices,
            updates,
            output_ref,
        )

    # CHECK-LABEL: test_scatter_ax0
    # CHECK-NOT: FAIL
    test_scatter_ax0()

    fn test_scatter_ax1() raises:
        print("== test_scatter_ax1")
        let data = VariadicList[Float32](1, 2, 3, 4, 5)
        let indices = VariadicList[Int32](1, 3)
        let updates = VariadicList[Float32](1.1, 2.1)
        let output_ref = VariadicList[Float32](1.0, 1.1, 3.0, 2.1, 5.0)
        test_case[DType.float32](
            TensorShape(1, 5),
            TensorShape(1, 2),
            1,
            data,
            indices,
            updates,
            output_ref,
        )

    # CHECK-LABEL: test_scatter_ax1
    # CHECK-NOT: FAIL
    test_scatter_ax1()

    fn test_scatter_neg_indices() raises:
        print("== test_scatter_neg_indices")
        let data = VariadicList[Float32](1, 2, 3, 4, 5)
        let indices = VariadicList[Int32](1, -3)
        let updates = VariadicList[Float32](1.1, 2.1)
        let output_ref = VariadicList[Float32](1.0, 1.1, 2.1, 4.0, 5.0)
        test_case[DType.float32](
            TensorShape(1, 5),
            TensorShape(1, 2),
            1,
            data,
            indices,
            updates,
            output_ref,
        )

    # CHECK-LABEL: test_scatter_neg_indices
    # CHECK-NOT: FAIL
    test_scatter_neg_indices()

    fn test_scatter_reduce_max() raises:
        print("== test_scatter_reduce_max")
        let data = VariadicList[Float32](1, 2, 3, 4, 5)
        let indices = VariadicList[Int32](1, 1)
        let updates = VariadicList[Float32](1.1, 2.1)
        let output_ref = VariadicList[Float32](1.0, 2.1, 3.0, 4.0, 5.0)

        @always_inline
        @parameter
        fn _max[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return max(v1, v2)

        test_case[DType.float32, _max](
            TensorShape(1, 5),
            TensorShape(1, 2),
            1,
            data,
            indices,
            updates,
            output_ref,
        )

    # CHECK-LABEL: test_scatter_reduce_max
    # CHECK-NOT: FAIL
    test_scatter_reduce_max()

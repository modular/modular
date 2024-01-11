# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full -I %S/.. %s | FileCheck %s

from math import max

from GatherScatter import gather_elements
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
    indices_vals: VariadicList[SIMD[DType.int32, 1]],
    output_ref_vals: VariadicList[SIMD[type, 1]],
) raises:
    var data = Tensor[type](input_shape)
    linear_fill(data, data_vals)
    var indices = Tensor[DType.int32](indices_shape)
    linear_fill(indices, indices_vals)
    let output = Tensor[type](indices_shape)

    gather_elements(
        data._to_ndbuffer[2](),
        indices._to_ndbuffer[2](),
        axis,
        output._to_ndbuffer[2](),
    )

    _ = data
    _ = indices

    var output_ref = Tensor[type](indices_shape)
    linear_fill(output_ref, output_ref_vals)

    for i in range(output.num_elements()):
        if output_ref._to_buffer()[i] != output._to_buffer()[i]:
            print_no_newline("FAIL: mismatch at idx ")
            print(i)


fn main() raises:
    fn test_gather_ax1() raises:
        print("== test_gather_ax1")
        let data = VariadicList[Float32](1, 2, 3, 4)
        let indices = VariadicList[Int32](0, 0, 1, 0)
        let output_ref = VariadicList[Float32](1, 1, 4, 3)
        test_case[DType.float32](
            TensorShape(2, 2),
            TensorShape(2, 2),
            1,
            data,
            indices,
            output_ref,
        )

    # CHECK-LABEL: test_gather_ax1
    # CHECK-NOT: FAIL
    test_gather_ax1()

    fn test_gather_ax0() raises:
        print("== test_gather_ax0")
        let data = VariadicList[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9)
        let indices = VariadicList[Int32](1, 2, 0, 2, 0, 0)
        let output_ref = VariadicList[Float32](4, 8, 3, 7, 2, 3)
        test_case[DType.float32](
            TensorShape(3, 3),
            TensorShape(2, 3),
            0,
            data,
            indices,
            output_ref,
        )

    # CHECK-LABEL: test_gather_ax0
    # CHECK-NOT: FAIL
    test_gather_ax0()

    fn test_gather_neg_indices() raises:
        print("== test_gather_neg_indices")
        let data = VariadicList[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9)
        let indices = VariadicList[Int32](-1, -2, 0, -2, 0, 0)
        let output_ref = VariadicList[Float32](7, 5, 3, 4, 2, 3)
        test_case[DType.float32](
            TensorShape(3, 3),
            TensorShape(2, 3),
            0,
            data,
            indices,
            output_ref,
        )

    # CHECK-LABEL: test_gather_neg_indices
    # CHECK-NOT: FAIL
    test_gather_neg_indices()

# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from nn.gather_scatter import gather_elements
from tensor import Tensor, TensorShape


fn test_case[
    type: DType,
](
    axis: Int,
    data: Tensor[type],
    indices: Tensor[DType.int32],
    output: Tensor[type],
) raises:
    var output_ref = output

    gather_elements(
        data._to_ndbuffer[2](),
        indices._to_ndbuffer[2](),
        axis,
        output._to_ndbuffer[2](),
    )

    for i in range(output.num_elements()):
        if output_ref._to_buffer()[i] != output._to_buffer()[i]:
            print("FAIL: mismatch at idx ", end="")
            print(i)


fn main() raises:
    fn test_gather_ax1() raises:
        print("== test_gather_ax1")
        var data = Tensor[DType.float32](
            TensorShape(2, 2), List[Float32](1, 2, 3, 4)
        )
        var indices = Tensor[DType.int32](
            TensorShape(2, 2), List[Int32](0, 0, 1, 0)
        )
        var output_ref = Tensor[DType.float32](
            TensorShape(2, 2), List[Float32](1, 1, 4, 3)
        )
        test_case[DType.float32](1, data, indices, output_ref)

    # CHECK-LABEL: test_gather_ax1
    # CHECK-NOT: FAIL
    test_gather_ax1()

    fn test_gather_ax0() raises:
        print("== test_gather_ax0")
        var data = Tensor[DType.float32](
            TensorShape(3, 3), List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9)
        )
        var indices = Tensor[DType.int32](
            TensorShape(2, 3), List[Int32](1, 2, 0, 2, 0, 0)
        )
        var output_ref = Tensor[DType.float32](
            TensorShape(2, 3), List[Float32](4, 8, 3, 7, 2, 3)
        )
        test_case[DType.float32](0, data, indices, output_ref)

    # CHECK-LABEL: test_gather_ax0
    # CHECK-NOT: FAIL
    test_gather_ax0()

    fn test_gather_neg_indices() raises:
        print("== test_gather_neg_indices")
        var data = Tensor[DType.float32](
            TensorShape(3, 3), List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9)
        )
        var indices = Tensor[DType.int32](
            TensorShape(2, 3), List[Int32](-1, -2, 0, -2, 0, 0)
        )
        var output_ref = Tensor[DType.float32](
            TensorShape(2, 3), List[Float32](7, 5, 3, 4, 2, 3)
        )
        test_case[DType.float32](0, data, indices, output_ref)

    # CHECK-LABEL: test_gather_neg_indices
    # CHECK-NOT: FAIL
    test_gather_neg_indices()

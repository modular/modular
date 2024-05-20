# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -I %S/../ %s | FileCheck %s

from nn.gather_scatter import gather_elements
from closed_source_utils._test_utils import array_equal, TestTensor
from buffer import NDBuffer
from buffer.list import DimList


fn test_case[
    type: DType
](
    axis: Int,
    data: TestTensor[type, 2],
    indices: TestTensor[DType.int32, 2],
    output: TestTensor[type, 2],
) raises:
    var output_ref = output

    gather_elements(
        data.ndbuffer,
        indices.ndbuffer,
        axis,
        output.ndbuffer,
    )

    if not array_equal(output, output_ref):
        print("FAIL")


fn main() raises:
    fn test_gather_ax1() raises:
        print("== test_gather_ax1")

        alias shape = DimList(2, 2)

        var data = TestTensor[DType.float32, 2](
            shape, List[Float32](1, 2, 3, 4)
        )
        var indices = TestTensor[DType.int32, 2](shape, List[Int32](0, 0, 1, 0))
        var output_ref = TestTensor[DType.float32, 2](
            shape, List[Float32](1, 1, 4, 3)
        )

        test_case[DType.float32](1, data, indices, output_ref)

    # CHECK-LABEL: test_gather_ax1
    # CHECK-NOT: FAIL
    test_gather_ax1()

    fn test_gather_ax0() raises:
        print("== test_gather_ax0")

        var data = TestTensor[DType.float32, 2](
            DimList(3, 3), List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9)
        )
        var indices = TestTensor[DType.int32, 2](
            DimList(2, 3), List[Int32](1, 2, 0, 2, 0, 0)
        )
        var output_ref = TestTensor[DType.float32, 2](
            DimList(2, 3), List[Float32](4, 8, 3, 7, 2, 3)
        )

        test_case[DType.float32](0, data, indices, output_ref)

    # CHECK-LABEL: test_gather_ax0
    # CHECK-NOT: FAIL
    test_gather_ax0()

    fn test_gather_neg_indices() raises:
        print("== test_gather_neg_indices")

        var data = TestTensor[DType.float32, 2](
            DimList(3, 3), List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9)
        )
        var indices = TestTensor[DType.int32, 2](
            DimList(2, 3), List[Int32](-1, -2, 0, -2, 0, 0)
        )
        var output_ref = TestTensor[DType.float32, 2](
            DimList(2, 3), List[Float32](7, 5, 3, 4, 2, 3)
        )

        test_case[DType.float32](0, data, indices, output_ref)

    # CHECK-LABEL: test_gather_neg_indices
    # CHECK-NOT: FAIL
    test_gather_neg_indices()

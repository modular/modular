# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: lit %s | FileCheck %s

from Buffer import Buffer, NDBuffer
from DType import DType
from Index import Index
from Int import Int
from IO import print
from List import create_kgen_list
from Memory import memset_zero
from Pad import pad


# CHECK-LABEL: test_pad_1d
fn test_pad_1d():
    print("== test_pad_1d\n")

    alias in_shape = create_kgen_list[__mlir_type.index](3)
    alias out_shape = create_kgen_list[__mlir_type.index](6)

    # Create an input matrix of the form
    # [1, 2, 3]
    var input = NDBuffer[1, in_shape, DType.index.value].stack_allocation()
    input.__setitem__(Index(0).as_tuple(), 1)
    input.__setitem__(Index(1).as_tuple(), 2)
    input.__setitem__(Index(2).as_tuple(), 3)

    # Create a padding array of the form
    # [1, 2]
    var paddings = Buffer[2, DType.index.value].stack_allocation()
    paddings.__setitem__(0, 1)
    paddings.__setitem__(1, 2)

    # Create an output matrix of the form
    # [0, 0, 0, 0, 0, 0]
    var output = NDBuffer[1, out_shape, DType.index.value].stack_allocation()
    memset_zero(output.data, output.size())

    # pad
    pad(output, input, paddings.data, 5)

    # output should have form
    # [5, 1, 2, 3, 5, 5]

    # check: 5
    print(output[0])
    # check: 1
    print(output[1])
    # check: 2
    print(output[2])
    # check: 3
    print(output[3])
    # check: 5
    print(output[4])
    # check: 4
    print(output[5])


# CHECK-LABEL: test_pad_2d
fn test_pad_2d():
    print("== test_pad_2d\n")

    alias in_shape = create_kgen_list[__mlir_type.index](2, 2)
    alias out_shape = create_kgen_list[__mlir_type.index](3, 4)

    # Create an input matrix of the form
    # [[1, 2],
    #  [3, 4]]
    var input = NDBuffer[2, in_shape, DType.index.value].stack_allocation()
    input.__setitem__(Index(0, 0).as_tuple(), 1)
    input.__setitem__(Index(0, 1).as_tuple(), 2)
    input.__setitem__(Index(1, 0).as_tuple(), 3)
    input.__setitem__(Index(1, 1).as_tuple(), 4)

    # Create a padding array of the form
    # [1, 0, 1, 1]
    var paddings = Buffer[4, DType.index.value].stack_allocation()
    paddings.__setitem__(0, 1)
    paddings.__setitem__(1, 0)
    paddings.__setitem__(2, 1)
    paddings.__setitem__(3, 1)

    # Create an output matrix of the form
    # [[0, 0, 0, 0]
    #  [0, 0, 0, 0]
    #  [0, 0, 0, 0]]
    var output = NDBuffer[2, out_shape, DType.index.value].stack_allocation()
    memset_zero(output.data, output.size())

    # pad
    pad(output, input, paddings.data, 6)

    # output should have form
    # [[6, 6, 6, 6]
    #  [6, 1, 2, 6]
    #  [6, 3, 4, 6]]

    # check: 6
    print(output[0, 0])
    # check: 6
    print(output[0, 1])
    # check: 6
    print(output[0, 2])
    # check: 6
    print(output[0, 3])
    # check: 6
    print(output[1, 0])
    # check: 1
    print(output[1, 1])
    # check: 2
    print(output[1, 2])
    # check: 6
    print(output[1, 3])
    # check: 6
    print(output[2, 0])
    # check: 3
    print(output[2, 1])
    # check: 4
    print(output[2, 2])
    # check: 6
    print(output[2, 3])


# CHECK-LABEL: test_pad_3d
fn test_pad_3d():
    print("== test_pad_3d\n")

    alias in_shape = create_kgen_list[__mlir_type.index](1, 2, 2)
    alias out_shape = create_kgen_list[__mlir_type.index](2, 3, 3)

    # Create an input matrix of the form
    # [[[1, 2],
    #   [3, 4]]]
    var input = NDBuffer[3, in_shape, DType.index.value].stack_allocation()
    input.__setitem__(Index(0, 0, 0).as_tuple(), 1)
    input.__setitem__(Index(0, 0, 1).as_tuple(), 2)
    input.__setitem__(Index(0, 1, 0).as_tuple(), 3)
    input.__setitem__(Index(0, 1, 1).as_tuple(), 4)

    # Create a padding array of the form
    # [1, 0, 0, 1, 1, 1, 0]
    var paddings = Buffer[6, DType.index.value].stack_allocation()
    paddings.__setitem__(0, 1)
    paddings.__setitem__(1, 0)
    paddings.__setitem__(2, 0)
    paddings.__setitem__(3, 1)
    paddings.__setitem__(4, 1)
    paddings.__setitem__(5, 0)

    # Create an output matrix of the form
    # [[[0, 0, 0]
    #   [0, 0, 0]
    #   [0, 0, 0]]
    #  [[0, 0, 0]
    #   [0, 0, 0]
    #   [0, 0, 0]]]
    var output = NDBuffer[3, out_shape, DType.index.value].stack_allocation()
    memset_zero(output.data, output.size())

    # pad
    pad(output, input, paddings.data, 7)

    # output should have form
    # [[[7, 7, 7]
    #   [7, 7, 7]
    #   [7, 7, 7]]
    #  [[7, 1, 2]
    #   [7, 3, 4]
    #   [7, 7, 7]]]

    # check: 7
    print(output[0, 0, 0])
    # check: 7
    print(output[0, 0, 1])
    # check: 7
    print(output[0, 0, 2])
    # check: 7
    print(output[0, 1, 0])
    # check: 7
    print(output[0, 1, 1])
    # check: 7
    print(output[0, 1, 2])
    # check: 7
    print(output[0, 2, 0])
    # check: 7
    print(output[0, 2, 1])
    # check: 7
    print(output[0, 2, 2])
    # check: 7
    print(output[1, 0, 0])
    # check: 1
    print(output[1, 0, 1])
    # check: 2
    print(output[1, 0, 2])
    # check: 7
    print(output[1, 1, 0])
    # check: 3
    print(output[1, 1, 1])
    # check: 4
    print(output[1, 1, 2])
    # check: 7
    print(output[1, 2, 0])
    # check: 7
    print(output[1, 2, 1])
    # check: 7
    print(output[1, 2, 2])


fn main() -> Int:
    test_pad_1d()
    test_pad_2d()
    test_pad_3d()
    return 0

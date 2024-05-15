# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.info import alignof
from utils import InlineArray
from math import isclose
from buffer.list import DimList
from buffer import NDBuffer


fn get_minmax[
    dtype: DType
](x: DTypePointer[dtype], N: Int) -> InlineArray[Scalar[dtype], 2]:
    var max_val = x[0]
    var min_val = x[0]
    for i in range(1, N):
        if x[i] > max_val:
            max_val = x[i]
        if x[i] < min_val:
            min_val = x[i]
    return InlineArray[Scalar[dtype], 2](min_val, max_val)


# TODO: use assert_true for comparisons atol_minmax,rtol_minmax = InlineArray[Scalar[dtype],2]
fn compare[
    dtype: DType, N: Int
](x: DTypePointer[dtype], y: DTypePointer[dtype], label: String):
    alias alignment = alignof[dtype]()
    alias unroll_factor = 8

    var atol = DTypePointer[dtype].alloc(N, alignment=alignment)
    var rtol = DTypePointer[dtype].alloc(N, alignment=alignment)

    # TODO: parallelize and unroll this loop
    for i in range(N):
        var d = abs(x[i] - y[i])
        var e = abs(d / y[i])
        atol[i] = d
        rtol[i] = e

    print(label)
    var atol_minmax = get_minmax[dtype](atol, N)
    var rtol_minmax = get_minmax[dtype](rtol, N)
    print("AbsErr-Min/Max", atol_minmax[0], atol_minmax[1])
    print("RelErr-Min/Max", rtol_minmax[0], rtol_minmax[1])
    print("==========================================================")
    atol.free()
    rtol.free()


fn array_equal[
    type: DType, rank: Int, output_shape: DimList
](
    output_x: DTypePointer[type],
    output_y: DTypePointer[type],
    num_elements: Int,
) -> Bool:
    for i in range(num_elements):
        if not isclose(output_x[i], output_y[i]):
            print("FAIL: mismatch at idx ", end="")
            print(i)
            return False
    return True


# TODO: call the above function in this
fn array_equal[
    type: DType, rank: Int, output_shape: DimList
](
    output_x: NDBuffer[type, rank, output_shape],
    output_y: NDBuffer[type, rank, output_shape],
) -> Bool:
    for i in range(output_x.num_elements()):
        if not isclose(output_x.data[i], output_y.data[i]):
            print("FAIL: mismatch at idx ", end="")
            print(output_x.get_nd_index(i))
            return False
    return True


fn create_buffer_from_list[
    dtype: DType
](values: List[Scalar[dtype]]) -> DTypePointer[dtype]:
    var N = len(values)
    var buffer = DTypePointer[dtype].alloc(N)
    for i in range(N):
        buffer[i] = values[i]
    return buffer

# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from memory import UnsafePointer
from nn.argsort import argsort
from testing import assert_true


# CHECK-LABEL: test_argsort
fn test_argsort[
    *,
    ascending: Bool,
    filler: fn (Int, Int) -> Float32,
]() raises:
    print("== test_argsort")

    alias n = 16384

    var input_ptr = UnsafePointer[Float32].alloc(n)
    var input = NDBuffer[DType.float32, 1, _, DimList(n)](input_ptr)

    var indices_ptr = UnsafePointer[Int32].alloc(n)
    var indices = NDBuffer[DType.int32, 1, _, DimList(n)](indices_ptr)
    indices.fill(0)

    for i in range(n):
        input[i] = filler(i, n)

    argsort[ascending=ascending](indices, input)

    for i in range(n):
        if i < n - 1:
            var lhs = input[Int(indices[i])]
            var rhs = input[Int(indices[i + 1])]

            @parameter
            if ascending:
                assert_true(
                    lhs < rhs,
                    msg=String(
                        "input[Int(indices[",
                        i,
                        "])] < input[Int(indices[",
                        i + 1,
                        "])] where the rhs is ",
                        rhs,
                        " and the lhs is ",
                        lhs,
                        " and the indices are ",
                        indices[i],
                        " and ",
                        indices[i + 1],
                        " and the ascending is ",
                        ascending,
                    ),
                )
            else:
                assert_true(
                    lhs > rhs,
                    msg=String(
                        "input[Int(indices[",
                        i,
                        "])] > input[Int(indices[",
                        i + 1,
                        "])] where the rhs is ",
                        rhs,
                        " and the lhs is ",
                        lhs,
                        " and the indices are ",
                        indices[i],
                        " and ",
                        indices[i + 1],
                        " and the ascending is ",
                        ascending,
                    ),
                )
        else:
            var lhs = input[Int(indices[i])]
            var rhs = input[Int(indices[0])]

            @parameter
            if ascending:
                assert_true(
                    lhs > rhs,
                    msg=String(
                        "input[Int(indices[",
                        i,
                        "])] > input[Int(indices[0])] where the rhs is ",
                        rhs,
                        " and the lhs is ",
                        lhs,
                        " and the indices are ",
                        indices[i],
                        " and ",
                        indices[0],
                        " and the ascending is ",
                        ascending,
                    ),
                )
            else:
                assert_true(
                    lhs < rhs,
                    msg=String(
                        "input[Int(indices[",
                        i,
                        "])] < input[Int(indices[0])] where the rhs is ",
                        rhs,
                        " and the lhs is ",
                        lhs,
                        " and the indices are ",
                        indices[i],
                        " and ",
                        indices[0],
                        " and the ascending is ",
                        ascending,
                    ),
                )

    input_ptr.free()
    indices_ptr.free()


fn main() raises:
    fn linear_filler(i: Int, n: Int) -> Float32:
        return i

    fn reverse_filler(i: Int, n: Int) -> Float32:
        return n - i

    test_argsort[ascending=True, filler=linear_filler]()
    test_argsort[ascending=True, filler=reverse_filler]()

    test_argsort[ascending=False, filler=linear_filler]()
    test_argsort[ascending=False, filler=reverse_filler]()

# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from layout.layout_tensor import (
    LayoutTensor,
    Layout,
    RuntimeLayout,
    RuntimeTuple,
    IntTuple,
    UNKNOWN_VALUE,
)
from memory import UnsafePointer, memcpy
from linalg.qr_factorization import qr_factorization, apply_q, form_q
from random import rand, seed
from testing import assert_almost_equal
import internal_utils
from os import abort


# A is a general matrix, B is a non-unit upper triangular matrix
fn trmm[
    dtype: DType,
    element_layout: Layout,
](
    A: LayoutTensor[dtype, element_layout=element_layout, **_],
    B: LayoutTensor[dtype, element_layout=element_layout, **_],
    C: LayoutTensor[dtype, element_layout=element_layout, **_],
):
    m, k1 = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    k, n = Int(B.runtime_layout.shape[0]), Int(B.runtime_layout.shape[1])
    min_kn = min(k, n)
    if k1 < min_kn:
        abort("trmm: A and B must have the at least the same number of columns")
    # C.fill(0.0) doesn't work
    for i in range(m):
        for j in range(n):
            C[i, j] = 0.0
    for i in range(m):
        for j in range(min_kn):
            for p in range(j + 1):
                C[i, j] += A[i, p] * B[p, j]


fn a_mul_bt[
    dtype: DType,
    element_layout: Layout,
](
    A: LayoutTensor[dtype, element_layout=element_layout, **_],
    B: LayoutTensor[dtype, element_layout=element_layout, **_],
    C: LayoutTensor[dtype, element_layout=element_layout, **_],
):
    m, k1 = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    n, k = Int(B.runtime_layout.shape[0]), Int(B.runtime_layout.shape[1])
    if k1 != k:
        abort("a_mul_bt: A and B must have the same number of columns")
    # C.fill(0.0) doesn't work
    for i in range(m):
        for j in range(n):
            C[i, j] = 0.0
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i, j] += A[i, p] * B[j, p]


def all_almost_id[
    dtype: DType,
    element_layout: Layout,
](
    A: LayoutTensor[dtype, element_layout=element_layout, **_],
    atol: Float64,
    rtol: Float64,
):
    m, n = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    for i in range(m):
        for j in range(n):
            reference = SIMD[dtype, A.element_layout.size()](
                1.0 if i == j else 0.0
            )
            assert_almost_equal(A[i, j], reference, atol=atol, rtol=rtol)


fn create_vector[
    dtype: DType, layout: Layout
](
    m: Int,
    ptr: UnsafePointer[Scalar[dtype]],
    out result: LayoutTensor[dtype, layout, ptr.origin],
):
    var dynamic_layout = __type_of(result.runtime_layout)(
        __type_of(result.runtime_layout.shape)(m),
        __type_of(result.runtime_layout.stride)(1),
    )
    return __type_of(result)(ptr, dynamic_layout)


fn create_tensor[
    dtype: DType, layout: Layout
](
    m: Int,
    n: Int,
    ptr: UnsafePointer[Scalar[dtype]],
    out result: LayoutTensor[dtype, layout, ptr.origin],
):
    var dynamic_layout = __type_of(result.runtime_layout)(
        __type_of(result.runtime_layout.shape)(m, n),
        __type_of(result.runtime_layout.stride)(1, m),
    )
    return __type_of(result)(ptr, dynamic_layout)


def main():
    atol = 1e-5
    rtol = 1e-3
    m, n = 80, 50
    min_mn = min(m, n)
    alias a_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    alias v_layout = Layout(UNKNOWN_VALUE)
    alias T = Scalar[DType.float32]
    var a_ptr = UnsafePointer[T]().alloc(m * n)
    var a_ptr_copy = UnsafePointer[T]().alloc(m * n)
    var v_ptr = UnsafePointer[T]().alloc(min_mn)
    seed(123)
    rand[DType.float32](a_ptr, m * n)
    var a = create_tensor[DType.float32, a_layout](m, n, a_ptr)
    memcpy(a_ptr_copy, a_ptr, m * n)
    # factorize
    var a_copy = create_tensor[DType.float32, a_layout](m, n, a_ptr_copy)
    var v = create_vector[DType.float32, v_layout](min_mn, v_ptr)
    qr_factorization[DType.float32](v, a)
    # form Q
    var q_ptr = UnsafePointer[T]().alloc(m * m)
    var q = create_tensor[DType.float32, a_layout](m, m, q_ptr)
    form_q[DType.float32](v, a, q)
    print("check backward stability")
    var q_mul_r_ptr = UnsafePointer[T]().alloc(m * n)
    var q_mul_r = create_tensor[DType.float32, a_layout](m, n, q_mul_r_ptr)
    trmm[DType.float32](q, a, q_mul_r)
    internal_utils.assert_almost_equal(
        q_mul_r.ptr, a_copy.ptr, m * n, atol=atol, rtol=rtol
    )
    print("check orthogonality")
    var q_mul_qt_ptr = UnsafePointer[T]().alloc(m * m)
    var q_mul_qt = create_tensor[DType.float32, a_layout](m, m, q_mul_qt_ptr)
    a_mul_bt[DType.float32](q, q, q_mul_qt)
    all_almost_id(q_mul_qt, atol=atol, rtol=rtol)

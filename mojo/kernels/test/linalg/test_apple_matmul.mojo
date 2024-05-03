# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Checks Apple cblas_sgemm matmul C = A*B when called from Matmul.mojo functions
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from buffer import NDBuffer
from buffer.list import DimList
from LinAlg.Matmul import (
    matmul,
    matmul_M,
)
from LinAlg.MatmulPack import (
    pack_b_ndbuffer,
    pack_b_ndbuffer_M,
    pack_matmul_b_shape_func,
    pack_matmul_b_shape_func_M,
)
from LinAlg.MatmulUtils import elementwise_epilogue_type
from collections import OptionalReg as Optional

from utils.index import Index, StaticIntTuple

from sys.info import os_is_macos

alias alignment = 64

alias some_constant = 30


fn gemm_naive(
    a: NDBuffer,
    b: NDBuffer,
    c: NDBuffer,
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a[i, p].cast[c.type]()
                var b_val = b[p, j].cast[c.type]()
                c[(i, j)] += a_val * b_val


fn gemm_naive_elementwise(
    a: NDBuffer, b: NDBuffer, c: NDBuffer, m: Int, n: Int, k: Int, val: Int
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a[i, p].cast[c.type]()
                var b_val = b[p, j].cast[c.type]()
                c[(i, j)] += a_val * b_val

    for i in range(m):
        for j in range(n):
            c[(i, j)] += val


fn test_matmul[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_b: Bool,
    b_packed: Bool,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    bp: NDBuffer[b_type, 2, DimList.create_unknown[2]()],
    m: Int,
    n: Int,
    k: Int,
    kernel_type_m: Int,
) -> Int:
    var c1_ptr = DTypePointer[c_type].alloc(m * n, alignment=alignment)
    var golden = NDBuffer[c_type, 2, c_shape](c1_ptr, Index(m, n))
    for i in range(m):
        for j in range(n):
            golden[StaticIntTuple[2]((i, j))] = 0

    if b_packed:
        if kernel_type_m != 0:
            pack_b_ndbuffer_M[
                a_type,
                a_shape,
                b_type,
                b_shape,
                c_type,
                c_shape,
            ](b, bp, kernel_type_m)
        else:
            pack_b_ndbuffer[
                a_type,
                a_shape,
                b_type,
                b_shape,
                c_type,
                c_shape,
            ](b, bp)

    if kernel_type_m != 0:
        matmul_M[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            transpose_b,
            b_packed=b_packed,
            elementwise_lambda_fn=epilogue_fn,
        ](
            c,
            a,
            rebind[NDBuffer[b_type, 2, b_shape]](bp),
            kernel_type_m,
        )
    else:
        matmul[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            transpose_b,
            b_packed=b_packed,
            elementwise_lambda_fn=epilogue_fn,
        ](c, a, rebind[NDBuffer[b_type, 2, b_shape]](bp))

    if epilogue_fn:
        gemm_naive_elementwise(a, b, golden, m, n, k, some_constant)
    else:
        gemm_naive(a, b, golden, m, n, k)

    var errors: Int = 0
    for i in range(m):
        for j in range(n):
            if c[i, j] != golden[i, j]:
                errors += 1

    if errors != 0:
        print("\nMatrices don't agree!")

    c1_ptr.free()
    return errors


fn test_matmul[
    lambdas_have_fusion: Bool,
    *,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    b_packed: Bool,
    mixed_kernels: Bool,
](m: Int, n: Int, k: Int):
    print("== test_matmul")
    var errors = 0
    var kernel_type_m = m if mixed_kernels else 0
    alias a_shape = DimList.create_unknown[2]()
    alias b_shape = DimList.create_unknown[2]()
    alias c_shape = DimList.create_unknown[2]()

    var a_ptr = DTypePointer[a_type].alloc(m * k, alignment=alignment)
    var b_ptr = DTypePointer[b_type].alloc(k * n, alignment=alignment)
    var b = NDBuffer[b_type, 2, b_shape](b_ptr, Index(k, n))

    var padded_n_k = StaticIntTuple[2]()
    if kernel_type_m != 0:
        padded_n_k = pack_matmul_b_shape_func_M[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            False,
            True,
        ](b, kernel_type_m)
    else:
        padded_n_k = pack_matmul_b_shape_func[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            False,
            True,
        ](b)

    var padded_n = padded_n_k[1] if b_packed else n
    var padded_k = padded_n_k[0] if b_packed else k

    var c0_ptr = DTypePointer[c_type].alloc(m * n, alignment=alignment)

    var bp_ptr = DTypePointer[b_type].alloc(
        padded_k * padded_n, alignment=alignment
    )

    var bp = NDBuffer[b_type, 2, DimList.create_unknown[2]()](
        bp_ptr, Index(padded_k, padded_n)
    )

    var a = NDBuffer[a_type, 2, a_shape](a_ptr, Index(m, k))

    var c = NDBuffer[c_type, 2, c_shape](c0_ptr, Index(m, n))

    # saturated VNNI only has a range [0,127] for the input a
    var vnni_range: Int = 256
    var cnt: Int = 0
    for i in range(m):
        for p in range(k):
            a[StaticIntTuple[2]((i, p))] = cnt % vnni_range
            cnt += 1

    cnt = 0
    for p in range(k):
        for j in range(n):
            b[StaticIntTuple[2]((p, j))] = cnt % 256 - 128
            bp[StaticIntTuple[2]((p, j))] = b[StaticIntTuple[2]((p, j))]
            cnt += 1

    for i in range(m):
        for j in range(n):
            c[StaticIntTuple[2]((i, j))] = 0

    @parameter
    @always_inline
    @__copy_capture(c)
    fn epilogue_fn[
        _type: DType, width: Int
    ](coords: StaticIntTuple[2], val: SIMD[_type, width]) capturing -> None:
        c.store(coords, rebind[SIMD[c_type, width]](val + some_constant))

    @parameter
    if lambdas_have_fusion:
        errors = test_matmul[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            False,  # transpose_b
            b_packed,  # b_packed
            epilogue_fn,
        ](
            c,
            a,
            b,
            bp,
            m,
            n,
            k,
            m if mixed_kernels else 0,
        )
    else:
        errors = test_matmul[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            False,  # transpose_b
            b_packed,  # b_packed
            None,
        ](
            c,
            a,
            b,
            bp,
            m,
            n,
            k,
            m if mixed_kernels else 0,
        )
    if errors > 0:
        return
    # CHECK: Success
    print("Success")

    a_ptr.free()
    b_ptr.free()
    bp_ptr.free()
    c0_ptr.free()


fn test_shapes[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    b_packed: Bool,
    mixed_kernels: Bool,
]():
    @parameter
    fn test_shapes_helper(m: Int, n: Int, k: Int):
        # Test without output fusion.
        test_matmul[
            False,
            a_type=a_type,
            b_type=b_type,
            c_type=c_type,
            b_packed=b_packed,
            mixed_kernels=mixed_kernels,
        ](m, n, k)
        # Test with output fusion.
        test_matmul[
            True,
            a_type=a_type,
            b_type=b_type,
            c_type=c_type,
            b_packed=b_packed,
            mixed_kernels=mixed_kernels,
        ](m, n, k)

    # Test various shapes.
    test_shapes_helper(256, 1024, 4096)
    test_shapes_helper(4, 5, 6)
    test_shapes_helper(15, 16, 17)
    test_shapes_helper(24, 32, 64)
    test_shapes_helper(61, 73, 79)
    test_shapes_helper(123, 456, 321)
    test_shapes_helper(256, 256, 256)
    test_shapes_helper(2, 65, 1200)


fn test_types[b_packed: Bool, mixed_kernels: Bool]():
    test_shapes[
        DType.float32,
        DType.float32,
        DType.float32,
        b_packed,
        mixed_kernels,
    ]()


fn main():
    @parameter
    if not os_is_macos():
        return

    test_types[b_packed=False, mixed_kernels=False]()
    test_types[b_packed=False, mixed_kernels=True]()
    # Note: b_packed = True doesn't apply for Apple cblas_sgemm. This is handled
    #       in the packing functions, and in get_kernel_config, if we are on
    #       MacOs and for DType.float32 a, b, c used in cblas_sgemm.
    test_types[b_packed=True, mixed_kernels=False]()
    test_types[b_packed=True, mixed_kernels=True]()

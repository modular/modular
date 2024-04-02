# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Checks x86 int8 matmul C = A*B with prepacked B
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from sys.info import has_avx2, has_neon_int8_matmul

from buffer import NDBuffer
from buffer.list import DimList
from Matmul import (
    matmul,
    matmul_M,
    pack_b_ndbuffer,
    pack_b_ndbuffer_M,
    pack_matmul_b_shape_func,
    pack_matmul_b_shape_func_M,
)

from utils.index import Index, StaticIntTuple

alias alignment = 64

alias a_type = DType.uint8
alias b_type = DType.int8
alias c_type = DType.int32

alias M1 = 17
alias M2 = 123
alias N = 143
alias K = 71


fn gemm_naive[](
    a: NDBuffer[_, 2, _],
    b: NDBuffer[_, 2, _],
    c: NDBuffer[_, 2, _],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a[i, p].cast[c.type]()
                var b_val = b[p, j].cast[c.type]()
                c[StaticIntTuple[2]((i, j))] += a_val * b_val


fn test_matmul[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_b: Bool,
    b_packed: Bool,
    saturated: Bool,
](m: Int, n: Int, k: Int, kernel_type_m: Int) -> Int:
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
            transpose_b,
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
            transpose_b,
            True,
        ](b)

    var padded_n = padded_n_k[1] if b_packed else n
    var padded_k = padded_n_k[0] if b_packed else k

    var bp_ptr = DTypePointer[b_type].alloc(
        padded_k * padded_n, alignment=alignment
    )
    var c0_ptr = DTypePointer[c_type].alloc(m * n, alignment=alignment)
    var c1_ptr = DTypePointer[c_type].alloc(m * n, alignment=alignment)

    var a = NDBuffer[a_type, 2, a_shape](a_ptr, Index(m, k))

    var bp = NDBuffer[b_type, 2, DimList.create_unknown[2]()](
        bp_ptr, Index(padded_k, padded_n)
    )
    var c = NDBuffer[c_type, 2, c_shape](c0_ptr, Index(m, n))

    var am = NDBuffer[a_type, 2, a_shape](a_ptr, Index(m, k))
    var bm = NDBuffer[b_type, 2, b_shape](b_ptr, Index(k, n))
    var bpm = NDBuffer[b_type, 2, DimList.create_unknown[2]()](
        bp_ptr, Index(k, n)
    )
    var cm0 = NDBuffer[c_type, 2, c_shape](c0_ptr, Index(m, n))
    var cm1 = NDBuffer[c_type, 2, c_shape](c1_ptr, Index(m, n))

    # saturated VNNI only has a range [0,127] for the input a
    var vnni_range: Int = 128 if saturated else 256
    var cnt: Int = 0
    for i in range(m):
        for p in range(k):
            # uint8 but limited to [0,127]
            am[StaticIntTuple[2]((i, p))] = cnt % vnni_range
            cnt += 1

    cnt = 0
    for p in range(k):
        for j in range(n):
            # int8 [-128, 127]
            bm[StaticIntTuple[2]((p, j))] = cnt % 256 - 128
            bpm[StaticIntTuple[2]((p, j))] = bm[StaticIntTuple[2]((p, j))]
            cnt += 1

    for i in range(m):
        for j in range(n):
            cm0[StaticIntTuple[2]((i, j))] = 0
            cm1[StaticIntTuple[2]((i, j))] = cm0[StaticIntTuple[2]((i, j))]

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
            saturated_vnni=saturated,
        ](c, a, rebind[NDBuffer[b_type, 2, b_shape]](bp), kernel_type_m)
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
            saturated_vnni=saturated,
        ](c, a, rebind[NDBuffer[b_type, 2, b_shape]](bp))

    gemm_naive(am, bm, cm1, m, n, k)

    var errors: Int = 0
    for i in range(m):
        for j in range(n):
            if cm0[i, j] != cm1[i, j]:
                errors += 1

    if errors != 0:
        print("\nMatrices don't agree!")

    a_ptr.free()
    b_ptr.free()
    bp_ptr.free()
    c0_ptr.free()
    c1_ptr.free()
    return errors


alias test_range = False


fn test_matmul[bPacked: Bool, saturated: Bool, mixed: Bool]() -> Int:
    # b_packed = False is not supported with i8mm yet
    var errors: Int = 0

    @parameter
    if test_range:
        for m in range(64):
            for n in range(64):
                for k in range(64):
                    var kernel_type_m = m if mixed else 0
                    errors += test_matmul[
                        a_type,
                        DimList.create_unknown[2](),
                        b_type,
                        DimList.create_unknown[2](),
                        c_type,
                        DimList.create_unknown[2](),
                        False,  # transpose_b
                        bPacked,  # b_packed
                        saturated=saturated,
                    ](m, n, k, kernel_type_m)
    else:
        var kernel_type_m1 = M1 if mixed else 0
        errors += test_matmul[
            a_type,
            DimList.create_unknown[2](),
            b_type,
            DimList.create_unknown[2](),
            c_type,
            DimList.create_unknown[2](),
            False,  # transpose_b
            bPacked,  # b_packed
            saturated=saturated,
        ](M1, N, K, kernel_type_m1)
        var kernel_type_m2 = M2 if mixed else 0
        errors += test_matmul[
            a_type,
            DimList.create_unknown[2](),
            b_type,
            DimList.create_unknown[2](),
            c_type,
            DimList.create_unknown[2](),
            False,  # transpose_b
            bPacked,  # b_packed
            saturated=saturated,
        ](M2, N, K, kernel_type_m2)

    return errors


fn test_matmul_mixed_static[bPacked: Bool, saturated: Bool]() -> Int:
    # b_packed = False is not supported with i8mm yet
    var errors: Int = 0

    errors += test_matmul[
        a_type,
        DimList(M1, K),
        b_type,
        DimList(K, N),
        c_type,
        DimList(M1, N),
        False,  # transpose_b
        bPacked,  # b_packed
        saturated=saturated,
    ](M1, N, K, 0)
    errors += test_matmul[
        a_type,
        DimList(M2, K),
        b_type,
        DimList(K, N),
        c_type,
        DimList(M2, N),
        False,  # transpose_b
        bPacked,  # b_packed
        saturated=saturated,
    ](M2, N, K, 0)

    return errors


fn test_matmul_vnni():
    print("== test_matmul_vnni")
    var errors = test_matmul[False, False, False]()
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_bpacked():
    print("== test_matmul_vnni_bpacked")
    var errors = test_matmul[True, False, False]()
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_saturated():
    print("== test_matmul_vnni_saturated")
    var errors = test_matmul[False, True, False]() if has_avx2() else 0
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_bpacked_saturated():
    print("== test_matmul_vnni_bpacked_saturated")
    var errors = test_matmul[True, True, False]() if has_avx2() else 0
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_mixed_dynamic():
    print("== test_matmul_vnni_mixed_dynamic")
    var errors = test_matmul[False, False, True]()
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_bpacked_mixed_dynamic():
    print("== test_matmul_vnni_bpacked_mixed_dynamic")
    var errors = test_matmul[True, False, True]()
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_saturated_mixed_dynamic():
    print("== test_matmul_vnni_saturated_mixed_dynamic")
    var errors = test_matmul[False, True, True]() if has_avx2() else 0
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_bpacked_saturated_mixed_dynamic():
    print("== test_matmul_vnni_bpacked_saturated_mixed_dynamic")
    var errors = test_matmul[True, True, True]() if has_avx2() else 0
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_mixed_static():
    print("== test_matmul_vnni_mixed_static")
    var errors = test_matmul_mixed_static[False, False]()
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_bpacked_mixed_static():
    print("== test_matmul_vnni_bpacked_mixed_static")
    var errors = test_matmul_mixed_static[True, False]()
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_saturated_mixed_static():
    print("== test_matmul_vnni_saturated_mixed_static")
    var errors = test_matmul_mixed_static[False, True]() if has_avx2() else 0
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_bpacked_saturated_mixed_static():
    print("== test_matmul_vnni_bpacked_saturated_mixed_static")
    var errors = test_matmul_mixed_static[True, True]() if has_avx2() else 0
    # CHECK: 0
    print(errors)


fn main():
    test_matmul_vnni()
    test_matmul_vnni_bpacked()
    test_matmul_vnni_saturated()
    test_matmul_vnni_bpacked_saturated()

    test_matmul_vnni_mixed_dynamic()
    test_matmul_vnni_bpacked_mixed_dynamic()
    test_matmul_vnni_saturated_mixed_dynamic()
    test_matmul_vnni_bpacked_saturated_mixed_dynamic()

    test_matmul_vnni_mixed_static()
    test_matmul_vnni_bpacked_mixed_static()
    test_matmul_vnni_saturated_mixed_static()
    test_matmul_vnni_bpacked_saturated_mixed_static()

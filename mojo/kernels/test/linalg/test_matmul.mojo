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


from Matmul import matmul, pack_b_ndbuffer_M, pack_matmul_b_shape_func_M
from memory.buffer import NDBuffer
from sys.info import has_avx2, has_neon_int8_matmul
from utils.index import Index, StaticIntTuple

alias alignment = 64

alias a_type = DType.uint8
alias b_type = DType.int8
alias c_type = DType.int32


fn gemm_naive[
    a_type: DType, b_type: DType, c_type: DType
](
    a: NDBuffer[a_type, 2, DimList.create_unknown[2]()],
    b: NDBuffer[b_type, 2, DimList.create_unknown[2]()],
    c: NDBuffer[c_type, 2, DimList.create_unknown[2]()],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                let a_val = a[i, p].cast[c_type]()
                let b_val = b[p, j].cast[c_type]()
                c[StaticIntTuple[2]((i, j))] += a_val * b_val


fn test_matmul[
    transpose_b: Bool,
    b_packed: Bool,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    saturated: Bool,
](m: Int, n: Int, k: Int) -> Int:
    let a_ptr = DTypePointer[a_type].aligned_alloc(alignment, m * k)
    let b_ptr = DTypePointer[b_type].aligned_alloc(alignment, k * n)
    let b = NDBuffer[b_type, 2](b_ptr, Index(k, n))

    var padded_n_k = StaticIntTuple[2]()
    padded_n_k = pack_matmul_b_shape_func_M[
        a_type,
        DimList.create_unknown[2](),
        b_type,
        DimList.create_unknown[2](),
        c_type,
        DimList.create_unknown[2](),
        transpose_b,
        True,
    ](b, m)

    let padded_n = padded_n_k[1] if b_packed else n
    let padded_k = padded_n_k[0] if b_packed else k

    let bp_ptr = DTypePointer[b_type].aligned_alloc(
        alignment, padded_k * padded_n
    )
    let c0_ptr = DTypePointer[c_type].aligned_alloc(alignment, m * n)
    let c1_ptr = DTypePointer[c_type].aligned_alloc(alignment, m * n)

    let a = NDBuffer[a_type, 2](a_ptr, Index(m, k))

    let bp = NDBuffer[b_type, 2](bp_ptr, Index(padded_k, padded_n))
    let c = NDBuffer[c_type, 2](c0_ptr, Index(m, n))

    let am = NDBuffer[a_type, 2, DimList.create_unknown[2]()](
        a_ptr, Index(m, k)
    )
    let bm = NDBuffer[b_type, 2, DimList.create_unknown[2]()](
        b_ptr, Index(k, n)
    )
    let bpm = NDBuffer[b_type, 2, DimList.create_unknown[2]()](
        bp_ptr, Index(k, n)
    )
    let cm0 = NDBuffer[c_type, 2, DimList.create_unknown[2]()](
        c0_ptr, Index(m, n)
    )
    let cm1 = NDBuffer[c_type, 2, DimList.create_unknown[2]()](
        c1_ptr, Index(m, n)
    )

    # saturated VNNI only has a range [0,127] for the input a
    let vnni_range: Int = 128 if saturated else 256
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
        pack_b_ndbuffer_M[
            a_type,
            DimList.create_unknown[2](),
            b_type,
            DimList.create_unknown[2](),
            c_type,
            DimList.create_unknown[2](),
        ](b, bp, m)

    matmul[
        a_type,
        DimList.create_unknown[2](),
        b_type,
        DimList.create_unknown[2](),
        c_type,
        DimList.create_unknown[2](),
        transpose_b,
        b_packed=b_packed,
        saturated_vnni=saturated,
    ](c, a, bp)

    gemm_naive[a_type, b_type, c_type](am, bm, cm1, m, n, k)

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


alias M = 123
alias N = 143
alias K = 71

alias test_range = False


fn test_matmul[bPacked: Bool, saturated: Bool]() -> Int:
    # b_packed = False is not supported with i8mm yet
    var errors: Int = 0

    @parameter
    if test_range:
        for m in range(64):
            for n in range(64):
                for k in range(64):
                    errors += test_matmul[
                        False,  # transpose_b
                        bPacked,  # b_packed
                        a_type,
                        b_type,
                        c_type,
                        saturated=saturated,
                    ](m, n, k)
    else:
        errors += test_matmul[
            False,  # transpose_b
            bPacked,  # b_packed
            a_type,
            b_type,
            c_type,
            saturated=saturated,
        ](M, N, K)

    return errors


fn test_matmul_vnni():
    print("== test_matmul_vnni")
    let errors = test_matmul[False, False]()
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_bpacked():
    print("== test_matmul_vnni_bpacked")
    let errors = test_matmul[True, False]()
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_saturated():
    print("== test_matmul_vnni_saturated")
    let errors = test_matmul[False, True]() if has_avx2() else 0
    # CHECK: 0
    print(errors)


fn test_matmul_vnni_bpacked_saturated():
    print("== test_matmul_vnni_bpacked_saturated")
    let errors = test_matmul[True, True]() if has_avx2() else 0
    # CHECK: 0
    print(errors)


fn main():
    test_matmul_vnni()
    test_matmul_vnni_bpacked()
    test_matmul_vnni_saturated()
    test_matmul_vnni_bpacked_saturated()

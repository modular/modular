# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Checks x86 int8 matmul C = A*B with prepacked B
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s


from utils.index import Index, StaticIntTuple
from memory.buffer import NDBuffer
from Matrix import Matrix
from Matmul import matmul, pack_b_ndbuffer, pack_matmul_b_shape_func
from runtime.llcl import Runtime, OwningOutputChainPtr

alias alignment = 64


fn gemm_naive[
    a_type: DType, b_type: DType, c_type: DType
](
    a: Matrix[DimList.create_unknown[2](), a_type, False],
    b: Matrix[DimList.create_unknown[2](), b_type, False],
    c: Matrix[DimList.create_unknown[2](), c_type, False],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                let a_val = a[i, p].cast[c_type]()
                let b_val = b[p, j].cast[c_type]()
                c[i, j] += a_val * b_val


fn test_matmul[
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Bool,
    b_packed: Bool,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    saturated: Bool,
]():

    let a_ptr = DTypePointer[a_type].aligned_alloc(alignment, m * k)
    let b_ptr = DTypePointer[b_type].aligned_alloc(alignment, k * n)
    let b = NDBuffer[2, DimList.create_unknown[2](), b_type](b_ptr, Index(k, n))

    var padded_n_k = StaticIntTuple[2]()
    padded_n_k = pack_matmul_b_shape_func[
        a_type, b_type, c_type, transpose_b, True
    ](b)

    let padded_n = padded_n_k[1] if b_packed else n
    let padded_k = padded_n_k[0] if b_packed else k

    let bp_ptr = DTypePointer[b_type].aligned_alloc(
        alignment, padded_k * padded_n
    )
    let c0_ptr = DTypePointer[c_type].aligned_alloc(alignment, m * n)
    let c1_ptr = DTypePointer[c_type].aligned_alloc(alignment, m * n)

    let a = NDBuffer[2, DimList.create_unknown[2](), a_type](a_ptr, Index(m, k))

    let bp = NDBuffer[2, DimList.create_unknown[2](), b_type](
        bp_ptr, Index(padded_k, padded_n)
    )
    let c = NDBuffer[2, DimList.create_unknown[2](), c_type](
        c0_ptr, Index(m, n)
    )

    let am = Matrix[DimList.create_unknown[2](), a_type, False](
        a_ptr, Index(m, k)
    )
    let bm = Matrix[DimList.create_unknown[2](), b_type, False](
        b_ptr, Index(k, n)
    )
    let bpm = Matrix[DimList.create_unknown[2](), b_type, False](
        bp_ptr, Index(k, n)
    )
    let cm0 = Matrix[DimList.create_unknown[2](), c_type, False](
        c0_ptr, Index(m, n)
    )
    let cm1 = Matrix[DimList.create_unknown[2](), c_type, False](
        c1_ptr, Index(m, n)
    )

    # saturated VNNI only has a range [0,127] for the input a
    let vnni_range: Int = 128 if saturated else 256
    var cnt: Int = 0
    for i in range(m):
        for p in range(k):
            # uint8 but limited to [0,127]
            am[i, p] = cnt % vnni_range
            cnt += 1

    cnt = 0
    for p in range(k):
        for j in range(n):
            # int8 [-128, 127]
            bm[p, j] = cnt % 256 - 128
            bpm[p, j] = bm[p, j]
            cnt += 1

    for i in range(m):
        for j in range(n):
            cm0[i, j] = 0
            cm1[i, j] = cm0[i, j]

    with Runtime() as runtime:
        if b_packed:
            let out_chain = OwningOutputChainPtr(runtime)
            pack_b_ndbuffer[a_type, b_type, c_type](b, bp, out_chain.borrow())
            out_chain.wait()
        let out_chain = OwningOutputChainPtr(runtime)
        matmul[a_type, b_type, c_type, False, transpose_b, b_packed, saturated](
            c, a, bp, out_chain.borrow()
        )
        out_chain.wait()

    gemm_naive[a_type, b_type, c_type](am, bm, cm1, m, n, k)

    var errors: Int = 0
    for i in range(m):
        for j in range(n):
            if cm0[i, j] != cm1[i, j]:
                errors += 1
    # CHECK: 0
    print(errors)
    if errors != 0:
        print("\nMatrices don't agree!")

    a_ptr.free()
    b_ptr.free()
    bp_ptr.free()
    c0_ptr.free()
    c1_ptr.free()


alias N = 257
alias M = 1023
alias K = 513


fn test_matmul_vnni():
    print("== test_matmul_vnni")
    test_matmul[
        N,
        M,
        K,
        False,  # transpose_b
        False,  # b_packed
        DType.uint8,
        DType.int8,
        DType.int32,
        saturated=False,
    ]()


fn test_matmul_vnni_bpacked():
    print("== test_matmul_vnni_bpacked")
    test_matmul[
        N,
        M,
        K,
        False,  # transpose_b
        True,  # b_packed
        DType.uint8,
        DType.int8,
        DType.int32,
        saturated=False,
    ]()


fn test_matmul_vnni_saturated():
    print("== test_matmul_vnni_saturated")
    test_matmul[
        N,
        M,
        K,
        False,  # transpose_b
        False,  # b_packed
        DType.uint8,
        DType.int8,
        DType.int32,
        True,
    ]()


fn test_matmul_vnni_bpacked_saturated():
    print("== test_matmul_vnni_bpacked_saturated")
    test_matmul[
        N,
        M,
        K,
        False,  # transpose_b
        True,  # b_packed
        DType.uint8,
        DType.int8,
        DType.int32,
        True,
    ]()


fn main():
    test_matmul_vnni()
    test_matmul_vnni_bpacked()
    test_matmul_vnni_saturated()
    test_matmul_vnni_bpacked_saturated()

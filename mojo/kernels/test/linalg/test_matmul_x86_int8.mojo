# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Checks x86 int8 matmul C = A*B with prepacked B
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: avx2
# RUN: %mojo -debug-level full %s | FileCheck %s

from Range import range
from SIMD import SIMD
from List import DimList
from Index import Index, StaticIntTuple
from Pointer import DTypePointer
from Buffer import NDBuffer, DynamicRankBuffer
from Matrix import Matrix
from Matmul import matmul_parallel_sync, _submatmul_sequential_sync, pack_b
from MatmulUtils import (
    MatmulConfig,
    calculate_tile_n_k,
    get_partitioned_matmul,
    get_matmul_config,
    search_mm_config,
    is_critical_stride,
    PartitionHeuristic,
)
from math import div_ceil, align_up, max, min
from IO import print, print_no_newline

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


fn _get_tile_n_k[
    config: MatmulConfig,
    transpose_b: Bool,
](n: Int, k: Int) -> StaticIntTuple[2]:
    @parameter
    if not transpose_b:
        return calculate_tile_n_k[
            config.pack_data_size, config.pack_inner_size
        ](n, k)
    else:
        return calculate_tile_n_k[
            config.pack_data_size, config.pack_inner_size
        ](k, n)


fn test_matmul[
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Bool,
    b_packed: Bool,
    a_type: DType,
    b_type: DType,
    c_type: DType,
]():
    alias config = search_mm_config[
        a_type, b_type, c_type, True, is_critical_stride(k)
    ]()
    let tile_n_k = _get_tile_n_k[config, transpose_b](n, k)
    let tile_n = tile_n_k[0]
    let tile_k = align_up(tile_n_k[1], 4) if b_packed else tile_n_k[1]
    let padded_n = div_ceil(n, tile_n) * tile_n if b_packed else n
    let padded_k = div_ceil(k, tile_k) * tile_k if b_packed else k

    # k%4 != 0 can overread into unallocated memory and fault
    # 3 bytes of padding fixes this. See issue 18784
    alias extra_bytes = 3
    let a_ptr = DTypePointer[a_type].aligned_alloc(
        alignment, m * k + extra_bytes
    )
    let b_ptr = DTypePointer[b_type].aligned_alloc(alignment, k * n)
    let bp_ptr = DTypePointer[b_type].aligned_alloc(
        alignment, padded_k * padded_n
    )
    let c0_ptr = DTypePointer[c_type].aligned_alloc(alignment, m * n)
    let c1_ptr = DTypePointer[c_type].aligned_alloc(alignment, m * n)

    let a = NDBuffer[2, DimList.create_unknown[2](), a_type](a_ptr, Index(m, k))
    let b = NDBuffer[2, DimList.create_unknown[2](), b_type](b_ptr, Index(k, n))
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

    var cnt: Int = 0
    for i in range(m):
        for p in range(k):
            # uint8 but limited to [0,127]
            am[i, p] = cnt % 128
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

    let sub_matmul_config = get_partitioned_matmul[PartitionHeuristic.MOJO](
        m, n, k, 0, 1
    )

    if b_packed:
        pack_b[
            transpose_b,
            config.simd_size,
            config.pack_inner_size,
            b_type,
            DimList.create_unknown[2](),
            DimList.create_unknown[2](),
        ](
            bp,
            b,
            tile_n,
            tile_k,
        )

    _submatmul_sequential_sync[
        a_type,
        b_type,
        c_type,
        False,  # transpose_a - not supported yet
        transpose_b,
        b_packed,
    ](c, a, bp, sub_matmul_config.shape, sub_matmul_config.offset)

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


fn test_matmul_vnni():
    print("== test_matmul_vnni")
    test_matmul[
        257,
        1023,
        513,
        False,  # transpose_b
        False,  # b_packed
        DType.uint8,
        DType.int8,
        DType.int32,
    ]()


fn test_matmul_vnni_bpacked():
    print("== test_matmul_vnni_bpacked")
    test_matmul[
        257,
        1023,
        513,
        False,  # transpose_b
        True,  # b_packed
        DType.uint8,
        DType.int8,
        DType.int32,
    ]()


fn main():
    test_matmul_vnni()
    test_matmul_vnni_bpacked()

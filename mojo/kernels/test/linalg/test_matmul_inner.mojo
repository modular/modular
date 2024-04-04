# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import div_ceil
from sys.info import has_neon, simdwidthof

from buffer import NDBuffer
from buffer.list import DimList
from LinAlg.Matmul import GemmShape, MatmulConfig, MatmulInnerLoopBPacked
from LinAlg.MatmulUtils import (
    get_matmul_kernel_shape,
    get_matmul_prefetch_b_distance_k,
)

from utils.index import Index

alias prefetch_b_distance_k: Int = get_matmul_prefetch_b_distance_k()

alias M: Int = 64
alias N: Int = 64
alias K: Int = 64


fn matmul_inner_loop[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_shape: DimList,
    b_shape: DimList,
    c_shape: DimList,
    tile_inner_size: Int,
    simd_size: Int,
    a_row_size: Int,
    pack_inner_size: Int,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b_packed: NDBuffer[b_type, 3, b_shape],
    m: Int,
    n: Int,
    k: Int,
):
    MatmulInnerLoopBPacked[
        a_shape,
        c_shape,
        b_shape,
        a_type,
        b_type,
        c_type,
        simd_size,
        a_row_size,
        pack_inner_size * simd_size,
        True,  # skip bound check
        prefetch_b_distance_k,
        False,  # saturated_vnni
    ].run(
        c,
        a,
        b_packed,
        # Below are configurations for outer loops, just
        #  use the trivial numbers for now.
        GemmShape(0, 0, 0),  # Tile offset.
        GemmShape(m, n, k),  # Global tile dimension.
        Index(n, k),  # Local tile dimension.
    )


# CHECK-LABEL: test_micro_kernel
fn test_micro_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
](m: Int, n: Int, k: Int):
    print("== test_micro_kernel")

    alias simd_size: Int = simdwidthof[c_type]()
    alias kernel_shape = get_matmul_kernel_shape[
        a_type, b_type, c_type, False
    ]()
    alias a_row_size = kernel_shape.a_row_size
    alias pack_inner_size = kernel_shape.pack_inner_size
    alias tile_inner_size: Int = pack_inner_size * simd_size

    var alignment = 64
    var a_ptr = DTypePointer[a_type].alloc(m * k, alignment=alignment)

    alias a_shape = DimList.create_unknown[2]()
    var a = NDBuffer[a_type, 2, a_shape](a_ptr, Index(m, k))
    a.fill(1)

    alias b_shape = DimList.create_unknown[3]()
    var b_packed_ptr = DTypePointer[b_type].alloc(
        (n // tile_inner_size) * k * tile_inner_size, alignment=alignment
    )
    var b_packed = NDBuffer[b_type, 3, b_shape](
        b_packed_ptr, Index(n // tile_inner_size, k, tile_inner_size)
    )
    b_packed.fill(1)

    alias c_shape = DimList.create_unknown[2]()
    var c_ptr = DTypePointer[c_type].alloc(m * n, alignment=alignment)
    var c = NDBuffer[c_type, 2, c_shape](c_ptr, Index(m, n))
    # var c = NDBuffer[c_type, 2, DimList(M, N)].aligned_stack_allocation[128]()
    c.fill(0)

    matmul_inner_loop[
        a_type,
        b_type,
        c_type,
        a_shape,
        b_shape,
        c_shape,
        tile_inner_size,
        simd_size,
        a_row_size,
        pack_inner_size,
    ](c, a, b_packed, m, n, k)

    # CHECK: 64.0
    print(c[0, 0])
    a_ptr.free()
    b_packed_ptr.free()
    c_ptr.free()


fn test_micro_kernel_static[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
](m: Int):
    alias simd_size: Int = simdwidthof[c_type]()
    alias kernel_shape = get_matmul_kernel_shape[
        a_type, b_type, c_type, False
    ]()
    alias a_row_size = kernel_shape.a_row_size
    alias pack_inner_size = kernel_shape.pack_inner_size
    alias tile_inner_size: Int = pack_inner_size * simd_size

    alias alignment = alignof[SIMD[c_type, simd_size]]()
    var a_ptr = DTypePointer[a_type].alloc(m * K, alignment=alignment)

    alias a_shape = DimList.create_unknown[2]()
    var a = NDBuffer[a_type, 2, a_shape](a_ptr, Index(m, K))
    a.fill(1)

    alias b_shape = DimList(N // tile_inner_size, K, tile_inner_size)
    var b_packed_ptr = DTypePointer[b_type].alloc(
        (N // tile_inner_size) * K * tile_inner_size, alignment=alignment
    )
    var b_packed = NDBuffer[b_type, 3, b_shape](
        b_packed_ptr, Index(N // tile_inner_size, K, tile_inner_size)
    )
    b_packed.fill(1)

    alias c_shape = DimList.create_unknown[2]()
    var c_ptr = DTypePointer[c_type].alloc(m * N, alignment=alignment)
    var c = NDBuffer[c_type, 2, c_shape](c_ptr, Index(m, N))
    c.fill(0)

    matmul_inner_loop[
        a_type,
        b_type,
        c_type,
        a_shape,
        b_shape,
        c_shape,
        tile_inner_size,
        simd_size,
        a_row_size,
        pack_inner_size,
    ](c, a, b_packed, m, N, K)

    a_ptr.free()
    b_packed_ptr.free()
    c_ptr.free()


@export(ABI="C")
fn kernel_export_dynamic(m: Int, n: Int, k: Int):
    test_micro_kernel[DType.float32, DType.float32, DType.float32](m, n, k)


@export(ABI="C")
fn kernel_export_static(m: Int):
    test_micro_kernel_static[DType.float32, DType.float32, DType.float32, M, N](
        m
    )


fn main():
    test_micro_kernel[DType.float32, DType.float32, DType.float32](M, N, K)

    # TODO(30525): Re-enable after we resolve llvm lowering issues.
    @parameter
    if not has_neon():
        test_micro_kernel[DType.bfloat16, DType.bfloat16, DType.bfloat16](
            M, N, K
        )
        test_micro_kernel[DType.bfloat16, DType.bfloat16, DType.float32](
            M, N, K
        )

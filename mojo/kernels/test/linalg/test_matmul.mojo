# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Checks x86 int8 matmul C = A*B with prepacked B
#
# ===----------------------------------------------------------------------=== #
# NOTE: Takes ~20 minutes to run with asan
# UNSUPPORTED: asan
# RUN: %mojo %s

from sys.info import has_avx2, has_neon_int8_matmul, os_is_macos

from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.matmul import _matmul_cpu, matmul
from linalg.packing import (
    _pack_b_ndbuffer_impl,
    _pack_matmul_b_shape_func_impl,
    pack_b_ndbuffer,
    pack_matmul_b_shape_func,
)
from memory import UnsafePointer
from testing import assert_almost_equal, assert_equal

from utils.index import Index, IndexList

alias alignment = 64


fn gemm_naive[](
    a: NDBuffer,
    b: NDBuffer,
    c: NDBuffer[mut=True, *_],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a[i, p].cast[c.type]()
                var b_val = b[p, j].cast[c.type]()
                c[i, j] += a_val * b_val


def test_matmul[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_b: Bool,
    b_packed: Bool,
    saturated: Bool,
](m: Int, n: Int, k: Int, kernel_type_m: Int):
    var a_ptr = UnsafePointer[Scalar[a_type], alignment=alignment].alloc(m * k)
    var b_ptr = UnsafePointer[Scalar[b_type], alignment=alignment].alloc(k * n)
    var b = NDBuffer[b_type, 2, _, b_shape](b_ptr, Index(k, n))

    var padded_n_k = IndexList[2]()
    if kernel_type_m != 0:
        padded_n_k = _pack_matmul_b_shape_func_impl[
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

    var bp_ptr = UnsafePointer[Scalar[b_type], alignment=alignment].alloc(
        padded_k * padded_n
    )
    var c0_ptr = UnsafePointer[Scalar[c_type], alignment=alignment].alloc(m * n)
    var c1_ptr = UnsafePointer[Scalar[c_type], alignment=alignment].alloc(m * n)

    var a = NDBuffer[a_type, 2, _, a_shape](a_ptr, Index(m, k))

    var bp = NDBuffer[b_type, 2, _, DimList.create_unknown[2]()](
        bp_ptr, Index(padded_k, padded_n)
    )
    var c = NDBuffer[c_type, 2, _, c_shape](c0_ptr, Index(m, n))

    var golden = NDBuffer[c_type, 2, _, c_shape](c1_ptr, Index(m, n))

    # saturated VNNI only has a range [0,127] for the input a
    var vnni_range: Int = 128 if saturated else 256
    var cnt: Int = 0
    for i in range(m):
        for p in range(k):
            # uint8 but limited to [0,127]
            a[IndexList[2]((i, p))] = cnt % vnni_range
            cnt += 1

    cnt = 0
    for p in range(k):
        for j in range(n):
            # int8 [-128, 127]
            b[IndexList[2]((p, j))] = cnt % 256 - 128
            bp[IndexList[2]((p, j))] = b[IndexList[2]((p, j))]
            cnt += 1

    for i in range(m):
        for j in range(n):
            c[IndexList[2]((i, j))] = 0
            golden[IndexList[2]((i, j))] = c[IndexList[2]((i, j))]

    @parameter
    if b_packed:
        if kernel_type_m != 0:
            _pack_b_ndbuffer_impl[
                a_type, a_shape, b_type, b_shape, c_type, c_shape, transpose_b
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
        _matmul_cpu[
            transpose_b=transpose_b,
            b_packed=b_packed,
            saturated_vnni=saturated,
        ](
            c,
            a,
            rebind[NDBuffer[b_type, 2, bp.origin, b_shape]](bp),
            kernel_type_m,
        )
    else:
        matmul[
            transpose_b=transpose_b,
            b_packed=b_packed,
            saturated_vnni=saturated,
        ](c, a, rebind[NDBuffer[b_type, 2, bp.origin, b_shape]](bp))

    gemm_naive(a, b, golden, m, n, k)

    for i in range(m):
        for j in range(n):
            var msg = String(
                "values do not agree for ",
                m,
                "x",
                n,
                "x",
                k,
                " using the dtype=",
                a_type,
                ",",
                b_type,
                ",",
                c_type,
            )

            @parameter
            if c_type.is_floating_point():
                assert_almost_equal(c[i, j], golden[i, j], msg)
            else:
                assert_equal(c[i, j], golden[i, j], msg)

    a_ptr.free()
    b_ptr.free()
    bp_ptr.free()
    c0_ptr.free()
    c1_ptr.free()


def test_matmul[
    *,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    b_packed: Bool,
    saturated: Bool,
    mixed_kernels: Bool,
](m: Int, n: Int, k: Int):
    alias a_shape = DimList.create_unknown[2]()
    alias b_shape = DimList.create_unknown[2]()
    alias c_shape = DimList.create_unknown[2]()
    test_matmul[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
        False,  # transpose_b
        b_packed,  # b_packed
        saturated=saturated,
    ](m, n, k, m if mixed_kernels else 0)


def test_shapes[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    b_packed: Bool,
    saturated: Bool,
    mixed_kernels: Bool,
]():
    @parameter
    fn test_shapes_helper(m: Int, n: Int, k: Int) raises:
        test_matmul[
            a_type=a_type,
            b_type=b_type,
            c_type=c_type,
            b_packed=b_packed,
            saturated=saturated,
            mixed_kernels=mixed_kernels,
        ](m, n, k)

    test_shapes_helper(256, 1024, 4096)
    test_shapes_helper(4, 5, 6)
    test_shapes_helper(15, 16, 17)
    test_shapes_helper(24, 32, 64)
    test_shapes_helper(61, 73, 79)
    test_shapes_helper(123, 456, 321)
    test_shapes_helper(256, 256, 256)
    test_shapes_helper(2, 65, 1200)
    test_shapes_helper(1, 5, 10)
    test_shapes_helper(1, 768, 384)
    test_shapes_helper(1, 1, 1)
    test_shapes_helper(768, 1, 384)
    test_shapes_helper(1, 384, 1)
    test_shapes_helper(384, 768, 1)


def test_types[b_packed: Bool, saturated: Bool, mixed_kernels: Bool]():
    @parameter
    if b_packed and os_is_macos():
        return

    test_shapes[
        DType.uint8,
        DType.uint8,
        DType.int32,
        b_packed,
        saturated,
        mixed_kernels,
    ]()
    test_shapes[
        DType.uint8, DType.int8, DType.int32, b_packed, saturated, mixed_kernels
    ]()
    test_shapes[
        DType.int8, DType.int8, DType.int32, b_packed, saturated, mixed_kernels
    ]()
    if not saturated:
        test_shapes[
            DType.float32,
            DType.float32,
            DType.float32,
            b_packed,
            saturated,
            mixed_kernels,
        ]()


def main():
    test_types[b_packed=False, saturated=False, mixed_kernels=False]()
    test_types[b_packed=True, saturated=False, mixed_kernels=False]()
    test_types[b_packed=False, saturated=True, mixed_kernels=False]()
    test_types[b_packed=True, saturated=True, mixed_kernels=False]()
    test_types[b_packed=False, saturated=False, mixed_kernels=True]()
    test_types[b_packed=True, saturated=False, mixed_kernels=True]()
    test_types[b_packed=False, saturated=True, mixed_kernels=True]()
    test_types[b_packed=True, saturated=True, mixed_kernels=True]()

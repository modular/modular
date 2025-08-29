# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
#
# Checks Apple cblas_sgemm matmul C = A*B and apple_gemv, when called from
# Matmul.mojo functions
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg

import benchmark
from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.bmm import batched_matmul
from linalg.matmul import _matmul_cpu, matmul
from linalg.packing import (
    _pack_b_ndbuffer_impl,
    _pack_matmul_b_shape_func_impl,
    pack_b_ndbuffer,
    pack_matmul_b_shape_func,
    pack_transposed_b_ndbuffer,
)
from linalg.utils import elementwise_epilogue_type
from testing import assert_almost_equal, assert_true

from utils.index import Index, IndexList

alias alignment = 64
alias some_constant = 20
alias do_benchmarking = False


@parameter
fn bench_run[
    func: fn () raises capturing [_] -> None
]() raises -> benchmark.Report:
    return benchmark.run[func](2, 1_000_000, 1, 3)


fn gemm_naive[
    transpose_b: Bool
](a: NDBuffer, b: NDBuffer, c: NDBuffer[mut=True, *_], m: Int, n: Int, k: Int):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a[i, p].cast[c.type]()
                var b_val = b[
                    IndexList[b.rank](j, p) if transpose_b else IndexList[
                        b.rank
                    ](p, j)
                ].cast[c.type]()
                c[i, j] += a_val * b_val


fn gemm_naive_elementwise[
    transpose_b: Bool
](
    a: NDBuffer,
    b: NDBuffer,
    c: NDBuffer[mut=True, *_],
    m: Int,
    n: Int,
    k: Int,
    val: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a[i, p].cast[c.type]()
                var b_val = b[
                    IndexList[b.rank](j, p) if transpose_b else IndexList[
                        b.rank
                    ](p, j)
                ].cast[c.type]()
                c[i, j] += a_val * b_val

    for i in range(m):
        for j in range(n):
            c[i, j] += val


def test_matmul[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_b: Bool,
    b_packed: Bool,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
](
    c: NDBuffer[mut=True, c_type, 2, _, c_shape],
    a: NDBuffer[a_type, 2, _, a_shape],
    b: NDBuffer[b_type, 2, _, b_shape],
    bp: NDBuffer[mut=True, b_type, 2, _, DimList.create_unknown[2]()],
    m: Int,
    n: Int,
    k: Int,
    kernel_type_m: Int,
) -> Int:
    var c1_ptr = UnsafePointer[Scalar[c_type], alignment=alignment].alloc(m * n)
    var golden = NDBuffer[c_type, 2, _, c_shape](c1_ptr, Index(m, n))
    for i in range(m):
        for j in range(n):
            golden[IndexList[2]((i, j))] = 0

    if b_packed:
        if not transpose_b:
            if kernel_type_m != 0:
                _pack_b_ndbuffer_impl[
                    a_type,
                    a_shape,
                    b_type,
                    b_shape,
                    c_type,
                    c_shape,
                    transpose_b,
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
        else:
            if kernel_type_m != 0:
                _pack_b_ndbuffer_impl[
                    a_type,
                    a_shape,
                    b_type,
                    b_shape,
                    c_type,
                    c_shape,
                    transpose_b,
                ](b, bp, kernel_type_m)
            else:
                pack_transposed_b_ndbuffer[
                    a_type,
                    a_shape,
                    b_type,
                    b_shape,
                    c_type,
                    c_shape,
                ](b, bp)

    @always_inline
    @__copy_capture(c, a, bp)
    @parameter
    fn bench_fn_matmul() raises:
        if kernel_type_m != 0:
            _matmul_cpu[
                transpose_b=transpose_b,
                b_packed=b_packed,
                elementwise_lambda_fn=epilogue_fn,
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
                elementwise_lambda_fn=epilogue_fn,
            ](c, a, rebind[NDBuffer[b_type, 2, bp.origin, b_shape]](bp))

    bench_fn_matmul()

    @parameter
    if do_benchmarking:
        var matmul_perf = bench_run[bench_fn_matmul]()
        benchmark.keep(c[0, 0])
        print(
            "Apple Matmul GFLOP/s for (M, N, K) = (",
            m,
            n,
            k,
            "): ",
            1e-9 * ((2 * m * k * n) / matmul_perf.mean()),
        )

    @parameter
    if epilogue_fn:
        gemm_naive_elementwise[transpose_b](
            a, b, golden, m, n, k, some_constant
        )
    else:
        gemm_naive[transpose_b](a, b, golden, m, n, k)

    var errors: Int = 0
    for i in range(m):
        for j in range(n):
            if c[i, j] != golden[i, j]:
                assert_almost_equal(
                    c[i, j],
                    golden[i, j],
                    msg=String(
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
                    ),
                )

    c1_ptr.free()
    return errors


def test_matmul[
    lambdas_have_fusion: Bool,
    *,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    b_packed: Bool,
    mixed_kernels: Bool,
    transpose_b: Bool,
](m: Int, n: Int, k: Int):
    print("== test_matmul")
    var errors = 0
    var kernel_type_m = m if mixed_kernels else 0
    alias a_shape = DimList.create_unknown[2]()
    alias b_shape = DimList.create_unknown[2]()
    alias c_shape = DimList.create_unknown[2]()

    var a_ptr = UnsafePointer[Scalar[a_type], alignment=alignment].alloc(m * k)
    var b_ptr = UnsafePointer[Scalar[b_type], alignment=alignment].alloc(k * n)
    var b = NDBuffer[b_type, 2, _, b_shape](
        b_ptr, Index(n, k) if transpose_b else Index(k, n)
    )

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

    var padded_n = (
        padded_n_k[1] if b_packed or (not b_packed and transpose_b) else n
    )
    var padded_k = (
        padded_n_k[0] if b_packed or (not b_packed and transpose_b) else k
    )

    var c0_ptr = UnsafePointer[Scalar[c_type], alignment=alignment].alloc(m * n)

    var bp_ptr = UnsafePointer[Scalar[b_type], alignment=alignment].alloc(
        padded_k * padded_n
    )

    var bp = NDBuffer[b_type, 2, _, DimList.create_unknown[2]()](
        bp_ptr, Index(padded_k, padded_n)
    )
    var a = NDBuffer[a_type, 2, _, a_shape](a_ptr, Index(m, k))

    var c = NDBuffer[c_type, 2, _, c_shape](c0_ptr, Index(m, n))

    for i in range(m):
        for p in range(k):
            a[IndexList[2]((i, p))] = Scalar[a_type](0.001) * i

    for p in range(n if transpose_b else k):
        for j in range(k if transpose_b else n):
            b[IndexList[2]((p, j))] = Scalar[b_type](0.002) * p
            if b_packed and not transpose_b:
                bp[IndexList[2]((j, p))] = b[IndexList[2]((p, j))]
            else:
                bp[IndexList[2]((p, j))] = b[IndexList[2]((p, j))]

    for i in range(m):
        for j in range(n):
            c[IndexList[2]((i, j))] = 0

    @parameter
    @always_inline
    @__copy_capture(c)
    fn epilogue_fn[
        _type: DType, width: Int, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[_type, width]) -> None:
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
            transpose_b,  # transpose_b
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
            transpose_b,  # transpose_b
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
    print("Success")

    a_ptr.free()
    b_ptr.free()
    bp_ptr.free()
    c0_ptr.free()


def test_shapes[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    b_packed: Bool,
    mixed_kernels: Bool,
]():
    @parameter
    def test_shapes_helper[transpose_b: Bool = False](m: Int, n: Int, k: Int):
        # Test without output fusion.
        test_matmul[
            False,
            a_type=a_type,
            b_type=b_type,
            c_type=c_type,
            b_packed=b_packed,
            mixed_kernels=mixed_kernels,
            transpose_b=transpose_b,
        ](m, n, k)
        # Test with output fusion.
        test_matmul[
            True,
            a_type=a_type,
            b_type=b_type,
            c_type=c_type,
            b_packed=b_packed,
            mixed_kernels=mixed_kernels,
            transpose_b=transpose_b,
        ](m, n, k)

    # Test various matmul and gemv shapes with and without transpose_b.

    # Test with transpose_b = False
    test_shapes_helper(256, 1024, 4096)
    test_shapes_helper(4, 5, 6)
    test_shapes_helper(15, 16, 17)
    test_shapes_helper(24, 32, 64)
    test_shapes_helper(61, 73, 79)
    test_shapes_helper(123, 456, 321)
    test_shapes_helper(256, 256, 256)
    test_shapes_helper(2, 65, 1200)

    # Test with transpose_b = True
    test_shapes_helper[True](256, 1024, 4096)
    test_shapes_helper[True](4, 5, 6)
    test_shapes_helper[True](15, 16, 17)
    test_shapes_helper[True](24, 32, 64)
    test_shapes_helper[True](61, 73, 79)
    test_shapes_helper[True](123, 456, 321)
    test_shapes_helper[True](256, 256, 256)
    test_shapes_helper[True](2, 65, 1200)

    # Test with transpose_b = False
    test_shapes_helper(1, 5120, 3072)
    test_shapes_helper(1, 3072, 3072)
    test_shapes_helper(1, 12288, 3072)
    test_shapes_helper(1, 3072, 12288)
    test_shapes_helper(1, 32768, 3072)

    # Test with transpose_b = True
    test_shapes_helper[True](1, 5120, 3072)
    test_shapes_helper[True](1, 3072, 3072)
    test_shapes_helper[True](1, 12288, 3072)
    test_shapes_helper[True](1, 3072, 12288)
    test_shapes_helper[True](1, 32768, 3072)


def test_types[b_packed: Bool, mixed_kernels: Bool]():
    test_shapes[
        DType.float32,
        DType.float32,
        DType.float32,
        b_packed,
        mixed_kernels,
    ]()


fn bmm_naive(
    c: NDBuffer[mut=True, *_],
    a: NDBuffer,
    b: NDBuffer,
    batches: Int,
    m: Int,
    n: Int,
    k: Int,
    val: Int = 0,
    transpose_b: Bool = False,
):
    for batch in range(batches):
        for i in range(m):
            for p in range(k):
                for j in range(n):
                    var a_val = a[batch, i, p].cast[c.type]()
                    var b_val = b[
                        IndexList[b.rank](
                            batch, j, p
                        ) if transpose_b else IndexList[b.rank](batch, p, j)
                    ].cast[c.type]()
                    c[batch, i, j] += a_val * b_val

    for batch in range(batches):
        for i in range(m):
            for j in range(n):
                c[batch, i, j] += val


def test_batched_matmul[
    has_lambda: Bool
](
    c: NDBuffer[mut=True, _, 3],
    a: NDBuffer[mut=True, _, 3],
    b: NDBuffer[mut=True, _, 3],
    batches: Int,
    m: Int,
    n: Int,
    k: Int,
):
    var golden_ptr = UnsafePointer[Scalar[c.type], alignment=alignment].alloc(
        batches * m * n
    )
    var golden = NDBuffer[c.type, 3](golden_ptr, Index(batches, m, n))

    for batch in range(batches):
        for i in range(m):
            for j in range(k):
                a[batch, i, j] = (i + j) * Scalar[a.type](0.001)

    for batch in range(batches):
        for i in range(k):
            for j in range(n):
                b[batch, i, j] = (i + k) * Scalar[b.type](0.001)

    for batch in range(batches):
        for i in range(m):
            for j in range(n):
                c[batch, i, j] = 0
                golden[batch, i, j] = 0

    @parameter
    @always_inline
    @__copy_capture(c)
    fn epilogue_fn[
        _type: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](coords: IndexList[rank], val: SIMD[_type, width],) -> None:
        c.store(
            rebind[IndexList[3]](coords),
            rebind[SIMD[c.type, width]](val + some_constant),
        )

    @always_inline
    @__copy_capture(c, a, b)
    @parameter
    fn bench_fn_batched_matmul() raises:
        @parameter
        if has_lambda:
            batched_matmul[
                transpose_a=False,
                transpose_b=False,
                elementwise_epilogue_fn=epilogue_fn,
            ](c, a, b)
        else:
            batched_matmul[
                transpose_a=False,
                transpose_b=False,
            ](c, a, b)

    bench_fn_batched_matmul()

    @parameter
    if do_benchmarking:
        var batched_matmul_perf = bench_run[bench_fn_batched_matmul]()
        benchmark.keep(c[0, 0, 0])
        print(
            "Apple Batched Matmul GFLOP/s for (BATCHES, M, N, K) = (",
            batches,
            m,
            n,
            k,
            "): ",
            1e-9 * ((2 * batches * m * k * n) / batched_matmul_perf.mean()),
        )

    @parameter
    if has_lambda:
        bmm_naive(golden, a, b, batches, m, n, k, some_constant)
    else:
        bmm_naive(golden, a, b, batches, m, n, k)

    var errors: Int = 0
    for batch in range(batches):
        for i in range(m):
            for j in range(n):
                if c[batch, i, j] != golden[batch, i, j]:
                    if errors < 10:
                        print(
                            c[batch, i, j],
                            golden[batch, i, j],
                            c[batch, i, j] - golden[batch, i, j],
                            "at",
                            batch,
                            i,
                            j,
                        )
                    errors += 1

    assert_true(
        errors == 0,
        String(
            "num of errors must be 0, but got ",
            errors,
            " for dimensions Batch=",
            batches,
            " M=",
            m,
            ", N=",
            n,
            ", K=",
            k,
        ),
    )

    golden_ptr.free()


def test_batched_matmul(batch: Int, m: Int, n: Int, k: Int):
    alias c_type = DType.float32
    alias a_type = DType.float32
    alias b_type = DType.float32

    var c_ptr = UnsafePointer[Scalar[c_type], alignment=alignment].alloc(
        batch * m * n
    )
    var a_ptr = UnsafePointer[Scalar[a_type], alignment=alignment].alloc(
        batch * m * k
    )
    var b_ptr = UnsafePointer[Scalar[b_type], alignment=alignment].alloc(
        batch * k * n
    )

    var c = NDBuffer[c_type, 3](c_ptr, Index(batch, m, n))
    var a = NDBuffer[a_type, 3](a_ptr, Index(batch, m, k))
    var b = NDBuffer[b_type, 3](b_ptr, Index(batch, k, n))

    test_batched_matmul[False](c, a, b, batch, m, n, k)
    test_batched_matmul[True](c, a, b, batch, m, n, k)

    c_ptr.free()
    b_ptr.free()
    a_ptr.free()


def test_batched_matmul():
    for batch in [1, 2, 4, 9, 12]:
        test_batched_matmul(batch, 256, 1024, 4096)
        test_batched_matmul(batch, 4, 5, 6)
        test_batched_matmul(batch, 15, 16, 17)
        test_batched_matmul(batch, 24, 32, 64)
        test_batched_matmul(batch, 61, 73, 79)
        test_batched_matmul(batch, 123, 456, 321)
        test_batched_matmul(batch, 256, 256, 256)
        test_batched_matmul(batch, 2, 65, 1200)


def main():
    test_types[b_packed=True, mixed_kernels=False]()
    test_types[b_packed=True, mixed_kernels=True]()
    test_types[b_packed=False, mixed_kernels=False]()
    test_types[b_packed=False, mixed_kernels=True]()

    test_batched_matmul()

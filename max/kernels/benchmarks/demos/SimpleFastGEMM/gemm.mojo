# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

# Meant to be run on an AVX512 system

from std.math import align_up
from std.sys import align_of, prefetch, simd_width_of
from std.sys.intrinsics import PrefetchOptions

import std.benchmark
from buffer import NDBuffer
from linalg.utils import (
    get_matmul_kernel_shape,
    get_matmul_prefetch_b_distance_k,
)

from std.utils.index import Index

comptime dtype = DType.float32
comptime simd_size = simd_width_of[dtype]()
comptime alignment = align_of[SIMD[dtype, simd_size]]()

comptime kernel_shape = get_matmul_kernel_shape[dtype, dtype, dtype, False]()
comptime MR = kernel_shape.simd_rows
comptime NR = kernel_shape.simd_cols * simd_size

# AVX512 values
# alias MR = 6
# alias NR = 64

comptime prefetch_distance = get_matmul_prefetch_b_distance_k()


fn print_mat(a_ptr: UnsafePointer[Scalar[dtype], _], m: Int, n: Int):
    var a = NDBuffer[rank=2, dtype](a_ptr, Index(m, n))
    for i in range(m):
        for j in range(n):
            print(a[i, j], end=" ")
        print("")


fn gemm_naive(
    a: NDBuffer[rank=2, dtype],
    b: NDBuffer[rank=2, dtype],
    c: NDBuffer[rank=2, dtype],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                c[i, j] += a[i, p] * b[p, j]


fn kernel(
    a_ptr: UnsafePointer[Scalar[dtype], _],
    b_ptr: UnsafePointer[Scalar[dtype], _],
    c_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
    n: Int,
    k: Int,
    kc: Int,
):
    var a = NDBuffer[rank=1, dtype](a_ptr, MR * k)
    var b = NDBuffer[rank=1, dtype](b_ptr, k * NR)
    var c = NDBuffer[rank=1, dtype](c_ptr, MR * n)

    var c_local = NDBuffer[
        rank=1, dtype, MutAnyOrigin, MR * NR
    ]().stack_allocation[alignment=alignment]()

    comptime NR2 = NR // simd_size

    comptime for idx0 in range(MR):
        for idx1 in range(NR2):
            var cv = c.load[width=simd_size](n * idx0 + simd_size * idx1)
            c_local.store(NR * idx0 + simd_size * idx1, cv)

    for pr in range(kc):
        comptime for i in range(NR2):
            prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](b_ptr + NR * pr + simd_size * (i + 16))

        comptime for idx0 in range(MR):
            for idx1 in range(NR2):
                var av = a[idx0 * k + pr].cast[dtype]()
                var bv = b.load[width=simd_size](NR * pr + simd_size * idx1)
                var cv = c_local.load[width=simd_size](
                    NR * idx0 + simd_size * idx1
                )
                cv += av * bv
                c_local.store(NR * idx0 + simd_size * idx1, cv)

    comptime for idx0 in range(MR):
        for idx1 in range(NR2):
            var cv = c_local.load[width=simd_size](NR * idx0 + simd_size * idx1)
            c.store(n * idx0 + simd_size * idx1, cv)


fn pack_B(
    b_ptr: UnsafePointer[Scalar[dtype], _],
    b2_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
    k: Int,
    n: Int,
    kc: Int,
    nc: Int,
):
    var b = NDBuffer[rank=1, dtype](b_ptr, k * n)
    var bc = NDBuffer[rank=1, dtype](b2_ptr, k * n)
    for pr in range(kc):
        for ir in range(nc // NR):
            for v in range(NR):
                bc[NR * (pr + kc * ir) + v] = b[pr * n + NR * ir + v]


fn prepack_B(
    b_ptr: UnsafePointer[Scalar[dtype], _],
    b2_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
    k: Int,
    n: Int,
    kc: Int,
    nc: Int,
):
    for pc in range(0, k, kc):
        for jc in range(0, n, nc):
            pack_B(b_ptr + pc * n + jc, b2_ptr + n * pc + jc * kc, k, n, kc, nc)


fn gemm(
    a_ptr: UnsafePointer[Scalar[dtype], _],
    b_ptr: UnsafePointer[Scalar[dtype], _],
    c_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
    m: Int,
    n: Int,
    k: Int,
    mc: Int,
    nc: Int,
    kc: Int,
):
    for ic in range(0, m, mc):
        for pc in range(0, k, kc):
            for jc in range(0, n, nc):
                for ir in range(0, mc, MR):
                    for jr in range(0, nc, NR):
                        kernel(
                            a_ptr + (ic + ir) * k + pc,
                            b_ptr + n * pc + jc * kc + jr * kc,
                            c_ptr + (ic + ir) * n + jc + jr,
                            n,
                            k,
                            kc,
                        )


def main() raises:
    var m = align_up(1024, MR)
    var n = align_up(1024, NR)
    var k: Int = 1024
    var mc: Int = m
    var nc: Int = NR
    var kc: Int = k
    if m % MR != 0:
        print("m must be multiple of 6")
        return
    if n % NR != 0:
        print("n must be a multiple of 64")
        return

    print(m, end="")
    print("x", end="")
    print(n, end="")
    print("x", end="")
    print(k)

    var a_ptr = alloc[Scalar[dtype]](m * k, alignment=alignment)
    var b_ptr = alloc[Scalar[dtype]](k * n, alignment=alignment)
    var b2_ptr = alloc[Scalar[dtype]](k * n, alignment=alignment)
    var c_ptr = alloc[Scalar[dtype]](m * n, alignment=alignment)
    var c2_ptr = alloc[Scalar[dtype]](m * n, alignment=alignment)
    var a = NDBuffer[rank=1, dtype](a_ptr, m * k)
    var b = NDBuffer[rank=1, dtype](b_ptr, k * n)
    var b2 = NDBuffer[rank=1, dtype](b2_ptr, k * n)
    var c = NDBuffer[rank=1, dtype](c_ptr, m * n)
    var c2 = NDBuffer[rank=1, dtype](c2_ptr, m * n)

    var am = NDBuffer[rank=2, dtype](a_ptr, Index(m, k))
    var bm = NDBuffer[rank=2, dtype](b_ptr, Index(k, n))
    var cm = NDBuffer[rank=2, dtype](c_ptr, Index(m, n))

    for i in range(m * k):
        a[i] = i
    for i in range(k * n):
        b[i] = i
        b2[i] = i
    for i in range(m * n):
        c[i] = i
        c2[i] = i

    prepack_B(b.data, b2.data, k, n, kc, nc)

    gemm_naive(am, bm, cm, m, n, k)
    gemm(a.data, b2.data, c2.data, m, n, k, mc, nc, kc)
    var errors: Int = 0
    for i in range(m * n):
        if c[i] != c2[i]:
            errors += 1
    print(errors, end="")
    print("/", end="")
    print(m * n, end="")
    print(" errors")

    @parameter
    fn bench_gemm():
        gemm(a.data, b2.data, c2.data, m, n, k, mc, nc, kc)

    var num_warmup: Int = 1
    var time = std.benchmark.run[func3=bench_gemm](num_warmup).mean()
    var flops = 2.0 * m * n * k / time / 1e9
    print(time, end="")
    print(" seconds")
    print(flops, end="")
    print(" GFLOPS")

    # assume turbo is disabled and the frequency set to 2.9 GHz
    var rpeak = flops / (2.9 * 64)
    print(rpeak, end="")
    print(" measured/peak FLOPS assuming 2.9 GHz")

    a_ptr.free()
    b_ptr.free()
    b2_ptr.free()
    c_ptr.free()
    c2_ptr.free()

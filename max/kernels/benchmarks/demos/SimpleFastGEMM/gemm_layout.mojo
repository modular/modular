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

# Meant to be run on an AVX512 system

from math import align_up
from sys import alignof, simdwidthof

import benchmark
from buffer import NDBuffer
from layout import *
from linalg.utils import (
    get_matmul_kernel_shape,
    get_matmul_prefetch_b_distance_k,
)

alias MR = 6
alias NR = 64

alias dtype = DType.float32
alias simd_size = simdwidthof[dtype]()
alias alignment = alignof[SIMD[dtype, simd_size]]()


fn gemm_naive[
    layout_b: Layout, origin: Origin
](
    c: NDBuffer[dtype, 2],  # M x N
    a: NDBuffer[dtype, 2],  # M x K
    b: LayoutTensor[dtype, layout_b, MutableAnyOrigin],  # N x K
):
    var M = c.dim(0)
    var N = b.dim(1)
    var K = b.dim(0)

    for mm in range(M):
        for kk in range(K):
            for nn in range(N):
                c[(mm, nn)] += a[mm, kk] * b[kk, nn]


fn kernel[
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
](
    c: LayoutTensor[dtype, layout_c],  # MR, NR
    a: LayoutTensor[dtype, layout_a],  # MR, K
    b_packed: LayoutTensor[dtype, layout_b],  # 1, K * NR
):
    var K = a.dim(1)

    var c_cache = TensorBuilder[MR, NR, dtype].OnStackAligned[alignment]()

    @parameter
    for m in range(MR):
        c_cache.store[NR](m, 0, c.load[NR](m, 0))

    for pr in range(K // NR):
        var a_tile = a.tile[MR, NR](0, pr)
        var b_row = b_packed.tile[1, NR * NR](0, pr)

        for k in range(NR):
            var b_next_tile = b_row.tile[1, NR](0, k + 4)

            @parameter
            for n in range(0, NR, simd_size):
                b_next_tile.prefetch(0, n)

            var b_tile = b_row.tile[1, NR](0, k)

            @parameter
            for m in range(MR):
                var av = a_tile[m, k]

                c_cache.store[NR](
                    m, 0, av * b_tile.load[NR](0, 0) + c_cache.load[NR](m, 0)
                )

    @parameter
    for m in range(MR):
        c.store[NR](m, 0, c_cache.load[NR](m, 0))


fn pack_b[
    layout_b: Layout,
    layout_packed: Layout,
](
    b: LayoutTensor[layout_b, dtype],  # K x N
    packed: LayoutTensor[layout_packed, dtype],  # N // NR x K * NR
):
    alias K = b.dim[0]()
    alias N = b.dim[1]()

    for jc in range(N // NR):
        for pr in range(K // NR):
            var b_tile = b.tile[NR, NR](pr, jc)
            var packed_row = packed.tile[1, NR * NR](jc, pr)

            for k in range(NR):
                var packed_tile = packed_row.tile[1, NR](0, k)
                for n in range(NR):
                    packed_tile[0, n] = b_tile[k, n]


fn gemm[
    N: Int,
    K: Int,
    layout_b: Layout,
](
    c: NDBuffer[dtype, 2],  # M x N
    a: NDBuffer[dtype, 2],  # M x K
    b_packed: LayoutTensor[layout_b, dtype],  # (N // NR) x (K * NR)
):
    var M = c.dim(0)

    for jc in range(N // NR):
        var b_tile = b_packed.tile[1, K * NR](jc, 0)

        # @parameter
        # fn process_row(ir: Int):
        for ir in range(M // MR):
            var a_tile = TensorBuilder[MR, K, dtype].Wrap(
                a.data.offset(K * MR * ir)
            )

            # var c_strip = TensorBuilder[MR, N, dtype].Wrap(
            #     c.data.offset(N * MR * ir)
            # )
            # var c_tile = c_strip.tile[MR, NR](0, jc)

            # Possibly a slightly more efficient way of building c_tile
            alias c_tile_layout = Layout(IntTuple(MR, NR), IntTuple(N, 1))
            var c_tile = LayoutTensor[c_tile_layout, dtype](
                c.data.offset(N * MR * ir + NR * jc)
            )

            kernel(c_tile, a_tile, b_tile)

        # sync_parallelize[process_row](M // MR)


# kgen --emit-asm open-source/max/max/kernels/benchmarks/demos/SimpleFastGEMM/gemm_layout.mojo >out.S
@export(ABI="C")
fn gemm_export_dynamic(
    a_ptr: UnsafePointer[Scalar[dtype]],
    b_packed_ptr: UnsafePointer[Scalar[dtype]],
    c_ptr: UnsafePointer[Scalar[dtype]],
    M: Int,
):
    alias N = 1024
    alias K = 1024
    var a = NDBuffer[dtype, 2](a_ptr, (M, N))
    var b_packed = TensorBuilder[N // NR, K * NR, dtype].Wrap(b_packed_ptr)
    var c = NDBuffer[dtype, 2](c_ptr, (M, N))
    gemm[N, K](c, a, b_packed)


fn main():
    alias M = align_up(1024, MR)
    alias N = align_up(1024, NR)
    alias K: Int = 1024

    if M % MR != 0:
        print("M must be multiple of", MR)
        return
    if N % NR != 0:
        print("N must be a multiple of", NR)
        return

    print(M, end="")
    print("x", end="")
    print(N, end="")
    print("x", end="")
    print(K)

    # FIXME: Something causes sporadic crashes on intel with TensorBuilder.Build()
    var a_ptr = UnsafePointer[Float32, alignment=alignment].alloc(M * K)
    var b_ptr = UnsafePointer[Float32, alignment=alignment].alloc(K * N)
    var b_packed_ptr = UnsafePointer[Float32, alignment=alignment].alloc(K * N)
    var c_ptr = UnsafePointer[Float32, alignment=alignment].alloc(M * N)
    var c2_ptr = UnsafePointer[Float32, alignment=alignment].alloc(M * N)

    var a = NDBuffer[dtype, 2](a_ptr, (M, K))

    var b = TensorBuilder[K, N, dtype].Wrap(b_ptr)
    var b_packed = TensorBuilder[N // NR, K * NR, dtype].Wrap(b_packed_ptr)

    var c = NDBuffer[dtype, 2](c_ptr, (M, N))
    var c2 = NDBuffer[dtype, 2](c2_ptr, (M, N))

    for j in range(M):
        for i in range(K):
            a[(j, i)] = K * j + i

    for j in range(K):
        for i in range(N):
            b[j, i] = N * j + i

    for j in range(M):
        for i in range(N):
            c[(j, i)] = c2[(j, i)] = 0

    pack_b(b, b_packed)

    gemm_naive(c, a, b)
    gemm[N, K](c2, a, b_packed)
    var errors: Int = 0
    for j in range(M):
        for i in range(N):
            if c[j, i] != c2[j, i]:
                errors += 1

    print(errors)
    print("/", end="")
    print(M * N, end="")
    print(" errors")

    @parameter
    fn bench_gemm():
        gemm[N, K](c2, a, b_packed)

    var num_warmup: Int = 1
    var time = benchmark.run[bench_gemm](num_warmup).mean()
    var flops = 2.0 * M * N * K / time / 1e9
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
    b_packed_ptr.free()
    c_ptr.free()
    c2_ptr.free()

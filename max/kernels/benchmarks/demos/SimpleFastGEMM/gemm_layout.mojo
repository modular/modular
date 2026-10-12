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
from std.memory import Layout as AllocLayout, alloc, dealloc
from std.sys import align_of, simd_width_of

import std.benchmark
from layout import Coord, Idx, TileTensor, row_major, stack_allocation
from layout.tensor_engine import DefaultEngine

comptime MR = 6
comptime NR = 64

comptime dtype = DType.float32
comptime simd_size = simd_width_of[dtype]()
comptime alignment = align_of[SIMD[dtype, simd_size]]()


def gemm_naive(
    c: TileTensor[
        mut=True, dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # M x N
    a: TileTensor[
        mut=False, dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # M x K
    b: TileTensor[
        mut=False, dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # K x N
):
    comptime assert c.rank == c.flat_rank == 2
    comptime assert a.rank == a.flat_rank == 2
    comptime assert b.rank == b.flat_rank == 2
    var M = Int(c.dim[0]())
    var N = Int(b.dim[1]())
    var K = Int(b.dim[0]())

    for mm in range(M):
        for kk in range(K):
            for nn in range(N):
                c[mm, nn] += a[mm, kk] * b[kk, nn]


def kernel(
    c: TileTensor[
        mut=True, dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # MR, NR
    a: TileTensor[
        mut=False, dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # MR, K
    b_packed: TileTensor[
        mut=False, dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # 1, K * NR
):
    comptime assert c.rank == c.flat_rank == 2
    comptime assert a.rank == a.flat_rank == 2
    comptime assert b_packed.rank == b_packed.flat_rank == 2
    var K = Int(a.dim[1]())

    var c_cache = stack_allocation[dtype, alignment=alignment](
        row_major[MR, NR]()
    )

    comptime for m in range(MR):
        c_cache.store[alignment=alignment]((m, 0), c.load[width=NR]((m, 0)))

    for pr in range(K // NR):
        var a_tile = a.tile[MR, NR](0, pr)
        var b_row = b_packed.tile[1, NR * NR](0, pr)

        for k in range(NR):
            if pr * NR + k + 4 < K:
                var b_next_tile = b_packed.tile[1, NR](0, pr * NR + k + 4)
                comptime for n in range(0, NR, simd_size):
                    b_next_tile.prefetch(Coord(0, n))

            var b_tile = b_row.tile[1, NR](0, k)

            comptime for m in range(MR):
                var av = a_tile[m, k]

                c_cache.store[alignment=alignment](
                    (m, 0),
                    av * b_tile.load[width=NR]((0, 0))
                    + c_cache.load[width=NR, alignment=alignment]((m, 0)),
                )

    comptime for m in range(MR):
        c.store((m, 0), c_cache.load[width=NR, alignment=alignment]((m, 0)))


def pack_b(
    b: TileTensor[
        mut=False, dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # K x N
    packed: TileTensor[
        mut=True, dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # N // NR x K * NR
):
    comptime assert b.rank == b.flat_rank == 2
    comptime assert packed.rank == packed.flat_rank == 2
    comptime K = b.static_shape[0]
    comptime N = b.static_shape[1]
    comptime assert K >= 0 and N >= 0, "packing requires static dimensions"

    for jc in range(N // NR):
        for pr in range(K // NR):
            var b_tile = b.tile[NR, NR](pr, jc)
            var packed_row = packed.tile[1, NR * NR](jc, pr)

            for k in range(NR):
                var packed_tile = packed_row.tile[1, NR](0, k)
                for n in range(NR):
                    packed_tile[0, n] = b_tile[k, n]


def gemm[
    N: Int, K: Int
](
    c: TileTensor[
        mut=True, dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # M x N
    a: TileTensor[
        mut=False, dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # M x K
    b_packed: TileTensor[
        mut=False, dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # (N // NR) x (K * NR)
):
    comptime assert c.rank == c.flat_rank == 2
    comptime assert a.rank == a.flat_rank == 2
    comptime assert b_packed.rank == b_packed.flat_rank == 2
    var M = Int(c.dim[0]())

    for jc in range(N // NR):
        var b_tile = b_packed.tile[1, K * NR](jc, 0)

        for ir in range(M // MR):
            var a_tile = a.tile[MR, K](ir, 0)
            var c_tile = c.tile[MR, NR](ir, jc)

            kernel(c_tile, a_tile, b_tile)


# kgen --emit=asm max/kernels/benchmarks/demos/SimpleFastGEMM/gemm_layout.mojo >out.S
@export
def gemm_export_dynamic(
    a_ptr: ImmPointer[Scalar[dtype], _],
    b_packed_ptr: ImmPointer[Scalar[dtype], _],
    c_ptr: MutPointer[Scalar[dtype], _],
    M: Int,
) abi("C"):
    comptime N = 1024
    comptime K = 1024
    var a = TileTensor(a_ptr, row_major(M, Idx[K]))
    var b_packed = TileTensor(b_packed_ptr, row_major[N // NR, K * NR]())
    var c = TileTensor(c_ptr, row_major(M, Idx[N]))
    gemm[N, K](c, a, b_packed)


def main() raises:
    comptime M = align_up(1024, MR)
    comptime N = align_up(1024, NR)
    comptime K: Int = 1024

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

    var a_alloc = alloc(
        AllocLayout[Float32, alignment=.of_bytes[alignment]()](count=M * K)
    ).into_managed()
    var b_alloc = alloc(
        AllocLayout[Float32, alignment=.of_bytes[alignment]()](count=K * N)
    ).into_managed()
    var b_packed_alloc = alloc(
        AllocLayout[Float32, alignment=.of_bytes[alignment]()](count=K * N)
    ).into_managed()
    var c_alloc = alloc(
        AllocLayout[Float32, alignment=.of_bytes[alignment]()](count=M * N)
    ).into_managed()
    var c2_alloc = alloc(
        AllocLayout[Float32, alignment=.of_bytes[alignment]()](count=M * N)
    ).into_managed()

    var a = TileTensor(a_alloc.unsafe_ptr(), row_major[M, K]())

    var b = TileTensor(b_alloc.unsafe_ptr(), row_major[K, N]())
    var b_packed = TileTensor(
        b_packed_alloc.unsafe_ptr(), row_major[N // NR, K * NR]()
    )

    var c = TileTensor(c_alloc.unsafe_ptr(), row_major[M, N]())
    var c2 = TileTensor(c2_alloc.unsafe_ptr(), row_major[M, N]())

    for j in range(M):
        for i in range(K):
            a[j, i] = Scalar[dtype](K * j + i)

    for j in range(K):
        for i in range(N):
            b[j, i] = Scalar[dtype](N * j + i)

    for j in range(M):
        for i in range(N):
            c[j, i] = 0
            c2[j, i] = 0

    pack_b(b.as_imm(), b_packed)

    gemm_naive(c, a.as_imm(), b.as_imm())
    gemm[N, K](c2, a.as_imm(), b_packed.as_imm())
    var errors: Int = 0
    for j in range(M):
        for i in range(N):
            if c[j, i] != c2[j, i]:
                errors += 1

    print(errors)
    print("/", end="")
    print(M * N, end="")
    print(" errors")

    def bench_gemm() {var}:
        gemm[N, K](c2, a.as_imm(), b_packed.as_imm())

    var num_warmup: Int = 1
    var time = std.benchmark.run(bench_gemm, num_warmup).mean()
    var flops = Float64(2 * M * N * K) / time / 1e9
    print(time, end="")
    print(" seconds")
    print(flops, end="")
    print(" GFLOPS")

    # assume turbo is disabled and the frequency set to 2.9 GHz
    var rpeak = flops / (2.9 * 64)
    print(rpeak, end="")
    print(" measured/peak FLOPS assuming 2.9 GHz")

    dealloc(a_alloc^)
    dealloc(b_alloc^)
    dealloc(b_packed_alloc^)
    dealloc(c_alloc^)
    dealloc(c2_alloc^)

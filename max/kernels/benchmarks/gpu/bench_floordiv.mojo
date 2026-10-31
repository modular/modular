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
"""Benchmarks signed floor division by a compile-time constant on the GPU.

The GPU counterpart of `Mojo/stdlib/benchmarks/math/bench_floordiv.mojo`. Each
thread runs `CHAINS` independent dependent chains of `r = (x ^ r) // d`, the
latency loop of `div_bench.cpp` from https://github.com/benjoffe/fastdiv, with
enough of them in flight to make the division the bottleneck. It compares
`x // d`, which lowers to Ben Joffe's multiply-high sequence, against the
truncating division plus sign fixup that `pop.floordiv` lowered to before, and
checks the two agree on every output.

```
bazel run //max/kernels/benchmarks:gpu/bench_floordiv
```
"""

from std.random import random_ui64, seed
from std.sys import bit_width_of, size_of

from max.benchmark import bencher_iter_custom
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.builtin._closure import __ownership_keepalive
from internal_utils import update_bench_config_args

comptime CHAINS = 4
comptime ITERS = 256
comptime BLOCK_DIM = 256
comptime GRID_DIM = 1024
comptime THREADS = BLOCK_DIM * GRID_DIM


@inline(.always)
def floordiv_generic[
    dtype: DType, width: Int, d: Scalar[dtype]
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime divisor = SIMD[dtype, width](d)
    var q = x / divisor
    var sign = (x ^ divisor) >> SIMD[dtype, width](bit_width_of[dtype]() - 1)
    return q + (q * divisor).eq(x).select(SIMD[dtype, width](0), sign)


@inline(.always)
def divide[
    dtype: DType, width: Int, d: Scalar[dtype], benjoffe: Bool
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime if benjoffe:
        return x // SIMD[dtype, width](d)
    else:
        return floordiv_generic[dtype, width, d](x)


def floordiv_kernel[
    dtype: DType, d: Scalar[dtype], benjoffe: Bool
](
    dst: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    src: ImmPointer[Scalar[dtype], ImmUntrackedOrigin],
):
    var offset = global_idx.x * CHAINS
    var x = src.unsafe_load[width=CHAINS](offset)
    var r = SIMD[dtype, CHAINS](0)
    for _ in range(ITERS):
        r = divide[dtype, CHAINS, d, benjoffe](x ^ r)
    dst.unsafe_store(offset, r)


@inline(.always)
def split[
    idx_dtype: DType, d: Int, benjoffe: Bool
](x: Scalar[idx_dtype]) -> Tuple[Scalar[idx_dtype], Scalar[idx_dtype]]:
    var q = divide[idx_dtype, 1, Scalar[idx_dtype](d), benjoffe](x)
    return q, x - q * Scalar[idx_dtype](d)


# An elementwise kernel whose index math divides by the compile-time shape:
# a copy of an (A, B, C, D) row-major tensor into its (D, C, B, A) transpose.
# The permutation keeps the compiler from folding the coordinates back into the
# linear index, as it can for a same-layout elementwise op.
def permute_kernel[
    dtype: DType,
    idx_dtype: DType,
    A: Int,
    B: Int,
    C: Int,
    D: Int,
    benjoffe: Bool,
](
    dst: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    src: ImmPointer[Scalar[dtype], ImmUntrackedOrigin],
):
    comptime I = Scalar[idx_dtype]
    var stride = I(GRID_DIM * BLOCK_DIM)
    var i = I(global_idx.x)
    while i < I(A * B * C * D):
        var abc, d = split[idx_dtype, D, benjoffe](i)
        var ab, c = split[idx_dtype, C, benjoffe](abc)
        var a, b = split[idx_dtype, B, benjoffe](ab)
        var out = ((d * I(C) + c) * I(B) + b) * I(A) + a
        dst[unsafe_offset=Int(out)] = src[unsafe_offset=Int(i)]
        i += stride


@inline(.never)
def bench_permute[
    idx_dtype: DType, A: Int, B: Int, C: Int, D: Int
](mut b: Bench, ctx: DeviceContext) raises:
    comptime dtype = DType.bfloat16
    comptime count = A * B * C * D
    var src_host = ctx.enqueue_create_host_buffer[dtype](count)
    var out_host = ctx.enqueue_create_host_buffer[dtype](count)
    var ref_host = ctx.enqueue_create_host_buffer[dtype](count)
    ctx.synchronize()
    for i in range(count):
        src_host[i] = Scalar[dtype](i % 251)
    var src = ctx.enqueue_create_buffer[dtype](count)
    var dst = ctx.enqueue_create_buffer[dtype](count)
    ctx.enqueue_copy(src, src_host)

    comptime generic_kernel = permute_kernel[
        dtype, idx_dtype, A, B, C, D, False
    ]
    comptime benjoffe_kernel = permute_kernel[
        dtype, idx_dtype, A, B, C, D, True
    ]

    def launch_generic(c: DeviceContext) raises {imm src, imm dst}:
        c.enqueue_function[generic_kernel](
            dst, src, grid_dim=GRID_DIM, block_dim=BLOCK_DIM
        )

    def launch_benjoffe(c: DeviceContext) raises {imm src, imm dst}:
        c.enqueue_function[benjoffe_kernel](
            dst, src, grid_dim=GRID_DIM, block_dim=BLOCK_DIM
        )

    launch_generic(ctx)
    ctx.enqueue_copy(ref_host, dst)
    launch_benjoffe(ctx)
    ctx.enqueue_copy(out_host, dst)
    ctx.synchronize()
    for i in range(count):
        if out_host[i] != ref_host[i]:
            raise Error(
                t"permute mismatch at {i}: got {out_host[i]},"
                t" want {ref_host[i]}"
            )

    var input_id = String(idx_dtype, "/", A, "x", B, "x", C, "x", D)
    var measures: List[ThroughputMeasure] = [
        ThroughputMeasure(BenchMetric.bytes, 2 * count * size_of[dtype]())
    ]

    def bench_generic(
        mut bencher: Bencher,
    ) raises {imm launch_generic, imm ctx}:
        bencher_iter_custom(bencher, launch_generic, ctx)

    def bench_benjoffe(
        mut bencher: Bencher,
    ) raises {imm launch_benjoffe, imm ctx}:
        bencher_iter_custom(bencher, launch_benjoffe, ctx)

    b.bench_function(
        bench_generic, BenchId("permute_generic", input_id), measures.copy()
    )
    b.bench_function(
        bench_benjoffe, BenchId("permute_benjoffe", input_id), measures.copy()
    )

    ctx.synchronize()
    __ownership_keepalive(src, dst)
    _ = src_host^
    _ = out_host^
    _ = ref_host^


@inline(.never)
def bench_divisor[
    dtype: DType, d: Scalar[dtype]
](mut b: Bench, ctx: DeviceContext) raises:
    comptime count = THREADS * CHAINS
    var src_host = ctx.enqueue_create_host_buffer[dtype](count)
    var out_host = ctx.enqueue_create_host_buffer[dtype](count)
    var ref_host = ctx.enqueue_create_host_buffer[dtype](count)
    ctx.synchronize()
    seed(0xDEADBEEF)
    for i in range(count):
        src_host[i] = random_ui64(0, UInt64.MAX).cast[dtype]()
    var src = ctx.enqueue_create_buffer[dtype](count)
    var dst = ctx.enqueue_create_buffer[dtype](count)
    ctx.enqueue_copy(src, src_host)

    comptime generic_kernel = floordiv_kernel[dtype, d, False]
    comptime benjoffe_kernel = floordiv_kernel[dtype, d, True]

    def launch_generic(c: DeviceContext) raises {imm src, imm dst}:
        c.enqueue_function[generic_kernel](
            dst, src, grid_dim=GRID_DIM, block_dim=BLOCK_DIM
        )

    def launch_benjoffe(c: DeviceContext) raises {imm src, imm dst}:
        c.enqueue_function[benjoffe_kernel](
            dst, src, grid_dim=GRID_DIM, block_dim=BLOCK_DIM
        )

    launch_generic(ctx)
    ctx.enqueue_copy(ref_host, dst)
    launch_benjoffe(ctx)
    ctx.enqueue_copy(out_host, dst)
    ctx.synchronize()
    for i in range(count):
        if out_host[i] != ref_host[i]:
            raise Error(
                t"floordiv mismatch for d={d} at {i}: got {out_host[i]},"
                t" want {ref_host[i]}"
            )

    var input_id = String(dtype, "/d", d)
    var measures: List[ThroughputMeasure] = [
        ThroughputMeasure(BenchMetric.elements, count * ITERS)
    ]

    def bench_generic(
        mut bencher: Bencher,
    ) raises {imm launch_generic, imm ctx}:
        bencher_iter_custom(bencher, launch_generic, ctx)

    def bench_benjoffe(
        mut bencher: Bencher,
    ) raises {imm launch_benjoffe, imm ctx}:
        bencher_iter_custom(bencher, launch_benjoffe, ctx)

    b.bench_function(
        bench_generic, BenchId("floordiv_generic", input_id), measures.copy()
    )
    b.bench_function(
        bench_benjoffe, BenchId("floordiv_benjoffe", input_id), measures.copy()
    )

    ctx.synchronize()
    __ownership_keepalive(src, dst)
    _ = src_host^
    _ = out_host^
    _ = ref_host^


def bench_dtype[dtype: DType](mut b: Bench, ctx: DeviceContext) raises:
    comptime for d in [3, 5, 7, 10, 13, 100, 86400, -3, -7, -10, -86400]:
        bench_divisor[dtype, Scalar[dtype](d)](b, ctx)


def main() raises:
    var b = Bench()
    update_bench_config_args(b)
    with DeviceContext() as ctx:
        bench_dtype[DType.int64](b, ctx)
        bench_dtype[DType.int32](b, ctx)
        # The shapes bench_elementwise.yaml uses, without their leading 1.
        comptime for idx_dtype in [DType.int64, DType.int32]:
            bench_permute[idx_dtype, 8, 3, 1025, 128](b, ctx)
            bench_permute[idx_dtype, 1, 1, 1024, 3072](b, ctx)
    b.dump_report()

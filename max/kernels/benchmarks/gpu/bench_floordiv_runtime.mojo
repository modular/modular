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
"""Benchmarks signed floor division by a *runtime* divisor on the GPU.

The GPU counterpart of `bench_floordiv_runtime.mojo` in
`Mojo/stdlib/benchmarks/math`. Each thread runs `CHAINS` dependent chains of
`r = (x ^ r) // d` with `d` passed to the kernel, so `pop.floordiv` takes its
runtime-divisor path. Compiling this file with and without that path measures
the two side by side.

```
./bazelw run //max/kernels/benchmarks:gpu/bench_floordiv_runtime -- 3 7 100
```
"""

from std.random import random_ui64, seed
from std.sys import argv, size_of
from std.time import perf_counter_ns

from max.gpu import global_idx
from max.gpu.host import DeviceContext

comptime CHAINS = 4
comptime ITERS = 256
comptime BLOCK_DIM = 256
comptime GRID_DIM = 1024
comptime THREADS = BLOCK_DIM * GRID_DIM
comptime REPEATS = 20
comptime ROUNDS = 5


def floordiv_kernel[
    dtype: DType
](
    dst: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    src: ImmPointer[Scalar[dtype], ImmUntrackedOrigin],
    divisor: Scalar[dtype],
):
    var offset = global_idx.x * CHAINS
    var x0 = src.unsafe_load(offset + 0)
    var x1 = src.unsafe_load(offset + 1)
    var x2 = src.unsafe_load(offset + 2)
    var x3 = src.unsafe_load(offset + 3)
    var r0 = Scalar[dtype](0)
    var r1 = Scalar[dtype](0)
    var r2 = Scalar[dtype](0)
    var r3 = Scalar[dtype](0)
    for _ in range(ITERS):
        r0 = (x0 ^ r0) // divisor
        r1 = (x1 ^ r1) // divisor
        r2 = (x2 ^ r2) // divisor
        r3 = (x3 ^ r3) // divisor
    dst.unsafe_store(offset + 0, r0)
    dst.unsafe_store(offset + 1, r1)
    dst.unsafe_store(offset + 2, r2)
    dst.unsafe_store(offset + 3, r3)


# The same recurrence in 64 bits on the host, for a sample of the threads.
def reference[
    dtype: DType
](x0: SIMD[dtype, CHAINS], divisor: Scalar[dtype]) -> SIMD[dtype, CHAINS]:
    var wide = SIMD[DType.int64, CHAINS](x0.cast[DType.int64]())
    var dw = SIMD[DType.int64, CHAINS](Int64(divisor))
    var r = SIMD[DType.int64, CHAINS](0)
    for _ in range(ITERS):
        r = (wide ^ r) // dw
    return r.cast[dtype]()


def main() raises:
    var divs = List[Scalar[DType.int32]]()
    for i in range(1, len(argv())):
        divs.append(Int32(Int(argv()[i])))
    if len(divs) == 0:
        divs.append(3)
        divs.append(7)
        divs.append(1024)
        divs.append(-7)

    comptime dtype = DType.int32
    comptime count = THREADS * CHAINS

    with DeviceContext() as ctx:
        var src_host = ctx.enqueue_create_host_buffer[dtype](count)
        var out_host = ctx.enqueue_create_host_buffer[dtype](count)
        ctx.synchronize()
        seed(0xDEADBEEF)
        for i in range(count):
            src_host[i] = random_ui64(0, UInt64.MAX).cast[dtype]()
        var src = ctx.enqueue_create_buffer[dtype](count)
        var dst = ctx.enqueue_create_buffer[dtype](count)
        ctx.enqueue_copy(src, src_host)
        ctx.synchronize()

        # Compile the kernel before timing anything.
        ctx.enqueue_function[floordiv_kernel[dtype]](
            dst, src, divs[0], grid_dim=GRID_DIM, block_dim=BLOCK_DIM
        )
        ctx.synchronize()

        print("d ns_per_div")
        for d in divs:
            var best = -1
            for _ in range(ROUNDS):
                var start = perf_counter_ns()
                for _ in range(REPEATS):
                    ctx.enqueue_function[floordiv_kernel[dtype]](
                        dst, src, d, grid_dim=GRID_DIM, block_dim=BLOCK_DIM
                    )
                ctx.synchronize()
                var elapsed = Int(perf_counter_ns() - start)
                if best < 0 or elapsed < best:
                    best = elapsed
            var elapsed = best

            # Check a sample of the threads against the 64-bit reference.
            ctx.enqueue_copy(out_host, dst)
            ctx.synchronize()
            # One thread in 64, aligned to the CHAINS-wide stores.
            var stride = CHAINS * (THREADS // 64)
            for i in range(0, count - CHAINS, stride):
                var x0 = SIMD[dtype, CHAINS](
                    src_host[i],
                    src_host[i + 1],
                    src_host[i + 2],
                    src_host[i + 3],
                )
                var got = SIMD[dtype, CHAINS](
                    out_host[i],
                    out_host[i + 1],
                    out_host[i + 2],
                    out_host[i + 3],
                )
                var want = reference[dtype](x0, d)
                if got != want:
                    raise Error(t"floordiv mismatch for d={d} at {i}")

            var divisions = REPEATS * THREADS * CHAINS * ITERS
            print(d, Float64(elapsed) / Float64(divisions))
        print("checks passed", size_of[dtype]())

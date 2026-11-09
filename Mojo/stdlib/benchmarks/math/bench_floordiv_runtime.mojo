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
"""Benchmarks signed floor division by a *runtime* divisor.

`x // d` with a runtime `d` takes the generic `pop.floordiv` lowering: a
truncating `sdiv` plus a sign fixup. Compiling this file with and without a
runtime-divisor multiply-high lowering measures the two side by side under one
source file.

The divisors come from the command line so the compiler cannot fold them:

```
mojo build -O3 bench_floordiv_runtime.mojo -o bench
./bench 3 7 100 86400
```

Each row prints the best-of-N nanoseconds per division for independent
(`tput`) and dependent (`lat`) chains, plus a checksum that must match across
compiler builds.
"""

from std.random import random_ui64, seed
from std.sys import argv
from std.sys._assembly import inlined_assembly
from std.time import perf_counter_ns

comptime COUNT = 1 << 16
comptime REPS = 5


# Register-only barriers, like `OPAQUE_IN` and `SINK` in div_bench.cpp.
@inline(.always)
def opaque(x: Int32) -> Int32:
    return inlined_assembly[
        "", Int32, constraints="=r,0", has_side_effect=True
    ](x)


@inline(.always)
def sink(x: Int32):
    inlined_assembly["", NoneType, constraints="r", has_side_effect=True](x)


@inline(.always)
def sink_ret(x: Int32) -> Int32:
    return inlined_assembly[
        "", Int32, constraints="=r,0", has_side_effect=True
    ](x)


# Eight independent chains, so the throughput number is not limited by the
# dependency chain of a single division.
@inline(.never)
def run_throughput(inputs: Span[Int32, _], d: Int32) -> Int32:
    var a0 = Int32(0)
    var a1 = Int32(0)
    var a2 = Int32(0)
    var a3 = Int32(0)
    var a4 = Int32(0)
    var a5 = Int32(0)
    var a6 = Int32(0)
    var a7 = Int32(0)
    # `unsafe_get` skips the bounds check, which otherwise costs more than the
    # division and keeps LLVM from vectorizing the loop.
    var i = 0
    var n = len(inputs)
    while i + 8 <= n:
        a0 ^= opaque(inputs.unsafe_get(i + 0)) // d
        a1 ^= opaque(inputs.unsafe_get(i + 1)) // d
        a2 ^= opaque(inputs.unsafe_get(i + 2)) // d
        a3 ^= opaque(inputs.unsafe_get(i + 3)) // d
        a4 ^= opaque(inputs.unsafe_get(i + 4)) // d
        a5 ^= opaque(inputs.unsafe_get(i + 5)) // d
        a6 ^= opaque(inputs.unsafe_get(i + 6)) // d
        a7 ^= opaque(inputs.unsafe_get(i + 7)) // d
        i += 8
    return a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7


# Same, but with a register barrier per element so LLVM cannot vectorize the
# loop (x86 turns a vectorizable `x // d` into packed double division).
@inline(.never)
def run_throughput_scalar(inputs: Span[Int32, _], d: Int32) -> Int32:
    var acc = Int32(0)
    var i = 0
    var n = len(inputs)
    while i + 8 <= n:
        acc ^= sink_ret(opaque(inputs.unsafe_get(i + 0)) // d)
        acc ^= sink_ret(opaque(inputs.unsafe_get(i + 1)) // d)
        acc ^= sink_ret(opaque(inputs.unsafe_get(i + 2)) // d)
        acc ^= sink_ret(opaque(inputs.unsafe_get(i + 3)) // d)
        acc ^= sink_ret(opaque(inputs.unsafe_get(i + 4)) // d)
        acc ^= sink_ret(opaque(inputs.unsafe_get(i + 5)) // d)
        acc ^= sink_ret(opaque(inputs.unsafe_get(i + 6)) // d)
        acc ^= sink_ret(opaque(inputs.unsafe_get(i + 7)) // d)
        i += 8
    return acc


# One dependent chain: each division feeds the next dividend.
@inline(.never)
def run_latency(inputs: Span[Int32, _], d: Int32) -> Int32:
    var r = Int32(0)
    for i in range(len(inputs)):
        r = (opaque(inputs.unsafe_get(i)) ^ r) // d
        sink(r)
    return r


# Reference floor division in 64 bits, which takes the unchanged generic
# lowering, so a wrong 32-bit lowering shows up here.
@inline(.never)
def floordiv_ref(x: Int32, d: Int32) -> Int32:
    return Int32(Int64(x) // Int64(d))


# Exercises the SIMD path. Note that `SIMD(d)` does not always reach the
# runtime-divisor lowering (see the PR's open issues); the lowering's splat case
# is covered by `pop-to-llvm/simd.mlir`, so a failure here is a real
# miscompile either way, but a pass does not prove the splat path ran.
def check_vector(inputs: List[Int32], d: Int32) raises:
    var dv = SIMD[DType.int32, 8](d)
    var n = len(inputs) - 7
    for i in range(0, n, 8):
        var xv = SIMD[DType.int32, 8](
            inputs[i],
            inputs[i + 1],
            inputs[i + 2],
            inputs[i + 3],
            inputs[i + 4],
            inputs[i + 5],
            inputs[i + 6],
            inputs[i + 7],
        )
        var qv = xv // dv
        for lane in range(8):
            var want = floordiv_ref(inputs[i + lane], d)
            if qv[lane] != want:
                raise Error(
                    t"vector floordiv mismatch: {inputs[i + lane]} // {d}"
                    t" gave {qv[lane]}, want {want}"
                )


# The dependent chain in 64 bits, so the latency loop's result is checked
# against a division the lowering does not touch.
def expected_latency(inputs: List[Int32], d: Int32) -> Int32:
    var r = Int32(0)
    for i in range(len(inputs)):
        r = Int32(Int64(inputs.unsafe_get(i) ^ r) // Int64(d))
    return r


def expected_checksum(inputs: List[Int32], d: Int32) -> Int32:
    var acc = Int32(0)
    var i = 0
    while i + 8 <= len(inputs):
        for lane in range(8):
            acc ^= floordiv_ref(inputs.unsafe_get(i + lane), d)
        i += 8
    return acc


def check(inputs: List[Int32], d: Int32) raises:
    var edges = List[Int32]()
    edges.append(Int32.MIN)
    edges.append(Int32.MIN + 1)
    edges.append(-2)
    edges.append(-1)
    edges.append(0)
    edges.append(1)
    edges.append(2)
    edges.append(Int32.MAX - 1)
    edges.append(Int32.MAX)
    edges.append(d - 1)
    edges.append(d)
    edges.append(d + 1)
    for i in range(len(inputs)):
        edges.append(inputs[i])
    for x in edges:
        var got = x // d
        var want = floordiv_ref(x, d)
        if got != want:
            raise Error(
                t"floordiv mismatch: {x} // {d} gave {got}, want {want}"
            )
    check_vector(inputs, d)


def make_inputs() -> List[Int32]:
    seed(0xDEADBEEF)
    var inputs = List[Int32](capacity=COUNT)
    for _ in range(COUNT):
        inputs.append(Int32(random_ui64(0, UInt64.MAX)))
    return inputs^


def best_throughput(inputs: Span[Int32, _], d: Int32) -> Tuple[Int, Int32]:
    var best = -1
    var checksum = Int32(0)
    for _ in range(REPS):
        var start = perf_counter_ns()
        checksum = run_throughput(inputs, d)
        var elapsed = Int(perf_counter_ns() - start)
        if best < 0 or elapsed < best:
            best = elapsed
    return best, checksum


def best_throughput_scalar(
    inputs: Span[Int32, _], d: Int32
) -> Tuple[Int, Int32]:
    var best = -1
    var checksum = Int32(0)
    for _ in range(REPS):
        var start = perf_counter_ns()
        checksum = run_throughput_scalar(inputs, d)
        var elapsed = Int(perf_counter_ns() - start)
        if best < 0 or elapsed < best:
            best = elapsed
    return best, checksum


def best_latency(inputs: Span[Int32, _], d: Int32) -> Tuple[Int, Int32]:
    var best = -1
    var checksum = Int32(0)
    for _ in range(REPS):
        var start = perf_counter_ns()
        checksum = run_latency(inputs, d)
        var elapsed = Int(perf_counter_ns() - start)
        if best < 0 or elapsed < best:
            best = elapsed
    return best, checksum


def main() raises:
    var divs = List[Int32]()
    var test_mode = False
    for i in range(1, len(argv())):
        var arg = String(argv()[i])
        if arg == "-t":
            # The smoke target runs every benchmark with `-t`: check the
            # lowering and skip the timing.
            test_mode = True
        else:
            divs.append(Int32(Int(arg)))
    if len(divs) == 0:
        # A positive divisor, a power of two, a negative one, and one that
        # needs the multiply rather than the shift.
        divs.append(3)
        divs.append(1024)
        divs.append(-7)

    var inputs = make_inputs()

    if test_mode:
        for d in divs:
            check(inputs, d)
        print("check OK")
        return

    # Bring the core to its full clock before timing anything.
    var start = perf_counter_ns()
    while perf_counter_ns() - start < 500_000_000:
        sink(run_throughput(inputs, divs[0]))

    for d in divs:
        check(inputs, d)
    print("check OK")

    print("d tput_ns scalar_tput_ns lat_ns checksum")
    for d in divs:
        var tput = best_throughput(inputs, d)
        var scalar = best_throughput_scalar(inputs, d)
        var lat = best_latency(inputs, d)
        var want = expected_checksum(inputs, d)
        if tput[1] != want or scalar[1] != want:
            raise Error(t"throughput checksum mismatch for {d}")
        var want_latency = expected_latency(inputs, d)
        if lat[1] != want_latency:
            raise Error(t"latency checksum mismatch for {d}")
        print(
            d,
            Float64(tput[0]) / Float64(COUNT),
            Float64(scalar[0]) / Float64(COUNT),
            Float64(lat[0]) / Float64(COUNT),
            tput[1],
        )

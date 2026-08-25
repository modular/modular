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

import std.time
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep

# Purposefully mixes power-of-two with non-power-of-two values.
comptime SIZES = [5, 8, 33, 64, 173, 256, 2111, 4096, 65536]


struct NonTrivial(Copyable, Defaultable):
    var value: Int

    def __init__(out self):
        self.value = 1

    def __init__(out self, *, copy: Self):
        self.value = copy.value + 1

    def __init__(out self, *, deinit move: Self):
        self.value = move.value + 1


def bench_move[
    T: Copyable & Defaultable & Deinitable, size: Int
](mut b: Bencher) raises:
    def benchmark(var array: Array[T, size]):
        var moved = array^
        keep(moved)

    b._iter_setup(
        setup=lambda () -> Array[T, size]: {fill = T()}, benchmark=benchmark
    )


def bench_copy[
    T: Copyable & Defaultable & Deinitable, size: Int
](mut b: Bencher) raises:
    def benchmark(var array: Array[T, size]):
        var copied = array.copy()
        keep(copied)

    b._iter_setup(
        setup=lambda () -> Array[T, size]: {fill = T()}, benchmark=benchmark
    )


def main() raises:
    var m = Bench(
        BenchConfig(
            min_runtime_secs=0.5, max_runtime_secs=2.0, max_iters=100_000
        )
    )
    comptime for size in SIZES:
        m.bench_function(
            bench_move[Int, size],
            BenchId("array_move/trivial/" + String(size)),
        )
        m.bench_function(
            bench_move[NonTrivial, size],
            BenchId("array_move/nontrivial/" + String(size)),
        )
        m.bench_function(
            bench_copy[Int, size],
            BenchId("array_copy/trivial/" + String(size)),
        )
        m.bench_function(
            bench_copy[NonTrivial, size],
            BenchId("array_copy/nontrivial/" + String(size)),
        )
    m.dump_report()

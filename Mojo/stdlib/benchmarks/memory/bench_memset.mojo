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
"""Benchmarks filling bytes with `unsafe_memset`, `unsafe_memset_zero` and
`Span.fill` on the CPU.

`Span.fill` is a plain store loop that the compiler can turn into a `memset`
call, so it is the reference for the two `unsafe_memset` entry points. The
fill length is a runtime value, so no case is specialized for its size. The
odd sizes leave a tail shorter than one vector.
"""

from std.benchmark import (
    Bench,
    BenchConfig,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    black_box,
    keep,
)
from std.memory import unsafe_memset, unsafe_memset_zero

comptime MAX_SIZE = 1 << 20


@inline(.never)
def fill_memset(buffer: MutSpan[Byte, _]):
    unsafe_memset(buffer.unsafe_ptr(), 0xA5, len(buffer))
    keep(buffer)


@inline(.never)
def fill_memset_zero(buffer: MutSpan[Byte, _]):
    unsafe_memset_zero(buffer.unsafe_ptr(), len(buffer))
    keep(buffer)


@inline(.never)
def fill_span(buffer: MutSpan[Byte, _]):
    buffer.fill(0xA5)
    keep(buffer)


def add_size[size: Int](mut m: Bench, buffer: MutSpan[Byte, _]) raises:
    var span = buffer.unsafe_subspan(offset=0, length=size)
    var measures: List[ThroughputMeasure] = [
        ThroughputMeasure(BenchMetric.bytes, size)
    ]

    def run_memset() {imm span}:
        fill_memset(black_box(span))

    def run_memset_zero() {imm span}:
        fill_memset_zero(black_box(span))

    def run_span_fill() {imm span}:
        fill_span(black_box(span))

    var input_id = String(size)
    m.bench_function(run_memset, BenchId("unsafe_memset", input_id), measures)
    m.bench_function(
        run_memset_zero, BenchId("unsafe_memset_zero", input_id), measures
    )
    m.bench_function(run_span_fill, BenchId("span_fill", input_id), measures)


def main() raises:
    # A fill takes nanoseconds, so without a floor on the runtime a sample is
    # too short for the timer to resolve.
    var m = Bench(BenchConfig(num_repetitions=3, min_runtime_secs=0.1))
    var buffer = List[Byte](length=MAX_SIZE, fill=0)
    comptime for size in [16, 63, 256, 1000, 4096, 1 << 16, MAX_SIZE]:
        add_size[size](m, Span(buffer))
    m.dump_report()

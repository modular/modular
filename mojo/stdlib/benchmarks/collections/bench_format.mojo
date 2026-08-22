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
"""Benchmarks for format string parsing.

Measures the performance of `String.format()` with varying amounts of literal
text between replacement fields, which exercises the SIMD-accelerated brace
scanning in `_find_next_brace()`.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Helpers
# ===-----------------------------------------------------------------------===#
def _make_format_string[
    literal_len: Int, n_fields: Int
]() -> String:
    """Build a format string with `n_fields` replacement fields separated by
    `literal_len` bytes of literal text.
    """
    var fmt = String(capacity_bytes=literal_len * (n_fields + 1) + n_fields * 4)
    var padding = String("x") * literal_len

    comptime for i in range(n_fields):
        fmt.write(padding, "{", i, "}")
    fmt.write(padding)
    return fmt^


# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#
def bench_format_literal[literal_len: Int](mut b: Bencher) raises:
    """Runtime format with `literal_len` bytes of padding between 4 fields."""
    var fmt = _make_format_string[literal_len, 4]()

    @always_inline
    def call_fn() raises {imm}:
        for _ in range(1_000):
            var res = fmt.format("v0", "v1", "v2", "v3")
            keep(res)

    b.iter(call_fn)
    keep(Bool(fmt))


def bench_format_runtime_short(mut b: Bencher) raises:
    """Runtime format with a typical short format string."""
    var fmt = String("Hello, {}! I am {} years old and I like {}.")

    @always_inline
    def call_fn() raises {imm}:
        for _ in range(1_000):
            var res = fmt.format("World", 42, "Mojo")
            keep(res)

    b.iter(call_fn)
    keep(Bool(fmt))


def bench_format_literal_short(mut b: Bencher) raises:
    """Format a short literal format string."""

    @always_inline
    def call_fn() raises {imm}:
        for _ in range(1_000):
            var res = "{} + {} = {}".format(
                black_box(1), black_box(2), black_box(3)
            )
            keep(res)

    b.iter(call_fn)


def bench_format_literal_long(mut b: Bencher) raises:
    """Format a literal format string with a longer literal prefix."""

    @always_inline
    def call_fn() raises {imm}:
        for _ in range(1_000):
            var res = (
                "The quick brown fox jumps over the lazy dog"
                " and the answer is {}"
            ).format(black_box(42))
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    var m = Bench(BenchConfig(num_repetitions=5))

    m.bench_function(
        bench_format_literal[4],
        BenchId("bench_format_literal[4B x 4]"),
    )
    m.bench_function(
        bench_format_literal[64],
        BenchId("bench_format_literal[64B x 4]"),
    )
    m.bench_function(
        bench_format_literal[512],
        BenchId("bench_format_literal[512B x 4]"),
    )
    m.bench_function(
        bench_format_literal[4096],
        BenchId("bench_format_literal[4096B x 4]"),
    )
    m.bench_function(
        bench_format_runtime_short,
        BenchId("bench_format_runtime_short"),
    )
    m.bench_function(
        bench_format_literal_short,
        BenchId("bench_format_literal_short"),
    )
    m.bench_function(
        bench_format_literal_long,
        BenchId("bench_format_literal_long"),
    )

    m.dump_report()

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
"""Benchmark `PythonObject(string)` conversion.

Measures the per-call cost of converting a Mojo `String` to a
`PythonObject`. The fast path SIMD-scans for ASCII and calls
`PyUnicode_FromKindAndData(kind=1, ...)` directly, skipping CPython's
UTF-8 decoder (validation, error callback setup) that
`PyUnicode_DecodeUTF8` runs.
"""

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    black_box,
    keep,
)
from std.os import abort
from std.python import Python, PythonObject


@parameter
def bench_string_to_pyobject_ascii_short(mut b: Bencher) raises:
    """`PythonObject(string)` on a short pure-ASCII input (fast path)."""
    _ = Python()
    var s = String("hello world")

    @always_inline
    def call_fn() {read}:
        try:
            for _ in range(1000):
                var x = PythonObject(black_box(s.as_string_slice()))
                keep(x)
        except e:
            abort(String(e))

    b.iter(call_fn)


@parameter
def bench_string_to_pyobject_ascii_large(mut b: Bencher) raises:
    """`PythonObject(string)` on a ~1 KiB pure-ASCII input (fast path).

    Exercises the vectorized SIMD loop in `_is_ascii` and the
    `PyUnicode_FromKindAndData` memcpy at a length where the scan is
    no longer dominated by entry/exit overhead.
    """
    _ = Python()
    var s = String("0123456789abcdef" * 64)  # 1024 bytes

    @always_inline
    def call_fn() {read}:
        try:
            for _ in range(1000):
                var x = PythonObject(black_box(s.as_string_slice()))
                keep(x)
        except e:
            abort(String(e))

    b.iter(call_fn)


@parameter
def bench_string_to_pyobject_non_ascii(mut b: Bencher) raises:
    """`PythonObject(string)` on a non-ASCII input (UTF-8 fallback)."""
    _ = Python()
    var s = String("héllo \U0001F525")

    @always_inline
    def call_fn() {read}:
        try:
            for _ in range(1000):
                var x = PythonObject(black_box(s.as_string_slice()))
                keep(x)
        except e:
            abort(String(e))

    b.iter(call_fn)


def main() raises:
    _ = Python()
    var m = Bench(BenchConfig(num_repetitions=20, max_runtime_secs=2.0))
    m.bench_function[bench_string_to_pyobject_ascii_short](
        BenchId("string_to_pyobject_ascii_short")
    )
    m.bench_function[bench_string_to_pyobject_ascii_large](
        BenchId("string_to_pyobject_ascii_large")
    )
    m.bench_function[bench_string_to_pyobject_non_ascii](
        BenchId("string_to_pyobject_non_ascii")
    )
    print(m)

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
"""Benchmark `String(py=...)` conversion.

Measures the per-call cost a `def_function`-style binding body pays
when it converts a borrowed `PythonObject` holding a Python `str` to a
native Mojo `String`. The fast path short-circuits the `__str__`
dispatch + temporary `PythonObject` allocation for exact PyUnicode
inputs.
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
def bench_string_from_pystr(mut b: Bencher) raises:
    """`String(py=...)` on an exact PyUnicode (fast path)."""
    _ = Python()
    var py_str = PythonObject("hello world")

    @always_inline
    def call_fn() {read}:
        try:
            for _ in range(1000):
                var x = String(py=black_box(py_str))
                keep(x)
        except e:
            abort(String(e))

    b.iter(call_fn)


def main() raises:
    _ = Python()
    var m = Bench(BenchConfig(num_repetitions=5, max_runtime_secs=2.0))
    m.bench_function[bench_string_from_pystr](BenchId("string_from_pystr"))
    print(m)

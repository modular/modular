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
"""Benchmark `Int(py=...)` conversion.

Measures the cost of converting a `PythonObject` holding a Python `int`
to a native Mojo `Int`. The `Int.__init__(py=...)` fast path checks
`PyLong_CheckExact` and calls `PyLong_AsSsize_t` directly, skipping the
`py.__int__()` -> `PyNumber_Long` round trip + temporary `PythonObject`
construction that the slow path pays for non-exact subclasses.

This is the per-arg cost that `def_function`-style bindings pay when
their handler body says `Int(py=arg)`.
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
def bench_int_from_pyint(mut b: Bencher) raises:
    """Isolated `Int(py=...)` conversion. Measures the primitive
    that the fast path optimizes."""
    _ = Python()
    var py_int = PythonObject(42)

    @always_inline
    def call_fn() {read}:
        try:
            for _ in range(1000):
                var x = Int(py=black_box(py_int))
                keep(x)
        except e:
            abort(String(e))

    b.iter(call_fn)


def main() raises:
    _ = Python()
    var m = Bench(BenchConfig(num_repetitions=5, max_runtime_secs=2.0))
    m.bench_function[bench_int_from_pyint](BenchId("int_from_pyint"))
    print(m)

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
"""Benchmark `PythonObject(value: Bool)` conversion.

Measures the per-call cost of converting a Mojo `Bool` to a `PythonObject`.
The fast path reads the cached `Py_True` / `Py_False` singletons directly
and pays a single `Py_IncRef`, skipping the `PyBool_FromLong` C-call
dispatch that the previous implementation went through.
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
def bench_bool_to_pyobject(mut b: Bencher) raises:
    """`PythonObject(value: Bool)` via the singleton fast path."""
    _ = Python()
    var v = True

    @always_inline
    def call_fn() {read}:
        try:
            for _ in range(1000):
                var x = PythonObject(black_box(v))
                keep(x)
        except e:
            abort(String(e))

    b.iter(call_fn)


@parameter
def bench_scalar_bool_to_pyobject(mut b: Bencher) raises:
    """`PythonObject(value: Scalar[DType.bool])` overload via the same
    singleton fast path (it goes through the shared `_bool_to_pyobject`
    helper after a `Bool(value)` cast)."""
    _ = Python()
    var v = Scalar[DType.bool](True)

    @always_inline
    def call_fn() {read}:
        try:
            for _ in range(1000):
                var x = PythonObject(black_box(v))
                keep(x)
        except e:
            abort(String(e))

    b.iter(call_fn)


def main() raises:
    _ = Python()
    var m = Bench(BenchConfig(num_repetitions=20, max_runtime_secs=2.0))
    m.bench_function[bench_bool_to_pyobject](BenchId("bool_to_pyobject"))
    m.bench_function[bench_scalar_bool_to_pyobject](
        BenchId("scalar_bool_to_pyobject")
    )
    print(m)

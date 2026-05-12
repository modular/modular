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

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


@parameter
def bench_int_from_pyint(mut b: Bencher) raises:
    """Isolated `Int(py=...)` conversion — measures the primitive
    that the fast path optimizes."""
    _ = Python()
    var py_int = PythonObject(42)

    @always_inline
    @parameter
    def call_fn() raises:
        for _ in range(1000):
            var x = Int(py=py_int)
            keep(x)

    b.iter[call_fn]()
    keep(py_int)


# End-to-end shape from issue #6521.
def add(a: PythonObject, b: PythonObject) raises -> PythonObject:
    return PythonObject(Int(py=a) + Int(py=b))


@parameter
def bench_def_function_add(mut b: Bencher) raises:
    """End-to-end Python -> Mojo `add(a, b)` through the standard
    `def_function` path. Each call does two `Int(py=...)` conversions
    plus a return-wrap, so the fast path savings stack."""
    _ = Python()
    var mb = PythonModuleBuilder("_pyobj_convert_bench")
    mb.def_function[add]("add")
    var mod = mb.finalize()
    var f = mod.add
    var arg_a = PythonObject(1)
    var arg_b = PythonObject(2)

    @always_inline
    @parameter
    def call_fn() raises:
        for _ in range(1000):
            var r = f(arg_a, arg_b)
            keep(r)

    b.iter[call_fn]()
    keep(mod)
    keep(arg_a)
    keep(arg_b)
    keep(f)


def main() raises:
    _ = Python()
    var m = Bench(BenchConfig(num_repetitions=5, max_runtime_secs=2.0))
    m.bench_function[bench_int_from_pyint](BenchId("int_from_pyint"))
    m.bench_function[bench_def_function_add](BenchId("def_function_add"))
    print(m)

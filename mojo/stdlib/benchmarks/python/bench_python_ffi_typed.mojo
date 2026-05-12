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
"""Benchmark typed-argument fast-path wrappers vs the standard
`def_function[user_func]` path.

A: stdlib `def_function[add]` where `add` is
   `def(PythonObject, PythonObject) raises -> PythonObject` and the body
   does `Int(py=a) + Int(py=b)` (allocates two PyLongs internally
   via PyNumber_Long).

B: typed wrapper registered via `def_py_c_function`, with `add`
   declared as `def(Int, Int) -> Int`. The wrapper calls
   `PyLong_AsSsize_t` directly on each tuple slot, skipping the
   PythonObject + PyNumber_Long round-trip.

This is item 6 from issue #6521.
"""

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    keep,
)
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


# --- Variant A: current def_function path with PythonObject args ---


def add_pyobj(a: PythonObject, b: PythonObject) raises -> PythonObject:
    return PythonObject(Int(py=a) + Int(py=b))


# --- Variant B: typed fast-path via `def_typed_function`, which picks
# the right wrapper based on the user's function signature.


def add_int(a: Int, b: Int) -> Int:
    return a + b


@parameter
def bench_a_def_function_add(mut b: Bencher) raises:
    _ = Python()
    var m = PythonModuleBuilder("_ffi_typed_bench_a")
    m.def_function[add_pyobj]("add")
    var mod = m.finalize()
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


@parameter
def bench_b_typed_int_int_add(mut b: Bencher) raises:
    _ = Python()
    var m = PythonModuleBuilder("_ffi_typed_bench_b")
    m.def_typed_function[add_int]("add")
    var mod = m.finalize()
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
    m.bench_function[bench_a_def_function_add](BenchId("a_def_function_add"))
    m.bench_function[bench_b_typed_int_int_add](BenchId("b_typed_int_int_add"))
    print(m)

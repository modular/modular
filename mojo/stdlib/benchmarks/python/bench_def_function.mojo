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
"""End-to-end benchmark of `PythonModuleBuilder.def_function` bindings.

Drives a Mojo function from Python via the registered `def_function`
trampoline, in the same shape as issue #6521's reproducer
(`def add(a: PythonObject, b: PythonObject) raises -> PythonObject:
return PythonObject(Int(py=a) + Int(py=b))`).

The bench loop runs in Mojo, so each iteration pays a Mojo -> Python
crossing on top of the Python -> Mojo overhead being measured. The
absolute numbers are therefore noisier than what an external Python
`timeit.repeat` driver would see; the file is here so future
optimizations to the binding plumbing can be A/B'd on the same shape.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


def add(a: PythonObject, b: PythonObject) raises -> PythonObject:
    return PythonObject(Int(py=a) + Int(py=b))


@parameter
def bench_def_function_add(mut b: Bencher) raises:
    """End-to-end `add(a, b)` through `def_function`."""
    _ = Python()
    var mb = PythonModuleBuilder("_def_function_bench")
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
    m.bench_function[bench_def_function_add](BenchId("def_function_add"))
    print(m)

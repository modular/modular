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
"""End-to-end benchmarks for `PythonModuleBuilder.def_function` bindings.

Each bench registers a Mojo function on a `PythonModuleBuilder` and
drives it from a tight loop, measuring the round-trip cost of a
Python -> Mojo call through the registered wrapper.

The loop runs in Mojo, so every iteration pays an extra Mojo -> Python
crossing on top of the Python -> Mojo overhead being measured. The
absolute numbers are therefore noisier than what an external Python
`timeit.repeat` driver would see; for clean end-to-end numbers drive
the registered module from Python instead.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from std.os import abort
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
    def call_fn() {read}:
        try:
            for _ in range(1000):
                var r = f(arg_a, arg_b)
                keep(r)
        except e:
            abort(String(e))

    b.iter(call_fn)
    _ = mod^
    _ = arg_a^
    _ = arg_b^
    _ = f^


def main() raises:
    _ = Python()
    var m = Bench(BenchConfig(num_repetitions=5, max_runtime_secs=2.0))
    m.bench_function[bench_def_function_add](BenchId("def_function_add"))
    print(m)

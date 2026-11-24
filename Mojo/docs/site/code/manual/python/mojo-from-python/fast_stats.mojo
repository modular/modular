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
# The page's two `fast_stats.mojo` blocks, merged into one file as the page
# says ("The interface goes in the same file").

from std.math import sqrt
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


struct RunningStats(Defaultable, Movable, Writable):
    var count: Int
    var mean: Float64
    var m2: Float64

    def __init__(out self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(mut self, value: Float64):
        self.count += 1
        var delta = value - self.mean
        self.mean += delta / Float64(self.count)
        self.m2 += delta * (value - self.mean)

    def merge(mut self, other: Self):
        if other.count == 0:
            return
        var total = self.count + other.count
        var delta = other.mean - self.mean
        var weight = Float64(self.count) * Float64(other.count) / Float64(total)
        self.mean += delta * Float64(other.count) / Float64(total)
        self.m2 += other.m2 + delta * delta * weight
        self.count = total

    def stddev(self) -> Float64:
        if self.count < 2:
            return 0.0
        return sqrt(self.m2 / Float64(self.count - 1))


def py_add(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    var stats = py_self.downcast_value_ptr[RunningStats]()
    for value in args:
        stats[].add(Float64(py=value))
    return None


def py_merge(
    self: Pointer[RunningStats, MutUnsafeAnyOrigin], other: PythonObject
) raises:
    self[].merge(other.downcast_value_ptr[RunningStats]()[])


def py_mean(
    self: Pointer[RunningStats, MutUnsafeAnyOrigin]
) raises -> PythonObject:
    return self[].mean


def py_stddev(
    self: Pointer[RunningStats, MutUnsafeAnyOrigin]
) raises -> PythonObject:
    return self[].stddev()


def py_from_values(values: PythonObject) raises -> PythonObject:
    var stats = RunningStats()
    for value in values:
        stats.add(Float64(py=value))
    return PythonObject(alloc=stats^)


@export
def PyInit_fast_stats() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("fast_stats")
        _ = (
            module.add_type[RunningStats]("RunningStats")
            .def_init_defaultable[RunningStats]()
            .def_py_method[py_add]("add", docstring="Add one or more samples.")
            .def_method[py_merge]("merge")
            .def_method[py_mean]("mean")
            .def_method[py_stddev]("stddev")
            .def_staticmethod[py_from_values](
                "from_values", docstring="Build stats from an iterable."
            )
        )
        return module.finalize()
    except e:
        abort(t"error creating fast_stats: {e}")

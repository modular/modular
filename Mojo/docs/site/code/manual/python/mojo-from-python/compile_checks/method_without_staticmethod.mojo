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
# Doesn't compile: an interface function declared inside the struct without
# `@staticmethod`. `test.py` imports it and expects `ImportError`.

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


struct Gauge(Defaultable, Movable, Writable):
    var level: Float64

    def __init__(out self):
        self.level = 0.0

    def py_level(
        self: Pointer[Gauge, MutUnsafeAnyOrigin]
    ) raises -> PythonObject:
        return self[].level


@export
def PyInit_method_without_staticmethod() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("method_without_staticmethod")
        _ = (
            module.add_type[Gauge]("Gauge")
            .def_init_defaultable[Gauge]()
            .def_method[Gauge.py_level]("level")
        )
        return module.finalize()
    except e:
        abort(t"error creating method_without_staticmethod: {e}")

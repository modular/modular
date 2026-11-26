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
# Compiles: the same struct as `method_without_staticmethod.mojo`, with
# `@staticmethod` added. Confirms the decorator is the only difference.

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


struct Gauge(Defaultable, Movable, Writable):
    var level: Float64

    def __init__(out self):
        self.level = 0.0

    @staticmethod
    def py_level(
        self: Pointer[Gauge, MutUnsafeAnyOrigin]
    ) raises -> PythonObject:
        return self[].level


@export
def PyInit_method_with_staticmethod() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("method_with_staticmethod")
        _ = (
            module.add_type[Gauge]("Gauge")
            .def_init_defaultable[Gauge]()
            .def_method[Gauge.py_level]("level")
        )
        return module.finalize()
    except e:
        abort(t"error creating method_with_staticmethod: {e}")

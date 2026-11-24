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
# Doesn't compile: `def_function()` with a closure that captures a value.

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_capturing_closure() abi("C") -> PythonObject:
    var offset = 1

    def shifted(x: PythonObject) raises {var offset} -> PythonObject:
        return Int(py=x) + offset

    try:
        var module = PythonModuleBuilder("capturing_closure")
        module.def_function[shifted]("shifted")
        return module.finalize()
    except e:
        abort(t"error creating capturing_closure: {e}")

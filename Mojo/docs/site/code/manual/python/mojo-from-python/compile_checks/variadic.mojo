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
# Doesn't compile: `def_function()` with a Mojo variadic parameter.

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


def count(*args: PythonObject) raises -> PythonObject:
    return len(args)


@export
def PyInit_variadic() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("variadic")
        module.def_function[count]("count")
        return module.finalize()
    except e:
        abort(t"error creating variadic: {e}")

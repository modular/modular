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
# Compiles: a nested function with an empty capture list is a plain function
# (see the closures page), so `def_function()` accepts it. Marks the boundary
# of the page's "not closures" claim.

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_nested_plain_function() abi("C") -> PythonObject:
    def shifted(x: PythonObject) raises {} -> PythonObject:
        return Int(py=x) + 1

    try:
        var module = PythonModuleBuilder("nested_plain_function")
        module.def_function[shifted]("shifted")
        return module.finalize()
    except e:
        abort(t"error creating nested_plain_function: {e}")

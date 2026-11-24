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
# Doesn't compile: `def_function()` with a Mojo keyword-only parameter.

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


def scaled(*, scale: PythonObject) raises -> PythonObject:
    return scale


@export
def PyInit_keyword_only() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("keyword_only")
        module.def_function[scaled]("scaled")
        return module.finalize()
    except e:
        abort(t"error creating keyword_only: {e}")

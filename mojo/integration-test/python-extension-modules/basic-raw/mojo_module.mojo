# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from os import abort

from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from python._cpython import PyObjectPtr


@export
fn PyInit_mojo_module() -> PythonObject:
    """Create a Python module with a function binding for `mojo_count_args`."""

    try:
        var b = PythonModuleBuilder("mojo_module")
        b.def_py_c_function(
            mojo_count_args,
            "mojo_count_args",
            docstring="Count the provided arguments",
        )
        return b.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


@export
fn mojo_count_args(py_self: PyObjectPtr, args: PyObjectPtr) -> PyObjectPtr:
    var cpython = Python().cpython()

    return PythonObject(cpython.PyObject_Length(args)).py_object

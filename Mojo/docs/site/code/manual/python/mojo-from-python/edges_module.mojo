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
# Bindings for the claims in the page's "Sharp edges" and "Worth knowing"
# sections. The page names these forms in prose but shows no full example, so
# this module isn't page code.

from std.os import abort
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import (
    ExceptionType,
    PythonModuleBuilder,
    raise_python_exception,
)


# The page's keyword lookup: `Int(py=kwargs["scale"])`.
def scaled(
    value: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    return Int(py=value) * Int(py=kwargs["scale"])


# The page's fix for the unnamed key: check first, then raise your own error.
def scaled_checked(
    value: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    if "scale" not in kwargs:
        raise Error("scaled_checked() needs a 'scale' keyword argument")
    return Int(py=value) * Int(py=kwargs["scale"])


# `def_py_function()` hands every positional argument over as one tuple.
def total(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    var sum = 0
    for value in args:
        sum += Int(py=value)
    return sum


def raise_value_error(
    py_self: PyObjectPtr, args: PyObjectPtr
) abi("C") -> PyObjectPtr:
    return raise_python_exception(
        Error("raised as ValueError"), ExceptionType.ValueError
    )


@export
def PyInit_edges_module() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("edges_module")
        module.def_function[scaled]("scaled")
        module.def_function[scaled_checked]("scaled_checked")
        module.def_py_function[total]("total")
        module.def_py_c_function(raise_value_error, "raise_value_error")
        return module.finalize()
    except e:
        abort(t"error creating edges_module: {e}")

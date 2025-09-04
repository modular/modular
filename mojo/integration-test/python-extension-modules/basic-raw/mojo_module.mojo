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

        # Test def_py_c_function
        b.def_py_c_function(
            mojo_count_args,
            "mojo_count_args",
            docstring="Count the provided arguments",
        )
        b.def_py_c_function(
            mojo_count_args_with_kwargs,
            "mojo_count_args_with_kwargs",
            docstring="Count the provided arguments and keyword arguments",
        )

        # Test def_py_c_method
        _ = (
            b.add_type[TestCounter]("TestCounter")
            .def_init_defaultable[TestCounter]()
            .def_py_c_method(
                counter_count_args,
                "count_args",
                docstring="Count the provided arguments",
            )
            .def_py_c_method(
                counter_count_args_with_kwargs,
                "count_args_with_kwargs",
                docstring="Count the provided arguments and keyword arguments",
            )
            .def_py_c_method[static_method=True](
                counter_static_count_args,
                "static_count_args",
                docstring="Count the provided arguments",
            )
            .def_py_c_method[static_method=True](
                counter_static_count_args_with_kwargs,
                "static_count_args_with_kwargs",
                docstring="Count the provided arguments and keyword arguments",
            )
        )

        return b.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


@export
fn mojo_count_args(py_self: PyObjectPtr, args: PyObjectPtr) -> PyObjectPtr:
    """Count the provided arguments.

    Return value: New reference.
    """
    return mojo_count_args_with_kwargs(py_self, args, {})


@export
fn mojo_count_args_with_kwargs(
    py_self: PyObjectPtr, args: PyObjectPtr, kwargs: PyObjectPtr
) -> PyObjectPtr:
    """Count the provided arguments and keyword arguments.

    Return value: New reference.
    """
    ref cpy = Python().cpython()
    var count = cpy.PyObject_Length(args) + (
        cpy.PyObject_Length(kwargs) if kwargs else 0
    )
    return cpy.PyLong_FromSsize_t(count)


struct TestCounter(Defaultable, ImplicitlyCopyable, Movable, Representable):
    var value: Int

    fn __init__(out self):
        self.value = 0

    fn __repr__(self) -> String:
        return String("TestCounter(value=") + String(self.value) + ")"


@export
fn counter_count_args(py_self: PyObjectPtr, args: PyObjectPtr) -> PyObjectPtr:
    """PyCFunction method to count arguments."""
    return mojo_count_args(py_self, args)


@export
fn counter_count_args_with_kwargs(
    py_self: PyObjectPtr, args: PyObjectPtr, kwargs: PyObjectPtr
) -> PyObjectPtr:
    return mojo_count_args_with_kwargs(py_self, args, kwargs)


@export
fn counter_static_count_args(
    py_self: PyObjectPtr, args: PyObjectPtr
) -> PyObjectPtr:
    return mojo_count_args(py_self, args)


@export
fn counter_static_count_args_with_kwargs(
    py_self: PyObjectPtr, args: PyObjectPtr, kwargs: PyObjectPtr
) -> PyObjectPtr:
    return mojo_count_args_with_kwargs(py_self, args, kwargs)

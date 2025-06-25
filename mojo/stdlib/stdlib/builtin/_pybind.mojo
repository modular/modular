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

from compile.reflection import get_type_name

import python._cpython as cp
from memory import stack_allocation
from python import (
    Python,
    PythonObject,
    ConvertibleFromPython,
)
from python.bindings import (  # Imported for use by the compiler
    PyMojoObject,
    PythonModuleBuilder,
    PythonTypeBuilder,
    _get_type_name,
    check_arguments_arity,
    PyFunctionRaising,
)
from python._cpython import (
    CPython,
    PyMethodDef,
    PyObjectPtr,
    PyType_Slot,
    PyType_Spec,
)


fn get_cpython() -> CPython:
    return Python().cpython()


# This function is used by the compiler to create a new module.
fn create_pybind_module[name: StaticString]() raises -> PythonObject:
    return Python.create_module(name)


fn fail_initialization(owned err: Error) -> PythonObject:
    # TODO(MSTDL-933): Add custom 'MojoError' type, and raise it here.
    cpython = get_cpython()
    error_type = cpython.get_error_global("PyExc_Exception")

    cpython.PyErr_SetString(
        error_type,
        err.unsafe_cstr_ptr(),
    )
    _ = err^
    return PythonObject(from_owned_ptr=PyObjectPtr())


fn gen_pytype_wrapper[
    T: Movable & Defaultable & Representable,
    name: StaticString,
](module: PythonObject) raises:
    # TODO(MOCO-1301): Add support for member method generation.
    # TODO(MOCO-1302): Add support for generating member field as computed properties.
    # TODO(MOCO-1307): Add support for constructor generation.

    var type_builder = PythonTypeBuilder.bind[T](name)
    _ = type_builder.finalize(module)


fn add_wrapper_to_module[
    wrapper_func: PyFunctionRaising, func_name: StaticString
](mut module_obj: PythonObject) raises:
    var b = PythonModuleBuilder(module_obj)
    b.def_py_function[wrapper_func](func_name)
    _ = b.finalize()


fn check_and_get_arg[
    T: AnyType
](
    func_name: StaticString,
    py_args: PythonObject,
    index: Int,
) raises -> UnsafePointer[T]:
    return py_args[index].downcast_value_ptr[T](func=func_name)


# NOTE:
#   @always_inline is needed so that the stack_allocation() that appears in
#   the definition below is valid in the _callers_ stack frame, effectively
#   allowing us to "return" a pointer to stack-allocated data from this
#   function.
@always_inline
fn check_and_get_or_convert_arg[
    T: ConvertibleFromPython
](
    func_name: StaticString,
    py_args: PythonObject,
    index: Int,
) raises -> UnsafePointer[T]:
    # Stack space to hold a converted value for this argument, if needed.
    var converted_arg_ptr: UnsafePointer[T] = stack_allocation[1, T]()

    try:
        return check_and_get_arg[T](func_name, py_args, index)
    except e:
        converted_arg_ptr.init_pointee_move(
            _try_convert_arg[T](
                func_name,
                py_args,
                index,
            )
        )
        # Return a pointer to stack data. Only valid because this function is
        # @always_inline.
        return converted_arg_ptr


fn _try_convert_arg[
    T: ConvertibleFromPython
](
    func_name: StringSlice,
    py_args: PythonObject,
    argidx: Int,
    out result: T,
) raises:
    try:
        result = T(py_args[argidx])
    except convert_err:
        raise Error(
            String.format(
                (
                    "TypeError: {}() expected argument at position {} to be"
                    " instance of (or convertible to) Mojo '{}'; got '{}'."
                    " (Note: attempted conversion failed due to: {})"
                ),
                func_name,
                argidx,
                get_type_name[T](),
                _get_type_name(py_args[argidx]),
                convert_err,
            )
        )

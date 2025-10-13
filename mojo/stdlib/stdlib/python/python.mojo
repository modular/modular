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
"""Implements Python interoperability.

You can import these APIs from the `python` package. For example:

```mojo
from python import Python
```
"""

from collections.dict import OwnedKwargsDict
from os import abort
from sys.ffi import _Global

from ._cpython import (
    CPython,
    Py_eval_input,
    Py_file_input,
    Py_ssize_t,
    PyMethodDef,
    PyObjectPtr,
)
from .python_object import PythonObject

alias _PYTHON_GLOBAL = _Global["Python", _init_python_global]


fn _init_python_global() -> _PythonGlobal:
    return _PythonGlobal()


struct _PythonGlobal(Defaultable, Movable):
    var cpython: CPython

    fn __init__(out self):
        self.cpython = CPython()

    fn __del__(deinit self):
        self.cpython.destroy()


fn _get_python_interface() raises -> Pointer[CPython, StaticConstantOrigin]:
    """Returns an immutable static pointer to the CPython global.

    The returned pointer is immutable to prevent invalid shared mutation of
    this global variable. Once it is initialized, it may not be mutated.
    """

    var ptr = _PYTHON_GLOBAL.get_or_create_indexed_ptr(_Global._python_idx)
    var ptr2 = (
        UnsafePointer(to=ptr[].cpython)
        .as_immutable()
        .unsafe_origin_cast[StaticConstantOrigin]()
    )
    return Pointer(to=ptr2[])


struct Python(Defaultable, ImplicitlyCopyable):
    """Provides methods that help you use Python code in Mojo."""

    var _impl: Pointer[CPython, StaticConstantOrigin]
    """The underlying implementation of Mojo's Python interface."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        try:
            self._impl = _get_python_interface()
        except e:
            abort[prefix="ERROR:"](String(e))
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(self)
            )

    fn __init__(out self, ref [StaticConstantOrigin]cpython: CPython):
        """Construct a `Python` instance from an existing reference
        to the lower-level singleton `CPython` instance.

        Args:
            cpython: Reference to the `CPython` singleton.
        """
        self._impl = Pointer(to=cpython)

    @always_inline
    fn cpython(self) -> ref [StaticConstantOrigin] CPython:
        """Handle to the low-level C API of the CPython interpreter present in
        the current process.

        Returns:
            Handle to the CPython interpreter instance in the current process.
        """
        return self._impl[]

    fn eval(self, var code: String) -> Bool:
        """Executes the given Python code.

        Args:
            code: The python code to execute.

        Returns:
            `True` if the code executed successfully or `False` if the code
            raised an exception.
        """
        ref cpython = self.cpython()
        # return 0 if the code executed successfully, -1 if it raised an exception.
        return cpython.PyRun_SimpleString(code^) == 0

    @staticmethod
    fn evaluate(
        var expr: String,
        file: Bool = False,
        name: StaticString = "__main__",
    ) raises -> PythonObject:
        """Executes the given Python code.

        Args:
            expr: The Python expression to evaluate.
            file: Evaluate as a file and return the module.
            name: The name of the module (most relevant if `file` is True).

        Returns:
            `PythonObject` containing the result of the evaluation.
        """
        ref cpy = Self().cpython()

        var mod = PythonObject(from_borrowed=cpy.PyImport_AddModule(name))
        var dict_ptr = cpy.PyModule_GetDict(mod._obj_ptr)
        if file:
            # We compile the code as provided and execute in the module
            # context. Note that this may be an existing module if the provided
            # module name is not unique. The name here is used only for this
            # code object, not the module itself.
            #
            # The Py_file_input is the code passed to the parsed to indicate
            # the initial state: this is essentially whether it is expecting
            # to compile an expression, a file or statements (e.g. repl).
            var code_ptr = cpy.Py_CompileString(
                expr^, "<evaluate>", Py_file_input
            )
            if not code_ptr:
                raise cpy.unsafe_get_error()
            # For this evaluation, we pass the dictionary both as the globals
            # and the locals. This is because the globals is defined as the
            # dictionary for the module scope, and locals is defined as the
            # dictionary for the *current* scope. Since we are executing at
            # the module scope for this eval, they should be the same object.
            var res_ptr = cpy.PyEval_EvalCode(code_ptr, dict_ptr, dict_ptr)
            cpy.Py_DecRef(code_ptr)
            if not res_ptr:
                raise cpy.unsafe_get_error()
            cpy.Py_DecRef(res_ptr)
            return mod
        else:
            # We use the result of evaluating the expression directly, and allow
            # all the globals/locals to be discarded. See above re: why the same
            # dictionary is being used here for both globals and locals.
            var res_ptr = cpy.PyRun_String(
                expr^, Py_eval_input, dict_ptr, dict_ptr
            )
            _ = mod^
            if not res_ptr:
                raise cpy.unsafe_get_error()
            return PythonObject(from_owned=res_ptr)

    @staticmethod
    fn add_to_path(dir_path: StringSlice) raises:
        """Adds a directory to the Python path.

        This might be necessary to import a Python module via `import_module()`.
        For example:

        ```mojo
        from python import Python

        # Specify path to `mypython.py` module
        Python.add_to_path("path/to/module")
        var mypython = Python.import_module("mypython")

        var c = mypython.my_algorithm(2, 3)
        ```

        Args:
            dir_path: The path to a Python module you want to import.
        """
        var sys = Python.import_module("sys")
        var directory: PythonObject = dir_path
        _ = sys.path.append(directory)

    # ===-------------------------------------------------------------------===#
    # PythonObject "Module" Operations
    # ===-------------------------------------------------------------------===#

    @staticmethod
    fn import_module(var module: String) raises -> PythonObject:
        """Imports a Python module.

        This provides you with a module object you can use just like you would
        in Python. For example:

        ```mojo
        from python import Python

        # This is equivalent to Python's `import numpy as np`
        np = Python.import_module("numpy")
        a = np.array([1, 2, 3])
        ```

        Args:
            module: The Python module name. This module must be visible from the
                list of available Python paths (you might need to add the
                module's path with `add_to_path()`).

        Returns:
            The Python module.
        """
        # Initialize the global interpreter and check for errors.
        ref cpy = Self().cpython()
        cpy.check_init_error()

        var module_ptr = cpy.PyImport_ImportModule(module^)
        if not module_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=module_ptr)

    @staticmethod
    fn create_module(name: StaticString) raises -> PythonObject:
        """Creates a Python module using the provided name.

        Inspired by https://github.com/pybind/pybind11/blob/a1d00916b26b187e583f3bce39cd59c3b0652c32/include/pybind11/pybind11.h#L1227

        TODO: allow specifying a doc-string to attach to the module upon creation or lazily added?

        Args:
            name: The Python module name.

        Returns:
            The Python module.
        """
        # Initialize the global interpreter and check for errors.
        ref cpy = Self().cpython()
        cpy.check_init_error()

        var module_ptr = cpy.PyModule_Create(name)
        if not module_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=module_ptr)

    @staticmethod
    fn add_functions(
        module: PythonObject,
        var functions: List[PyMethodDef],
    ) raises:
        """Adds functions to a Python module object.

        Args:
            module: The Python module object.
            functions: List of function data.

        Raises:
            If we fail to add the functions to the module.
        """

        # Write a zeroed entry at the end as a terminator.
        functions.append(PyMethodDef())

        # FIXME(MSTDL-910):
        #   This is an intentional memory leak, because we don't store this
        #   in a global variable (yet).
        var ptr: UnsafePointer[PyMethodDef] = functions.steal_data()

        return Self._unsafe_add_functions(module, ptr)

    @staticmethod
    fn _unsafe_add_functions(
        module: PythonObject,
        functions: UnsafePointer[PyMethodDef],
    ) raises:
        """Adds functions to a Python module object.

        Safety:
            The provided `functions` pointer must point to data that lives
            for the duration of the associated Python interpreter session.

        Args:
            module: The Python module object.
            functions: A null terminated pointer to function data.

        Raises:
            If we fail to add the functions to the module.
        """
        ref cpython = Python().cpython()

        var result = cpython.PyModule_AddFunctions(
            # Safety: `module` pointer lives long enough because its reference
            #   argument.
            module._obj_ptr,
            functions,
        )

        if result != 0:
            raise cpython.get_error()

    @staticmethod
    fn add_object(
        module: PythonObject,
        var name: String,
        value: PythonObject,
    ) raises:
        """Add a new object to `module` with the given name and value.

        The provided object can be any type of Python object: an instance,
        a type object, a function, etc.

        The added value will be inserted into the `__dict__` of the provided
        module.

        Args:
            module: The Python module to modify.
            name: The name of the new object.
            value: The python object value.
        """

        ref cpython = Python().cpython()

        var result = cpython.PyModule_AddObjectRef(
            module._obj_ptr,
            name.unsafe_cstr_ptr(),
            value._obj_ptr,
        )

        if result != 0:
            raise cpython.get_error()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @doc_private
    @staticmethod
    fn _dict[
        V: ConvertibleToPython & Copyable & Movable = PythonObject
    ](kwargs: OwnedKwargsDict[V]) raises -> PyObjectPtr:
        """Construct a Python dictionary from keyword arguments.

        Return value: New reference.
        """
        ref cpy = Python().cpython()
        var dict_obj = cpy.PyDict_New()

        for entry in kwargs.items():
            var key = cpy.PyUnicode_DecodeUTF8(entry.key.as_string_slice())
            if not key:
                raise cpy.unsafe_get_error()

            var val = entry.value.copy().to_python_object()
            var errno = cpy.PyDict_SetItem(dict_obj, key, val._obj_ptr)
            cpy.Py_DecRef(key)
            _ = val
            if errno == -1:
                raise cpy.unsafe_get_error()

        return dict_obj

    @staticmethod
    fn dict[
        V: ConvertibleToPython & Copyable & Movable = PythonObject
    ](**kwargs: V) raises -> PythonObject:
        """Construct an Python dictionary from keyword arguments.

        Parameters:
            V: The type of the values in the dictionary. Must implement the
                `ConvertibleToPython`, `Copyable`, and `Movable` traits.

        Args:
            kwargs: The keyword arguments to construct the dictionary with.

        Returns:
            The constructed Python dictionary.

        Raises:
            On failure to construct the dictionary or convert the values to
            Python objects.
        """
        return PythonObject(from_owned=Self._dict(kwargs))

    @staticmethod
    fn dict[
        K: ConvertibleToPython & Copyable & Movable = PythonObject,
        V: ConvertibleToPython & Copyable & Movable = PythonObject,
    ](tuples: Span[Tuple[K, V]]) raises -> PythonObject:
        """Construct an Python dictionary from a list of key-value tuples.

        Parameters:
            K: The type of the keys in the dictionary. Must implement the
                `ConvertibleToPython`, `Copyable`, and `Movable` traits.
            V: The type of the values in the dictionary. Must implement the
                `ConvertibleToPython`, `Copyable`, and `Movable` traits.

        Args:
            tuples: The list of key-value tuples to construct the dictionary
                with.

        Returns:
            The constructed Python dictionary.

        Raises:
            On failure to construct the dictionary or convert the keys or values
            to Python objects.
        """

        ref cpython = Python().cpython()
        var dict_obj_ptr = cpython.PyDict_New()
        if not dict_obj_ptr:
            raise Error("internal error: PyDict_New failed")

        for i in range(len(tuples)):
            var key = tuples[i][0].copy().to_python_object()
            var val = tuples[i][1].copy().to_python_object()
            var errno = cpython.PyDict_SetItem(
                dict_obj_ptr, key._obj_ptr, val._obj_ptr
            )
            _ = key
            _ = val
            if errno == -1:
                raise cpython.unsafe_get_error()

        return PythonObject(from_owned=dict_obj_ptr)

    @staticmethod
    fn list[
        T: ConvertibleToPython & Copyable & Movable
    ](values: Span[T]) raises -> PythonObject:
        """Initialize the object from a list of values.

        Parameters:
            T: The span element type.

        Args:
            values: The values to initialize the list with.

        Returns:
            A PythonObject representing the list.
        """
        ref cpython = Python().cpython()
        var obj_ptr = cpython.PyList_New(len(values))

        for i in range(len(values)):
            var obj = values[i].copy().to_python_object()
            _ = cpython.PyList_SetItem(obj_ptr, i, obj.steal_data())

        return PythonObject(from_owned=obj_ptr)

    @staticmethod
    fn _list[
        *Ts: ConvertibleToPython & Copyable
    ](
        values: VariadicPack[True, _, ConvertibleToPython & Copyable, *Ts]
    ) raises -> PythonObject:
        """Initialize the object from a list literal.

        Parameters:
            Ts: The list element types.

        Args:
            values: The values to initialize the list with.

        Returns:
            A PythonObject representing the list.
        """
        ref cpython = Python().cpython()
        var obj_ptr = cpython.PyList_New(len(values))

        @parameter
        for i in range(len(VariadicList(Ts))):
            var obj = values[i].copy().to_python_object()
            _ = cpython.PyList_SetItem(obj_ptr, i, obj.steal_data())

        return PythonObject(from_owned=obj_ptr)

    @always_inline
    @staticmethod
    fn list[
        *Ts: ConvertibleToPython & Copyable
    ](var *values: *Ts) raises -> PythonObject:
        """Construct an Python list of objects.

        Parameters:
            Ts: The list element types.

        Args:
            values: The values to initialize the list with.

        Returns:
            The constructed Python list.
        """
        return Self._list(values)

    @staticmethod
    fn _tuple[
        *Ts: ConvertibleToPython & Copyable
    ](
        values: VariadicPack[True, _, ConvertibleToPython & Copyable, *Ts]
    ) raises -> PythonObject:
        """Initialize the object from a tuple literal.

        Parameters:
            Ts: The tuple element types.

        Args:
            values: The values to initialize the tuple with.

        Returns:
            A PythonObject representing the tuple.
        """
        ref cpython = Python().cpython()
        var obj_ptr = cpython.PyTuple_New(len(values))

        @parameter
        for i in range(len(VariadicList(Ts))):
            var obj = values[i].copy().to_python_object()
            _ = cpython.PyTuple_SetItem(obj_ptr, i, obj.steal_data())

        return PythonObject(from_owned=obj_ptr)

    @always_inline
    @staticmethod
    fn tuple[
        *Ts: ConvertibleToPython & Copyable
    ](var *values: *Ts) raises -> PythonObject:
        """Construct an Python tuple of objects.

        Parameters:
            Ts: The list element types.

        Args:
            values: The values to initialize the tuple with.

        Returns:
            The constructed Python tuple.
        """
        return Self._tuple(values)

    @no_inline
    fn as_string_slice(
        self, str_obj: PythonObject
    ) -> StringSlice[ImmutableAnyOrigin]:
        """Return a string representing the given Python object.

        Args:
            str_obj: The Python object.

        Returns:
            Mojo string representing the given Python object.
        """
        ref cpython = self.cpython()
        return cpython.PyUnicode_AsUTF8AndSize(str_obj._obj_ptr)

    @staticmethod
    fn type(obj: PythonObject) -> PythonObject:
        """Return Type of this PythonObject.

        Args:
            obj: PythonObject we want the type of.

        Returns:
            A PythonObject that holds the type object.
        """
        ref cpython = Python().cpython()
        return PythonObject(from_owned=cpython.PyObject_Type(obj._obj_ptr))

    @staticmethod
    fn none() -> PythonObject:
        """Get a `PythonObject` representing `None`.

        Returns:
            `PythonObject` representing `None`.
        """
        return PythonObject(None)

    @staticmethod
    fn str(obj: PythonObject) raises -> PythonObject:
        """Convert a PythonObject to a Python `str`.

        Args:
            obj: The PythonObject to convert.

        Returns:
            A Python `str` object.

        Raises:
            An error if the conversion failed.
        """
        ref cpython = Python().cpython()
        var py_str_ptr = cpython.PyObject_Str(obj._obj_ptr)
        if not py_str_ptr:
            raise cpython.get_error()

        return PythonObject(from_owned=py_str_ptr)

    @staticmethod
    fn int(obj: PythonObject) raises -> PythonObject:
        """Convert a PythonObject to a Python `int` (i.e. arbitrary precision
        integer).

        Args:
            obj: The PythonObject to convert.

        Raises:
            If the conversion to `int` fails.

        Returns:
            A PythonObject representing the result of the conversion to `int`.
        """
        ref cpython = Python().cpython()
        var py_obj_ptr = cpython.PyNumber_Long(obj._obj_ptr)
        if not py_obj_ptr:
            raise cpython.get_error()

        return PythonObject(from_owned=py_obj_ptr)

    @staticmethod
    fn float(obj: PythonObject) raises -> PythonObject:
        """Convert a PythonObject to a Python `float` object.

        Args:
            obj: The PythonObject to convert.

        Returns:
            A Python `float` object.

        Raises:
            If the conversion fails.
        """
        ref cpython = Python().cpython()

        var float_obj = cpython.PyNumber_Float(obj._obj_ptr)
        if not float_obj:
            raise cpython.get_error()

        return PythonObject(from_owned=float_obj)

    # ===-------------------------------------------------------------------===#
    # Checked Conversions
    # ===-------------------------------------------------------------------===#

    @staticmethod
    fn py_long_as_ssize_t(obj: PythonObject) raises -> Py_ssize_t:
        """Get the value of a Python `long` object.

        Args:
            obj: The Python `long` object.

        Raises:
            If `obj` is not a Python `long` object, or if the `long` object
            value overflows `Py_ssize_t`.

        Returns:
            The value of the `long` object as a `Py_ssize_t`.
        """
        ref cpython = Python().cpython()
        var long: Py_ssize_t = cpython.PyLong_AsSsize_t(obj._obj_ptr)
        if long == -1 and cpython.PyErr_Occurred():
            # Note that -1 does not guarantee an error, it just means we need to
            # check if there was an exception.
            raise cpython.unsafe_get_error()

        return long

    @staticmethod
    fn is_true(obj: PythonObject) raises -> Bool:
        """Check if the PythonObject is truthy.

        Args:
            obj: The PythonObject to check.

        Returns:
            True if the PythonObject is truthy and False otherwise.

        Raises:
            If the boolean value of the PythonObject cannot be determined.
        """
        # TODO: decide if this method should be actually exposed as public,
        # and add tests if so.
        ref cpython = Python().cpython()
        var result = cpython.PyObject_IsTrue(obj._obj_ptr)
        if result == -1:
            raise cpython.get_error()

        return result == 1

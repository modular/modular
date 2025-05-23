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

from builtin.identifiable import TypeIdentifiable
from collections.string.string_slice import get_static_string
from os import abort
from sys.ffi import c_int, _Global
from sys.info import sizeof

from memory import UnsafePointer
from python import Python, PythonConvertible, PythonObject, TypedPythonObject
from python._cpython import (
    Py_TPFLAGS_DEFAULT,
    PyCFunction,
    PyMethodDef,
    PyObject,
    PyObjectPtr,
    PyTypeObject,
    PyType_Slot,
    PyType_Spec,
    destructor,
    newfunc,
)
from python.python_object import PyFunction, PyFunctionRaising, PythonModule

# ===-----------------------------------------------------------------------===#
# Global `PyTypeObject` Registration
# ===-----------------------------------------------------------------------===#

alias MOJO_PYTHON_TYPE_OBJECTS = _Global[
    "MOJO_PYTHON_TYPE_OBJECTS",
    Dict[StaticString, TypedPythonObject["Type"]],
    _init_python_type_objects,
]
"""Mapping of Mojo type identifiers to unique `PyTypeObject*` binding
that Mojo type to this CPython interpreter instance."""


fn _init_python_type_objects() -> Dict[StaticString, TypedPythonObject["Type"]]:
    return Dict[StaticString, TypedPythonObject["Type"]]()


fn _register_py_type_object(
    type_id: StaticString,
    owned type_obj: TypedPythonObject["Type"],
) raises:
    """Register a Python type object for the identified Mojo type.

    The provided Python type object describes how a wrapped Mojo value can
    be used from within Python code.

    Args:
        type_id: The unique type id of a Mojo type.
        type_obj: The Python type object that binds the Mojo type identified
          by `type_id`.

    Raises:
        If a Python type object has already been registered in the current
        session for the provided type id.
    """
    var type_dict = MOJO_PYTHON_TYPE_OBJECTS.get_or_create_ptr()

    if type_id in type_dict[]:
        raise Error(
            (
                "Error building multiple Python type objects bound to"
                " Mojo type with id: "
            ),
            type_id,
        )

    type_dict[][type_id] = type_obj^


fn lookup_py_type_object[
    T: TypeIdentifiable
]() raises -> TypedPythonObject["Type"]:
    """Retrieve a reference to the unique Python type describing Python objects
    containing Mojo values of type `T`.

    This function looks up the Python type object that was previously registered
    for the Mojo type `T` using a `PythonTypeBuilder`. The returned type object
    can be used to create Python objects that wrap Mojo values of type `T`.

    Parameters:
        T: The Mojo type to look up. Must implement the `TypeIdentifiable` trait
           to provide a unique type identifier.

    Returns:
        A `TypedPythonObject["Type"]` representing the Python type object that
        binds the Mojo type `T` to the current CPython interpreter instance.

    Raises:
        If no `PythonTypeBuilder` was ever finalized for type `T`, or if no
        Python type object has been registered for the provided type identifier.
    """
    var type_dict = MOJO_PYTHON_TYPE_OBJECTS.get_or_create_ptr()

    # FIXME(MSTDL-1580):
    #   This should use a unique compiler type ID, not the Python name of this
    #   type.
    if entry := type_dict[].find(T.TYPE_ID):
        return entry.take()

    raise Error(
        "No Python type object registered for Mojo type with id: ", T.TYPE_ID
    )


# ===-----------------------------------------------------------------------===#
# Mojo Object
# ===-----------------------------------------------------------------------===#

# Must be ABI compatible with `initproc`
alias Typed_initproc = fn (
    PyObjectPtr,
    TypedPythonObject["Tuple"],
    # Will be NULL if no keyword arguments were passed.
    PyObjectPtr,
) -> c_int

# Must be ABI compatible with `newfunc`
alias Typed_newfunc = fn (
    UnsafePointer[PyTypeObject],
    TypedPythonObject["Tuple"],
    PyObjectPtr,
) -> PyObjectPtr


struct PyMojoObject[T: AnyType]:
    """Storage backing a PyObject* wrapping a Mojo value.

    This struct represents the C-level layout of a Python object that contains
    a wrapped Mojo value. It must be ABI-compatible with CPython's PyObject
    structure to enable seamless interoperability between Mojo and Python.

    The struct follows Python's object model where all Python objects begin
    with a PyObject header (ob_base), followed by type-specific data. In this
    case, the type-specific data is a Mojo value of type T.

    See https://docs.python.org/3/c-api/structures.html#c.PyObject for more details.

    Parameters:
        T: The Mojo type being wrapped. Can be any type that satisfies `AnyType`.
    """

    var ob_base: PyObject
    """The standard Python object header containing reference count and type information.

    This must be the first field to maintain ABI compatibility with Python's object layout.
    All Python objects begin with this header structure.
    """

    var mojo_value: T
    """The actual Mojo value being wrapped and exposed to Python.

    This field stores the Mojo data that Python code can interact with through
    the registered type methods and bindings.
    """


fn _default_tp_new_wrapper[
    T: Defaultable & Movable
](
    subtype: UnsafePointer[PyTypeObject],
    args: TypedPythonObject["Tuple"],
    keyword_args: PyObjectPtr,
) -> PyObjectPtr:
    """Python-compatible wrapper around a Mojo initializer function.

    This function serves as the `tp_new` slot for Python type objects that
    wrap Mojo values. It creates new Python objects containing default-initialized
    Mojo values of type `T`. The function follows Python's object creation
    protocol and handles error cases by setting appropriate Python exceptions.

    This wrapper is designed to be used with Python types that don't accept
    any initialization arguments, creating objects with default Mojo values.

    Parameters:
        T: The wrapped Mojo type that must be `Defaultable` and `Movable`.

    Args:
        subtype: Pointer to the Python type object for which to create an instance.
                This allows for proper subtype handling in Python's type system.
        args: Tuple of positional arguments passed from Python. Must be empty
              for this default initializer.
        keyword_args: Pointer to keyword arguments dictionary passed from Python.
                     Must be NULL for this default initializer.

    Returns:
        A new Python object pointer containing a default-initialized Mojo value
        of type `T`, or a null pointer if an error occurs during creation.

    Note:
        This function sets a Python `ValueError` exception if any arguments
        are provided, since the default initializer expects no parameters.
        The returned object follows Python's reference counting rules where
        the caller takes ownership of the new reference.
    """

    var cpython = Python().cpython()

    try:
        if len(args) != 0 or keyword_args != PyObjectPtr():
            raise "unexpected arguments passed to default initializer function of wrapped Mojo type"

        # Create a new Python object with a default initialized Mojo value.
        return PythonObject._unsafe_alloc(subtype, T()).steal_data()

    except e:
        # TODO(MSTDL-933): Add custom 'MojoError' type, and raise it here.
        var error_type = cpython.get_error_global("PyExc_ValueError")
        cpython.PyErr_SetString(
            error_type,
            e.unsafe_cstr_ptr(),
        )
        return PyObjectPtr()


fn _tp_dealloc_wrapper[T: Defaultable & Representable](py_self: PyObjectPtr):
    """Python-compatible wrapper for deallocating a `PyMojoObject`.

    This function serves as the tp_dealloc slot for Python type objects that
    wrap Mojo values. It properly destroys the wrapped Mojo value and frees
    the Python object memory.

    Parameters:
        T: The wrapped Mojo type that must be `Defaultable` and `Representable`.

    Args:
        py_self: Pointer to the Python object to be deallocated.
    """
    var self_obj_ptr = py_self.unsized_obj_ptr.bitcast[PyMojoObject[T]]()
    var self_ptr = UnsafePointer[T](to=self_obj_ptr[].mojo_value)

    # TODO(MSTDL-633):
    #   Is this always safe? Wrap in GIL, because this could
    #   evaluate arbitrary code?
    # Destroy this `Person` instance.
    self_ptr.destroy_pointee()

    var cpython = Python().cpython()

    cpython.PyObject_Free(py_self.unsized_obj_ptr.bitcast[NoneType]())


fn _tp_repr_wrapper[
    T: Defaultable & Representable
](py_self: PyObjectPtr) -> PyObjectPtr:
    """Python-compatible wrapper for generating string representation of a
    `PyMojoObject`.

    This function serves as the `tp_repr` slot for Python type objects that
    wrap Mojo values. It calls the Mojo `repr()` function on the wrapped value
    and returns the result as a Python string object.

    Parameters:
        T: The wrapped Mojo type that must be `Defaultable` and `Representable`.

    Args:
        py_self: Pointer to the Python object to get representation for.

    Returns:
        A new Python string object containing the string representation,
        or null pointer if an error occurs.
    """
    var self_obj_ptr = py_self.unsized_obj_ptr.bitcast[PyMojoObject[T]]()
    var self_ptr = UnsafePointer[T](to=self_obj_ptr[].mojo_value)

    var repr_str: String = repr(self_ptr[])

    return PythonObject(string=repr_str).steal_data()


# ===-----------------------------------------------------------------------===#
# Builders
# ===-----------------------------------------------------------------------===#


struct PythonModuleBuilder:
    """A builder for creating Python modules with Mojo function and type bindings.

    This builder provides a high-level API for declaring Python bindings for Mojo
    functions and types within a Python module. It manages the registration of
    functions, types, and their associated metadata, then finalizes everything
    into a complete Python module object.

    The builder follows a declarative pattern where you:
    1. Create a builder instance with a module name
    2. Add function bindings using `def_function()`, `def_py_function()`, `def_py_c_function()`
    3. Add type bindings using `add_type[T]()` and configure them
    4. Call `finalize()` to finish building the Python module.

    Example:
        ```mojo
        from python._bindings import PythonModuleBuilder

        var builder = PythonModuleBuilder("my_module")
        builder.def_function[my_func]("my_func", "Documentation for my_func")

        _ = builder.add_type[MyType]("MyType").def_method[my_method]("my_method")

        var module = builder.finalize()
        ```

    Note:
        After calling `finalize()`, the builder's internal state is cleared and
        it should not be reused for creating additional modules.

        TODO: This should be enforced programmatically in the future.
    """

    var module: PythonModule
    """The Python module being built."""

    var functions: List[PyMethodDef]
    """List of function definitions that will be exposed in the module."""

    var type_builders: List[PythonTypeBuilder]
    """List of type builders for types that will be exposed in the module."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self, name: StaticString) raises:
        """Construct a Python module builder with the given module name.

        Args:
            name: The name of the module.

        Raises:
            If the module creation fails.
        """
        self = Self(PythonModule(name))

    fn __init__(out self, module: PythonModule):
        """Construct a Python module builder with the given module.

        Args:
            module: The module to build.
        """
        self.module = module
        self.functions = List[PyMethodDef]()
        self.type_builders = List[PythonTypeBuilder]()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn add_type[
        T: Movable & Defaultable & Representable & TypeIdentifiable
    ](mut self, type_name: StaticString) -> ref [
        self.type_builders
    ] PythonTypeBuilder:
        """Add a type to the module and return a builder for it.

        Parameters:
            T: The mojo type to bind in the module.

        Args:
            type_name: The name of the type to expose in the module.

        Returns:
            A reference to a type builder registered in the module builder.
        """
        self.type_builders.append(PythonTypeBuilder.bind[T](type_name))
        return self.type_builders[-1]

    fn def_py_c_function(
        mut self: Self,
        func: PyCFunction,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PyCFunction signature in the
        module.

        Args:
            func: The function to declare a binding for.
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        self.functions.append(PyMethodDef.function(func, func_name, docstring))

    fn def_py_function[
        func: PyFunction
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PyFunction signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        self.def_py_c_function(
            _py_c_function_wrapper[func], func_name, docstring
        )

    fn def_py_function[
        func: PyFunctionRaising
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PyFunctionRaising signature in
        the module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        self.def_py_c_function(
            _py_c_function_wrapper[func], func_name, docstring
        )

    # ===-------------------------------------------------------------------===#
    # def_function with return, raising
    # ===-------------------------------------------------------------------===#

    # TODO: declare these as a single method using variadics
    fn def_function[
        func: fn () raises -> PythonObject
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut py_args: TypedPythonObject["Tuple"]
        ) raises -> PythonObject:
            check_arguments_arity(0, py_args)
            return func()

        self.def_py_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (mut PythonObject) raises -> PythonObject
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut py_args: TypedPythonObject["Tuple"]
        ) raises -> PythonObject:
            check_arguments_arity(1, py_args)
            var arg = py_args[0]
            return func(arg)

        self.def_py_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (mut PythonObject, mut PythonObject) raises -> PythonObject
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut py_args: TypedPythonObject["Tuple"]
        ) raises -> PythonObject:
            check_arguments_arity(2, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            return func(arg0, arg1)

        self.def_py_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (
            mut PythonObject, mut PythonObject, mut PythonObject
        ) raises -> PythonObject
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut py_args: TypedPythonObject["Tuple"]
        ) raises -> PythonObject:
            check_arguments_arity(3, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]
            return func(arg0, arg1, arg2)

        self.def_py_function[wrapper](func_name, docstring)

    # ===-------------------------------------------------------------------===#
    # def_function with return, not raising
    # ===-------------------------------------------------------------------===#

    # TODO: declare these as a single method using variadics
    fn def_function[
        func: fn () -> PythonObject
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper() raises -> PythonObject:
            return func()

        self.def_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (mut PythonObject) -> PythonObject
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(mut a0: PythonObject) raises -> PythonObject:
            return func(a0)

        self.def_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (mut PythonObject, mut PythonObject) -> PythonObject
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(
            mut a0: PythonObject, mut a1: PythonObject
        ) raises -> PythonObject:
            return func(a0, a1)

        self.def_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (
            mut PythonObject, mut PythonObject, mut PythonObject
        ) -> PythonObject
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(
            mut a0: PythonObject, mut a1: PythonObject, mut a2: PythonObject
        ) raises -> PythonObject:
            return func(a0, a1, a2)

        self.def_function[wrapper](func_name, docstring)

    # ===-------------------------------------------------------------------===#
    # def_function with no return, raising
    # ===-------------------------------------------------------------------===#

    # TODO: declare these as a single method using variadics
    fn def_function[
        func: fn () raises
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper() raises -> PythonObject:
            func()
            return PythonObject(None)

        self.def_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (mut PythonObject) raises
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(mut a0: PythonObject) raises -> PythonObject:
            func(a0)
            return PythonObject(None)

        self.def_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (mut PythonObject, mut PythonObject) raises
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(
            mut a0: PythonObject, mut a1: PythonObject
        ) raises -> PythonObject:
            func(a0, a1)
            return PythonObject(None)

        self.def_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (mut PythonObject, mut PythonObject, mut PythonObject) raises
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(
            mut a0: PythonObject, mut a1: PythonObject, mut a2: PythonObject
        ) raises -> PythonObject:
            func(a0, a1, a2)
            return PythonObject(None)

        self.def_function[wrapper](func_name, docstring)

    # ===-------------------------------------------------------------------===#
    # def_function with no return, not raising
    # ===-------------------------------------------------------------------===#

    # TODO: declare these as a single method using variadics
    fn def_function[
        func: fn ()
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper() raises:
            func()

        self.def_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (mut PythonObject)
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(mut a0: PythonObject) raises:
            func(a0)

        self.def_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (mut PythonObject, mut PythonObject)
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(mut a0: PythonObject, mut a1: PythonObject) raises:
            func(a0, a1)

        self.def_function[wrapper](func_name, docstring)

    fn def_function[
        func: fn (mut PythonObject, mut PythonObject, mut PythonObject)
    ](
        mut self: Self,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ):
        """Declare a binding for a function with PythonObject signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(
            mut a0: PythonObject, mut a1: PythonObject, mut a2: PythonObject
        ) raises:
            func(a0, a1, a2)

        self.def_function[wrapper](func_name, docstring)

    fn finalize(mut self) raises -> PythonModule:
        """Finalize the module builder, creating the module object.


        All types and functions added to the builder will be built and exposed
        in the module. After calling this method, the builder's internal state
        is cleared and it should not be reused for creating additional modules.

        Returns:
            The finalized Python module containing all registered functions and types.

        Raises:
            If the module creation fails or if we fail to add any of the
            declared functions or types to the module.
        """

        Python.add_functions(self.module, self.functions)
        self.functions.clear()

        for builder in self.type_builders:
            builder[].finalize(self.module)
        self.type_builders.clear()

        return self.module


struct PythonTypeBuilder(Movable, Copyable):
    """A builder for a Python 'type' binding.

    This is typically used to build a type description of a `PyMojoObject[T]`.

    This builder is used to declare method bindings for a Python type, and then
    create the type binding.

    Finalizing builder created with `PythonTypeObject.bind[T]()` will globally
    register the resulting Python 'type' object as the single canonical type
    object for the Mojo type `T`. Subsequent attempts to register a Python type
    for `T` will raise an exception.

    Registering a Python type object for `T` is necessary to be able to
    construct a `PythonObject` from an instance of `T`, or to downcast an
    existing `PythonObject` to a pointer to the inner `T` value.
    """

    var type_name: StaticString
    """The name the type will be exposed as in the Python module."""

    var _type_id: Optional[StaticString]
    """The unique type identifier for the Mojo type being bound, if any."""

    var basicsize: Int
    """The required allocation size to hold an instance of this type as a Python object."""

    var _slots: List[PyType_Slot]
    """List of Python type slots that define the behavior of the type."""

    var methods: List[PyMethodDef]
    """List of method definitions that will be exposed on the Python type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self, type_name: StaticString, *, basicsize: Int):
        """Construct a new builder for a Python type binding.

        Args:
            type_name: The name the type will be exposed as in the Python module.
            basicsize: The required allocation size to hold an instance of this
              type as a Python object.
        """

        self.type_name = type_name
        self._type_id = None
        self.basicsize = basicsize
        self._slots = List[PyType_Slot]()
        self.methods = List[PyMethodDef]()

    @staticmethod
    fn bind[
        T: Movable & Defaultable & Representable & TypeIdentifiable
    ](type_name: StaticString) -> PythonTypeBuilder:
        """Construct a new builder for a Python type that binds a Mojo type.

        Parameters:
            T: The mojo type to bind.

        Args:
            type_name: The name the type will be exposed as in the Python module.

        Returns:
            A new type builder instance.
        """
        var b = PythonTypeBuilder(
            type_name,
            basicsize=sizeof[PyMojoObject[T]](),
        )
        b._slots = List[PyType_Slot](
            # All wrapped Mojo types are allocated generically.
            PyType_Slot.tp_new(_default_tp_new_wrapper[T]),
            PyType_Slot.tp_dealloc(_tp_dealloc_wrapper[T]),
            PyType_Slot.tp_repr(_tp_repr_wrapper[T]),
        )
        b.methods = List[PyMethodDef]()
        b._type_id = T.TYPE_ID

        return b^

    fn finalize(mut self) raises -> TypedPythonObject["Type"]:
        """Finalize the builder and create a Python type object.

        This method completes the construction of a Python type object from the
        builder's configuration.

        The method ensures that each Mojo type has exactly one corresponding
        Python type object by registering the created type in a global registry.
        This prevents accidental creation of multiple type objects for the same
        Mojo type, which would break Python's type system assumptions.

        Returns:
            A `TypedPythonObject["Type"]` representing the newly created Python
            type object that can be used to create instances or register with
            Python modules.

        Raises:
            If the Python type object creation fails, typically due to invalid
            type specifications or Python C API errors.

        Note:
            After calling this method, the builder's internal state may be
            modified (methods list is consumed), so the builder should not be
            reused for creating additional type objects.

            TODO: This should be enforced programmatically in the future.
        """
        var cpython = Python().cpython()

        if self.methods:
            self.methods.append(PyMethodDef())  # Zeroed item as terminator
            # FIXME: Avoid leaking the methods data pointer in this way.
            self._slots.append(
                PyType_Slot.tp_methods(self.methods.steal_data())
            )

        # Zeroed item terminator
        self._slots.append(PyType_Slot.null())

        var type_spec = PyType_Spec(
            # FIXME(MOCO-1306): This should be `T.__name__`.
            self.type_name.unsafe_ptr().bitcast[sys.ffi.c_char](),
            self.basicsize,
            0,
            Py_TPFLAGS_DEFAULT,
            # Note: This pointer is only "read-only" by PyType_FromSpec.
            self._slots.unsafe_ptr(),
        )

        # Construct a Python 'type' object from our type spec.
        var type_obj = cpython.PyType_FromSpec(UnsafePointer(to=type_spec))

        if type_obj.is_null():
            raise cpython.get_error()

        var typed_type_obj = TypedPythonObject["Type"](
            unsafe_unchecked_from=PythonObject(from_owned_ptr=type_obj)
        )

        # Every Mojo type that is exposed to Python must have EXACTLY ONE
        # `PyTypeObject` instance that represents it. That is important for
        # correctness. This check here ensures that the user is not accidentally
        # creating multiple `PyTypeObject` instances that bind the same Mojo
        # type.
        if type_id := self._type_id:
            _register_py_type_object(type_id[], typed_type_obj)

        return typed_type_obj^

    fn finalize(mut self, module: PythonModule) raises:
        """Finalize the builder and add the created type to a Python module.

        This method completes the type building process by calling the parameterless
        `finalize()` method to create the Python type object, then automatically
        adds the resulting type to the specified Python module using the builder's
        configured type name. After successful completion, the builder's method
        list is cleared to prevent accidental reuse.

        This is a convenience method that combines type finalization and module
        registration in a single operation, which is the most common use case
        when creating Python-accessible Mojo types.

        Args:
            module: The Python module to which the finalized type will be added.
                   The type will be accessible from Python code that imports
                   this module using the name specified during builder construction.

        Raises:
            If the type object creation fails (see `finalize()` for details) or
            if adding the type to the module fails, typically due to name conflicts
            or module state issues.

        Note:
            After calling this method, the builder's internal state is modified
            (methods list is cleared), so the builder should not be reused for
            creating additional type objects. If you need the type object for
            further operations, use the parameterless `finalize()` method instead
            and manually add it to the module.
        """
        var type_obj = self.finalize()
        Python.add_object(module, self.type_name, type_obj)
        self.methods.clear()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn def_py_c_method(
        mut self,
        method: PyCFunction,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PyObjectPtr signature for the
        type.

        Args:
            method: The method to declare a binding for.
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """
        self.methods.append(
            PyMethodDef.function(method, method_name, docstring)
        )
        return self

    fn def_py_method[
        method: PyFunction
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PyObject signature for the type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        return self.def_py_c_method(
            _py_c_function_wrapper[method], method_name, docstring
        )

    fn def_py_method[
        method: PyFunctionRaising
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PyObject signature for the type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        return self.def_py_c_method(
            _py_c_function_wrapper[method], method_name, docstring
        )

    # ===-------------------------------------------------------------------===#
    # def_method with return, raising
    # ===-------------------------------------------------------------------===#

    # TODO: declare these as a single method using variadics
    fn def_method[
        method: fn (mut PythonObject) raises -> PythonObject
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut py_args: TypedPythonObject["Tuple"]
        ) raises -> PythonObject:
            check_arguments_arity(0, py_args)
            return method(py_self)

        return self.def_py_method[wrapper](method_name, docstring)

    fn def_method[
        method: fn (mut PythonObject, mut PythonObject) raises -> PythonObject
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut py_args: TypedPythonObject["Tuple"]
        ) raises -> PythonObject:
            check_arguments_arity(1, py_args)
            var a0 = py_args[0]
            return method(py_self, a0)

        return self.def_py_method[wrapper](method_name, docstring)

    fn def_method[
        method: fn (
            mut PythonObject, mut PythonObject, mut PythonObject
        ) raises -> PythonObject
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut py_args: TypedPythonObject["Tuple"]
        ) raises -> PythonObject:
            check_arguments_arity(2, py_args)
            var a0 = py_args[0]
            var a1 = py_args[1]
            return method(py_self, a0, a1)

        return self.def_py_method[wrapper](method_name, docstring)

    # ===-------------------------------------------------------------------===#
    # def_method with return, not raising
    # ===-------------------------------------------------------------------===#

    # TODO: declare these as a single method using variadics
    fn def_method[
        method: fn (mut PythonObject) -> PythonObject
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(mut py_self: PythonObject) raises -> PythonObject:
            return method(py_self)

        return self.def_method[wrapper](method_name, docstring)

    fn def_method[
        method: fn (mut PythonObject, mut PythonObject) -> PythonObject
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut a0: PythonObject
        ) raises -> PythonObject:
            return method(py_self, a0)

        return self.def_method[wrapper](method_name, docstring)

    fn def_method[
        method: fn (
            mut PythonObject, mut PythonObject, mut PythonObject
        ) -> PythonObject
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject,
            mut a0: PythonObject,
            mut a1: PythonObject,
        ) raises -> PythonObject:
            return method(py_self, a0, a1)

        return self.def_method[wrapper](method_name, docstring)

    # ===-------------------------------------------------------------------===#
    # def_method with no return, raising
    # ===-------------------------------------------------------------------===#

    # TODO: declare these as a single method using variadics
    fn def_method[
        method: fn (mut PythonObject) raises
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(mut py_self: PythonObject) raises -> PythonObject:
            method(py_self)
            return PythonObject(None)

        return self.def_method[wrapper](method_name, docstring)

    fn def_method[
        method: fn (mut PythonObject, mut PythonObject) raises
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut a0: PythonObject
        ) raises -> PythonObject:
            method(py_self, a0)
            return PythonObject(None)

        return self.def_method[wrapper](method_name, docstring)

    fn def_method[
        method: fn (mut PythonObject, mut PythonObject, mut PythonObject) raises
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject,
            mut a0: PythonObject,
            mut a1: PythonObject,
        ) raises -> PythonObject:
            method(py_self, a0, a1)
            return PythonObject(None)

        return self.def_method[wrapper](method_name, docstring)

    # ===-------------------------------------------------------------------===#
    # def_method with no return, not raising
    # ===-------------------------------------------------------------------===#

    # TODO: declare these as a single method using variadics
    fn def_method[
        method: fn (mut PythonObject)
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(mut py_self: PythonObject) raises:
            method(py_self)

        return self.def_method[wrapper](method_name, docstring)

    fn def_method[
        method: fn (mut PythonObject, mut PythonObject)
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(mut py_self: PythonObject, mut a0: PythonObject) raises:
            method(py_self, a0)

        return self.def_method[wrapper](method_name, docstring)

    fn def_method[
        method: fn (mut PythonObject, mut PythonObject, mut PythonObject)
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        Parameters:
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject,
            mut a0: PythonObject,
            mut a1: PythonObject,
        ) raises:
            method(py_self, a0, a1)

        return self.def_method[wrapper](method_name, docstring)


# ===-----------------------------------------------------------------------===#
# PyCFunction Wrappers
# ===-----------------------------------------------------------------------===#


fn _py_c_function_wrapper[
    user_func: PyFunction
](py_self_ptr: PyObjectPtr, args_ptr: PyObjectPtr) -> PyObjectPtr:
    """Wrapper function that adapts a Mojo `PyFunction` to be callable from
    Python.

    This function creates a bridge between Python's C API calling convention
    and Mojo's `PyFunction` signature. It handles the conversion of raw Python
    object pointers to typed Mojo objects, calls the user function, and
    properly manages object lifetimes to prevent reference counting issues.

    The instantiated type of this generic function is a `PyCFunction`,
    suitable for being called from Python's C extension mechanism.

    Parameters:
        user_func: The Mojo function to wrap. Must have signature
                  `fn(PythonObject, TypedPythonObject["Tuple"]) -> PythonObject`.

    Args:
        py_self_ptr: Pointer to the Python object representing 'self' in the
                    method call. This is borrowed from the caller.
        args_ptr: Pointer to a Python tuple containing the positional arguments
                 passed to the function. This is borrowed from the caller.

    Returns:
        A new Python object pointer containing the result of the user function,
        or a null pointer if an error occurred during execution.

    Note:
        This function carefully manages object ownership according to Python's
        reference counting rules. The input arguments are borrowed references
        that must not be decremented, while the return value is a new reference
        that the caller will own.
    """

    #   > When a C function is called from Python, it borrows references to its
    #   > arguments from the caller. The caller owns a reference to the object,
    #   > so the read-only reference’s lifetime is guaranteed until the function
    #   > returns. Only when such a read-only reference must be stored or passed
    #   > on, it must be turned into an owned reference by calling Py_INCREF().
    #   >
    #   >  -- https://docs.python.org/3/extending/extending.html#ownership-rules

    # SAFETY:
    #   Here we illegally (but carefully) construct _owned_ `PythonObject`
    #   values from the read-only object reference arguments. We are careful
    #   down below to prevent the destructor for these objects from running
    #   so that we do not illegally decrement the reference count of these
    #   objects we do not own.
    #
    #   This is valid to do, because these are passed using the `read-only`
    #   argument convention to `user_func`, so logically they are treated
    #   as Python read-only references.
    var py_self = PythonObject(from_owned_ptr=py_self_ptr)
    var args = TypedPythonObject["Tuple"](
        unsafe_unchecked_from=PythonObject(from_owned_ptr=args_ptr)
    )

    # SAFETY:
    #   Call the user provided function, and take ownership of the
    #   PyObjectPtr of the returned PythonObject.
    var result = user_func(py_self, args).steal_data()

    # Do not destroy the provided PyObjectPtr arguments, since they
    # actually have ownership of the underlying object.
    __disable_del py_self

    # SAFETY:
    #   Prevent `args` AND `args._obj` from being destroyed, since we don't
    #   own them.
    var _obj = args._obj^
    __disable_del args
    __disable_del _obj
    return result


# Wrap a `raises` function
fn _py_c_function_wrapper[
    user_func: PyFunctionRaising
](py_self_ptr: PyObjectPtr, py_args_ptr: PyObjectPtr) -> PyObjectPtr:
    """Create a Python C API compatible wrapper for a Mojo function that can raise exceptions.

    This function wraps a Mojo function that follows the `PyFunctionRaising` signature
    (can raise exceptions) and makes it compatible with Python's C API calling convention.

    Parameters:
        user_func: The Mojo function to wrap. Must follow the `PyFunctionRaising`
                  signature: `fn(PythonObject, TypedPythonObject["Tuple"]) raises -> PythonObject`

    Args:
        py_self_ptr: Pointer to the Python object representing 'self' (borrowed reference).
        py_args_ptr: Pointer to a Python tuple containing the function arguments (borrowed reference).

    Returns:
        A new Python object pointer containing the function result, or NULL if an exception occurred.
        The caller takes ownership of the returned reference.
    """

    fn wrapper(
        mut py_self: PythonObject, mut args: TypedPythonObject["Tuple"]
    ) -> PythonObject:
        var cpython = Python().cpython()

        var state = cpython.PyGILState_Ensure()

        try:
            var result = user_func(py_self, args)
            return result
        except e:
            # TODO(MSTDL-933): Add custom 'MojoError' type, and raise it here.
            var error_type = cpython.get_error_global("PyExc_Exception")

            cpython.PyErr_SetString(
                error_type,
                e.unsafe_cstr_ptr(),
            )

            # Return a NULL `PyObject*`.
            return PythonObject(from_owned_ptr=PyObjectPtr())
        finally:
            cpython.PyGILState_Release(state)

    # TODO:
    #   Does this lead to multiple levels of indirect function calls for
    #   `raises` functions? Could we fix that by marking `wrapper` here as
    #   `@always_inline`?
    # Call the non-`raises` overload of `_py_c_function_wrapper`.
    return _py_c_function_wrapper[wrapper](py_self_ptr, py_args_ptr)


fn check_arguments_arity(
    arity: Int,
    args: TypedPythonObject["Tuple"],
) raises:
    """Validate that the provided arguments match the expected function arity.

    This function checks if the number of arguments in the provided tuple matches
    the expected arity for a function call. If the counts don't match, it raises
    a descriptive error message similar to Python's built-in TypeError messages.

    Args:
        arity: The expected number of arguments for the function.
        args: A tuple containing the actual arguments passed to the function.

    Raises:
        Error: If the argument count doesn't match the expected arity. The error
               message follows Python's convention for TypeError messages, indicating
               whether too few or too many arguments were provided.
    """
    # TODO: try to extract the current function name from cpython
    return check_arguments_arity(arity, args, "<mojo function>")


fn check_arguments_arity(
    arity: Int,
    args: TypedPythonObject["Tuple"],
    func_name: StringSlice,
) raises:
    """Validate that the provided arguments match the expected function arity.

    This function checks if the number of arguments in the provided tuple matches
    the expected arity for a function call. If the counts don't match, it raises
    a descriptive error message similar to Python's built-in TypeError messages.

    Args:
        arity: The expected number of arguments for the function.
        args: A tuple containing the actual arguments passed to the function.
        func_name: The name of the function being called, used in error messages
                  to provide better debugging information.

    Raises:
        Error: If the argument count doesn't match the expected arity. The error
               message follows Python's convention for TypeError messages, indicating
               whether too few or too many arguments were provided, along with the
               specific function name.
    """

    var arg_count = len(args)

    # The error messages raised below are intended to be similar to the
    # equivalent errors in Python.
    if arg_count != arity:
        if arg_count < arity:
            var missing_arg_count = arity - arg_count

            raise Error(
                String.format(
                    "TypeError: {}() missing {} required positional {}",
                    func_name,
                    missing_arg_count,
                    _pluralize(missing_arg_count, "argument", "arguments"),
                )
            )
        else:
            raise Error(
                String.format(
                    "TypeError: {}() takes {} positional {} but {} were given",
                    func_name,
                    arity,
                    _pluralize(arity, "argument", "arguments"),
                    arg_count,
                )
            )


fn _get_type_name(obj: PythonObject) raises -> String:
    var cpython = Python().cpython()

    var actual_type = cpython.Py_TYPE(obj.unsafe_as_py_object_ptr())
    var actual_type_name = PythonObject(
        from_owned_ptr=cpython.PyType_GetName(actual_type)
    )

    return String(actual_type_name)


fn _pluralize(
    count: Int,
    singular: StaticString,
    plural: StaticString,
) -> StaticString:
    if count == 1:
        return singular
    else:
        return plural

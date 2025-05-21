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

from collections import Optional
from collections.string.string_slice import get_static_string
from os import abort
from sys.ffi import c_int
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
    """Storage backing a PyObject* wrapping a Mojo value."""

    var ob_base: PyObject
    var mojo_value: T


fn default_tp_new_wrapper[
    T: Defaultable
](
    subtype: UnsafePointer[PyTypeObject],
    args: TypedPythonObject["Tuple"],
    keyword_args: PyObjectPtr,
) -> PyObjectPtr:
    """Python-compatible wrapper around a Mojo initializer function.

    Parameters:
        T: The wrapped Mojo type.
    """

    var cpython = Python().cpython()

    # Allocates and zero-initializes the new object.
    # For some objects, zeroed values are valid. But that isn't guaranteed
    # for any given Mojo object, so we further call `T`'s default initializer.
    var py_self = cpython.PyType_GenericAlloc(subtype, 0)

    # If we failed to allocate, return NULL.
    if not py_self.unsized_obj_ptr:
        return py_self

    try:
        if len(args) != 0 or keyword_args != PyObjectPtr():
            raise "unexpected arguments passed to default initializer function of wrapped Mojo type"

        var obj_ptr: UnsafePointer[T] = py_self.unchecked_cast_to_mojo_value[
            T
        ]()

        # Call the user-provided initialization function on uninit memory.
        __get_address_as_uninit_lvalue(obj_ptr.address) = T()
        return py_self

    except e:
        # Free the object memory we just allocated but failed to initialize.
        cpython.PyObject_Free(py_self.unsized_obj_ptr.bitcast[NoneType]())

        # TODO(MSTDL-933): Add custom 'MojoError' type, and raise it here.
        var error_type = cpython.get_error_global("PyExc_ValueError")
        cpython.PyErr_SetString(
            error_type,
            e.unsafe_cstr_ptr(),
        )
        return PyObjectPtr()


fn tp_dealloc_wrapper[T: Defaultable & Representable](py_self: PyObjectPtr):
    var self_ptr: UnsafePointer[T] = py_self.unchecked_cast_to_mojo_value[T]()

    # TODO(MSTDL-633):
    #   Is this always safe? Wrap in GIL, because this could
    #   evaluate arbitrary code?
    # Destroy this `Person` instance.
    self_ptr.destroy_pointee()

    var cpython = Python().cpython()

    cpython.PyObject_Free(py_self.unsized_obj_ptr.bitcast[NoneType]())


fn tp_repr_wrapper[
    T: Defaultable & Representable
](py_self: PyObjectPtr) -> PyObjectPtr:
    var self_ptr: UnsafePointer[T] = py_self.unchecked_cast_to_mojo_value[T]()

    var repr_str: String = repr(self_ptr[])

    return PythonObject(string=repr_str).steal_data()


# ===-----------------------------------------------------------------------===#
# Builders
# ===-----------------------------------------------------------------------===#


struct PythonModuleBuilder:
    """API for declaring and creating Python bindings for a module."""

    var module: PythonModule
    var functions: List[PyMethodDef]
    var type_builders: List[PythonTypeBuilder]

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
        T: Defaultable & Representable
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
            py_c_function_wrapper[func], func_name, docstring
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
            py_c_function_wrapper[func], func_name, docstring
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
        in the module.

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
    """

    var type_name: StaticString
    var basicsize: Int
    var _slots: List[PyType_Slot]
    var methods: List[PyMethodDef]

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
        self.basicsize = basicsize
        self._slots = List[PyType_Slot]()
        self.methods = List[PyMethodDef]()

    @staticmethod
    fn bind[
        T: Defaultable & Representable
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
            PyType_Slot.tp_new(default_tp_new_wrapper[T]),
            PyType_Slot.tp_dealloc(tp_dealloc_wrapper[T]),
            PyType_Slot.tp_repr(tp_repr_wrapper[T]),
        )
        b.methods = List[PyMethodDef]()

        return b^

    fn finalize(mut self) raises -> TypedPythonObject["Type"]:
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

        return TypedPythonObject["Type"](
            unsafe_unchecked_from=PythonObject(from_owned_ptr=type_obj)
        )

    fn finalize(mut self, module: PythonModule) raises:
        """Finalize the builder, creating the type binding with the registered
        methods.

        Raises:
            If we fail to add the type to the module.
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
            py_c_function_wrapper[method], method_name, docstring
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
            py_c_function_wrapper[method], method_name, docstring
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


fn py_c_function_wrapper[
    user_func: PyFunction
](py_self_ptr: PyObjectPtr, args_ptr: PyObjectPtr) -> PyObjectPtr:
    """The instantiated type of this generic function is a `PyCFunction`,
    suitable for being called from Python.
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
fn py_c_function_wrapper[
    user_func: PyFunctionRaising
](py_self_ptr: PyObjectPtr, py_args_ptr: PyObjectPtr) -> PyObjectPtr:
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
    # Call the non-`raises` overload of `py_c_function_wrapper`.
    return py_c_function_wrapper[wrapper](py_self_ptr, py_args_ptr)


fn check_arguments_arity(
    arity: Int,
    args: TypedPythonObject["Tuple"],
) raises:
    """Raise an error if the provided argument count does not match the expected
    function arity.

    If this function returns normally (without raising), then the argument
    count is exactly equal to the expected arity.
    """
    # TODO: try to extract the current function name from cpython
    return check_arguments_arity(arity, args, "<mojo function>")


fn check_arguments_arity(
    arity: Int,
    args: TypedPythonObject["Tuple"],
    func_name: StringSlice,
) raises:
    """Raise an error if the provided argument count does not match the expected
    function arity.

    If this function returns normally (without raising), then the argument
    count is exactly equal to the expected arity.
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


fn check_argument_type[
    T: AnyType
](
    func_name: StaticString,
    type_name_id: StaticString,
    obj: PythonObject,
) raises -> UnsafePointer[T]:
    """Raise an error if the provided Python object does not contain a wrapped
    instance of the Mojo `T` type.
    """

    var opt: Optional[UnsafePointer[T]] = obj.py_object.try_cast_to_mojo_value[
        T
    ](type_name_id)

    if not opt:
        raise Error(
            String.format(
                "TypeError: {}() expected Mojo '{}' type argument, got '{}'",
                func_name,
                type_name_id,
                _get_type_name(obj),
            )
        )

    # SAFETY: We just validated that this Optional is not empty.
    return opt.unsafe_take()


fn _pluralize(
    count: Int,
    singular: StaticString,
    plural: StaticString,
) -> StaticString:
    if count == 1:
        return singular
    else:
        return plural

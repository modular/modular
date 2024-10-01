# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import builtin

from sys import exit
from sys.info import sizeof
from sys.ffi import OpaquePointer, c_int
from memory import UnsafePointer

from os import abort

from python import Python, PythonObject, TypedPythonObject
from python._cpython import (
    PyMethodDef,
    PyObjectPtr,
    create_wrapper_function,
    PyType_Spec,
    PyObject,
    Py_TPFLAGS_DEFAULT,
    PyType_Slot,
    Py_tp_new,
    Py_tp_init,
    Py_tp_dealloc,
    Py_tp_methods,
)


@export
fn PyInit_feature_overview() -> PythonObject:
    # Initialize the global runtime (including the memory allocator)
    _ = builtin._startup._init_global_runtime(UnsafePointer[NoneType]())

    # ----------------------------------
    # Create a Python module
    # ----------------------------------

    # This will initialize the Python interpreter and create
    # an extension module with the provided name.
    var module: TypedPythonObject["Module"]

    try:
        module = Python.create_module("bindings")
    except e:
        return abort[PythonObject]("failed to create Python module: " + str(e))

    # ----------------------------------
    # Populate the Python module
    # ----------------------------------

    var funcs = List[PyMethodDef](
        PyMethodDef.function[
            create_wrapper_function[case_return_arg_tuple](),
            "case_return_arg_tuple",
        ](),
        PyMethodDef.function[
            create_wrapper_function[case_raise_empty_error](),
            "case_raise_empty_error",
        ](),
        PyMethodDef.function[
            create_wrapper_function[case_raise_string_error](),
            "case_raise_string_error",
        ](),
        PyMethodDef.function[
            create_wrapper_function[case_mojo_raise](),
            "case_mojo_raise",
        ](),
    )

    try:
        Python.add_functions(module, funcs)
    except e:
        abort("Error adding functions to PyModule: " + str(e))

    add_person_type(module)

    return module


# ===----------------------------------------------------------------------=== #
# Functions
# ===----------------------------------------------------------------------=== #


@export
fn case_return_arg_tuple(
    py_self: PythonObject,
    args: TypedPythonObject["Tuple"],
) -> PythonObject:
    var cpython = Python().impl.cpython()

    return args


@export
fn case_raise_empty_error(
    py_self: PythonObject,
    args: TypedPythonObject["Tuple"],
) -> PythonObject:
    var cpython = Python().impl.cpython()

    var error_type = cpython.get_error_global("PyExc_ValueError")

    cpython.PyErr_SetNone(error_type)

    return PythonObject(PyObjectPtr())


@export
fn case_raise_string_error(
    py_self: PythonObject,
    args: TypedPythonObject["Tuple"],
) -> PythonObject:
    var cpython = Python().impl.cpython()

    var error_type = cpython.get_error_global("PyExc_ValueError")

    cpython.PyErr_SetString(error_type, "sample value error".unsafe_cstr_ptr())

    return PythonObject(PyObjectPtr())


# Tests `create_wrapper_function()` of a `raises` function.
@export
fn case_mojo_raise(
    py_self: PythonObject,
    args: TypedPythonObject["Tuple"],
) raises -> PythonObject:
    raise "Mojo error"


# ===----------------------------------------------------------------------=== #
# Custom Types
# ===----------------------------------------------------------------------=== #


@value
struct Person:
    var name: String
    var age: Int


struct MojoPersonObject:
    var ob_base: PyObject
    var person: Person

    @staticmethod
    fn obj_init(
        self_: PyObjectPtr,
        args: TypedPythonObject["Tuple"],
        kwds: PythonObject,
    ) -> c_int:
        var self0 = self_.value.bitcast[MojoPersonObject]()

        # Field ptrs
        var name_ptr = UnsafePointer[String].address_of(self0[].person.name)
        var age_ptr = UnsafePointer[Int].address_of(self0[].person.age)

        name_ptr.init_pointee_move("John Smith")
        age_ptr.init_pointee_move(123)

        return 0

    @staticmethod
    fn obj_destroy(self_: PyObjectPtr):
        var obj0: UnsafePointer[MojoPersonObject] = self_.value.bitcast[
            MojoPersonObject
        ]()

        # TODO(MSTDL-633):
        #   Is this always safe? Wrap in GIL, because this could
        #   evaluate arbitrary code?
        # Destroy this `Person` instance.
        obj0.destroy_pointee()

    @staticmethod
    fn obj_name(
        self_: PythonObject, args: TypedPythonObject["Tuple"]
    ) -> PythonObject:
        var self0 = self_.unsafe_as_py_object_ptr().value.bitcast[
            MojoPersonObject
        ]()

        return PythonObject(self0[].person.name).steal_data()


fn add_person_type(inout module: TypedPythonObject["Module"]):
    var cpython = Python().impl.cpython()

    # ----------------------------------------------
    # Construct a 'type' object describing `Person`
    # ----------------------------------------------

    var methods = List[PyMethodDef](
        PyMethodDef.function[
            create_wrapper_function[MojoPersonObject.obj_name](),
            "name",
        ](),
        # Zeroed item as terminator
        PyMethodDef(),
    )

    var slots = List[PyType_Slot](
        PyType_Slot(
            Py_tp_new, cpython.lib.get_symbol[NoneType]("PyType_GenericNew")
        ),
        PyType_Slot(
            Py_tp_init, rebind[OpaquePointer](MojoPersonObject.obj_init)
        ),
        PyType_Slot(
            Py_tp_dealloc, rebind[OpaquePointer](MojoPersonObject.obj_destroy)
        ),
        PyType_Slot(Py_tp_methods, rebind[OpaquePointer](methods.steal_data())),
        PyType_Slot.null(),
    )

    var type_spec = PyType_Spec {
        name: "Person".unsafe_cstr_ptr(),
        basicsize: sizeof[MojoPersonObject](),
        itemsize: 0,
        flags: Py_TPFLAGS_DEFAULT,
        # FIXME: Don't leak this pointer, use globals instead.
        slots: slots.steal_data(),
    }

    # Heap allocate the type specification metadata.
    var type_spec_ptr = UnsafePointer[PyType_Spec].alloc(1)
    type_spec_ptr.init_pointee_move(type_spec)

    # Construct a Python 'type' object from our Person type spec.
    var type_obj = cpython.PyType_FromSpec(type_spec_ptr)

    # ----------------------------------
    # Register the type in the module
    # ----------------------------------

    try:
        Python.add_object(module, "Person", type_obj)
    except e:
        abort("error adding object: " + str(e))

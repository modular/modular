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
"""
Mojo bindings functions and types from the CPython C API.

Documentation for these functions can be found online at:
  <https://docs.python.org/3/c-api/stable.html#contents-of-limited-api>
"""

from collections import InlineArray
from os import abort, getenv, setenv
from os.path import dirname
from pathlib import Path
from sys import external_call
from sys.arg import argv
from sys.ffi import (
    DLHandle,
    c_char,
    c_double,
    c_int,
    c_long,
    c_size_t,
    c_ssize_t,
    c_uint,
)

from utils import Variant

alias Py_ssize_t = c_ssize_t
alias Py_hash_t = Py_ssize_t

# ===-----------------------------------------------------------------------===#
# Raw Bindings
# ===-----------------------------------------------------------------------===#

# ref: https://github.com/python/cpython/blob/main/Include/compile.h
alias Py_single_input: c_int = 256
alias Py_file_input: c_int = 257
alias Py_eval_input: c_int = 258
alias Py_func_type_input: c_int = 345

# 0 when Stackless Python is disabled
# ref: https://github.com/python/cpython/blob/main/Include/object.h
alias Py_TPFLAGS_DEFAULT = 0


# TODO(MOCO-1138):
#   This should be a C ABI function pointer, not a Mojo ABI function.
# ref: https://docs.python.org/3/c-api/structures.html#c.PyCFunction
alias PyCFunction = fn (PyObjectPtr, PyObjectPtr) -> PyObjectPtr
alias PyCFunctionWithKeywords = fn (
    PyObjectPtr, PyObjectPtr, PyObjectPtr
) -> PyObjectPtr

# Flag passed to newmethodobject
# ref: https://github.com/python/cpython/blob/main/Include/methodobject.h
alias METH_VARARGS = 0x01
alias METH_KEYWORDS = 0x02
alias METH_STATIC = 0x20


# GIL
@fieldwise_init
@register_passable("trivial")
struct PyGILState_STATE:
    """Represents the state of the Python Global Interpreter Lock (GIL).

    This struct is used to store and manage the state of the GIL, which is
    crucial for thread-safe operations in Python.

    References:
    - https://github.com/python/cpython/blob/d45225bd66a8123e4a30314c627f2586293ba532/Include/pystate.h#L76
    """

    # typedef enum {
    #   PyGILState_LOCKED, PyGILState_UNLOCKED
    # } PyGILState_STATE;

    var current_state: c_int
    """The current state of the GIL."""

    alias PyGILState_LOCKED = c_int(0)
    alias PyGILState_UNLOCKED = c_int(1)


struct PyThreadState:
    """This data structure represents the state of a single thread.

    It's an opaque struct.

    References:
    - https://docs.python.org/3/c-api/init.html#c.PyThreadState
    """

    # TODO: add this public data member
    # PyInterpreterState *interp
    pass


@fieldwise_init
@register_passable("trivial")
struct PyObjectPtr(
    Boolable,
    Defaultable,
    EqualityComparable,
    ImplicitlyCopyable,
    Intable,
    Movable,
    Stringable,
    Writable,
):
    """Equivalent to `PyObject*` in C.

    It is crucial that this type has the same size and alignment as `PyObject*`
    for FFI ABI correctness.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _unsized_obj_ptr: UnsafePointer[PyObject]
    """Raw pointer to the underlying PyObject struct instance.

    It is not valid to read or write a `PyObject` directly from this pointer.

    This is because `PyObject` is an "unsized" or "incomplete" type: typically,
    any allocation containing a `PyObject` contains additional fields holding
    information specific to that Python object instance, e.g. containing its
    "true" value.

    The value behind this pointer is only safe to interact with directly when
    it has been downcasted to a concrete Python object type backing struct, in
    a context where the user has ensured the object value is of that type.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        """Initialize a null PyObjectPtr."""
        self._unsized_obj_ptr = {}

    @always_inline
    fn __init__[T: AnyType, //](out self, *, upcast_from: UnsafePointer[T]):
        self._unsized_obj_ptr = upcast_from.bitcast[PyObject]()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __eq__(self, rhs: Self) -> Bool:
        """Compare two PyObjectPtr for equality.

        Args:
            rhs: The right-hand side PyObjectPtr to compare.

        Returns:
            Bool: True if the pointers are equal, False otherwise.
        """
        return self._unsized_obj_ptr == rhs._unsized_obj_ptr

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __bool__(self) -> Bool:
        return Bool(self._unsized_obj_ptr)

    @always_inline
    fn __int__(self) -> Int:
        return Int(self._unsized_obj_ptr)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn bitcast[T: AnyType](self) -> UnsafePointer[T]:
        """Bitcasts the `PyObjectPtr` to a pointer of type `T`.

        Parameters:
            T: The target type to cast to.

        Returns:
            A pointer to the underlying object as type `T`.
        """
        return self._unsized_obj_ptr.bitcast[T]()

    fn write_to(self, mut writer: Some[Writer]):
        """Formats to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write(self._unsized_obj_ptr)


@fieldwise_init
@register_passable
struct PythonVersion(ImplicitlyCopyable, Movable):
    """Represents a Python version with major, minor, and patch numbers."""

    var major: Int
    """The major version number."""
    var minor: Int
    """The minor version number."""
    var patch: Int
    """The patch version number."""

    fn __init__(out self, version: StringSlice):
        """Initialize a PythonVersion object from a version string.

        Args:
            version: A string representing the Python version (e.g., "3.9.5").

        The version string is parsed to extract major, minor, and patch numbers.
        If parsing fails for any component, it defaults to -1.
        """
        var components = InlineArray[Int, 3](fill=-1)
        var start = 0
        var next_idx = 0
        var i = 0
        while next_idx < len(version) and i < 3:
            if version[next_idx] == "." or (
                version[next_idx] == " " and i == 2
            ):
                var c = version[start:next_idx]
                try:
                    components[i] = atol(c)
                except:
                    components[i] = -1
                i += 1
                start = next_idx + 1
            next_idx += 1
        self = PythonVersion(components[0], components[1], components[2])


fn _py_get_version(lib: DLHandle) -> StaticString:
    return StaticString(
        unsafe_from_utf8_ptr=lib.call[
            "Py_GetVersion",
            UnsafePointer[c_char, mut=False, origin=StaticConstantOrigin],
        ]()
    )


@fieldwise_init
struct PyMethodDef(Defaultable, ImplicitlyCopyable, Movable):
    """Represents a Python method definition. This struct is used to define
    methods for Python modules or types.

    Notes:
        [Reference](
        https://docs.python.org/3/c-api/structures.html#c.PyMethodDef
        ).
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var method_name: UnsafePointer[
        c_char, mut=False, origin=StaticConstantOrigin
    ]
    """A pointer to the name of the method as a C string.

    Notes:
        called `ml_name` in CPython.
    """

    var method_impl: OpaquePointer
    """A function pointer to the implementation of the method."""

    var method_flags: c_int
    """Flags indicating how the method should be called.

    References:
    - https://docs.python.org/3/c-api/structures.html#c.PyMethodDef"""

    var method_docstring: UnsafePointer[
        c_char, mut=False, origin=StaticConstantOrigin
    ]
    """The docstring for the method."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Constructs a zero initialized PyModuleDef.

        This is suitable for use terminating an array of PyMethodDef values.
        """
        self.method_name = UnsafePointer[c_char]()
        self.method_impl = OpaquePointer()
        self.method_flags = 0
        self.method_docstring = UnsafePointer[c_char]()

    @staticmethod
    fn function[
        static_method: Bool = False
    ](
        func: Variant[PyCFunction, PyCFunctionWithKeywords],
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> Self:
        """Create a PyMethodDef for a function.

        Parameters:
            static_method: Whether the function is a static method. Default is
                False.

        Arguments:
            func: The function to wrap.
            func_name: The name of the function.
            docstring: The docstring for the function.
        """
        # TODO(MSTDL-896):
        #   Support a way to get the name of the function from its parameter
        #   type, similar to `get_linkage_name()`?

        var with_kwargs = func.isa[PyCFunctionWithKeywords]()
        var func_ptr = rebind[OpaquePointer](
            func[PyCFunctionWithKeywords]
        ) if with_kwargs else rebind[OpaquePointer](func[PyCFunction])

        var flags = (
            METH_VARARGS
            | (METH_STATIC if static_method else 0)
            | (METH_KEYWORDS if with_kwargs else 0)
        )
        return PyMethodDef(
            func_name.unsafe_ptr().bitcast[c_char](),
            func_ptr,
            flags,
            docstring.unsafe_ptr().bitcast[c_char](),
        )


fn _null_fn_ptr[T: AnyTrivialRegType]() -> T:
    return __mlir_op.`pop.pointer.bitcast`[_type=T](
        __mlir_attr.`#interp.pointer<0> : !kgen.pointer<none>`
    )


alias PyTypeObjectPtr = UnsafePointer[PyTypeObject]


struct PyTypeObject:
    """The opaque C structure of the objects used to describe types.

    References:
    - https://docs.python.org/3/c-api/type.html#c.PyTypeObject
    """

    # TODO(MSTDL-877):
    #   Fill this out based on
    #   https://docs.python.org/3/c-api/typeobj.html#pytypeobject-definition
    pass


@fieldwise_init
@register_passable("trivial")
struct PyType_Spec:
    """Structure defining a type's behavior.

    References:
    - https://docs.python.org/3/c-api/type.html#c.PyType_Spec
    """

    var name: UnsafePointer[c_char, mut=False]
    var basicsize: c_int
    var itemsize: c_int
    var flags: c_uint
    var slots: UnsafePointer[PyType_Slot]


# https://github.com/python/cpython/blob/main/Include/typeslots.h
alias Py_tp_dealloc = 52
alias Py_tp_init = 60
alias Py_tp_methods = 64
alias Py_tp_new = 65
alias Py_tp_repr = 66

# https://docs.python.org/3/c-api/typeobj.html#slot-type-typedefs

alias destructor = fn (PyObjectPtr) -> None
"""`typedef void (*destructor)(PyObject*)`"""
alias reprfunc = fn (PyObjectPtr) -> PyObjectPtr
"""`typedef PyObject *(*reprfunc)(PyObject*)`"""
alias Typed_initproc = fn (
    PyObjectPtr,
    PyObjectPtr,
    PyObjectPtr,  # NULL if no keyword arguments were passed
) -> c_int
"""`typedef int (*initproc)(PyObject*, PyObject*, PyObject*)`"""
alias Typed_newfunc = fn (
    PyTypeObjectPtr,
    PyObjectPtr,
    PyObjectPtr,
) -> PyObjectPtr
"""`typedef PyObject *(*newfunc)(PyTypeObject*, PyObject*, PyObject*)`"""


@fieldwise_init
@register_passable("trivial")
struct PyType_Slot(ImplicitlyCopyable, Movable):
    """Structure defining optional functionality of a type, containing a slot ID
    and a value pointer.

    References:
    - https://docs.python.org/3/c-api/type.html#c.PyType_Slot
    - https://docs.python.org/3/c-api/typeobj.html#type-object-structures
    """

    var slot: c_int
    var pfunc: OpaquePointer

    @staticmethod
    fn tp_dealloc(func: destructor) -> Self:
        return PyType_Slot(Py_tp_dealloc, rebind[OpaquePointer](func))

    @staticmethod
    fn tp_init(func: Typed_initproc) -> Self:
        return PyType_Slot(Py_tp_init, rebind[OpaquePointer](func))

    @staticmethod
    fn tp_methods(methods: UnsafePointer[PyMethodDef]) -> Self:
        return PyType_Slot(Py_tp_methods, rebind[OpaquePointer](methods))

    @staticmethod
    fn tp_new(func: Typed_newfunc) -> Self:
        return PyType_Slot(Py_tp_new, rebind[OpaquePointer](func))

    @staticmethod
    fn tp_repr(func: reprfunc) -> Self:
        return PyType_Slot(Py_tp_repr, rebind[OpaquePointer](func))

    @staticmethod
    fn null() -> Self:
        return PyType_Slot(0, OpaquePointer())


@fieldwise_init
struct PyObject(
    Defaultable,
    ImplicitlyCopyable,
    Movable,
    Representable,
    Stringable,
    Writable,
):
    """All object types are extensions of this type. This is a type which
    contains the information Python needs to treat a pointer to an object as an
    object. In a normal “release” build, it contains only the object's reference
    count and a pointer to the corresponding type object. Nothing is actually
    declared to be a PyObject, but every pointer to a Python object can be cast
    to a PyObject.

    References:
    - https://docs.python.org/3/c-api/structures.html#c.PyObject
    """

    var object_ref_count: Py_ssize_t
    var object_type: PyTypeObjectPtr

    fn __init__(out self):
        self.object_ref_count = 0
        self.object_type = {}

    @no_inline
    fn __str__(self) -> String:
        """Get the PyModuleDef_Base as a string.

        Returns:
            A string representation.
        """

        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the `PyObject` as a string. Returns the same `String` as
        `__str__`.

        Returns:
            A string representation.
        """
        return String(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn write_to(self, mut writer: Some[Writer]):
        """Formats to the provided Writer.

        Args:
            writer: The object to write to.
        """

        writer.write("PyObject(")
        writer.write("object_ref_count=", self.object_ref_count, ",")
        writer.write("object_type=", self.object_type)
        writer.write(")")


# Mojo doesn't have macros, so we define it here for ease.
struct PyModuleDef_Base(
    Defaultable, Movable, Representable, Stringable, Writable
):
    """PyModuleDef_Base.

    References:
    - https://github.com/python/cpython/blob/833c58b81ebec84dc24ef0507f8c75fe723d9f66/Include/moduleobject.h#L39
    - https://pyo3.rs/main/doc/pyo3/ffi/struct.pymoduledef_base
    - `PyModuleDef_HEAD_INIT` default inits all of its members (https://github.com/python/cpython/blob/833c58b81ebec84dc24ef0507f8c75fe723d9f66/Include/moduleobject.h#L60)
    """

    var object_base: PyObject
    """The initial segment of every `PyObject` in CPython."""

    # TODO(MOCO-1138): This is a C ABI function pointer, not Mojo a function.
    alias _init_fn_type = fn () -> PyObjectPtr
    var init_fn: Self._init_fn_type
    """The function used to re-initialize the module."""

    var index: Py_ssize_t
    """The module's index into its interpreter's `modules_by_index` cache."""

    var dict_copy: PyObjectPtr
    """A copy of the module's `__dict__` after the first time it was loaded."""

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    fn __init__(out self):
        self.object_base = {}
        self.init_fn = _null_fn_ptr[Self._init_fn_type]()
        self.index = 0
        self.dict_copy = {}

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @no_inline
    fn __str__(self) -> String:
        """Get the PyModuleDef_Base as a string.

        Returns:
            A string representation.
        """

        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the PyMdouleDef_Base as a string. Returns the same `String` as
        `__str__`.

        Returns:
            A string representation.
        """
        return String(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn write_to(self, mut writer: Some[Writer]):
        """Formats to the provided Writer.

        Args:
            writer: The object to write to.
        """

        writer.write("PyModuleDef_Base(")
        writer.write("object_base=", self.object_base, ",")
        writer.write("init_fn=<unprintable>", ",")
        writer.write("index=", self.index, ",")
        writer.write("dict_copy=", self.dict_copy)
        writer.write(")")


@fieldwise_init
struct PyModuleDef_Slot:
    """A struct representing a slot in the module definition.

    References:
    - https://docs.python.org/3/c-api/module.html#c.PyModuleDef_Slot
    """

    var slot: c_int
    var value: OpaquePointer


struct PyModuleDef(Movable, Representable, Stringable, Writable):
    """The Python module definition structs that holds all of the information
    needed to create a module.

    References:
    - https://docs.python.org/3/c-api/module.html#c.PyModuleDef
    """

    var base: PyModuleDef_Base

    var name: UnsafePointer[c_char, mut=False]
    """Name for the new module."""

    var docstring: UnsafePointer[c_char, mut=False]
    """Points to the contents of the docstring for the module."""

    var size: Py_ssize_t
    """Size of per-module data."""

    var methods: UnsafePointer[PyMethodDef]
    """A pointer to a table of module-level functions. Can be null if there
    are no functions present."""

    var slots: UnsafePointer[PyModuleDef_Slot]
    """An array of slot definitions for multi-phase initialization, terminated
    by a `{0, NULL}` entry."""

    # TODO(MOCO-1138): These are C ABI function pointers, not Mojo functions.
    alias _visitproc_fn_type = fn (PyObjectPtr, OpaquePointer) -> c_int
    alias _traverse_fn_type = fn (
        PyObjectPtr, Self._visitproc_fn_type, OpaquePointer
    ) -> c_int
    var traverse_fn: Self._traverse_fn_type
    """A traversal function to call during GC traversal of the module object,
    or `NULL` if not needed."""

    alias _clear_fn_type = fn (PyObjectPtr) -> c_int
    var clear_fn: Self._clear_fn_type
    """A clear function to call during GC clearing of the module object,
    or `NULL` if not needed."""

    alias _free_fn_type = fn (OpaquePointer) -> OpaquePointer
    var free_fn: Self._free_fn_type
    """A function to call during deallocation of the module object,
    or `NULL` if not needed."""

    fn __init__(out self, name: StaticString):
        self.base = {}
        self.name = name.unsafe_ptr().bitcast[c_char]()
        self.docstring = {}
        # setting `size` to -1 means that the module does not support sub-interpreters
        self.size = -1
        self.methods = {}
        self.slots = {}
        self.traverse_fn = _null_fn_ptr[Self._traverse_fn_type]()
        self.clear_fn = _null_fn_ptr[Self._clear_fn_type]()
        self.free_fn = _null_fn_ptr[Self._free_fn_type]()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @no_inline
    fn __str__(self) -> String:
        """Get the PyModuleDefe as a string.

        Returns:
            A string representation.
        """

        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the PyMdouleDef as a string. Returns the same `String` as
        `__str__`.

        Returns:
            A string representation.
        """
        return String(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn write_to(self, mut writer: Some[Writer]):
        """Formats to the provided Writer.

        Args:
            writer: The object to write to.
        """

        writer.write("PyModuleDef(")
        writer.write("base=", self.base, ",")
        writer.write("name=", self.name, ",")
        writer.write("docstring=", self.docstring, ",")
        writer.write("size=", self.size, ",")
        writer.write("methods=", self.methods, ",")
        writer.write("slots=", self.slots, ",")
        writer.write("traverse_fn=<unprintable>", ",")
        writer.write("clear_fn=<unprintable>", ",")
        writer.write("free_fn=<unprintable>")
        writer.write(")")


# ===-------------------------------------------------------------------===#
# CPython C API Functions
# ===-------------------------------------------------------------------===#


struct ExternalFunction[
    name: StaticString,
    type: AnyTrivialRegType,
]:
    @staticmethod
    @always_inline
    fn load(lib: DLHandle) -> type:
        """Loads this external function from an opened dynamic library."""
        return lib._get_function[name, type]()


# external functions for the CPython C API
# ordered based on https://docs.python.org/3/c-api/index.html

# The Very High Level Layer
alias PyRun_SimpleString = ExternalFunction[
    "PyRun_SimpleString",
    # int PyRun_SimpleString(const char *command)
    fn (UnsafePointer[c_char, mut=False]) -> c_int,
]
alias PyRun_String = ExternalFunction[
    "PyRun_String",
    # PyObject *PyRun_String(const char *str, int start, PyObject *globals, PyObject *locals)
    fn (
        UnsafePointer[c_char, mut=False],
        c_int,
        PyObjectPtr,
        PyObjectPtr,
    ) -> PyObjectPtr,
]
alias Py_CompileString = ExternalFunction[
    "Py_CompileString",
    # PyObject *Py_CompileString(const char *str, const char *filename, int start)
    fn (
        UnsafePointer[c_char, mut=False],
        UnsafePointer[c_char, mut=False],
        c_int,
    ) -> PyObjectPtr,
]
alias PyEval_EvalCode = ExternalFunction[
    "PyEval_EvalCode",
    # PyObject *PyEval_EvalCode(PyObject *co, PyObject *globals, PyObject *locals)
    fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> PyObjectPtr,
]

# Reference Counting
alias Py_IncRef = ExternalFunction[
    "Py_IncRef",
    # void Py_IncRef(PyObject *o)
    fn (PyObjectPtr) -> None,
]
alias Py_DecRef = ExternalFunction[
    "Py_DecRef",
    # void Py_DecRef(PyObject *o)
    fn (PyObjectPtr) -> None,
]

# Exception Handling
# - Printing and clearing
alias PyErr_Clear = ExternalFunction[
    "PyErr_Clear",
    # void PyErr_Clear()
    fn () -> None,
]
# - Raising exceptions
alias PyErr_SetString = ExternalFunction[
    "PyErr_SetString",
    # void PyErr_SetString(PyObject *type, const char *message)
    fn (PyObjectPtr, UnsafePointer[c_char, mut=False]) -> None,
]
alias PyErr_SetNone = ExternalFunction[
    "PyErr_SetNone",
    # void PyErr_SetNone(PyObject *type)
    fn (PyObjectPtr) -> None,
]
# - Querying the error indicator
alias PyErr_Occurred = ExternalFunction[
    "PyErr_Occurred",
    # PyObject *PyErr_Occurred()
    fn () -> PyObjectPtr,
]
alias PyErr_GetRaisedException = ExternalFunction[
    "PyErr_GetRaisedException",
    # PyObject *PyErr_GetRaisedException()
    fn () -> PyObjectPtr,
]
alias PyErr_Fetch = ExternalFunction[
    "PyErr_Fetch",
    # void PyErr_Fetch(PyObject **ptype, PyObject **pvalue, PyObject **ptraceback)
    fn (
        UnsafePointer[PyObjectPtr],
        UnsafePointer[PyObjectPtr],
        UnsafePointer[PyObjectPtr],
    ) -> None,
]

# Initialization, Finalization, and Threads
alias PyEval_SaveThread = ExternalFunction[
    "PyEval_SaveThread",
    # PyThreadState *PyEval_SaveThread()
    fn () -> UnsafePointer[PyThreadState],
]
alias PyEval_RestoreThread = ExternalFunction[
    "PyEval_RestoreThread",
    # void PyEval_RestoreThread(PyThreadState *tstate)
    fn (UnsafePointer[PyThreadState]) -> None,
]
alias PyGILState_Ensure = ExternalFunction[
    "PyGILState_Ensure",
    # PyGILState_STATE PyGILState_Ensure()
    fn () -> PyGILState_STATE,
]
alias PyGILState_Release = ExternalFunction[
    "PyGILState_Release",
    # void PyGILState_Release(PyGILState_STATE)
    fn (PyGILState_STATE) -> None,
]

# Importing Modules
alias PyImport_ImportModule = ExternalFunction[
    "PyImport_ImportModule",
    # PyObject *PyImport_ImportModule(const char *name)
    fn (UnsafePointer[c_char, mut=False]) -> PyObjectPtr,
]
alias PyImport_AddModule = ExternalFunction[
    "PyImport_AddModule",
    # PyObject *PyImport_AddModule(const char *name)
    fn (UnsafePointer[c_char, mut=False]) -> PyObjectPtr,
]

# Abstract Objects Layer
# Object Protocol
alias PyObject_HasAttrString = ExternalFunction[
    "PyObject_HasAttrString",
    # int PyObject_HasAttrString(PyObject *o, const char *attr_name)
    fn (PyObjectPtr, UnsafePointer[c_char, mut=False]) -> c_int,
]
alias PyObject_GetAttrString = ExternalFunction[
    "PyObject_GetAttrString",
    # PyObject *PyObject_GetAttrString(PyObject *o, const char *attr_name)
    fn (PyObjectPtr, UnsafePointer[c_char, mut=False]) -> PyObjectPtr,
]
alias PyObject_SetAttrString = ExternalFunction[
    "PyObject_SetAttrString",
    # int PyObject_SetAttrString(PyObject *o, const char *attr_name, PyObject *v)
    fn (PyObjectPtr, UnsafePointer[c_char, mut=False], PyObjectPtr) -> c_int,
]
alias PyObject_Str = ExternalFunction[
    "PyObject_Str",
    # PyObject *PyObject_Str(PyObject *o)
    fn (PyObjectPtr) -> PyObjectPtr,
]
alias PyObject_Hash = ExternalFunction[
    "PyObject_Hash",
    # Py_hash_t PyObject_Hash(PyObject *o)
    fn (PyObjectPtr) -> Py_hash_t,
]
alias PyObject_IsTrue = ExternalFunction[
    "PyObject_IsTrue",
    # int PyObject_IsTrue(PyObject *o)
    fn (PyObjectPtr) -> c_int,
]
alias PyObject_Type = ExternalFunction[
    "PyObject_Type",
    # PyTypeObject *PyObject_Type(PyObject *o)
    fn (PyObjectPtr) -> PyObjectPtr,
]
alias PyObject_Length = ExternalFunction[
    "PyObject_Length",
    # Py_ssize_t PyObject_Length(PyObject *o)
    fn (PyObjectPtr) -> Py_ssize_t,
]
alias PyObject_GetItem = ExternalFunction[
    "PyObject_GetItem",
    # PyObject *PyObject_GetItem(PyObject *o, PyObject *key)
    fn (PyObjectPtr, PyObjectPtr) -> PyObjectPtr,
]
alias PyObject_SetItem = ExternalFunction[
    "PyObject_SetItem",
    # int PyObject_SetItem(PyObject *o, PyObject *key, PyObject *v)
    fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> c_int,
]
alias PyObject_GetIter = ExternalFunction[
    "PyObject_GetIter",
    # PyObject *PyObject_GetIter(PyObject *o)
    fn (PyObjectPtr) -> PyObjectPtr,
]

# Call Protocol
alias PyObject_Call = ExternalFunction[
    "PyObject_Call",
    # PyObject *PyObject_Call(PyObject *callable, PyObject *args, PyObject *kwargs)
    fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> PyObjectPtr,
]
alias PyObject_CallObject = ExternalFunction[
    "PyObject_CallObject",
    # PyObject *PyObject_CallObject(PyObject *callable, PyObject *args)
    fn (PyObjectPtr, PyObjectPtr) -> PyObjectPtr,
]

# Number Protocol
alias PyNumber_Long = ExternalFunction[
    "PyNumber_Long",
    # PyObject *PyNumber_Long(PyObject *o)
    fn (PyObjectPtr) -> PyObjectPtr,
]
alias PyNumber_Float = ExternalFunction[
    "PyNumber_Float",
    # PyObject *PyNumber_Float(PyObject *o)
    fn (PyObjectPtr) -> PyObjectPtr,
]

# Iterator Protocol
alias PyIter_Check = ExternalFunction[
    "PyIter_Check",
    # int PyIter_Check(PyObject *o)
    fn (PyObjectPtr) -> c_int,
]
alias PyIter_Next = ExternalFunction[
    "PyIter_Next",
    # PyObject *PyIter_Next(PyObject *o)
    fn (PyObjectPtr) -> PyObjectPtr,
]

# Concrete Objects Layer
# Type Objects
alias PyType_GenericAlloc = ExternalFunction[
    "PyType_GenericAlloc",
    # PyObject *PyType_GenericAlloc(PyTypeObject *type, Py_ssize_t nitems)
    fn (PyTypeObjectPtr, Py_ssize_t) -> PyObjectPtr,
]
alias PyType_GetName = ExternalFunction[
    "PyType_GetName",
    # PyObject *PyType_GetName(PyTypeObject *type)
    fn (PyTypeObjectPtr) -> PyObjectPtr,
]
alias PyType_FromSpec = ExternalFunction[
    "PyType_FromSpec",
    # PyObject *PyType_FromSpec(PyType_Spec *spec)
    fn (UnsafePointer[PyType_Spec]) -> PyObjectPtr,
]

# Integer Objects
alias PyLong_FromSsize_t = ExternalFunction[
    "PyLong_FromSsize_t",
    # PyObject *PyLong_FromSsize_t(Py_ssize_t v)
    fn (Py_ssize_t) -> PyObjectPtr,
]
alias PyLong_FromSize_t = ExternalFunction[
    "PyLong_FromSize_t",
    # PyObject *PyLong_FromSize_t(size_t v)
    fn (c_size_t) -> PyObjectPtr,
]
alias PyLong_AsSsize_t = ExternalFunction[
    "PyLong_AsSsize_t",
    # Py_ssize_t PyLong_AsSsize_t(PyObject *pylong)
    fn (PyObjectPtr) -> Py_ssize_t,
]

# Boolean Objects
alias PyBool_FromLong = ExternalFunction[
    "PyBool_FromLong",
    # PyObject *PyBool_FromLong(long v)
    fn (c_long) -> PyObjectPtr,
]

# Floating-Point Objects
alias PyFloat_FromDouble = ExternalFunction[
    "PyFloat_FromDouble",
    # PyObject *PyFloat_FromDouble(double v)
    fn (c_double) -> PyObjectPtr,
]
alias PyFloat_AsDouble = ExternalFunction[
    "PyFloat_AsDouble",
    # double PyFloat_AsDouble(PyObject *pyfloat)
    fn (PyObjectPtr) -> c_double,
]

# Unicode Objects and Codecs
alias PyUnicode_DecodeUTF8 = ExternalFunction[
    "PyUnicode_DecodeUTF8",
    # PyObject *PyUnicode_DecodeUTF8(const char *str, Py_ssize_t size, const char *errors)
    fn (
        UnsafePointer[c_char, mut=False],
        Py_ssize_t,
        UnsafePointer[c_char, mut=False],
    ) -> PyObjectPtr,
]
alias PyUnicode_AsUTF8AndSize = ExternalFunction[
    "PyUnicode_AsUTF8AndSize",
    # const char *PyUnicode_AsUTF8AndSize(PyObject *unicode, Py_ssize_t *size)
    fn (
        PyObjectPtr,
        UnsafePointer[Py_ssize_t],
    ) -> UnsafePointer[c_char, mut=False],
]

# Tuple Objects
alias PyTuple_New = ExternalFunction[
    "PyTuple_New",
    # PyObject *PyTuple_New(Py_ssize_t len)
    fn (Py_ssize_t) -> PyObjectPtr,
]
alias PyTuple_GetItem = ExternalFunction[
    "PyTuple_GetItem",
    # PyObject *PyTuple_GetItem(PyObject *p, Py_ssize_t pos)
    fn (PyObjectPtr, Py_ssize_t) -> PyObjectPtr,
]
alias PyTuple_SetItem = ExternalFunction[
    "PyTuple_SetItem",
    # int PyTuple_SetItem(PyObject *p, Py_ssize_t pos, PyObject *o)
    fn (PyObjectPtr, Py_ssize_t, PyObjectPtr) -> c_int,
]

# List Objects
alias PyList_New = ExternalFunction[
    "PyList_New",
    # PyObject *PyList_New(Py_ssize_t len)
    fn (Py_ssize_t) -> PyObjectPtr,
]
alias PyList_GetItem = ExternalFunction[
    "PyList_GetItem",
    # PyObject *PyList_GetItem(PyObject *list, Py_ssize_t index)
    fn (PyObjectPtr, Py_ssize_t) -> PyObjectPtr,
]
alias PyList_SetItem = ExternalFunction[
    "PyList_SetItem",
    # int PyList_SetItem(PyObject *list, Py_ssize_t index, PyObject *item)
    fn (PyObjectPtr, Py_ssize_t, PyObjectPtr) -> c_int,
]

# Dictionary Objects
alias PyDict_New = ExternalFunction[
    "PyDict_New",
    # PyObject *PyDict_New()
    fn () -> PyObjectPtr,
]
alias PyDict_SetItem = ExternalFunction[
    "PyDict_SetItem",
    # int PyDict_SetItem(PyObject *p, PyObject *key, PyObject *val)
    fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> c_int,
]
alias PyDict_GetItemWithError = ExternalFunction[
    "PyDict_GetItemWithError",
    # PyObject *PyDict_GetItemWithError(PyObject *p, PyObject *key)
    fn (PyObjectPtr, PyObjectPtr) -> PyObjectPtr,
]
alias PyDict_Next = ExternalFunction[
    "PyDict_Next",
    # int PyDict_Next(PyObject *p, Py_ssize_t *ppos, PyObject **pkey, PyObject **pvalue)
    fn (
        PyObjectPtr,
        UnsafePointer[Py_ssize_t],
        UnsafePointer[PyObjectPtr],
        UnsafePointer[PyObjectPtr],
    ) -> c_int,
]

# Set Objects
alias PySet_New = ExternalFunction[
    "PySet_New",
    # PyObject *PySet_New(PyObject *iterable)
    fn (PyObjectPtr) -> PyObjectPtr,
]
alias PySet_Add = ExternalFunction[
    "PySet_Add",
    # int PySet_Add(PyObject *set, PyObject *key)
    fn (PyObjectPtr, PyObjectPtr) -> c_int,
]

# Module Objects
alias PyModule_GetDict = ExternalFunction[
    "PyModule_GetDict",
    # PyObject *PyModule_GetDict(PyObject *module)
    fn (PyObjectPtr) -> PyObjectPtr,
]
alias PyModule_Create2 = ExternalFunction[
    "PyModule_Create2",
    # PyObject *PyModule_Create2(PyModuleDef *def, int module_api_version)
    fn (UnsafePointer[PyModuleDef], c_int) -> PyObjectPtr,
]
alias PyModule_AddFunctions = ExternalFunction[
    "PyModule_AddFunctions",
    # int PyModule_AddFunctions(PyObject *module, PyMethodDef *functions)
    fn (PyObjectPtr, UnsafePointer[PyMethodDef]) -> c_int,
]
alias PyModule_AddObjectRef = ExternalFunction[
    "PyModule_AddObjectRef",
    # int PyModule_AddObjectRef(PyObject *module, const char *name, PyObject *value)
    fn (PyObjectPtr, UnsafePointer[c_char, mut=False], PyObjectPtr) -> c_int,
]

# Slice Objects
alias PySlice_New = ExternalFunction[
    "PySlice_New",
    # PyObject *PySlice_New(PyObject *start, PyObject *stop, PyObject *step)
    fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> PyObjectPtr,
]

# Capsules
alias PyCapsule_Destructor = (
    # typedef void (*PyCapsule_Destructor)(PyObject *)
    destructor
)
alias PyCapsule_New = ExternalFunction[
    "PyCapsule_New",
    # PyObject *PyCapsule_New(void *pointer, const char *name, PyCapsule_Destructor destructor)
    fn (
        OpaquePointer,
        UnsafePointer[c_char, mut=False],
        PyCapsule_Destructor,
    ) -> PyObjectPtr,
]
alias PyCapsule_GetPointer = ExternalFunction[
    "PyCapsule_GetPointer",
    # void *PyCapsule_GetPointer(PyObject *capsule, const char *name)
    fn (PyObjectPtr, UnsafePointer[c_char, mut=False]) -> OpaquePointer,
]

# Memory Management
alias PyObject_Free = ExternalFunction[
    "PyObject_Free",
    # void PyObject_Free(void *p)
    fn (OpaquePointer) -> None,
]

# Object Implementation Support
# Common Object Structures
alias Py_Is = ExternalFunction[
    "Py_Is",
    # int Py_Is(PyObject *x, PyObject *y)
    fn (PyObjectPtr, PyObjectPtr) -> c_int,
]


fn _PyErr_GetRaisedException_dummy() -> PyObjectPtr:
    return abort[PyObjectPtr](
        "PyErr_GetRaisedException is not available in this Python version"
    )


fn _PyType_GetName_dummy(type: PyTypeObjectPtr) -> PyObjectPtr:
    return abort[PyObjectPtr](
        "PyType_GetName is not available in this Python version"
    )


fn _PyModule_AddObjectRef_dummy(
    module: PyObjectPtr,
    name: UnsafePointer[c_char, mut=False],
    value: PyObjectPtr,
) -> c_int:
    return abort[c_int](
        "PyModule_AddObjectRef is not available in this Python version"
    )


fn _Py_Is_dummy(x: PyObjectPtr, y: PyObjectPtr) -> c_int:
    return abort[c_int]("Py_Is is not available in this Python version")


# ===-------------------------------------------------------------------===#
# Context Managers for Python GIL and Threading
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct GILAcquired(Movable):
    """Context manager for Python Global Interpreter Lock (GIL) operations.

    This struct provides automatic GIL management inspired by nanobind/pybind11.
    It ensures the GIL is acquired on construction and released on destruction,
    making it safe to use Python objects within the managed scope.

    Example:
        ```mojo
        var python = Python()
        with GILAcquired(Python(python)):
            # Python objects can be safely accessed here
            var py_obj = python.cpython().Py_None()
        # GIL is automatically released here
        ```
    """

    var python: Python
    """Reference to the CPython instance."""
    var gil_state: PyGILState_STATE
    """The GIL state returned by PyGILState_Ensure."""

    fn __init__(out self, python: Python):
        """Acquire the GIL and initialize the context manager.

        Args:
            python: The CPython instance to use for GIL operations.
        """
        self.python = python
        self.gil_state = PyGILState_STATE(PyGILState_STATE.PyGILState_UNLOCKED)

    fn __enter__(mut self):
        """Acquire the GIL."""
        self.gil_state = self.python.cpython().PyGILState_Ensure()

    fn __exit__(mut self):
        """Release the GIL."""
        self.python.cpython().PyGILState_Release(self.gil_state)


@fieldwise_init
struct GILReleased(Movable):
    """Context manager for Python thread state operations.

    This struct provides automatic thread state management for scenarios where
    you need to temporarily release the GIL to allow other threads to run,
    then restore the thread state. This is useful for long-running operations
    that don't need to access Python objects.

    Example:
        ```mojo
        var python = Python()
        with GILReleased(python):
            # GIL is released here, other threads can run
            # Perform CPU-intensive work without Python object access
            perform_heavy_computation()
        # Thread state is automatically restored here
        ```
    """

    var python: Python
    """Reference to the CPython instance."""
    var thread_state: UnsafePointer[PyThreadState]
    """The thread state returned by PyEval_SaveThread."""

    fn __init__(out self, python: Python):
        """Save the current thread state and release the GIL.

        Args:
            python: The Python instance to use for GIL operations.
        """
        self.python = python
        self.thread_state = {}

    fn __enter__(mut self):
        """Save the current thread state and release the GIL."""
        self.thread_state = self.python.cpython().PyEval_SaveThread()

    fn __exit__(mut self):
        """Restore the thread state and acquire the GIL."""
        self.python.cpython().PyEval_RestoreThread(self.thread_state)


@fieldwise_init
struct CPython(Defaultable, Movable):
    """Handle to the CPython interpreter present in the current process.

    This type is non-copyable due to its large size. Please refer to it only
    using either a reference, or the `Python` handle type."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var lib: DLHandle
    """The handle to the CPython shared library."""
    var version: PythonVersion
    """The version of the Python runtime."""
    var init_error: StringSlice[StaticConstantOrigin]
    """An error message if initialization failed."""

    # fields holding function pointers to CPython C API functions
    # ordered based on https://docs.python.org/3/c-api/index.html

    # The Very High Level Layer
    var _PyRun_SimpleString: PyRun_SimpleString.type
    var _PyRun_String: PyRun_String.type
    var _Py_CompileString: Py_CompileString.type
    var _PyEval_EvalCode: PyEval_EvalCode.type
    # Reference Counting
    var _Py_IncRef: Py_IncRef.type
    var _Py_DecRef: Py_DecRef.type
    # Exception Handling
    var _PyErr_Clear: PyErr_Clear.type
    var _PyErr_SetString: PyErr_SetString.type
    var _PyErr_SetNone: PyErr_SetNone.type
    var _PyErr_Occurred: PyErr_Occurred.type
    var _PyErr_GetRaisedException: PyErr_GetRaisedException.type
    var _PyErr_Fetch: PyErr_Fetch.type
    # Initialization, Finalization, and Threads
    var _PyEval_SaveThread: PyEval_SaveThread.type
    var _PyEval_RestoreThread: PyEval_RestoreThread.type
    var _PyGILState_Ensure: PyGILState_Ensure.type
    var _PyGILState_Release: PyGILState_Release.type
    # Importing Modules
    var _PyImport_ImportModule: PyImport_ImportModule.type
    var _PyImport_AddModule: PyImport_AddModule.type
    # Abstract Objects Layer
    # Object Protocol
    var _PyObject_HasAttrString: PyObject_HasAttrString.type
    var _PyObject_GetAttrString: PyObject_GetAttrString.type
    var _PyObject_SetAttrString: PyObject_SetAttrString.type
    var _PyObject_Str: PyObject_Str.type
    var _PyObject_Hash: PyObject_Hash.type
    var _PyObject_IsTrue: PyObject_IsTrue.type
    var _PyObject_Type: PyObject_Type.type
    var _PyObject_Length: PyObject_Length.type
    var _PyObject_GetItem: PyObject_GetItem.type
    var _PyObject_SetItem: PyObject_SetItem.type
    var _PyObject_GetIter: PyObject_GetIter.type
    # Call Protocol
    var _PyObject_Call: PyObject_Call.type
    var _PyObject_CallObject: PyObject_CallObject.type
    # Number Protocol
    var _PyNumber_Long: PyNumber_Long.type
    var _PyNumber_Float: PyNumber_Float.type
    # Iterator Protocol
    var _PyIter_Check: PyIter_Check.type
    var _PyIter_Next: PyIter_Next.type
    # Concrete Objects Layer
    # Type Objects
    var _PyType_GenericAlloc: PyType_GenericAlloc.type
    var _PyType_GetName: PyType_GetName.type
    var _PyType_FromSpec: PyType_FromSpec.type
    # The None Object
    var _Py_None: PyObjectPtr
    # Integer Objects
    var _PyLong_FromSsize_t: PyLong_FromSsize_t.type
    var _PyLong_FromSize_t: PyLong_FromSize_t.type
    var _PyLong_AsSsize_t: PyLong_AsSsize_t.type
    # Boolean Objects
    var _PyBool_FromLong: PyBool_FromLong.type
    # Floating-Point Objects
    var _PyFloat_FromDouble: PyFloat_FromDouble.type
    var _PyFloat_AsDouble: PyFloat_AsDouble.type
    # Unicode Objects and Codecs
    var _PyUnicode_DecodeUTF8: PyUnicode_DecodeUTF8.type
    var _PyUnicode_AsUTF8AndSize: PyUnicode_AsUTF8AndSize.type
    # Tuple Objects
    var _PyTuple_New: PyTuple_New.type
    var _PyTuple_GetItem: PyTuple_GetItem.type
    var _PyTuple_SetItem: PyTuple_SetItem.type
    # List Objects
    var _PyList_New: PyList_New.type
    var _PyList_GetItem: PyList_GetItem.type
    var _PyList_SetItem: PyList_SetItem.type
    # Dictionary Objects
    var _PyDict_Type: PyTypeObjectPtr
    var _PyDict_New: PyDict_New.type
    var _PyDict_SetItem: PyDict_SetItem.type
    var _PyDict_GetItemWithError: PyDict_GetItemWithError.type
    var _PyDict_Next: PyDict_Next.type
    # Set Objects
    var _PySet_New: PySet_New.type
    var _PySet_Add: PySet_Add.type
    # Module Objects
    var _PyModule_GetDict: PyModule_GetDict.type
    var _PyModule_Create2: PyModule_Create2.type
    var _PyModule_AddFunctions: PyModule_AddFunctions.type
    var _PyModule_AddObjectRef: PyModule_AddObjectRef.type
    # Slice Objects
    var _PySlice_New: PySlice_New.type
    # Capsules
    var _PyCapsule_New: PyCapsule_New.type
    var _PyCapsule_GetPointer: PyCapsule_GetPointer.type
    # Memory Management
    var _PyObject_Free: PyObject_Free.type
    # Object Implementation Support
    # Common Object Structures
    var _Py_Is: Py_Is.type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        # Add directory of target file to top of sys.path to find python modules
        var file_dir = dirname(argv()[0])
        if Path(file_dir).is_dir() or file_dir == "":
            var python_path = getenv("PYTHONPATH")
            # A leading `:` will put the current dir at the top of sys.path.
            # If we're doing `mojo run main.mojo` or `./main`, the returned
            # `dirname` will be an empty string.
            if file_dir == "" and not python_path:
                file_dir = ":"
            if python_path:
                _ = setenv("PYTHONPATH", String(file_dir, ":", python_path))
            else:
                _ = setenv("PYTHONPATH", file_dir)

        # TODO(MOCO-772) Allow raises to propagate through function pointers
        # and make this initialization a raising function.
        self.init_error = StaticString(
            unsafe_from_utf8_ptr=external_call[
                "KGEN_CompilerRT_Python_SetPythonPath",
                UnsafePointer[c_char, mut=False, origin=StaticConstantOrigin],
            ]()
        )

        var python_lib = getenv("MOJO_PYTHON_LIBRARY")

        # Note:
        #   MOJO_PYTHON_LIBRARY can be "" when the current Mojo program
        #   is a dynamic library being loaded as a Python extension module,
        #   and we need to find CPython symbols that are statically linked
        #   into the `python` main executable. On those platforms where
        #   `python` executable can be statically linked (Linux), it's
        #   important that we don't load a second copy of CPython symbols
        #   into the process by loading the `libpython` dynamic library.
        try:
            # Try to load the library from the current process.
            self.lib = DLHandle()
            if not self.lib.check_symbol("Py_Initialize"):
                # If the library is not present in the current process, try to load it from the environment variable.
                self.lib = DLHandle(python_lib)
        except e:
            self.lib = abort[DLHandle](
                String("Failed to load libpython from", python_lib, ":\n", e)
            )

        if not self.init_error:
            if not self.lib.check_symbol("Py_Initialize"):
                self.init_error = "compatible Python library not found"
            self.lib.call["Py_Initialize"]()
            self.version = PythonVersion(_py_get_version(self.lib))
        else:
            self.version = PythonVersion(0, 0, 0)

        # The Very High Level Layer
        self._PyRun_SimpleString = PyRun_SimpleString.load(self.lib)
        self._PyRun_String = PyRun_String.load(self.lib)
        self._Py_CompileString = Py_CompileString.load(self.lib)
        self._PyEval_EvalCode = PyEval_EvalCode.load(self.lib)
        # Reference Counting
        self._Py_IncRef = Py_IncRef.load(self.lib)
        self._Py_DecRef = Py_DecRef.load(self.lib)
        # Exception Handling
        self._PyErr_Clear = PyErr_Clear.load(self.lib)
        self._PyErr_SetString = PyErr_SetString.load(self.lib)
        self._PyErr_SetNone = PyErr_SetNone.load(self.lib)
        self._PyErr_Occurred = PyErr_Occurred.load(self.lib)
        if self.version.minor >= 12:
            self._PyErr_GetRaisedException = PyErr_GetRaisedException.load(
                self.lib
            )
        else:
            self._PyErr_GetRaisedException = _PyErr_GetRaisedException_dummy
        self._PyErr_Fetch = PyErr_Fetch.load(self.lib)
        # Initialization, Finalization, and Threads
        self._PyEval_SaveThread = PyEval_SaveThread.load(self.lib)
        self._PyEval_RestoreThread = PyEval_RestoreThread.load(self.lib)
        self._PyGILState_Ensure = PyGILState_Ensure.load(self.lib)
        self._PyGILState_Release = PyGILState_Release.load(self.lib)
        # Importing Modules
        self._PyImport_ImportModule = PyImport_ImportModule.load(self.lib)
        self._PyImport_AddModule = PyImport_AddModule.load(self.lib)
        # Abstract Objects Layer
        # Object Protocol
        self._PyObject_HasAttrString = PyObject_HasAttrString.load(self.lib)
        self._PyObject_GetAttrString = PyObject_GetAttrString.load(self.lib)
        self._PyObject_SetAttrString = PyObject_SetAttrString.load(self.lib)
        self._PyObject_Str = PyObject_Str.load(self.lib)
        self._PyObject_Hash = PyObject_Hash.load(self.lib)
        self._PyObject_IsTrue = PyObject_IsTrue.load(self.lib)
        self._PyObject_Type = PyObject_Type.load(self.lib)
        self._PyObject_Length = PyObject_Length.load(self.lib)
        self._PyObject_GetItem = PyObject_GetItem.load(self.lib)
        self._PyObject_SetItem = PyObject_SetItem.load(self.lib)
        self._PyObject_GetIter = PyObject_GetIter.load(self.lib)
        # Call Protocol
        self._PyObject_Call = PyObject_Call.load(self.lib)
        self._PyObject_CallObject = PyObject_CallObject.load(self.lib)
        # Number Protocol
        self._PyNumber_Long = PyNumber_Long.load(self.lib)
        self._PyNumber_Float = PyNumber_Float.load(self.lib)
        # Iterator Protocol
        self._PyIter_Check = PyIter_Check.load(self.lib)
        self._PyIter_Next = PyIter_Next.load(self.lib)
        # Concrete Objects Layer
        # Type Objects
        self._PyType_GenericAlloc = PyType_GenericAlloc.load(self.lib)
        if self.version.minor >= 11:
            self._PyType_GetName = PyType_GetName.load(self.lib)
        else:
            self._PyType_GetName = _PyType_GetName_dummy
        self._PyType_FromSpec = PyType_FromSpec.load(self.lib)
        # The None Object
        if self.version.minor >= 13:
            # Py_GetConstantBorrowed is part of the Stable ABI since version 3.13
            # References:
            # - https://docs.python.org/3/c-api/object.html#c.Py_GetConstantBorrowed
            # - https://docs.python.org/3/c-api/object.html#c.Py_CONSTANT_NONE

            # PyObject *Py_GetConstantBorrowed(unsigned int constant_id)
            self._Py_None = self.lib.call[
                "Py_GetConstantBorrowed", PyObjectPtr
            ](0)
        else:
            # PyObject *Py_None
            self._Py_None = PyObjectPtr(
                self.lib.get_symbol[PyObject]("_Py_NoneStruct")
            )
        # Integer Objects
        self._PyLong_FromSsize_t = PyLong_FromSsize_t.load(self.lib)
        self._PyLong_FromSize_t = PyLong_FromSize_t.load(self.lib)
        self._PyLong_AsSsize_t = PyLong_AsSsize_t.load(self.lib)
        # Boolean Objects
        self._PyBool_FromLong = PyBool_FromLong.load(self.lib)
        # Floating-Point Objects
        self._PyFloat_FromDouble = PyFloat_FromDouble.load(self.lib)
        self._PyFloat_AsDouble = PyFloat_AsDouble.load(self.lib)
        # Unicode Objects and Codecs
        self._PyUnicode_DecodeUTF8 = PyUnicode_DecodeUTF8.load(self.lib)
        self._PyUnicode_AsUTF8AndSize = PyUnicode_AsUTF8AndSize.load(self.lib)
        # Tuple Objects
        self._PyTuple_New = PyTuple_New.load(self.lib)
        self._PyTuple_GetItem = PyTuple_GetItem.load(self.lib)
        self._PyTuple_SetItem = PyTuple_SetItem.load(self.lib)
        # List Objects
        self._PyList_New = PyList_New.load(self.lib)
        self._PyList_GetItem = PyList_GetItem.load(self.lib)
        self._PyList_SetItem = PyList_SetItem.load(self.lib)
        # Dictionary Objects
        self._PyDict_Type = PyTypeObjectPtr(
            # PyTypeObject PyDict_Type
            self.lib.get_symbol[PyTypeObject]("PyDict_Type")
        )
        self._PyDict_New = PyDict_New.load(self.lib)
        self._PyDict_SetItem = PyDict_SetItem.load(self.lib)
        self._PyDict_GetItemWithError = PyDict_GetItemWithError.load(self.lib)
        self._PyDict_Next = PyDict_Next.load(self.lib)
        # Set Objects
        self._PySet_New = PySet_New.load(self.lib)
        self._PySet_Add = PySet_Add.load(self.lib)
        # Module Objects
        self._PyModule_GetDict = PyModule_GetDict.load(self.lib)
        self._PyModule_Create2 = PyModule_Create2.load(self.lib)
        self._PyModule_AddFunctions = PyModule_AddFunctions.load(self.lib)
        if self.version.minor >= 10:
            self._PyModule_AddObjectRef = PyModule_AddObjectRef.load(self.lib)
        else:
            self._PyModule_AddObjectRef = _PyModule_AddObjectRef_dummy
        # Slice Objects
        self._PySlice_New = PySlice_New.load(self.lib)
        # Capsules
        self._PyCapsule_New = PyCapsule_New.load(self.lib)
        self._PyCapsule_GetPointer = PyCapsule_GetPointer.load(self.lib)
        # Memory Management
        self._PyObject_Free = PyObject_Free.load(self.lib)
        # Object Implementation Support
        # Common Object Structures
        if self.version.minor >= 10:
            self._Py_Is = Py_Is.load(self.lib)
        else:
            self._Py_Is = _Py_Is_dummy

    fn __del__(deinit self):
        pass

    fn destroy(mut self):
        # https://docs.python.org/3/c-api/init.html#c.Py_FinalizeEx
        self.lib.call["Py_FinalizeEx"]()
        self.lib.close()

    fn check_init_error(self) raises:
        """Used for entry points that initialize Python on first use, will
        raise an error if one occurred when initializing the global CPython.
        """
        if self.init_error:
            var mojo_python = getenv("MOJO_PYTHON")
            var python_lib = getenv("MOJO_PYTHON_LIBRARY")
            var python_exe = getenv("PYTHONEXECUTABLE")
            raise Error(
                self.init_error,
                "\nMOJO_PYTHON: " if mojo_python else "",
                mojo_python if mojo_python else "",
                "\nMOJO_PYTHON_LIBRARY: " if python_lib else "",
                python_lib if python_lib else "",
                "\npython executable: " if python_exe else "",
                python_exe if python_exe else "",
                "\n\nMojo/Python interop error, troubleshooting docs at:",
                "\n    https://modul.ar/fix-python\n",
            )

    fn unsafe_get_error(self) -> Error:
        """Get the `Error` object corresponding to the current CPython
        interpreter error state.

        Safety:
            The caller MUST be sure that the CPython interpreter is in an error
            state before calling this function.

        This function will clear the CPython error.

        Returns:
            `Error` object describing the CPython error.
        """

        @parameter
        fn err_occurred() -> Bool:
            return self.PyErr_Occurred()

        debug_assert[err_occurred](
            "invalid unchecked conversion of Python error to Mojo error",
        )

        var err_ptr: PyObjectPtr
        # NOTE: PyErr_Fetch is deprecated since Python 3.12.
        var old_python = self.version.minor < 12
        if old_python:
            err_ptr = self.PyErr_Fetch()
        else:
            err_ptr = self.PyErr_GetRaisedException()
        debug_assert(
            Bool(err_ptr), "Python exception occurred but null was returned"
        )

        var error: String
        try:
            error = String(PythonObject(from_owned=err_ptr))
        except e:
            return abort[Error](
                "internal error: Python exception occurred but cannot be"
                " converted to String"
            )

        if old_python:
            self.PyErr_Clear()
        return Error(error^)

    fn get_error(self) -> Error:
        """Return an `Error` object from the CPython interpreter if it's in an
        error state, or an internal error if it's not.

        This should be used when you expect CPython to be in an error state,
        but want to fail gracefully if it's not.

        Returns:
            An `Error` object from the CPython interpreter if it's in an
            error state, or an internal error if it's not.
        """
        if self.PyErr_Occurred():
            return self.unsafe_get_error()
        return Error("internal error: expected CPython exception not found")

    fn get_error_global(
        self,
        global_name: StringSlice,
    ) -> PyObjectPtr:
        """Get a Python read-only reference to the specified global exception
        object.
        """

        # Get pointer to the immortal `global_name` PyObject struct
        # instance.
        var ptr = self.lib.get_symbol[PyObjectPtr](global_name)

        if not ptr:
            abort(
                "error: unable to get pointer to CPython `"
                + String(global_name)
                + "` global"
            )

        return ptr[]

    # ===-------------------------------------------------------------------===#
    # Python/C API
    # ref: https://docs.python.org/3/c-api/index.html
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # The Very High Level Layer
    # ref: https://docs.python.org/3/c-api/veryhigh.html
    # ===-------------------------------------------------------------------===#

    fn PyRun_SimpleString(self, var command: String) -> c_int:
        """This is a simplified interface to `PyRun_SimpleStringFlags()` below,
        leaving the `PyCompilerFlags*` argument set to `NULL`.

        References:
        - https://docs.python.org/3/c-api/veryhigh.html#c.PyRun_SimpleString
        """
        return self._PyRun_SimpleString(command.unsafe_cstr_ptr())

    fn PyRun_String(
        self,
        var str: String,
        start: c_int,
        globals: PyObjectPtr,
        locals: PyObjectPtr,
    ) -> PyObjectPtr:
        """Execute Python source code from `str` in the context specified by
        the objects `globals` and `locals`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/veryhigh.html#c.PyRun_String
        """
        return self._PyRun_String(str.unsafe_cstr_ptr(), start, globals, locals)

    fn Py_CompileString(
        self,
        var str: String,
        var filename: String,
        start: c_int,
    ) -> PyObjectPtr:
        """Parse and compile the Python source code in `str`, returning the
        resulting code object.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/veryhigh.html#c.Py_CompileString
        """
        return self._Py_CompileString(
            str.unsafe_cstr_ptr(),
            filename.unsafe_cstr_ptr(),
            start,
        )

    fn PyEval_EvalCode(
        self,
        co: PyObjectPtr,
        globals: PyObjectPtr,
        locals: PyObjectPtr,
    ) -> PyObjectPtr:
        """Evaluate a precompiled code object, given a particular environment
        for its evaluation.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/veryhigh.html#c.PyEval_EvalCode
        """
        return self._PyEval_EvalCode(co, globals, locals)

    # ===-------------------------------------------------------------------===#
    # Reference Counting
    # ref: https://docs.python.org/3/c-api/refcounting.html
    # ===-------------------------------------------------------------------===#

    fn Py_IncRef(self, ptr: PyObjectPtr):
        """Indicate taking a new strong reference to the object `ptr` points to.

        A function version of `Py_XINCREF()`, which is no-op if `ptr` is `NULL`.

        References:
        - https://docs.python.org/3/c-api/refcounting.html#c.Py_IncRef
        - https://docs.python.org/3/c-api/refcounting.html#c.Py_XINCREF
        """
        self._Py_IncRef(ptr)

    fn Py_DecRef(self, ptr: PyObjectPtr):
        """Release a strong reference to the object `ptr` points to.

        A function version of `Py_XDECREF()`, which is no-op if `ptr` is `NULL`.

        References:
        - https://docs.python.org/3/c-api/refcounting.html#c.Py_DecRef
        - https://docs.python.org/3/c-api/refcounting.html#c.Py_XDECREF
        """
        self._Py_DecRef(ptr)

    # This function assumes a specific way PyObjectPtr is implemented, namely
    # that the refcount has offset 0 in that structure. That generally doesn't
    # have to always be the case - but often it is and it's convenient for
    # debugging. We shouldn't rely on this function anywhere - its only purpose
    # is debugging.
    fn _Py_REFCNT(self, ptr: PyObjectPtr) -> Py_ssize_t:
        if not ptr:
            return -1
        # NOTE:
        #   The "obvious" way to write this would be:
        #       return ptr._unsized_obj_ptr[].object_ref_count
        #   However, that is not valid, because, as the name suggest, a PyObject
        #   is an "unsized" or "incomplete" type, meaning that a pointer to an
        #   instance of that type doesn't point at the entire allocation of the
        #   underlying "concrete" object instance.
        #
        #   To avoid concerns about whether that's UB or not in Mojo, this
        #   this by just assumes the first field will be the ref count, and
        #   treats the object pointer "as if" it was a pointer to just the first
        #   field.
        # TODO(MSTDL-950): Should use something like `addr_of!`
        return ptr.bitcast[Py_ssize_t]()[]

    # ===-------------------------------------------------------------------===#
    # Exception Handling
    # ref: https://docs.python.org/3/c-api/exceptions.html
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # - Printing and clearing
    # ===-------------------------------------------------------------------===#

    fn PyErr_Clear(self):
        """Clear the error indicator. If the error indicator is not set, there
        is no effect.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Clear
        """
        self._PyErr_Clear()

    # ===-------------------------------------------------------------------===#
    # - Raising exceptions
    # ===-------------------------------------------------------------------===#

    fn PyErr_SetString(
        self,
        type: PyObjectPtr,
        message: UnsafePointer[c_char, mut=False],
    ):
        """This is the most common way to set the error indicator. The first
        argument specifies the exception type; it is normally one of the
        standard exceptions, e.g. `PyExc_RuntimeError`. You need not create a
        new strong reference to it (e.g. with `Py_INCREF()`). The second
        argument is an error message; it is decoded from `'utf-8'`.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetString
        """
        self._PyErr_SetString(type, message)

    fn PyErr_SetNone(self, type: PyObjectPtr):
        """This is a shorthand for `PyErr_SetObject(type, Py_None)`.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetNone
        """
        self._PyErr_SetNone(type)

    # ===-------------------------------------------------------------------===#
    # - Querying the error indicator
    # ===-------------------------------------------------------------------===#

    # TODO: fix the return type
    fn PyErr_Occurred(self) -> Bool:
        """Test whether the error indicator is set. If set, return the exception
        type (the first argument to the last call to one of the `PyErr_Set*`
        functions or to `PyErr_Restore()`). If not set, return `NULL`.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Occurred
        """
        return Bool(self._PyErr_Occurred())

    fn PyErr_GetRaisedException(self) -> PyObjectPtr:
        """Return the exception currently being raised, clearing the error
        indicator at the same time. Return `NULL` if the error indicator is not
        set.

        Return value: New reference. Part of the Stable ABI since version 3.12.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_GetRaisedException
        """
        return self._PyErr_GetRaisedException()

    # TODO: fix the signature to take the type, value, and traceback as args
    fn PyErr_Fetch(self) -> PyObjectPtr:
        """Retrieve the error indicator into three variables whose addresses
        are passed.

        Deprecated since version 3.12.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Fetch
        """
        var type = PyObjectPtr()
        var value = PyObjectPtr()
        var traceback = PyObjectPtr()

        self._PyErr_Fetch(
            UnsafePointer(to=type),
            UnsafePointer(to=value),
            UnsafePointer(to=traceback),
        )

        return value

    # ===-------------------------------------------------------------------===#
    # Initialization, Finalization, and Threads
    # ref: https://docs.python.org/3/c-api/init.html
    # ===-------------------------------------------------------------------===#

    fn PyEval_SaveThread(self) -> UnsafePointer[PyThreadState]:
        """Release the global interpreter lock (if it has been created) and
        reset the thread state to `NULL`, returning the previous thread state
        (which is not `NULL`).

        References:
        - https://docs.python.org/3/c-api/init.html#c.PyEval_SaveThread
        """
        return self._PyEval_SaveThread()

    fn PyEval_RestoreThread(self, state: UnsafePointer[PyThreadState]):
        """Acquire the global interpreter lock (if it has been created) and
        set the thread state to tstate, which must not be `NULL`.

        References:
        - https://docs.python.org/3/c-api/init.html#c.PyEval_RestoreThread
        """
        self._PyEval_RestoreThread(state)

    fn PyGILState_Ensure(self) -> PyGILState_STATE:
        """Ensure that the current thread is ready to call the Python C API
        regardless of the current state of Python, or of the global interpreter
        lock.

        References:
        - https://docs.python.org/3/c-api/init.html#c.PyGILState_Ensure
        """
        return self._PyGILState_Ensure()

    fn PyGILState_Release(self, state: PyGILState_STATE):
        """Release any resources previously acquired.

        References:
        - https://docs.python.org/3/c-api/init.html#c.PyGILState_Release
        """
        self._PyGILState_Release(state)

    # ===-------------------------------------------------------------------===#
    # Importing Modules
    # ref: https://docs.python.org/3/c-api/import.html
    # ===-------------------------------------------------------------------===#

    fn PyImport_ImportModule(self, var name: String) -> PyObjectPtr:
        """This is a wrapper around `PyImport_Import()` which takes a `const char*`
        as an argument instead of a `PyObject*`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/import.html#c.PyImport_ImportModule
        """
        return self._PyImport_ImportModule(name.unsafe_cstr_ptr())

    fn PyImport_AddModule(self, var name: String) -> PyObjectPtr:
        """Return the module object corresponding to a module name.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/import.html#c.PyImport_AddModule
        """
        return self._PyImport_AddModule(name.unsafe_cstr_ptr())

    # ===-------------------------------------------------------------------===#
    # Abstract Objects Layer
    # ref: https://docs.python.org/3/c-api/abstract.html
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # Object Protocol
    # ref: https://docs.python.org/3/c-api/object.html
    # ===-------------------------------------------------------------------===#

    fn PyObject_HasAttrString(
        self, obj: PyObjectPtr, var name: String
    ) -> c_int:
        """Returns `1` if `obj` has the attribute `name`, and `0` otherwise.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_HasAttrString
        """
        return self._PyObject_HasAttrString(obj, name.unsafe_cstr_ptr())

    fn PyObject_GetAttrString(
        self, obj: PyObjectPtr, var name: String
    ) -> PyObjectPtr:
        """Retrieve an attribute named `name` from object `obj`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_GetAttrString
        """
        return self._PyObject_GetAttrString(obj, name.unsafe_cstr_ptr())

    fn PyObject_SetAttrString(
        self, obj: PyObjectPtr, var name: String, value: PyObjectPtr
    ) -> c_int:
        """Set the value of the attribute named `name`, for object `obj`, to
        `value`.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_SetAttrString
        """
        return self._PyObject_SetAttrString(obj, name.unsafe_cstr_ptr(), value)

    fn PyObject_Str(self, obj: PyObjectPtr) -> PyObjectPtr:
        """Compute a string representation of object `obj`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_Str
        """
        return self._PyObject_Str(obj)

    fn PyObject_Hash(self, obj: PyObjectPtr) -> Py_hash_t:
        """Compute and return the hash value of an object `obj`.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_Hash
        """
        return self._PyObject_Hash(obj)

    fn PyObject_IsTrue(self, obj: PyObjectPtr) -> c_int:
        """Returns `1` if the object `obj` is considered to be true, and `0`
        otherwise.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_IsTrue
        """
        return self._PyObject_IsTrue(obj)

    fn PyObject_Type(self, obj: PyObjectPtr) -> PyObjectPtr:
        """When `obj` is non-`NULL`, returns a type object corresponding to the
        object type of object `obj`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_Type
        """
        return self._PyObject_Type(obj)

    fn PyObject_Length(self, obj: PyObjectPtr) -> Py_ssize_t:
        """Return the length of object `obj`.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_Length
        """
        return self._PyObject_Length(obj)

    fn PyObject_GetItem(
        self, obj: PyObjectPtr, key: PyObjectPtr
    ) -> PyObjectPtr:
        """Return element of `obj` corresponding to the object `key` or `NULL`
        on failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_GetItem
        """
        return self._PyObject_GetItem(obj, key)

    fn PyObject_SetItem(
        self, obj: PyObjectPtr, key: PyObjectPtr, value: PyObjectPtr
    ) -> c_int:
        """Map the object `key` to `value`. Raise an exception and return `-1`
        on failure; return `0` on success.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_SetItem
        """
        return self._PyObject_SetItem(obj, key, value)

    fn PyObject_GetIter(self, obj: PyObjectPtr) -> PyObjectPtr:
        """This is equivalent to the Python expression `iter(obj)`. It returns
        a new iterator for the object argument, or the object itself if the
        object is already an iterator.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_GetIter
        """
        return self._PyObject_GetIter(obj)

    # ===-------------------------------------------------------------------===#
    # Call Protocol
    # ref: https://docs.python.org/3/c-api/call.html
    # ===-------------------------------------------------------------------===#

    fn PyObject_Call(
        self,
        callable: PyObjectPtr,
        args: PyObjectPtr,
        kwargs: PyObjectPtr,
    ) -> PyObjectPtr:
        """Call a callable Python object `callable`, with arguments given by the
        tuple `args`, and named arguments given by the dictionary `kwargs`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/call.html#c.PyObject_Call
        """
        return self._PyObject_Call(callable, args, kwargs)

    fn PyObject_CallObject(
        self,
        callable: PyObjectPtr,
        args: PyObjectPtr,
    ) -> PyObjectPtr:
        """Call a callable Python object `callable`, with arguments given by the
        tuple `args`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/call.html#c.PyObject_CallObject
        """
        return self._PyObject_CallObject(callable, args)

    # ===-------------------------------------------------------------------===#
    # Number Protocol
    # ref: https://docs.python.org/3/c-api/number.html
    # ===-------------------------------------------------------------------===#

    fn PyNumber_Long(self, obj: PyObjectPtr) -> PyObjectPtr:
        """Returns the `obj` converted to an integer object on success,
        or `NULL` on failure. This is the equivalent of the Python expression
        `int(obj)`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Long
        """
        return self._PyNumber_Long(obj)

    fn PyNumber_Float(self, obj: PyObjectPtr) -> PyObjectPtr:
        """Returns the `o` converted to a float object on success, or `NULL` on
        failure. This is the equivalent of the Python expression `float(obj)`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Float
        """
        return self._PyNumber_Float(obj)

    # ===-------------------------------------------------------------------===#
    # Iterator Protocol
    # ref: https://docs.python.org/3/c-api/iter.html
    # ===-------------------------------------------------------------------===#

    fn PyIter_Check(self, obj: PyObjectPtr) -> c_int:
        """Return non-zero if the object `obj` can be safely passed to `PyIter_Next()`,
        and `0` otherwise.

        References:
        - https://docs.python.org/3/c-api/iter.html#c.PyIter_Check
        """
        return self._PyIter_Check(obj)

    fn PyIter_Next(self, obj: PyObjectPtr) -> PyObjectPtr:
        """Return the next value from the iterator `obj`. The object must be an
        iterator according to `PyIter_Check()`. If there are no remaining values,
        returns `NULL` with no exception set. If an error occurs while retrieving
        the item, returns `NULL` and passes along the exception.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/iter.html#c.PyIter_Next
        """
        return self._PyIter_Next(obj)

    # ===-------------------------------------------------------------------===#
    # Concrete Objects Layer
    # ref: https://docs.python.org/3/c-api/concrete.html
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # Type Objects
    # ref: https://docs.python.org/3/c-api/type.html
    # ===-------------------------------------------------------------------===#

    fn PyType_GenericAlloc(
        self,
        type: PyTypeObjectPtr,
        nitems: Py_ssize_t,
    ) -> PyObjectPtr:
        """Generic handler for the `tp_alloc` slot of a type object.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/type.html#c.PyType_GenericAlloc
        """
        return self._PyType_GenericAlloc(type, nitems)

    fn PyType_GetName(self, type: UnsafePointer[PyTypeObject]) -> PyObjectPtr:
        """Return the type's name.

        Return value: New reference. Part of the Stable ABI since version 3.11.
        This function is patched to work with Python 3.10 and earlier versions.

        References:
        - https://docs.python.org/3/c-api/type.html#c.PyType_GetName
        """
        if self.version.minor < 11:
            return self.PyObject_GetAttrString(
                rebind[PyObjectPtr](type), "__name__"
            )
        return self._PyType_GetName(type)

    fn PyType_FromSpec(self, spec: UnsafePointer[PyType_Spec]) -> PyObjectPtr:
        """Equivalent to `PyType_FromMetaclass(NULL, NULL, spec, NULL)`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/type.html#c.PyType_FromSpec
        """
        return self._PyType_FromSpec(spec)

    # ===-------------------------------------------------------------------===#
    # The None Object
    # ref: https://docs.python.org/3/c-api/none.html
    # ===-------------------------------------------------------------------===#

    fn Py_None(self) -> PyObjectPtr:
        """The Python `None` object, denoting lack of value.

        References:
        - https://docs.python.org/3/c-api/none.html#c.Py_None
        """
        return self._Py_None

    # ===-------------------------------------------------------------------===#
    # Integer Objects
    # ref: https://docs.python.org/3/c-api/long.html
    # ===-------------------------------------------------------------------===#

    fn PyLong_FromSsize_t(self, value: Py_ssize_t) -> PyObjectPtr:
        """Return a new `PyLongObject` object from a C `Py_ssize_t`, or `NULL`
        on failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/long.html#c.PyLong_FromSsize_t
        """
        return self._PyLong_FromSsize_t(value)

    fn PyLong_FromSize_t(self, value: c_size_t) -> PyObjectPtr:
        """Return a new `PyLongObject` object from a C `size_t`, or `NULL` on
        failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/long.html#c.PyLong_FromSize_t
        """
        return self._PyLong_FromSize_t(value)

    fn PyLong_AsSsize_t(self, pylong: PyObjectPtr) -> Py_ssize_t:
        """Return a C `Py_ssize_t` representation of `pylong`.

        Raise `OverflowError` if the value of `pylong` is out of range for
        a `Py_ssize_t`.

        Returns `-1` on error. Use `PyErr_Occurred()` to disambiguate.

        References:
        - https://docs.python.org/3/c-api/long.html#c.PyLong_AsSsize_t
        """
        return self._PyLong_AsSsize_t(pylong)

    # ===-------------------------------------------------------------------===#
    # Boolean Objects
    # ref: https://docs.python.org/3/c-api/bool.html
    # ===-------------------------------------------------------------------===#

    fn PyBool_FromLong(self, value: c_long) -> PyObjectPtr:
        """Return `Py_True` or `Py_False`, depending on the truth value
        of `value`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/bool.html#c.PyBool_FromLong
        """
        return self._PyBool_FromLong(value)

    # ===-------------------------------------------------------------------===#
    # Floating-Point Objects
    # ref: https://docs.python.org/3/c-api/float.html
    # ===-------------------------------------------------------------------===#

    fn PyFloat_FromDouble(self, value: c_double) -> PyObjectPtr:
        """Create a PyFloatObject object from `value`, or `NULL` on failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/float.html#c.PyFloat_FromDouble
        """
        return self._PyFloat_FromDouble(value)

    fn PyFloat_AsDouble(self, pyfloat: PyObjectPtr) -> c_double:
        """Return a C double representation of the contents of `pyfloat`.

        This method returns `-1.0` upon failure, so one should call
        `PyErr_Occurred()` to check for errors.

        References:
        - https://docs.python.org/3/c-api/float.html#c.PyFloat_AsDouble
        """
        return self._PyFloat_AsDouble(pyfloat)

    # ===-------------------------------------------------------------------===#
    # Unicode Objects and Codecs
    # ref: https://docs.python.org/3/c-api/unicode.html
    # ===-------------------------------------------------------------------===#

    # TODO: fix the signature to take str, size, and errors as args
    fn PyUnicode_DecodeUTF8(self, s: StringSlice) -> PyObjectPtr:
        """Create a Unicode object by decoding size bytes of the UTF-8 encoded
        string slice `s`. Return `NULL` if an exception was raised by the codec.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_DecodeUTF8
        """
        return self._PyUnicode_DecodeUTF8(
            s.unsafe_ptr().bitcast[c_char](),
            Py_ssize_t(s.byte_length()),
            "strict".unsafe_cstr_ptr(),
        )

    # TODO: fix signature to take unicode and size as args
    fn PyUnicode_AsUTF8AndSize(
        self, obj: PyObjectPtr
    ) -> StringSlice[ImmutableAnyOrigin]:
        """Return a pointer to the UTF-8 encoding of the Unicode object, and
        store the size of the encoded representation (in bytes) in `size`.

        References:
        - https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_AsUTF8AndSize
        """
        var length = Py_ssize_t(0)
        var ptr = self._PyUnicode_AsUTF8AndSize(obj, UnsafePointer(to=length))
        return StringSlice[ImmutableAnyOrigin](
            ptr=ptr.bitcast[Byte](), length=UInt(length)
        )

    # ===-------------------------------------------------------------------===#
    # Tuple Objects
    # ref: https://docs.python.org/3/c-api/tuple.html
    # ===-------------------------------------------------------------------===#

    fn PyTuple_New(self, length: Py_ssize_t) -> PyObjectPtr:
        """Return a new tuple object of size `length`, or `NULL` with an
        exception set on failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/tuple.html#c.PyTuple_New
        """
        return self._PyTuple_New(length)

    fn PyTuple_GetItem(
        self,
        tuple: PyObjectPtr,
        pos: Py_ssize_t,
    ) -> PyObjectPtr:
        """Return the object at position `pos` in the tuple `tuple`.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/tuple.html#c.PyTuple_GetItem
        """
        return self._PyTuple_GetItem(tuple, pos)

    fn PyTuple_SetItem(
        self,
        tuple: PyObjectPtr,
        pos: Py_ssize_t,
        value: PyObjectPtr,
    ) -> c_int:
        """Insert a reference to object `value` at position `pos` of the tuple
        `tuple`.

        This function "steals" a reference to `value` and discards a reference
        to an item already in the tuple at the affected position.

        References:
        - https://docs.python.org/3/c-api/tuple.html#c.PyTuple_SetItem
        """
        return self._PyTuple_SetItem(tuple, pos, value)

    # ===-------------------------------------------------------------------===#
    # List Objects
    # ref: https://docs.python.org/3/c-api/list.html
    # ===-------------------------------------------------------------------===#

    fn PyList_New(self, length: Py_ssize_t) -> PyObjectPtr:
        """Return a new list of length `length` on success, or `NULL` on
        failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/list.html#c.PyList_New
        """
        return self._PyList_New(length)

    fn PyList_GetItem(
        self,
        list: PyObjectPtr,
        index: Py_ssize_t,
    ) -> PyObjectPtr:
        """Return the object at position `index` in the list `list`.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/list.html#c.PyList_GetItem
        """
        return self._PyList_GetItem(list, index)

    fn PyList_SetItem(
        self,
        list: PyObjectPtr,
        index: Py_ssize_t,
        value: PyObjectPtr,
    ) -> c_int:
        """Set the item at index `index` in `list` to `value`.

        This function "steals" a reference to `value` and discards a reference
        to an item already in the list at the affected position.

        References:
        - https://docs.python.org/3/c-api/list.html#c.PyList_SetItem
        """
        return self._PyList_SetItem(list, index, value)

    # ===-------------------------------------------------------------------===#
    # Dictionary Objects
    # ref: https://docs.python.org/3/c-api/dict.html
    # ===-------------------------------------------------------------------===#

    fn PyDict_Type(self) -> PyTypeObjectPtr:
        """This instance of `PyTypeObject` represents the Python dictionary type.

        References:
        - https://docs.python.org/3/c-api/dict.html#c.PyDict_Type
        """
        return self._PyDict_Type

    fn PyDict_New(self) -> PyObjectPtr:
        """Return a new empty dictionary, or `NULL` on failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/dict.html#c.PyDict_New
        """
        return self._PyDict_New()

    fn PyDict_SetItem(
        self,
        dict: PyObjectPtr,
        key: PyObjectPtr,
        value: PyObjectPtr,
    ) -> c_int:
        """Insert `value` into the dictionary `dict` with a key of `key`.

        This function *does not* steal a reference to `value`.

        References:
        - https://docs.python.org/3/c-api/dict.html#c.PyDict_SetItem
        """
        return self._PyDict_SetItem(dict, key, value)

    fn PyDict_GetItemWithError(
        self,
        dict: PyObjectPtr,
        key: PyObjectPtr,
    ) -> PyObjectPtr:
        """Return the object from dictionary `dict` which has a key `key`.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/dict.html#c.PyDict_GetItemWithError
        """
        return self._PyDict_GetItemWithError(dict, key)

    fn PyDict_Next(
        self,
        dict: PyObjectPtr,
        pos: UnsafePointer[Py_ssize_t],
        key: UnsafePointer[PyObjectPtr],
        value: UnsafePointer[PyObjectPtr],
    ) -> c_int:
        """Iterate over all key-value pairs in the dictionary `dict`.

        References:
        - https://docs.python.org/3/c-api/dict.html#c.PyDict_Next
        """
        return self._PyDict_Next(dict, pos, key, value)

    # ===-------------------------------------------------------------------===#
    # Set Objects
    # ref: https://docs.python.org/3/c-api/set.html
    # ===-------------------------------------------------------------------===#

    fn PySet_New(self, iterable: PyObjectPtr) -> PyObjectPtr:
        """Return a new `set` containing objects returned by the `iterable`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/set.html#c.PySet_New
        """
        return self._PySet_New(iterable)

    fn PySet_Add(self, set: PyObjectPtr, key: PyObjectPtr) -> c_int:
        """Add `key` to a `set` instance.

        References:
        - https://docs.python.org/3/c-api/set.html#c.PySet_Add
        """
        return self._PySet_Add(set, key)

    # ===-------------------------------------------------------------------===#
    # Module Objects
    # ref: https://docs.python.org/3/c-api/module.html
    # ===-------------------------------------------------------------------===#

    fn PyModule_GetDict(self, module: PyObjectPtr) -> PyObjectPtr:
        """Return the dictionary object that implements `module`'s namespace;
        this object is the same as the `__dict__` attribute of the module
        object.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/module.html#c.PyModule_GetDict).
        """
        return self._PyModule_GetDict(module)

    fn PyModule_Create(self, name: StaticString) -> PyObjectPtr:
        """Create a new module object.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/module.html#c.PyModule_Create
        """

        # NOTE: See https://github.com/pybind/pybind11/blob/a1d00916b26b187e583f3bce39cd59c3b0652c32/include/pybind11/pybind11.h#L1326
        # for what we want to do here.
        var module_def_ptr = UnsafePointer[PyModuleDef].alloc(1)
        module_def_ptr.init_pointee_move(PyModuleDef(name))

        # TODO: set gil stuff
        # Note: Python automatically calls https://docs.python.org/3/c-api/module.html#c.PyState_AddModule
        # after the caller imports said module.

        # TODO: it would be nice to programmatically call a CPython API to get the value here
        # but I think it's only defined via the `PYTHON_API_VERSION` macro that ships with Python.
        # if this mismatches with the user's Python, then a `RuntimeWarning` is emitted according to the
        # docs.
        alias module_api_version: c_int = 1013
        return self._PyModule_Create2(module_def_ptr, module_api_version)

    fn PyModule_AddFunctions(
        self,
        module: PyObjectPtr,
        functions: UnsafePointer[PyMethodDef],
    ) -> c_int:
        """Add the functions from the `NULL` terminated `functions` array to
        module.

        References:
        - https://docs.python.org/3/c-api/module.html#c.PyModule_AddFunctions
        """
        return self._PyModule_AddFunctions(module, functions)

    fn PyModule_AddObjectRef(
        self,
        module: PyObjectPtr,
        name: UnsafePointer[c_char, mut=False],
        value: PyObjectPtr,
    ) -> c_int:
        """Add an object to `module` as `name`.

        References:
        - https://docs.python.org/3/c-api/module.html#c.PyModule_AddObjectRef
        """
        return self._PyModule_AddObjectRef(module, name, value)

    # ===-------------------------------------------------------------------===#
    # Slice Objects
    # ref: https://docs.python.org/3/c-api/slice.html
    # ===-------------------------------------------------------------------===#

    fn PySlice_New(
        self,
        start: PyObjectPtr,
        stop: PyObjectPtr,
        step: PyObjectPtr,
    ) -> PyObjectPtr:
        """Return a new slice object with the given values.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/slice.html#c.PySlice_New
        """
        return self._PySlice_New(start, stop, step)

    # ===-------------------------------------------------------------------===#
    # Capsules
    # ref: https://docs.python.org/3/c-api/capsule.html
    # ===-------------------------------------------------------------------===#

    fn PyCapsule_New(
        self,
        pointer: OpaquePointer,
        var name: String,
        destructor: PyCapsule_Destructor,
    ) -> PyObjectPtr:
        """Create a `PyCapsule` encapsulating the pointer. The pointer argument
        may not be `NULL`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/capsule.html#c.PyCapsule_New
        """
        return self._PyCapsule_New(pointer, name.unsafe_cstr_ptr(), destructor)

    fn PyCapsule_GetPointer(
        self,
        capsule: PyObjectPtr,
        var name: String,
    ) raises -> OpaquePointer:
        """Retrieve the pointer stored in the capsule. On failure, set an
        exception and return `NULL`.

        References:
        - https://docs.python.org/3/c-api/capsule.html#c.PyCapsule_GetPointer
        """
        var r = self._PyCapsule_GetPointer(capsule, name.unsafe_cstr_ptr())
        if self.PyErr_Occurred():
            raise self.get_error()
        return r

    # ===-------------------------------------------------------------------===#
    # Memory Management
    # ref: https://docs.python.org/3/c-api/memory.html
    # ===-------------------------------------------------------------------===#

    fn PyObject_Free(self, ptr: OpaquePointer):
        """Frees the memory block pointed to by `ptr`, which must have been
        returned by a previous call to `PyObject_Malloc()`, `PyObject_Realloc()`
        or PyObject_Calloc()`.

        References:
        - https://docs.python.org/3/c-api/memory.html#c.PyObject_Free
        """
        self._PyObject_Free(ptr)

    # ===-------------------------------------------------------------------===#
    # Object Implementation Support
    # ref: https://docs.python.org/3/c-api/objimpl.html
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # Common Object Structures
    # ref: https://docs.python.org/3/c-api/structures.html
    # ===-------------------------------------------------------------------===#

    fn Py_Is(self, x: PyObjectPtr, y: PyObjectPtr) -> c_int:
        """Test if the `x` object is the `y` object, the same as `x is y` in
        Python.

        Part of the Stable ABI since version 3.10.
        This function is patched to work with Python 3.9 and earlier versions.

        References:
        - https://docs.python.org/3/c-api/structures.html#c.Py_Is
        """
        if self.version.minor < 11:
            return c_int(Int(x == y))
        return self._Py_Is(x, y)

    fn Py_TYPE(self, obj: PyObjectPtr) -> PyTypeObjectPtr:
        """Get the type of the Python object `obj`.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/structures.html#c.Py_TYPE
        - https://docs.python.org/3/c-api/typeobj.html#c.Py_TYPE
        """
        # Note:
        #   The `Py_TYPE` function is a `static` function in the C API, so
        #   we can't call it directly. Instead we reproduce its (trivial)
        #   behavior here.
        # TODO(MSTDL-977):
        #   Investigate doing this without hard-coding private API details.

        # TODO(MSTDL-950): Should use something like `addr_of!`
        return obj._unsized_obj_ptr[].object_type

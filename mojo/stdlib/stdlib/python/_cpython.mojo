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

from collections import InlineArray, Optional
from collections.string.string_slice import get_static_string
from os import abort, getenv, setenv
from os.path import dirname
from pathlib import Path
from sys import external_call
from sys.arg import argv
from sys.ffi import (
    DLHandle,
    OpaquePointer,
    c_char,
    c_int,
    c_long,
    c_size_t,
    c_ssize_t,
    c_uint,
)

from memory import UnsafePointer
from python.bindings import (
    Typed_initproc,
    Typed_newfunc,
)
from builtin.identifiable import TypeIdentifiable

# ===-----------------------------------------------------------------------===#
# Raw Bindings
# ===-----------------------------------------------------------------------===#

# https://github.com/python/cpython/blob/d45225bd66a8123e4a30314c627f2586293ba532/Include/compile.h#L7
alias Py_single_input = 256
alias Py_file_input = 257
alias Py_eval_input = 258
alias Py_func_type_input = 345

alias Py_tp_dealloc = 52
alias Py_tp_init = 60
alias Py_tp_methods = 64
alias Py_tp_new = 65
alias Py_tp_repr = 66

alias Py_TPFLAGS_DEFAULT = 0

alias Py_ssize_t = c_ssize_t
alias Py_hash_t = Py_ssize_t

# TODO(MOCO-1138):
#   This should be a C ABI function pointer, not a Mojo ABI function.
alias PyCFunction = fn (PyObjectPtr, PyObjectPtr) -> PyObjectPtr
"""[Reference](https://docs.python.org/3/c-api/structures.html#c.PyCFunction).
"""

alias METH_VARARGS = 0x1

alias destructor = fn (PyObjectPtr) -> None

alias reprfunc = fn (PyObjectPtr) -> PyObjectPtr

alias initproc = fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> c_int
alias newfunc = fn (
    UnsafePointer[PyTypeObject], PyObjectPtr, PyObjectPtr
) -> PyObjectPtr


# GIL
@fieldwise_init
@register_passable("trivial")
struct PyGILState_STATE:
    """Represents the state of the Python Global Interpreter Lock (GIL).

    Notes:
        This struct is used to store and manage the state of the GIL, which is
        crucial for thread-safe operations in Python. [Reference](
        https://github.com/python/cpython/blob/d45225bd66a8123e4a30314c627f2586293ba532/Include/pystate.h#L76
        ).
    """

    var current_state: c_int
    """The current state of the GIL."""

    alias PyGILState_LOCKED = c_int(0)
    alias PyGILState_UNLOCKED = c_int(1)


struct PyThreadState:
    """Opaque struct."""

    pass


@fieldwise_init
@register_passable("trivial")
struct PyKeysValuePair:
    """Represents a key-value pair in a Python dictionary iteration.

    This struct is used to store the result of iterating over a Python dictionary,
    containing the key, value, current position, and success status of the iteration.
    """

    var key: PyObjectPtr
    """The key of the current dictionary item."""
    var value: PyObjectPtr
    """The value of the current dictionary item."""
    var position: c_int
    """The current position in the dictionary iteration."""
    var success: Bool
    """Indicates whether the iteration was successful."""


@fieldwise_init
@register_passable("trivial")
struct PyObjectPtr(Copyable, Movable):
    """Equivalent to `PyObject*` in C.

    It is crucial that this type has the same size and alignment as `PyObject*`
    for FFI ABI correctness.

    This struct provides methods for initialization, null checking,
    equality comparison, and conversion to integer representation.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var unsized_obj_ptr: UnsafePointer[PyObject]

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
        self.unsized_obj_ptr = UnsafePointer[PyObject]()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __eq__(self, rhs: PyObjectPtr) -> Bool:
        """Compare two PyObjectPtr for equality.

        Args:
            rhs: The right-hand side PyObjectPtr to compare.

        Returns:
            Bool: True if the pointers are equal, False otherwise.
        """
        return Int(self.unsized_obj_ptr) == Int(rhs.unsized_obj_ptr)

    fn __ne__(self, rhs: PyObjectPtr) -> Bool:
        """Compare two PyObjectPtr for inequality.

        Args:
            rhs: The right-hand side PyObjectPtr to compare.

        Returns:
            Bool: True if the pointers are not equal, False otherwise.
        """
        return not (self == rhs)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn is_null(self) -> Bool:
        """Check if the pointer is null.

        Returns:
            Bool: True if the pointer is null, False otherwise.
        """
        return Int(self.unsized_obj_ptr) == 0

    # TODO: Consider removing this and inlining Int(p.value) into callers
    fn _get_ptr_as_int(self) -> Int:
        """Get the pointer value as an integer.

        Returns:
            Int: The integer representation of the pointer.
        """
        return Int(self.unsized_obj_ptr)


@fieldwise_init
@register_passable
struct PythonVersion(Copyable, Movable):
    """Represents a Python version with major, minor, and patch numbers."""

    var major: Int
    """The major version number."""
    var minor: Int
    """The minor version number."""
    var patch: Int
    """The patch version number."""

    @implicit
    fn __init__(out self, version: StringSlice):
        """Initialize a PythonVersion object from a version string.

        Args:
            version: A string representing the Python version (e.g., "3.9.5").

        The version string is parsed to extract major, minor, and patch numbers.
        If parsing fails for any component, it defaults to -1.
        """
        var components = InlineArray[Int, 3](-1)
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


fn _py_finalize(lib: DLHandle):
    lib.call["Py_Finalize"]()


@fieldwise_init
struct PyMethodDef(Copyable, Movable):
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

    # TODO(MSTDL-887): Support keyword-argument only methods
    var method_impl: PyCFunction
    """A function pointer to the implementation of the method."""

    var method_flags: c_int
    """Flags indicating how the method should be called. [Reference](
    https://docs.python.org/3/c-api/structures.html#c.PyMethodDef)."""

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
        self.method_impl = _null_fn_ptr[PyCFunction]()
        self.method_flags = 0
        self.method_docstring = UnsafePointer[c_char]()

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @staticmethod
    fn function(
        func: PyCFunction,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> Self:
        """Create a PyMethodDef for a function.

        Arguments:
            func: The function to wrap.
            func_name: The name of the function.
            docstring: The docstring for the function.
        """
        # TODO(MSTDL-896):
        #   Support a way to get the name of the function from its parameter
        #   type, similar to `get_linkage_name()`?

        # FIXME: PyMethodDef is capturing the pointer without an origin.

        return PyMethodDef(
            func_name.unsafe_ptr().bitcast[c_char](),
            func,
            METH_VARARGS,
            docstring.unsafe_ptr().bitcast[c_char](),
        )


fn _null_fn_ptr[T: AnyTrivialRegType]() -> T:
    return __mlir_op.`pop.pointer.bitcast`[_type=T](
        __mlir_attr.`#interp.pointer<0> : !kgen.pointer<none>`
    )


struct PyTypeObject:
    """The opaque C structure of the objects used to describe types.

    Notes:
        [Reference](https://docs.python.org/3/c-api/type.html#c.PyTypeObject).
    """

    # TODO(MSTDL-877):
    #   Fill this out based on
    #   https://docs.python.org/3/c-api/typeobj.html#pytypeobject-definition
    pass


@fieldwise_init
@register_passable("trivial")
struct PyType_Spec:
    """Structure defining a type's behavior.

    Notes:
        [Reference](https://docs.python.org/3/c-api/type.html#c.PyType_Spec).
    """

    var name: UnsafePointer[c_char]
    var basicsize: c_int
    var itemsize: c_int
    var flags: c_uint
    var slots: UnsafePointer[PyType_Slot]


@fieldwise_init
@register_passable("trivial")
struct PyType_Slot(Copyable, Movable):
    """Structure defining optional functionality of a type, containing a slot ID
    and a value pointer.

    Notes:
        [Reference](https://docs.python.org/3/c-api/type.html#c.PyType_Slot).
    """

    var slot: c_int
    var pfunc: OpaquePointer

    @staticmethod
    fn tp_new(func: Typed_newfunc) -> Self:
        return PyType_Slot(Py_tp_new, rebind[OpaquePointer](func))

    @staticmethod
    fn tp_init(func: Typed_initproc) -> Self:
        return PyType_Slot(Py_tp_init, rebind[OpaquePointer](func))

    @staticmethod
    fn tp_dealloc(func: destructor) -> Self:
        return PyType_Slot(Py_tp_dealloc, rebind[OpaquePointer](func))

    @staticmethod
    fn tp_methods(methods: UnsafePointer[PyMethodDef]) -> Self:
        return PyType_Slot(Py_tp_methods, rebind[OpaquePointer](methods))

    @staticmethod
    fn tp_repr(func: reprfunc) -> Self:
        return PyType_Slot(Py_tp_repr, rebind[OpaquePointer](func))

    @staticmethod
    fn null() -> Self:
        return PyType_Slot(0, OpaquePointer())


@fieldwise_init
struct PyObject(Stringable, Representable, Writable, Copyable, Movable):
    """All object types are extensions of this type. This is a type which
    contains the information Python needs to treat a pointer to an object as an
    object. In a normal “release” build, it contains only the object's reference
    count and a pointer to the corresponding type object. Nothing is actually
    declared to be a PyObject, but every pointer to a Python object can be cast
    to a PyObject.

    Notes:
        [Reference](https://docs.python.org/3/c-api/structures.html#c.PyObject).
    """

    var object_ref_count: Int
    var object_type: UnsafePointer[PyTypeObject]

    fn __init__(out self):
        self.object_ref_count = 0
        self.object_type = UnsafePointer[PyTypeObject]()

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

    fn write_to[W: Writer](self, mut writer: W):
        """Formats to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("PyObject(")
        writer.write("object_ref_count=", self.object_ref_count, ",")
        writer.write("object_type=", self.object_type)
        writer.write(")")


# Mojo doesn't have macros, so we define it here for ease.
struct PyModuleDef_Base(Stringable, Representable, Writable):
    """PyModuleDef_Base.

    Notes:
        [Reference 1](
        https://github.com/python/cpython/blob/833c58b81ebec84dc24ef0507f8c75fe723d9f66/Include/moduleobject.h#L39
        ). [Reference 2](
        https://pyo3.rs/main/doc/pyo3/ffi/struct.pymoduledef_base
        ). `PyModuleDef_HEAD_INIT` defaults all of its members, [Reference 3](
        https://github.com/python/cpython/blob/833c58b81ebec84dc24ef0507f8c75fe723d9f66/Include/moduleobject.h#L60
        ).
    """

    var object_base: PyObject
    """The initial segment of every `PyObject` in CPython."""

    # TODO(MOCO-1138): This is a C ABI function pointer, not Mojo a function.
    alias _init_fn_type = fn () -> UnsafePointer[PyObject]
    """The function used to re-initialize the module."""
    var init_fn: Self._init_fn_type

    var index: Py_ssize_t
    """The module's index into its interpreter's modules_by_index cache."""

    var dict_copy: UnsafePointer[PyObject]
    """A copy of the module's __dict__ after the first time it was loaded."""

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    fn __init__(out self):
        self.object_base = PyObject()
        self.init_fn = _null_fn_ptr[Self._init_fn_type]()
        self.index = 0
        self.dict_copy = UnsafePointer[PyObject]()

    fn __moveinit__(out self, owned existing: Self):
        self.object_base = existing.object_base
        self.init_fn = existing.init_fn
        self.index = existing.index
        self.dict_copy = existing.dict_copy

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

    fn write_to[W: Writer](self, mut writer: W):
        """Formats to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

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
    """[Reference](
    https://docs.python.org/3/c-api/module.html#c.PyModuleDef_Slot).
    """

    var slot: c_int
    var value: OpaquePointer


struct PyModuleDef(Stringable, Representable, Writable, Movable):
    """The Python module definition structs that holds all of the information
    needed to create a module.

    Notes:
        [Reference](https://docs.python.org/3/c-api/module.html#c.PyModuleDef).
    """

    var base: PyModuleDef_Base

    var name: UnsafePointer[c_char]
    """[Reference](https://docs.python.org/3/c-api/structures.html#c.PyMethodDef
    )."""

    var docstring: UnsafePointer[c_char]
    """Points to the contents of the docstring for the module."""

    var size: Py_ssize_t

    var methods: UnsafePointer[PyMethodDef]
    """A pointer to a table of module-level functions.  Can be null if there
    are no functions present."""

    var slots: UnsafePointer[PyModuleDef_Slot]

    # TODO(MOCO-1138): These are C ABI function pointers, not Mojo functions.
    alias _visitproc_fn_type = fn (PyObjectPtr, OpaquePointer) -> c_int
    alias _traverse_fn_type = fn (
        PyObjectPtr, Self._visitproc_fn_type, OpaquePointer
    ) -> c_int
    var traverse_fn: Self._traverse_fn_type

    alias _clear_fn_type = fn (PyObjectPtr) -> c_int
    var clear_fn: Self._clear_fn_type

    alias _free_fn_type = fn (OpaquePointer) -> OpaquePointer
    var free_fn: Self._free_fn_type

    @implicit
    fn __init__(out self, name: StaticString):
        self.base = PyModuleDef_Base()
        self.name = name.unsafe_ptr().bitcast[c_char]()
        self.docstring = UnsafePointer[c_char]()
        # means that the module does not support sub-interpreters
        self.size = -1
        self.methods = UnsafePointer[PyMethodDef]()
        self.slots = UnsafePointer[PyModuleDef_Slot]()

        self.slots = UnsafePointer[PyModuleDef_Slot]()
        self.traverse_fn = _null_fn_ptr[Self._traverse_fn_type]()
        self.clear_fn = _null_fn_ptr[Self._clear_fn_type]()
        self.free_fn = _null_fn_ptr[Self._free_fn_type]()

    fn __moveinit__(out self, owned existing: Self):
        self.base = existing.base^
        self.name = existing.name
        self.docstring = existing.docstring
        self.size = existing.size
        self.methods = existing.methods
        self.slots = existing.slots
        self.traverse_fn = existing.traverse_fn
        self.clear_fn = existing.clear_fn
        self.free_fn = existing.free_fn

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

    fn write_to[W: Writer](self, mut writer: W):
        """Formats to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

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
    fn load(lib: DLHandle) -> type:
        """Loads this external function from an opened dynamic library."""
        return lib._get_function[name, type]()


# int PyList_SetItem(PyObject *list, Py_ssize_t index, PyObject *item)
alias PyList_SetItem = ExternalFunction[
    "PyList_SetItem",
    fn (PyObjectPtr, Py_ssize_t, PyObjectPtr) -> c_int,
]

alias Py_IncRef = ExternalFunction[
    "Py_IncRef",
    fn (PyObjectPtr) -> None,
]

alias Py_DecRef = ExternalFunction[
    "Py_DecRef",
    fn (PyObjectPtr) -> None,
]

alias PyLong_FromSsize_t = ExternalFunction[
    "PyLong_FromSsize_t",
    fn (c_ssize_t) -> PyObjectPtr,
]


@fieldwise_init
struct CPython(Copyable, Movable):
    """Handle to the CPython interpreter present in the current process."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var lib: DLHandle
    """The handle to the CPython shared library."""
    var logging_enabled: Bool
    """Whether logging is enabled."""
    var version: PythonVersion
    """The version of the Python runtime."""
    var total_ref_count: UnsafePointer[Int]
    """The total reference count of all Python objects."""
    var init_error: StringSlice[StaticConstantOrigin]
    """An error message if initialization failed."""

    var Py_IncRef_func: Py_IncRef.type
    var Py_DecRef_func: Py_DecRef.type
    var PyLong_FromSsize_t_func: PyLong_FromSsize_t.type
    var PyList_SetItem_func: PyList_SetItem.type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        var logging_enabled = getenv("MODULAR_CPYTHON_LOGGING") == "ON"
        if logging_enabled:
            print("CPython init")
            print("MOJO_PYTHON:", getenv("MOJO_PYTHON"))
            print("MOJO_PYTHON_LIBRARY:", getenv("MOJO_PYTHON_LIBRARY"))

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
                _ = setenv("PYTHONPATH", file_dir + ":" + python_path)
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

        if logging_enabled:
            print("PYTHONEXECUTABLE:", getenv("PYTHONEXECUTABLE"))
            print("libpython selected:", python_lib)

        try:
            # Note:
            #   MOJO_PYTHON_LIBRARY can be "" when the current Mojo program
            #   is a dynamic library being loaded as a Python extension module,
            #   and we need to find CPython symbols that are statically linked
            #   into the `python` main executable. On those paltforms where
            #   `python` executable can be statically linked (Linux), it's
            #   important that we don't load a second copy of CPython symbols
            #   into the process by loading the `libpython` dynamic library.
            self.lib = DLHandle(python_lib) if python_lib != "" else DLHandle()
        except e:
            self.lib = abort[DLHandle](
                String("Failed to load libpython from", python_lib, ":\n", e)
            )
        self.total_ref_count = UnsafePointer[Int].alloc(1)
        self.logging_enabled = logging_enabled
        if not self.init_error:
            if not self.lib.check_symbol("Py_Initialize"):
                self.init_error = "compatible Python library not found"
            self.lib.call["Py_Initialize"]()
            self.version = PythonVersion(_py_get_version(self.lib))
        else:
            self.version = PythonVersion(0, 0, 0)

        self.Py_IncRef_func = Py_IncRef.load(self.lib)
        self.Py_DecRef_func = Py_DecRef.load(self.lib)
        self.PyLong_FromSsize_t_func = PyLong_FromSsize_t.load(self.lib)
        self.PyList_SetItem_func = PyList_SetItem.load(self.lib)

    fn __del__(owned self):
        pass

    @staticmethod
    fn destroy(mut existing: CPython):
        if existing.logging_enabled:
            print("CPython destroy")
            var remaining_refs = existing.total_ref_count.take_pointee()
            print("Number of remaining refs:", remaining_refs)
            # Technically not necessary since we're working with register
            # passable types, by it's good practice to re-initialize the
            # pointer after a consuming move.
            existing.total_ref_count.init_pointee_move(remaining_refs)
        _py_finalize(existing.lib)
        existing.lib.close()
        existing.total_ref_count.free()

    fn check_init_error(self) raises:
        """Used for entry points that initialize Python on first use, will
        raise an error if one occurred when initializing the global CPython.
        """
        if self.init_error:
            var error = String(self.init_error)
            var mojo_python = getenv("MOJO_PYTHON")
            var python_lib = getenv("MOJO_PYTHON_LIBRARY")
            var python_exe = getenv("PYTHONEXECUTABLE")
            if mojo_python:
                error += "\nMOJO_PYTHON: " + mojo_python
            if python_lib:
                error += "\nMOJO_PYTHON_LIBRARY: " + python_lib
            if python_exe:
                error += "\npython executable: " + python_exe
            error += "\n\nMojo/Python interop error, troubleshooting docs at:"
            error += "\n    https://modul.ar/fix-python\n"
            raise error

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
        debug_assert(
            self.PyErr_Occurred(),
            "invalid unchecked conversion of Python error to Mojo error",
        )

        # TODO(MSTDL-1479): PyErr_Fetch is deprecated since Python 3.12.
        var err_ptr = self.PyErr_Fetch()
        debug_assert(
            not err_ptr.is_null(),
            "Python exception occurred but PyErr_Fetch returned null",
        )

        var error: Error
        try:
            error = String(PythonObject(from_owned_ptr=err_ptr))
        except e:
            return abort[Error](
                "internal error: Python exception occurred but cannot be"
                " converted to String"
            )

        self.PyErr_Clear()
        return error

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

    # ===-------------------------------------------------------------------===#
    # Logging
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn log[*Ts: Writable](self, *args: *Ts):
        """If logging is enabled, print the given arguments as a log message.

        Parameters:
            Ts: The argument types.

        Arguments:
            args: The arguments to log.
        """
        if not self.logging_enabled:
            return

        # TODO(MOCO-358):
        #   Once Mojo argument splatting is supported, this should just
        #   be: `print(*args)`
        @parameter
        for i in range(args.__len__()):
            print(args[i], sep="", end="", flush=False)

        print(flush=True)

    # ===-------------------------------------------------------------------===#
    # Reference count management
    # ===-------------------------------------------------------------------===#

    fn _inc_total_rc(self):
        var v = self.total_ref_count.take_pointee()
        self.total_ref_count.init_pointee_move(v + 1)

    fn _dec_total_rc(self):
        var v = self.total_ref_count.take_pointee()
        self.total_ref_count.init_pointee_move(v - 1)

    fn Py_IncRef(self, ptr: PyObjectPtr):
        """[Reference](
        https://docs.python.org/3/c-api/refcounting.html#c.Py_IncRef).
        """

        self.log(ptr._get_ptr_as_int(), " INCREF refcnt:", self._Py_REFCNT(ptr))

        self.Py_IncRef_func(ptr)
        self._inc_total_rc()

    fn Py_DecRef(self, ptr: PyObjectPtr):
        """[Reference](
        https://docs.python.org/3/c-api/refcounting.html#c.Py_DecRef).
        """

        self.log(ptr._get_ptr_as_int(), " DECREF refcnt:", self._Py_REFCNT(ptr))
        self.Py_DecRef_func(ptr)
        self._dec_total_rc()

    # This function assumes a specific way PyObjectPtr is implemented, namely
    # that the refcount has offset 0 in that structure. That generally doesn't
    # have to always be the case - but often it is and it's convenient for
    # debugging. We shouldn't rely on this function anywhere - its only purpose
    # is debugging.
    fn _Py_REFCNT(self, ptr: PyObjectPtr) -> Int:
        if ptr._get_ptr_as_int() == 0:
            return -1
        # NOTE:
        #   The "obvious" way to write this would be:
        #       return ptr.unsized_obj_ptr[].object_ref_count
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
        return ptr.unsized_obj_ptr.bitcast[Int]()[]

    # ===-------------------------------------------------------------------===#
    # Python GIL and threading
    # ===-------------------------------------------------------------------===#

    fn PyGILState_Ensure(self) -> PyGILState_STATE:
        """[Reference](
        https://docs.python.org/3/c-api/init.html#c.PyGILState_Ensure).
        """
        return self.lib.call["PyGILState_Ensure", PyGILState_STATE]()

    fn PyGILState_Release(self, state: PyGILState_STATE):
        """[Reference](
        https://docs.python.org/3/c-api/init.html#c.PyGILState_Release).
        """
        self.lib.call["PyGILState_Release"](state)

    fn PyEval_SaveThread(self) -> UnsafePointer[PyThreadState]:
        """[Reference](
        https://docs.python.org/3/c-api/init.html#c.PyEval_SaveThread).
        """

        return self.lib.call[
            "PyEval_SaveThread", UnsafePointer[PyThreadState]
        ]()

    fn PyEval_RestoreThread(self, state: UnsafePointer[PyThreadState]):
        """[Reference](
        https://docs.python.org/3/c-api/init.html#c.PyEval_RestoreThread).
        """
        self.lib.call["PyEval_RestoreThread"](state)

    # ===-------------------------------------------------------------------===#
    # Python Set operations
    # ===-------------------------------------------------------------------===#

    fn PySet_New(self) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/set.html#c.PySet_New).
        """

        var r = self.lib.call["PySet_New", PyObjectPtr](PyObjectPtr())

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PySet_New, refcnt:",
            self._Py_REFCNT(r),
        )

        self._inc_total_rc()
        return r

    # int PySet_Add(PyObject *set, PyObject *key)
    fn PySet_Add(self, set: PyObjectPtr, element: PyObjectPtr) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/set.html#c.PySet_Add).
        """

        var r = self.lib.call["PySet_Add", c_int](set, element)
        self.log(
            set._get_ptr_as_int(),
            " PySet_Add, element: ",
            element._get_ptr_as_int(),
        )
        return r

    # ===-------------------------------------------------------------------===#
    # Python Dict operations
    # ===-------------------------------------------------------------------===#

    fn PyDict_New(self) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_New).
        """

        var r = self.lib.call["PyDict_New", PyObjectPtr]()

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyDict_New, refcnt:",
            self._Py_REFCNT(r),
        )

        self._inc_total_rc()
        return r

    # int PyDict_SetItem(PyObject *p, PyObject *key, PyObject *val)
    fn PyDict_SetItem(
        self, dict_obj: PyObjectPtr, key: PyObjectPtr, value: PyObjectPtr
    ) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_SetItem).
        """

        var r = self.lib.call["PyDict_SetItem", c_int](dict_obj, key, value)

        self.log(
            "PyDict_SetItem, key: ",
            key._get_ptr_as_int(),
            " value: ",
            value._get_ptr_as_int(),
        )

        return r

    fn PyDict_GetItemWithError(
        self, dict_obj: PyObjectPtr, key: PyObjectPtr
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_GetItemWithError).
        """

        var r = self.lib.call["PyDict_GetItemWithError", PyObjectPtr](
            dict_obj, key
        )
        self.log("PyDict_GetItemWithError, key: ", key._get_ptr_as_int())
        return r

    fn PyDict_Check(self, maybe_dict: PyObjectPtr) -> Bool:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_Check).
        """

        var my_type = self.PyObject_Type(maybe_dict)
        var my_type_as_int = my_type._get_ptr_as_int()
        var dict_type = self.PyDict_Type()
        var result = my_type_as_int == dict_type._get_ptr_as_int()
        self.Py_DecRef(my_type)
        return result

    fn PyDict_Type(self) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_Type).
        """
        return self.lib.call["PyDict_Type", PyObjectPtr]()

    # int PyDict_Next(PyObject *p, Py_ssize_t *ppos, PyObject **pkey, PyObject **pvalue)
    fn PyDict_Next(self, dictionary: PyObjectPtr, p: Int) -> PyKeysValuePair:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_Next).
        """
        var key = PyObjectPtr()
        var value = PyObjectPtr()
        var v = p
        var position = UnsafePointer[Int](to=v)
        var result = self.lib.call["PyDict_Next", c_int](
            dictionary,
            position,
            UnsafePointer(to=key),
            UnsafePointer(to=value),
        )

        self.log(
            dictionary._get_ptr_as_int(),
            " NEWREF PyDict_Next",
            dictionary._get_ptr_as_int(),
            "refcnt:",
            self._Py_REFCNT(dictionary),
            " key: ",
            key._get_ptr_as_int(),
            ", refcnt(key):",
            self._Py_REFCNT(key),
            "value:",
            value._get_ptr_as_int(),
            "refcnt(value)",
            self._Py_REFCNT(value),
        )

        _ = v
        return PyKeysValuePair(
            key,
            value,
            position.take_pointee(),
            result == 1,
        )

    # ===-------------------------------------------------------------------===#
    # Python Module operations
    # ===-------------------------------------------------------------------===#

    fn PyImport_ImportModule(
        self,
        owned name: String,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/import.html#c.PyImport_ImportModule).
        """

        var r = self.lib.call["PyImport_ImportModule", PyObjectPtr](
            name.unsafe_cstr_ptr()
        )

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyImport_ImportModule, str:",
            name,
            ", refcnt:",
            self._Py_REFCNT(r),
        )

        self._inc_total_rc()
        return r

    fn PyImport_AddModule(self, owned name: String) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/import.html#c.PyImport_AddModule).
        """
        return self.lib.call["PyImport_AddModule", PyObjectPtr](
            name.unsafe_cstr_ptr()
        )

    fn PyModule_Create(
        self,
        name: StaticString,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/module.html#c.PyModule_Create).
        """

        # TODO: See https://docs.python.org/3/c-api/module.html#c.PyModule_Create
        # and https://github.com/pybind/pybind11/blob/a1d00916b26b187e583f3bce39cd59c3b0652c32/include/pybind11/pybind11.h#L1326
        # for what we want to do essentially here.
        var module_def_ptr = UnsafePointer[PyModuleDef].alloc(1)
        var module_def = PyModuleDef(name)
        module_def_ptr.init_pointee_move(module_def^)

        # TODO: set gil stuff
        # Note: Python automatically calls https://docs.python.org/3/c-api/module.html#c.PyState_AddModule
        # after the caller imports said module.

        # TODO: it would be nice to programatically call a CPython API to get the value here
        # but I think it's only defined via the `PYTHON_API_VERSION` macro that ships with Python.
        # if this mismatches with the user's Python, then a `RuntimeWarning` is emitted according to the
        # docs.
        var module_api_version = 1013
        return self.lib.call["PyModule_Create2", PyObjectPtr](
            module_def_ptr, module_api_version
        )

    fn PyModule_AddFunctions(
        self,
        mod: PyObjectPtr,
        functions: UnsafePointer[PyMethodDef],
    ) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/module.html#c.PyModule_AddFunctions).
        """
        return self.lib.call["PyModule_AddFunctions", c_int](mod, functions)

    fn PyModule_AddObjectRef(
        self,
        module: PyObjectPtr,
        name: UnsafePointer[c_char, **_],
        value: PyObjectPtr,
    ) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/module.html#c.PyModule_AddObjectRef).
        """

        return self.lib.call["PyModule_AddObjectRef", c_int](
            module, name, value
        )

    fn PyModule_GetDict(self, name: PyObjectPtr) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/module.html#c.PyModule_GetDict).
        """
        return self.lib.call["PyModule_GetDict", PyObjectPtr](name)

    # ===-------------------------------------------------------------------===#
    # Python Type operations
    # ===-------------------------------------------------------------------===#

    fn Py_TYPE(self, ob_raw: PyObjectPtr) -> UnsafePointer[PyTypeObject]:
        """Get the PyTypeObject field of a Python object."""

        # Note:
        #   The `Py_TYPE` function is a `static` function in the C API, so
        #   we can't call it directly. Instead we reproduce its (trivial)
        #   behavior here.
        # TODO(MSTDL-977):
        #   Investigate doing this without hard-coding private API details.

        # TODO(MSTDL-950): Should use something like `addr_of!`
        return ob_raw.unsized_obj_ptr[].object_type

    fn PyType_GetName(self, type: UnsafePointer[PyTypeObject]) -> PyObjectPtr:
        return self.lib.call["PyType_GetName", PyObjectPtr](type)

    fn PyType_FromSpec(self, spec: UnsafePointer[PyType_Spec]) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/type.html#c.PyType_FromSpec).
        """
        return self.lib.call["PyType_FromSpec", PyObjectPtr](spec)

    fn PyType_GenericAlloc(
        self,
        type: UnsafePointer[PyTypeObject],
        nitems: Py_ssize_t,
    ) -> PyObjectPtr:
        return self.lib.call["PyType_GenericAlloc", PyObjectPtr](type, nitems)

    # ===-------------------------------------------------------------------===#
    # Python Evaluation
    # ===-------------------------------------------------------------------===#

    fn PyRun_SimpleString(self, owned str: String) -> Bool:
        """Executes the given Python code.

        Args:
            str: The python code to execute.

        Returns:
            `True` if the code executed successfully or `False` if the code
            raised an exception.

        Notes:
            [Reference](
            https://docs.python.org/3/c-api/veryhigh.html#c.PyRun_SimpleString).
        """
        return (
            self.lib.call["PyRun_SimpleString", c_int](str.unsafe_cstr_ptr())
            == 0
        )

    fn PyRun_String(
        self,
        owned str: String,
        globals: PyObjectPtr,
        locals: PyObjectPtr,
        run_mode: Int,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/veryhigh.html#c.PyRun_String).
        """
        var result = self.lib.call["PyRun_String", PyObjectPtr](
            str.unsafe_cstr_ptr(), Int32(run_mode), globals, locals
        )

        self.log(
            result._get_ptr_as_int(),
            " NEWREF PyRun_String, str:",
            str,
            ", ptr: ",
            result._get_ptr_as_int(),
            ", refcnt:",
            self._Py_REFCNT(result),
        )

        self._inc_total_rc()
        return result

    fn PyEval_EvalCode(
        self,
        co: PyObjectPtr,
        globals: PyObjectPtr,
        locals: PyObjectPtr,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/veryhigh.html#c.PyEval_EvalCode).
        """
        var result = self.lib.call["PyEval_EvalCode", PyObjectPtr](
            co, globals, locals
        )
        self._inc_total_rc()
        return result

    fn PyEval_GetBuiltins(self) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/reflection.html#c.PyEval_GetBuiltins).
        """
        return self.lib.call["PyEval_GetBuiltins", PyObjectPtr]()

    fn Py_CompileString(
        self,
        owned str: String,
        owned filename: String,
        compile_mode: Int,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/veryhigh.html#c.Py_CompileString).
        """

        var r = self.lib.call["Py_CompileString", PyObjectPtr](
            str.unsafe_cstr_ptr(),
            filename.unsafe_cstr_ptr(),
            Int32(compile_mode),
        )
        self._inc_total_rc()
        return r

    # ===-------------------------------------------------------------------===#
    # Python Object operations
    # ===-------------------------------------------------------------------===#

    fn Py_Is(
        self,
        rhs: PyObjectPtr,
        lhs: PyObjectPtr,
    ) -> Bool:
        """[Reference](
        https://docs.python.org/3/c-api/structures.html#c.Py_Is).
        """

        if self.version.minor >= 10:
            # int Py_Is(PyObject *x, PyObject *y)
            return self.lib.call["Py_Is", c_int](rhs, lhs) > 0
        else:
            return rhs == lhs

    fn PyObject_Type(self, obj: PyObjectPtr) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_Type).
        """

        var p = self.lib.call["PyObject_Type", PyObjectPtr](obj)
        self._inc_total_rc()
        return p

    fn PyObject_Free(self, p: OpaquePointer):
        """[Reference](
        https://docs.python.org/3/c-api/memory.html#c.PyObject_Free).
        """
        self.lib.call["PyObject_Free"](p)

    fn PyObject_Str(self, obj: PyObjectPtr) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_Str).
        """

        var p = self.lib.call["PyObject_Str", PyObjectPtr](obj)
        self._inc_total_rc()
        return p

    fn PyObject_GetItem(
        self, obj: PyObjectPtr, key: PyObjectPtr
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_GetItem).
        """

        var r = self.lib.call["PyObject_GetItem", PyObjectPtr](obj, key)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyObject_GetItem, key:",
            key._get_ptr_as_int(),
            ", refcnt:",
            self._Py_REFCNT(r),
            ", parent obj:",
            obj._get_ptr_as_int(),
        )

        self._inc_total_rc()
        return r

    fn PyObject_SetItem(
        self, obj: PyObjectPtr, key: PyObjectPtr, value: PyObjectPtr
    ) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_SetItem).
        """

        var r = self.lib.call["PyObject_SetItem", c_int](obj, key, value)

        self.log(
            "PyObject_SetItem result:",
            r,
            ", key:",
            key._get_ptr_as_int(),
            ", value:",
            value._get_ptr_as_int(),
            ", parent obj:",
            obj._get_ptr_as_int(),
        )
        return r

    fn PyObject_HasAttrString(
        self,
        obj: PyObjectPtr,
        owned name: String,
    ) -> c_int:
        """Returns `1` if `obj` has the attribute `attr_name`, and `0` otherwise.

        [Reference](https://docs.python.org/3/c-api/object.html#c.PyObject_HasAttrString).
        """
        # int PyObject_HasAttrString(PyObject *o, const char *attr_name)
        return self.lib.call["PyObject_HasAttrString", c_int](
            obj, name.unsafe_cstr_ptr()
        )

    fn PyObject_GetAttrString(
        self,
        obj: PyObjectPtr,
        owned name: String,
    ) -> PyObjectPtr:
        """Retrieve an attribute named `name` from object `obj`.

        [Reference](https://docs.python.org/3/c-api/object.html#c.PyObject_GetAttrString).
        """
        # PyObject *PyObject_GetAttrString(PyObject *o, const char *attr_name)
        var r = self.lib.call["PyObject_GetAttrString", PyObjectPtr](
            obj, name.unsafe_cstr_ptr()
        )

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyObject_GetAttrString, str:",
            name,
            ", refcnt:",
            self._Py_REFCNT(r),
            ", parent obj:",
            obj._get_ptr_as_int(),
        )

        self._inc_total_rc()
        return r

    fn PyObject_SetAttrString(
        self,
        obj: PyObjectPtr,
        owned name: String,
        new_value: PyObjectPtr,
    ) -> c_int:
        """Set the value of the attribute named `name`, for object `obj`, to the value `new_value`.

        [Reference](https://docs.python.org/3/c-api/object.html#c.PyObject_SetAttrString).
        """
        # int PyObject_SetAttrString(PyObject *o, const char *attr_name, PyObject *v)
        var r = self.lib.call["PyObject_SetAttrString", c_int](
            obj, name.unsafe_cstr_ptr(), new_value
        )

        self.log(
            "PyObject_SetAttrString str:",
            name,
            ", parent obj:",
            obj._get_ptr_as_int(),
            ", new value:",
            new_value._get_ptr_as_int(),
            " new value ref count: ",
            self._Py_REFCNT(new_value),
        )

        return r

    fn PyObject_CallObject(
        self,
        callable_obj: PyObjectPtr,
        args: PyObjectPtr,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/call.html#c.PyObject_CallObject).
        """

        var r = self.lib.call["PyObject_CallObject", PyObjectPtr](
            callable_obj, args
        )

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyObject_CallObject, refcnt:",
            self._Py_REFCNT(r),
            ", callable obj:",
            callable_obj._get_ptr_as_int(),
        )

        self._inc_total_rc()
        return r

    fn PyObject_Call(
        self,
        callable_obj: PyObjectPtr,
        args: PyObjectPtr,
        kwargs: PyObjectPtr,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/call.html#c.PyObject_Call).
        """

        var r = self.lib.call["PyObject_Call", PyObjectPtr](
            callable_obj, args, kwargs
        )

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyObject_Call, refcnt:",
            self._Py_REFCNT(r),
            ", callable obj:",
            callable_obj._get_ptr_as_int(),
        )

        self._inc_total_rc()
        return r

    fn PyObject_IsTrue(self, obj: PyObjectPtr) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_IsTrue).
        """
        return self.lib.call["PyObject_IsTrue", c_int](obj)

    fn PyObject_Length(self, obj: PyObjectPtr) -> Int:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_Length).
        """
        return Int(self.lib.call["PyObject_Length", Int](obj))

    fn PyObject_Hash(self, obj: PyObjectPtr) -> Py_hash_t:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_Hash).
        """
        return self.lib.call["PyObject_Hash", Py_hash_t](obj)

    fn PyObject_GetIter(
        self, traversable_py_object: PyObjectPtr
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_GetIter).
        """
        var iterator = self.lib.call["PyObject_GetIter", PyObjectPtr](
            traversable_py_object
        )

        self.log(
            iterator._get_ptr_as_int(),
            " NEWREF PyObject_GetIter, refcnt:",
            self._Py_REFCNT(iterator),
            "referencing ",
            traversable_py_object._get_ptr_as_int(),
            "refcnt of traversable: ",
            self._Py_REFCNT(traversable_py_object),
        )

        self._inc_total_rc()
        return iterator

    # ===-------------------------------------------------------------------===#
    # Tuple Objects
    # ref: https://docs.python.org/3/c-api/tuple.html
    # ===-------------------------------------------------------------------===#

    fn PyTuple_New(self, length: Py_ssize_t) -> PyObjectPtr:
        """Return a new tuple object of size `length`, or `NULL` with an exception set on failure.

        [Reference](https://docs.python.org/3/c-api/tuple.html#c.PyTuple_New).
        """

        # PyObject *PyTuple_New(Py_ssize_t len)
        var r = self.lib.call["PyTuple_New", PyObjectPtr](length)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyTuple_New, refcnt:",
            self._Py_REFCNT(r),
            ", tuple size:",
            length,
        )

        self._inc_total_rc()
        return r

    fn PyTuple_GetItem(
        self,
        tuple: PyObjectPtr,
        pos: Py_ssize_t,
    ) -> PyObjectPtr:
        """Return the object at position `pos` in the tuple pointed to by `tuple`.

        Returns borrowed reference.

        [Reference](https://docs.python.org/3/c-api/tuple.html#c.PyTuple_GetItem).
        """

        # PyObject *PyTuple_GetItem(PyObject *p, Py_ssize_t pos)
        return self.lib.call["PyTuple_GetItem", PyObjectPtr](tuple, pos)

    fn PyTuple_SetItem(
        self,
        tuple: PyObjectPtr,
        pos: Py_ssize_t,
        value: PyObjectPtr,
    ) -> c_int:
        """Insert a reference to object `value` at position `pos` of the tuple pointed to by `tuple`.

        [Reference](https://docs.python.org/3/c-api/tuple.html#c.PyTuple_SetItem).
        """

        # PyTuple_SetItem steals the reference - the value object will be
        # destroyed along with the tuple
        self._dec_total_rc()

        # int PyTuple_SetItem(PyObject *p, Py_ssize_t pos, PyObject *o)
        return self.lib.call["PyTuple_SetItem", c_int](tuple, pos, value)

    # ===-------------------------------------------------------------------===#
    # List Objects
    # ref: https://docs.python.org/3/c-api/list.html
    # ===-------------------------------------------------------------------===#

    fn PyList_New(self, length: Py_ssize_t) -> PyObjectPtr:
        """Return a new list of length `length` on success, or `NULL` on failure.

        [Reference](https://docs.python.org/3/c-api/list.html#c.PyList_New).
        """

        # PyObject *PyList_New(Py_ssize_t len)
        var r = self.lib.call["PyList_New", PyObjectPtr](length)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyList_New, refcnt:",
            self._Py_REFCNT(r),
            ", list size:",
            length,
        )

        self._inc_total_rc()
        return r

    fn PyList_GetItem(
        self,
        list_obj: PyObjectPtr,
        index: Py_ssize_t,
    ) -> PyObjectPtr:
        """Return the object at position `index` in the list pointed to by `list_obj`.

        Returns a borrowed reference instead of a strong reference.

        [Reference](https://docs.python.org/3/c-api/list.html#c.PyList_GetItem).
        """

        # PyObject *PyList_GetItem(PyObject *list, Py_ssize_t index)
        return self.lib.call["PyList_GetItem", PyObjectPtr](list_obj, index)

    fn PyList_SetItem(
        self,
        list_obj: PyObjectPtr,
        index: Py_ssize_t,
        value: PyObjectPtr,
    ) -> c_int:
        """Set the item at index `index` in list to `value`.

        [Reference](https://docs.python.org/3/c-api/list.html#c.PyList_SetItem).
        """

        # PyList_SetItem steals the reference - the element object will be
        # destroyed along with the list
        self._dec_total_rc()
        return self.PyList_SetItem_func(list_obj, index, value)

    # ===-------------------------------------------------------------------===#
    # Concrete Objects
    # ref: https://docs.python.org/3/c-api/concrete.html
    # ===-------------------------------------------------------------------===#

    fn Py_None(self) -> PyObjectPtr:
        """Get a None value, of type NoneType. [Reference](
        https://docs.python.org/3/c-api/none.html#c.Py_None)."""

        # Get pointer to the immortal `None` PyObject struct instance.
        # Note:
        #   The name of this global is technical a private part of the
        #   CPython API, but unfortunately the only stable ways to access it are
        #   macros.
        # TODO(MSTDL-977):
        #   Investigate doing this without hard-coding private API details.
        var ptr = self.lib.get_symbol[PyObject]("_Py_NoneStruct")

        if not ptr:
            abort("error: unable to get pointer to CPython `None` struct")

        return PyObjectPtr(ptr)

    # ===-------------------------------------------------------------------===#
    # Boolean Objects
    # ===-------------------------------------------------------------------===#

    fn PyBool_FromLong(self, value: c_long) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/bool.html#c.PyBool_FromLong).
        """

        var r = self.lib.call["PyBool_FromLong", PyObjectPtr](value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyBool_FromLong, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    fn PyBool_Check(self, obj: PyObjectPtr) -> Bool:
        """[Reference](
        https://docs.python.org/3/c-api/bool.html#c.PyBool_Check).
        """
        return self.lib.call["PyBool_Check", c_int](obj) != 0

    # ===-------------------------------------------------------------------===#
    # Integer Objects
    # ===-------------------------------------------------------------------===#

    fn PyLong_FromSsize_t(self, value: c_ssize_t) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/long.html#c.PyLong_FromSsize_t).
        """

        var r = self.PyLong_FromSsize_t_func(value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyLong_FromSsize_t, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    fn PyLong_FromSize_t(self, value: c_size_t) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/long.html#c.PyLong_FromSize_t).
        """

        var r = self.lib.call["PyLong_FromSize_t", PyObjectPtr](value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyLong_FromSize_t, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    fn PyLong_AsSsize_t(self, py_object: PyObjectPtr) -> c_ssize_t:
        """[Reference](
        https://docs.python.org/3/c-api/long.html#c.PyLong_AsSsize_t).
        """
        return self.lib.call["PyLong_AsSsize_t", c_ssize_t](py_object)

    fn PyNumber_Long(self, py_object: PyObjectPtr) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/number.html#c.PyNumber_Long).
        """
        return self.lib.call["PyNumber_Long", PyObjectPtr](py_object)

    # ===-------------------------------------------------------------------===#
    # Floating-Point Objects
    # ===-------------------------------------------------------------------===#

    fn PyNumber_Float(self, obj: PyObjectPtr) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/number.html#c.PyNumber_Float).
        """
        return self.lib.call["PyNumber_Float", PyObjectPtr](obj)

    fn PyFloat_FromDouble(self, value: Float64) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/float.html#c.PyFloat_FromDouble).
        """

        var r = self.lib.call["PyFloat_FromDouble", PyObjectPtr](value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyFloat_FromDouble, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    fn PyFloat_AsDouble(self, py_object: PyObjectPtr) -> Float64:
        """[Reference](
        https://docs.python.org/3/c-api/float.html#c.PyFloat_AsDouble).
        """
        return self.lib.call["PyFloat_AsDouble", Float64](py_object)

    # ===-------------------------------------------------------------------===#
    # Unicode Objects
    # ===-------------------------------------------------------------------===#

    fn PyUnicode_DecodeUTF8(self, strslice: StringSlice) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_DecodeUTF8).
        """
        var r = self.lib.call["PyUnicode_DecodeUTF8", PyObjectPtr](
            strslice.unsafe_ptr().bitcast[Int8](),
            strslice.byte_length(),
            "strict".unsafe_cstr_ptr(),
        )

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyUnicode_DecodeUTF8, refcnt:",
            self._Py_REFCNT(r),
            ", str:",
            strslice,
        )

        self._inc_total_rc()
        return r

    fn PySlice_FromSlice(self, slice: Slice) -> PyObjectPtr:
        # Convert Mojo Slice to Python slice parameters
        # Note: Deliberately avoid using `span.indices()` here and instead pass
        # the Slice parameters directly to Python. Python's C implementation
        # already handles such conditions, allowing Python to apply its own slice
        # handling.
        var py_start = self.Py_None()
        var py_stop = self.Py_None()
        var py_step = self.Py_None()

        if slice.start:
            py_start = self.PyLong_FromSsize_t(c_ssize_t(slice.start.value()))
        if slice.end:
            py_stop = self.PyLong_FromSsize_t(c_ssize_t(slice.end.value()))
        if slice.end:
            py_step = self.PyLong_FromSsize_t(c_ssize_t(slice.step.value()))

        var py_slice = self.PySlice_New(py_start, py_stop, py_step)

        if py_start != self.Py_None():
            self.Py_DecRef(py_start)
        if py_stop != self.Py_None():
            self.Py_DecRef(py_stop)
        self.Py_DecRef(py_step)

        return py_slice

    fn PyUnicode_AsUTF8AndSize(
        self, py_object: PyObjectPtr
    ) -> StringSlice[__origin_of(py_object.unsized_obj_ptr.origin)]:
        """[Reference](
        https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_AsUTF8AndSize).
        """

        var length = Int(0)
        var ptr = self.lib.call[
            "PyUnicode_AsUTF8AndSize", UnsafePointer[c_char]
        ](py_object, UnsafePointer(to=length)).bitcast[UInt8]()
        return StringSlice[__origin_of(py_object.unsized_obj_ptr.origin)](
            ptr=ptr, length=length
        )

    # ===-------------------------------------------------------------------===#
    # Python Error operations
    # ===-------------------------------------------------------------------===#

    fn PyErr_Clear(self):
        """[Reference](
        https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Clear).
        """
        self.lib.call["PyErr_Clear"]()

    fn PyErr_Occurred(self) -> Bool:
        """[Reference](
        https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Occurred).
        """
        return not self.lib.call["PyErr_Occurred", PyObjectPtr]().is_null()

    fn PyErr_Fetch(self) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Fetch).
        """
        var type = PyObjectPtr()
        var value = PyObjectPtr()
        var traceback = PyObjectPtr()

        self.lib.call["PyErr_Fetch"](
            UnsafePointer(to=type),
            UnsafePointer(to=value),
            UnsafePointer(to=traceback),
        )
        var r = value

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyErr_Fetch, refcnt:",
            self._Py_REFCNT(r),
        )

        self._inc_total_rc()
        _ = type
        _ = value
        _ = traceback
        return r

    fn PyErr_SetNone(self, type: PyObjectPtr):
        """[Reference](
        https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetNone).
        """
        self.lib.call["PyErr_SetNone"](type)

    fn PyErr_SetString(
        self,
        type: PyObjectPtr,
        message: UnsafePointer[c_char],
    ):
        """[Reference](
        https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetString).
        """
        self.lib.call["PyErr_SetString"](type, message)

    # ===-------------------------------------------------------------------===#
    # Python Error types
    # ===-------------------------------------------------------------------===#

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
    # Python Iterator operations
    # ===-------------------------------------------------------------------===#

    fn PyIter_Next(self, iterator: PyObjectPtr) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/iter.html#c.PyIter_Next).
        """

        var next_obj = self.lib.call["PyIter_Next", PyObjectPtr](iterator)

        self.log(
            next_obj._get_ptr_as_int(),
            " NEWREF PyIter_Next from ",
            iterator._get_ptr_as_int(),
            ", refcnt(obj):",
            self._Py_REFCNT(next_obj),
            "refcnt(iter)",
            self._Py_REFCNT(iterator),
        )

        if next_obj._get_ptr_as_int() != 0:
            self._inc_total_rc()
        return next_obj

    fn PyIter_Check(self, obj: PyObjectPtr) -> Bool:
        """[Reference](
        https://docs.python.org/3/c-api/iter.html#c.PyIter_Check).
        """
        return self.lib.call["PyIter_Check", c_int](obj) != 0

    fn PySequence_Check(self, obj: PyObjectPtr) -> Bool:
        """[Reference](
        https://docs.python.org/3/c-api/sequence.html#c.PySequence_Check).
        """
        return self.lib.call["PySequence_Check", c_int](obj) != 0

    # ===-------------------------------------------------------------------===#
    # Python Slice Creation
    # ===-------------------------------------------------------------------===#

    fn PySlice_New(
        self, start: PyObjectPtr, stop: PyObjectPtr, step: PyObjectPtr
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/slice.html#c.PySlice_New).
        """
        var r = self.lib.call["PySlice_New", PyObjectPtr](start, stop, step)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PySlice_New, refcnt:",
            self._Py_REFCNT(r),
            ", start:",
            start._get_ptr_as_int(),
            ", stop:",
            stop._get_ptr_as_int(),
            ", step:",
            step._get_ptr_as_int(),
        )

        self._inc_total_rc()
        return r

    # ===-------------------------------------------------------------------===#
    # Capsules
    # ref: https://docs.python.org/3/c-api/capsule.html
    # ===-------------------------------------------------------------------===#

    fn PyCapsule_New(
        mut self,
        pointer: OpaquePointer,
        owned name: String,
        destructor: destructor,
    ) -> PyObjectPtr:
        """Create a PyCapsule to communicate to another C extension the C API in `pointer`, identified by `name` and with the custom destructor in `destructor`.

        [Reference](https://docs.python.org/3/c-api/capsule.html#c.PyCapsule_New).
        """
        # PyObject *PyCapsule_New(void *pointer, const char *name, PyCapsule_Destructor destructor)
        var new_capsule = self.lib.call["PyCapsule_New", PyObjectPtr](
            pointer, name.unsafe_cstr_ptr(), destructor
        )
        self._inc_total_rc()
        return new_capsule

    fn PyCapsule_GetPointer(
        mut self,
        capsule: PyObjectPtr,
        owned name: String,
    ) -> OpaquePointer:
        """Extract the pointer to another C extension from a PyCapsule `capsule` with the given `name`.

        [Reference](https://docs.python.org/3/c-api/capsule.html#c.PyCapsule_GetPointer).
        """
        # void *PyCapsule_GetPointer(PyObject *capsule, const char *name)
        return self.lib.call["PyCapsule_GetPointer", OpaquePointer](
            capsule, name.unsafe_cstr_ptr()
        )

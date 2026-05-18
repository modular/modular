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

# ===----------------------------------------------------------------------=== #
# CPython slot adapter functions introduced in:
# https://github.com/modular/modular/pull/5562
#
# These adapt user-friendly Mojo function signatures to the low-level C ABI
# expected by each CPython type slot.  They are passed to
# PontoneerTypeBuilder.def_method as template parameters.
# ===----------------------------------------------------------------------=== #

from std.ffi import c_int, c_long
from std.memory import OpaquePointer, UnsafePointer
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import (
    PyObject,
    PyObjectPtr,
    Py_ssize_t,
    PySlotIndex,
    PyType_Slot,
)
from std.python.bindings import PythonTypeBuilder
from std.python.conversions import ConvertibleToPython
from std.utils import Variant

from .utils import PySlotError


@always_inline
def _unwrap_self[
    T: ImplicitlyDestructible
](py_self: PyObjectPtr) raises -> UnsafePointer[T, MutAnyOrigin]:
    """Downcast a raw PyObjectPtr to a typed Mojo pointer."""
    return PythonObject(from_borrowed=py_self).downcast_value_ptr[T]()


def _mp_length_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin]
    ) thin raises PySlotError -> Int,
](py_self: PyObjectPtr) abi("C") -> Py_ssize_t:
    """CPython `lenfunc` adapter for the `mp_length` slot (__len__).

    `method` declares `raises PySlotError`; the wrapper dispatches on the
    variant to the matching `PyExc_*` global. `not_implemented` has no
    Python meaning here and maps to `RuntimeError`.

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function
            `def(self: UnsafePointer[self_type, MutAnyOrigin]) raises PySlotError -> Int`.

    Returns:
        Length as `Py_ssize_t`, or -1 with an exception set on error.
    """
    ref cpython = Python().cpython()
    var self_ptr: UnsafePointer[self_type, MutAnyOrigin]
    try:
        self_ptr = _unwrap_self[self_type](py_self)
    except e:
        var error_type = cpython.get_error_global("PyExc_RuntimeError")
        var msg = String(e)
        cpython.PyErr_SetString(
            error_type, msg.as_c_string_slice().unsafe_ptr()
        )
        return Py_ssize_t(-1)
    try:
        return Py_ssize_t(method(self_ptr))
    except e:
        var error_type = cpython.get_error_global(e.pyexc_global_name())
        cpython.PyErr_SetString(
            error_type, e.msg.as_c_string_slice().unsafe_ptr()
        )
        return Py_ssize_t(-1)


def _mp_subscript_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject
    ) thin raises PySlotError -> PythonObject,
](py_self: PyObjectPtr, key: PyObjectPtr) abi("C") -> PyObjectPtr:
    """CPython `binaryfunc` adapter for the `mp_subscript` slot (__getitem__).

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function
            `def(self: UnsafePointer[self_type, MutAnyOrigin], key: PythonObject) raises PySlotError -> PythonObject`.

    Returns:
        New reference to the result, or null with an exception set on error.
    """
    ref cpython = Python().cpython()
    var self_ptr: UnsafePointer[self_type, MutAnyOrigin]
    try:
        self_ptr = _unwrap_self[self_type](py_self)
    except e:
        var error_type = cpython.get_error_global("PyExc_RuntimeError")
        var msg = String(e)
        cpython.PyErr_SetString(
            error_type, msg.as_c_string_slice().unsafe_ptr()
        )
        return PyObjectPtr()
    try:
        var result = method(self_ptr, PythonObject(from_borrowed=key))
        return result.steal_data()
    except e:
        var error_type = cpython.get_error_global(e.pyexc_global_name())
        cpython.PyErr_SetString(
            error_type, e.msg.as_c_string_slice().unsafe_ptr()
        )
        return PyObjectPtr()


def _mp_ass_subscript_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin],
        PythonObject,
        Variant[PythonObject, Int],
    ) thin raises PySlotError -> None,
](py_self: PyObjectPtr, key: PyObjectPtr, value: PyObjectPtr) abi("C") -> c_int:
    """CPython `objobjargproc` adapter for the `mp_ass_subscript` slot.

    When `value` is NULL the operation is a deletion (__delitem__); the `method`
    receives `Variant[PythonObject, Int](Int(0))` as the third argument.
    Otherwise the operation is an assignment (__setitem__) and `method` receives
    `Variant[PythonObject, Int](value_object)`.

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function with signature
            `def(self, key, value: Variant[PythonObject, Int]) raises PySlotError -> None`.

    Returns:
        0 on success, -1 with an exception set on error.
    """
    comptime PassedValue = Variant[PythonObject, Int]
    ref cpython = Python().cpython()
    var self_ptr: UnsafePointer[self_type, MutAnyOrigin]
    var key_obj: PythonObject
    var passed_value: PassedValue
    try:
        self_ptr = _unwrap_self[self_type](py_self)
        key_obj = PythonObject(from_borrowed=key)
        passed_value = PassedValue(
            PythonObject(from_borrowed=value)
        ) if value else PassedValue(Int(0))
    except e:
        var error_type = cpython.get_error_global("PyExc_RuntimeError")
        var msg = String(e)
        cpython.PyErr_SetString(
            error_type, msg.as_c_string_slice().unsafe_ptr()
        )
        return c_int(-1)
    try:
        method(self_ptr, key_obj, passed_value)
        return c_int(0)
    except e:
        var error_type = cpython.get_error_global(e.pyexc_global_name())
        cpython.PyErr_SetString(
            error_type, e.msg.as_c_string_slice().unsafe_ptr()
        )
        return c_int(-1)


def _unaryfunc_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin]
    ) thin raises PySlotError -> PythonObject,
](py_self: PyObjectPtr) abi("C") -> PyObjectPtr:
    """CPython `unaryfunc` adapter for unary nb_ slots (__neg__, __abs__, etc.).

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function
            `def(self: UnsafePointer[self_type, MutAnyOrigin]) raises PySlotError -> PythonObject`.

    Returns:
        New reference to the result, or null with an exception set on error.
    """
    ref cpython = Python().cpython()
    var self_ptr: UnsafePointer[self_type, MutAnyOrigin]
    try:
        self_ptr = _unwrap_self[self_type](py_self)
    except e:
        var error_type = cpython.get_error_global("PyExc_RuntimeError")
        var msg = String(e)
        cpython.PyErr_SetString(
            error_type, msg.as_c_string_slice().unsafe_ptr()
        )
        return PyObjectPtr()
    try:
        var result = method(self_ptr)
        return result.steal_data()
    except e:
        var error_type = cpython.get_error_global(e.pyexc_global_name())
        cpython.PyErr_SetString(
            error_type, e.msg.as_c_string_slice().unsafe_ptr()
        )
        return PyObjectPtr()


def _binaryfunc_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject
    ) thin raises PySlotError -> PythonObject,
](lhs: PyObjectPtr, rhs: PyObjectPtr) abi("C") -> PyObjectPtr:
    """CPython `binaryfunc` adapter for binary nb_ slots (__add__, __mul__, etc.).

    If `method` raises `PySlotError.not_implemented()` the wrapper returns
    `Py_NotImplemented`, signalling Python to try the reflected operation.
    Other variants map to the corresponding `PyExc_*`.

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function
            `def(self: UnsafePointer[self_type, MutAnyOrigin], other: PythonObject) raises PySlotError -> PythonObject`.

    Returns:
        New reference to the result, `Py_NotImplemented`, or null on error.
    """
    ref cpython = Python().cpython()
    var self_ptr: UnsafePointer[self_type, MutAnyOrigin]
    var rhs_obj: PythonObject
    try:
        self_ptr = _unwrap_self[self_type](lhs)
        rhs_obj = PythonObject(from_borrowed=rhs)
    except:
        # CPython invokes binary `nb_*` slots in reflected dispatch with the
        # original `(v, w)` order — so the LHS may not be our type. Treat
        # any prep failure as `NotImplemented` so CPython can try the other
        # operand or raise `TypeError`.
        return cpython.Py_NewRef(cpython.Py_NotImplemented())
    try:
        var result = method(self_ptr, rhs_obj)
        return result.steal_data()
    except e:
        if e._variant == PySlotError._NOT_IMPLEMENTED:
            return cpython.Py_NewRef(cpython.Py_NotImplemented())
        var error_type = cpython.get_error_global(e.pyexc_global_name())
        cpython.PyErr_SetString(
            error_type, e.msg.as_c_string_slice().unsafe_ptr()
        )
        return PyObjectPtr()


def _ternaryfunc_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject, PythonObject
    ) thin raises PySlotError -> PythonObject,
](py_self: PyObjectPtr, other: PyObjectPtr, mod: PyObjectPtr) abi(
    "C"
) -> PyObjectPtr:
    """CPython `ternaryfunc` adapter for nb_power / nb_inplace_power (__pow__).

    If `method` raises `PySlotError.not_implemented()` the wrapper returns
    `Py_NotImplemented`, signalling Python to try the reflected operation.
    Other variants map to the corresponding `PyExc_*`.

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function
            `def(self, other, mod: PythonObject) raises PySlotError -> PythonObject`
            where `mod` is typically `None` unless the three-argument form
            `pow(base, exp, mod)` is used.

    Returns:
        New reference to the result, `Py_NotImplemented`, or null on error.
    """
    ref cpython = Python().cpython()
    var self_ptr: UnsafePointer[self_type, MutAnyOrigin]
    var other_obj: PythonObject
    var mod_obj: PythonObject
    try:
        self_ptr = _unwrap_self[self_type](py_self)
        other_obj = PythonObject(from_borrowed=other)
        mod_obj = PythonObject(from_borrowed=mod)
    except:
        # See `_binaryfunc_wrapper`: reflected dispatch may call this slot
        # with a LHS that isn't our type.
        return cpython.Py_NewRef(cpython.Py_NotImplemented())
    try:
        var result = method(self_ptr, other_obj, mod_obj)
        return result.steal_data()
    except e:
        if e._variant == PySlotError._NOT_IMPLEMENTED:
            return cpython.Py_NewRef(cpython.Py_NotImplemented())
        var error_type = cpython.get_error_global(e.pyexc_global_name())
        cpython.PyErr_SetString(
            error_type, e.msg.as_c_string_slice().unsafe_ptr()
        )
        return PyObjectPtr()


def _inquiry_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin]
    ) thin raises PySlotError -> Bool,
](py_self: PyObjectPtr) abi("C") -> c_int:
    """CPython `inquiry` adapter for the `nb_bool` slot (__bool__).

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function
            `def(self: UnsafePointer[self_type, MutAnyOrigin]) raises PySlotError -> Bool`.

    Returns:
        1 for True, 0 for False, -1 with an exception set on error.
    """
    ref cpython = Python().cpython()
    var self_ptr: UnsafePointer[self_type, MutAnyOrigin]
    try:
        self_ptr = _unwrap_self[self_type](py_self)
    except e:
        var error_type = cpython.get_error_global("PyExc_RuntimeError")
        var msg = String(e)
        cpython.PyErr_SetString(
            error_type, msg.as_c_string_slice().unsafe_ptr()
        )
        return c_int(-1)
    try:
        var result = method(self_ptr)
        return c_int(1) if result else c_int(0)
    except e:
        var error_type = cpython.get_error_global(e.pyexc_global_name())
        cpython.PyErr_SetString(
            error_type, e.msg.as_c_string_slice().unsafe_ptr()
        )
        return c_int(-1)


def _ssizeargfunc_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], Int
    ) thin raises PySlotError -> PythonObject,
](py_self: PyObjectPtr, index: Py_ssize_t) abi("C") -> PyObjectPtr:
    """CPython `ssizeargfunc` adapter for sq_item, sq_repeat, sq_inplace_repeat.

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function
            `def(self: UnsafePointer[self_type, MutAnyOrigin], index: Int) raises PySlotError -> PythonObject`.

    Returns:
        New reference to the result, or null with an exception set on error.
    """
    ref cpython = Python().cpython()
    var self_ptr: UnsafePointer[self_type, MutAnyOrigin]
    try:
        self_ptr = _unwrap_self[self_type](py_self)
    except e:
        var error_type = cpython.get_error_global("PyExc_RuntimeError")
        var msg = String(e)
        cpython.PyErr_SetString(
            error_type, msg.as_c_string_slice().unsafe_ptr()
        )
        return PyObjectPtr()
    try:
        var result = method(self_ptr, Int(index))
        return result.steal_data()
    except e:
        var error_type = cpython.get_error_global(e.pyexc_global_name())
        cpython.PyErr_SetString(
            error_type, e.msg.as_c_string_slice().unsafe_ptr()
        )
        return PyObjectPtr()


def _ssizeobjargproc_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], Int, Variant[PythonObject, Int]
    ) thin raises PySlotError -> None,
](py_self: PyObjectPtr, index: Py_ssize_t, value: PyObjectPtr) abi(
    "C"
) -> c_int:
    """CPython `ssizeobjargproc` adapter for the `sq_ass_item` slot.

    When `value` is NULL the operation is a deletion; the `method` receives
    `Variant[PythonObject, Int](Int(0))` as the third argument.  Otherwise
    the operation is an assignment and `method` receives the value object.

    The user method declares `raises PySlotError`; the wrapper dispatches on
    the variant to the matching `PyExc_*` global. Raising
    `PySlotError.not_implemented()` from this slot has no Python meaning
    (`sq_ass_item` cannot return `NotImplemented`), so it is mapped to
    `RuntimeError`.

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function with signature
            `def(self, index: Int, value: Variant[PythonObject, Int]) raises PySlotError -> None`.

    Returns:
        0 on success, -1 with an exception set on error.
    """
    comptime PassedValue = Variant[PythonObject, Int]
    ref cpython = Python().cpython()
    # Compute args outside the typed `try` so the only call that can raise
    # inside it is `method(...)`, which lets the compiler infer
    # `except e:` as `PySlotError`. `_unwrap_self` and `PythonObject(...)`
    # construction raise plain `Error`; wrap them in their own block.
    var self_ptr: UnsafePointer[self_type, MutAnyOrigin]
    var passed_value: PassedValue
    try:
        self_ptr = _unwrap_self[self_type](py_self)
        passed_value = PassedValue(
            PythonObject(from_borrowed=value)
        ) if value else PassedValue(Int(0))
    except e:
        var error_type = cpython.get_error_global("PyExc_RuntimeError")
        var msg = String(e)
        cpython.PyErr_SetString(
            error_type, msg.as_c_string_slice().unsafe_ptr()
        )
        return c_int(-1)
    try:
        method(self_ptr, Int(index), passed_value)
        return c_int(0)
    except e:
        # `not_implemented` has no meaning for sq_ass_item; it falls through
        # to RuntimeError via PySlotError.pyexc_global_name().
        var error_type = cpython.get_error_global(e.pyexc_global_name())
        cpython.PyErr_SetString(
            error_type, e.msg.as_c_string_slice().unsafe_ptr()
        )
        return c_int(-1)


def _objobjproc_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject
    ) thin raises PySlotError -> Bool,
](py_self: PyObjectPtr, other: PyObjectPtr) abi("C") -> c_int:
    """CPython `objobjproc` adapter for the `sq_contains` slot (__contains__).

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function
            `def(self: UnsafePointer[self_type, MutAnyOrigin], item: PythonObject) raises PySlotError -> Bool`.

    Returns:
        1 if contained, 0 if not, -1 with an exception set on error.
    """
    ref cpython = Python().cpython()
    var self_ptr: UnsafePointer[self_type, MutAnyOrigin]
    var other_obj: PythonObject
    try:
        self_ptr = _unwrap_self[self_type](py_self)
        other_obj = PythonObject(from_borrowed=other)
    except e:
        var error_type = cpython.get_error_global("PyExc_RuntimeError")
        var msg = String(e)
        cpython.PyErr_SetString(
            error_type, msg.as_c_string_slice().unsafe_ptr()
        )
        return c_int(-1)
    try:
        var result = method(self_ptr, other_obj)
        return c_int(1) if result else c_int(0)
    except e:
        var error_type = cpython.get_error_global(e.pyexc_global_name())
        cpython.PyErr_SetString(
            error_type, e.msg.as_c_string_slice().unsafe_ptr()
        )
        return c_int(-1)


def _richcompare_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject, Int
    ) thin raises PySlotError -> Bool,
](py_self: PyObjectPtr, py_other: PyObjectPtr, op: c_int) abi(
    "C"
) -> PyObjectPtr:
    """CPython `richcmpfunc` adapter for the `tp_richcompare` slot.

    If `method` raises `PySlotError.not_implemented()` the wrapper returns
    `Py_NotImplemented`, signalling Python to try the reflected operation.
    Other variants map to the corresponding `PyExc_*`.

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function
            `def(self, other: PythonObject, op: Int) raises PySlotError -> Bool`
            where `op` is one of `RichCompareOps.Py_LT` … `Py_GE`.

    Returns:
        `Py_True`/`Py_False`, `Py_NotImplemented`, or null on error.
    """
    ref cpython = Python().cpython()
    var self_ptr: UnsafePointer[self_type, MutAnyOrigin]
    var other_obj: PythonObject
    try:
        self_ptr = _unwrap_self[self_type](py_self)
        other_obj = PythonObject(from_borrowed=py_other)
    except:
        # `tp_richcompare` is symmetric in CPython; if the LHS isn't our
        # type, return `NotImplemented` so the interpreter can try the
        # reflected comparison on the other operand.
        return cpython.Py_NewRef(cpython.Py_NotImplemented())
    try:
        var result = method(self_ptr, other_obj, Int(op))
        return cpython.PyBool_FromLong(c_long(Int(result)))
    except e:
        if e._variant == PySlotError._NOT_IMPLEMENTED:
            return cpython.Py_NewRef(cpython.Py_NotImplemented())
        var error_type = cpython.get_error_global(e.pyexc_global_name())
        cpython.PyErr_SetString(
            error_type, e.msg.as_c_string_slice().unsafe_ptr()
        )
        return PyObjectPtr()


# ===----------------------------------------------------------------------=== #
# Non-raising → raising lift helpers
# ===----------------------------------------------------------------------=== #


def _lift_to_int[
    T: ImplicitlyDestructible,
    method: def(UnsafePointer[T, MutAnyOrigin]) thin -> Int,
](ptr: UnsafePointer[T, MutAnyOrigin]) raises PySlotError -> Int:
    return method(ptr)


def _lift_to_obj[
    T: ImplicitlyDestructible,
    method: def(UnsafePointer[T, MutAnyOrigin]) thin -> PythonObject,
](ptr: UnsafePointer[T, MutAnyOrigin]) raises PySlotError -> PythonObject:
    return method(ptr)


def _lift_to_bool[
    T: ImplicitlyDestructible,
    method: def(UnsafePointer[T, MutAnyOrigin]) thin -> Bool,
](ptr: UnsafePointer[T, MutAnyOrigin]) raises PySlotError -> Bool:
    return method(ptr)


def _lift_obj_to_obj[
    T: ImplicitlyDestructible,
    method: def(
        UnsafePointer[T, MutAnyOrigin], PythonObject
    ) thin -> PythonObject,
](
    ptr: UnsafePointer[T, MutAnyOrigin], other: PythonObject
) raises PySlotError -> PythonObject:
    return method(ptr, other)


def _lift_obj_to_bool[
    T: ImplicitlyDestructible,
    method: def(UnsafePointer[T, MutAnyOrigin], PythonObject) thin -> Bool,
](
    ptr: UnsafePointer[T, MutAnyOrigin], other: PythonObject
) raises PySlotError -> Bool:
    return method(ptr, other)


def _lift_obj_var_to_none[
    T: ImplicitlyDestructible,
    method: def(
        UnsafePointer[T, MutAnyOrigin], PythonObject, Variant[PythonObject, Int]
    ) thin -> None,
](
    ptr: UnsafePointer[T, MutAnyOrigin],
    key: PythonObject,
    val: Variant[PythonObject, Int],
) raises PySlotError -> None:
    method(ptr, key, val)


def _lift_int_to_obj[
    T: ImplicitlyDestructible,
    method: def(UnsafePointer[T, MutAnyOrigin], Int) thin -> PythonObject,
](
    ptr: UnsafePointer[T, MutAnyOrigin], index: Int
) raises PySlotError -> PythonObject:
    return method(ptr, index)


def _lift_int_var_to_none[
    T: ImplicitlyDestructible,
    method: def(
        UnsafePointer[T, MutAnyOrigin], Int, Variant[PythonObject, Int]
    ) thin -> None,
](
    ptr: UnsafePointer[T, MutAnyOrigin],
    index: Int,
    val: Variant[PythonObject, Int],
) raises PySlotError -> None:
    method(ptr, index, val)


def _lift_obj_int_to_bool[
    T: ImplicitlyDestructible,
    method: def(UnsafePointer[T, MutAnyOrigin], PythonObject, Int) thin -> Bool,
](
    ptr: UnsafePointer[T, MutAnyOrigin], other: PythonObject, op: Int
) raises PySlotError -> Bool:
    return method(ptr, other, op)


def _lift_obj_obj_to_obj[
    T: ImplicitlyDestructible,
    method: def(
        UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
    ) thin -> PythonObject,
](
    ptr: UnsafePointer[T, MutAnyOrigin], a: PythonObject, b: PythonObject
) raises PySlotError -> PythonObject:
    return method(ptr, a, b)


# ===----------------------------------------------------------------------=== #
# Value-receiver → pointer-receiver lift helpers
# ===----------------------------------------------------------------------=== #


def _lift_val_to_int[
    T: ImplicitlyDestructible,
    method: def(T) thin raises PySlotError -> Int,
](ptr: UnsafePointer[T, MutAnyOrigin]) raises PySlotError -> Int:
    return method(ptr[])


def _lift_val_to_obj[
    T: ImplicitlyDestructible,
    method: def(T) thin raises PySlotError -> PythonObject,
](ptr: UnsafePointer[T, MutAnyOrigin]) raises PySlotError -> PythonObject:
    return method(ptr[])


def _lift_val_to_bool[
    T: ImplicitlyDestructible,
    method: def(T) thin raises PySlotError -> Bool,
](ptr: UnsafePointer[T, MutAnyOrigin]) raises PySlotError -> Bool:
    return method(ptr[])


def _lift_val_obj_to_obj[
    T: ImplicitlyDestructible,
    method: def(T, PythonObject) thin raises PySlotError -> PythonObject,
](
    ptr: UnsafePointer[T, MutAnyOrigin], other: PythonObject
) raises PySlotError -> PythonObject:
    return method(ptr[], other)


def _lift_val_obj_to_bool[
    T: ImplicitlyDestructible,
    method: def(T, PythonObject) thin raises PySlotError -> Bool,
](
    ptr: UnsafePointer[T, MutAnyOrigin], other: PythonObject
) raises PySlotError -> Bool:
    return method(ptr[], other)


def _lift_val_obj_var_to_none[
    T: ImplicitlyDestructible,
    method: def(
        T, PythonObject, Variant[PythonObject, Int]
    ) thin raises PySlotError -> None,
](
    ptr: UnsafePointer[T, MutAnyOrigin],
    key: PythonObject,
    val: Variant[PythonObject, Int],
) raises PySlotError -> None:
    method(ptr[], key, val)


def _lift_mut_obj_var_to_none[
    T: ImplicitlyDestructible,
    method: def(
        mut T, PythonObject, Variant[PythonObject, Int]
    ) thin raises PySlotError -> None,
](
    ptr: UnsafePointer[T, MutAnyOrigin],
    key: PythonObject,
    val: Variant[PythonObject, Int],
) raises PySlotError -> None:
    method(ptr[], key, val)


def _lift_val_int_to_obj[
    T: ImplicitlyDestructible,
    method: def(T, Int) thin raises PySlotError -> PythonObject,
](
    ptr: UnsafePointer[T, MutAnyOrigin], index: Int
) raises PySlotError -> PythonObject:
    return method(ptr[], index)


def _lift_val_int_var_to_none[
    T: ImplicitlyDestructible,
    method: def(
        T, Int, Variant[PythonObject, Int]
    ) thin raises PySlotError -> None,
](
    ptr: UnsafePointer[T, MutAnyOrigin],
    index: Int,
    val: Variant[PythonObject, Int],
) raises PySlotError -> None:
    method(ptr[], index, val)


def _lift_mut_int_var_to_none[
    T: ImplicitlyDestructible,
    method: def(
        mut T, Int, Variant[PythonObject, Int]
    ) thin raises PySlotError -> None,
](
    ptr: UnsafePointer[T, MutAnyOrigin],
    index: Int,
    val: Variant[PythonObject, Int],
) raises PySlotError -> None:
    method(ptr[], index, val)


def _lift_val_obj_int_to_bool[
    T: ImplicitlyDestructible,
    method: def(T, PythonObject, Int) thin raises PySlotError -> Bool,
](
    ptr: UnsafePointer[T, MutAnyOrigin], other: PythonObject, op: Int
) raises PySlotError -> Bool:
    return method(ptr[], other, op)


def _lift_val_obj_obj_to_obj[
    T: ImplicitlyDestructible,
    method: def(
        T, PythonObject, PythonObject
    ) thin raises PySlotError -> PythonObject,
](
    ptr: UnsafePointer[T, MutAnyOrigin], a: PythonObject, b: PythonObject
) raises PySlotError -> PythonObject:
    return method(ptr[], a, b)


def _lift_mut_obj_to_obj[
    T: ImplicitlyDestructible,
    method: def(mut T, PythonObject) thin raises PySlotError -> PythonObject,
](
    ptr: UnsafePointer[T, MutAnyOrigin], other: PythonObject
) raises PySlotError -> PythonObject:
    return method(ptr[], other)


def _lift_mut_obj_obj_to_obj[
    T: ImplicitlyDestructible,
    method: def(
        mut T, PythonObject, PythonObject
    ) thin raises PySlotError -> PythonObject,
](
    ptr: UnsafePointer[T, MutAnyOrigin], a: PythonObject, b: PythonObject
) raises PySlotError -> PythonObject:
    return method(ptr[], a, b)


# ===----------------------------------------------------------------------=== #
# ConvertibleToPython return-type lift helpers
# ===----------------------------------------------------------------------=== #

comptime _CPython = ConvertibleToPython & ImplicitlyCopyable


def _to_py[R: _CPython](var value: R) raises PySlotError -> PythonObject:
    """Call `to_python_object` and translate failures to `PySlotError`.

    `ConvertibleToPython.to_python_object` raises plain `Error`; this shim
    re-raises as `PySlotError.value_error` so it composes with the
    typed-raises chain inside `_conv_*` helpers.
    """
    try:
        return value^.to_python_object()
    except e:
        raise PySlotError.value_error(String(e))


def _conv_ptr_r_unary[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(UnsafePointer[T, MutAnyOrigin]) thin raises PySlotError -> R,
](ptr: UnsafePointer[T, MutAnyOrigin]) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr))


def _conv_ptr_nr_unary[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(UnsafePointer[T, MutAnyOrigin]) thin -> R,
](ptr: UnsafePointer[T, MutAnyOrigin]) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr))


def _conv_val_r_unary[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(T) thin raises PySlotError -> R,
](ptr: UnsafePointer[T, MutAnyOrigin]) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr[]))


def _conv_ptr_r_binary[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(
        UnsafePointer[T, MutAnyOrigin], PythonObject
    ) thin raises PySlotError -> R,
](
    ptr: UnsafePointer[T, MutAnyOrigin], other: PythonObject
) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr, other))


def _conv_ptr_nr_binary[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(UnsafePointer[T, MutAnyOrigin], PythonObject) thin -> R,
](
    ptr: UnsafePointer[T, MutAnyOrigin], other: PythonObject
) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr, other))


def _conv_val_r_binary[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(T, PythonObject) thin raises PySlotError -> R,
](
    ptr: UnsafePointer[T, MutAnyOrigin], other: PythonObject
) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr[], other))


def _conv_ptr_r_int_arg[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(
        UnsafePointer[T, MutAnyOrigin], Int
    ) thin raises PySlotError -> R,
](
    ptr: UnsafePointer[T, MutAnyOrigin], index: Int
) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr, index))


def _conv_ptr_nr_int_arg[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(UnsafePointer[T, MutAnyOrigin], Int) thin -> R,
](
    ptr: UnsafePointer[T, MutAnyOrigin], index: Int
) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr, index))


def _conv_val_r_int_arg[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(T, Int) thin raises PySlotError -> R,
](
    ptr: UnsafePointer[T, MutAnyOrigin], index: Int
) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr[], index))


def _conv_ptr_r_ternary[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(
        UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
    ) thin raises PySlotError -> R,
](
    ptr: UnsafePointer[T, MutAnyOrigin], a: PythonObject, b: PythonObject
) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr, a, b))


def _conv_ptr_nr_ternary[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(
        UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
    ) thin -> R,
](
    ptr: UnsafePointer[T, MutAnyOrigin], a: PythonObject, b: PythonObject
) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr, a, b))


def _conv_val_r_ternary[
    T: ImplicitlyDestructible,
    R: _CPython,
    method: def(T, PythonObject, PythonObject) thin raises PySlotError -> R,
](
    ptr: UnsafePointer[T, MutAnyOrigin], a: PythonObject, b: PythonObject
) raises PySlotError -> PythonObject:
    return _to_py[R](method(ptr[], a, b))


# Unary


# ===----------------------------------------------------------------------=== #
# Slot-install helpers — insert typed C function pointers into a builder
# ===----------------------------------------------------------------------=== #


struct _SlotInstaller:
    """Static-method namespace for inserting CPython type-slot function
    pointers into a `PythonTypeBuilder`. Each method wraps a method
    template parameter in the matching adapter and registers the result.
    """

    @staticmethod
    def unary[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin]
        ) thin raises PySlotError -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `unaryfunc` slot into the builder pointed to by `ptr`."""
        comptime _unaryfunc = def(PyObjectPtr) thin abi("C") -> PyObjectPtr
        var fn_ptr: _unaryfunc = _unaryfunc_wrapper[self_type, method]
        ptr[]._insert_slot(
            PyType_Slot(slot, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr))
        )

    @staticmethod
    def binary[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], PythonObject
        ) thin raises PySlotError -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `binaryfunc` slot into the builder pointed to by `ptr`."""
        comptime _binaryfunc = def(PyObjectPtr, PyObjectPtr) thin abi(
            "C"
        ) -> PyObjectPtr
        var fn_ptr: _binaryfunc = _binaryfunc_wrapper[self_type, method]
        ptr[]._insert_slot(
            PyType_Slot(slot, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr))
        )

    @staticmethod
    def ternary[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], PythonObject, PythonObject
        ) thin raises PySlotError -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `ternaryfunc` slot into the builder pointed to by `ptr`."""
        comptime _ternaryfunc = def(
            PyObjectPtr, PyObjectPtr, PyObjectPtr
        ) thin abi("C") -> PyObjectPtr
        var fn_ptr: _ternaryfunc = _ternaryfunc_wrapper[self_type, method]
        ptr[]._insert_slot(
            PyType_Slot(slot, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr))
        )

    @staticmethod
    def inquiry[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin]
        ) thin raises PySlotError -> Bool,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert an `inquiry` slot into the builder pointed to by `ptr`."""
        comptime _inquiry = def(PyObjectPtr) thin abi("C") -> c_int
        var fn_ptr: _inquiry = _inquiry_wrapper[self_type, method]
        ptr[]._insert_slot(
            PyType_Slot(slot, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr))
        )

    @staticmethod
    def richcompare[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], PythonObject, Int
        ) thin raises PySlotError -> Bool,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `richcmpfunc` slot (`tp_richcompare`) into the builder pointed to by `ptr`.
        """
        comptime _richcmpfunc = def(PyObjectPtr, PyObjectPtr, c_int) thin abi(
            "C"
        ) -> PyObjectPtr
        var fn_ptr: _richcmpfunc = _richcompare_wrapper[self_type, method]
        ptr[]._insert_slot(
            PyType_Slot(
                PySlotIndex.tp_richcompare,
                rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
            )
        )

    @staticmethod
    def lenfunc[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin]
        ) thin raises PySlotError -> Int,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `lenfunc` slot (`mp_length`) into the builder pointed to by `ptr`.
        """
        comptime _lenfunc = def(PyObjectPtr) thin abi("C") -> Py_ssize_t
        var fn_ptr: _lenfunc = _mp_length_wrapper[self_type, method]
        ptr[]._insert_slot(
            PyType_Slot(
                PySlotIndex.mp_length,
                rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
            )
        )

    @staticmethod
    def mp_getitem[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], PythonObject
        ) thin raises PySlotError -> PythonObject,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `binaryfunc` slot (`mp_subscript`) into the builder pointed to by `ptr`.
        """
        comptime _binaryfunc = def(PyObjectPtr, PyObjectPtr) thin abi(
            "C"
        ) -> PyObjectPtr
        var fn_ptr: _binaryfunc = _mp_subscript_wrapper[self_type, method]
        ptr[]._insert_slot(
            PyType_Slot(
                PySlotIndex.mp_getitem,
                rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
            )
        )

    @staticmethod
    def objobjargproc[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin],
            PythonObject,
            Variant[PythonObject, Int],
        ) thin raises PySlotError -> None,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert an `objobjargproc` slot (`mp_ass_subscript`) into the builder pointed to by `ptr`.
        """
        comptime _objobjargproc = def(
            PyObjectPtr, PyObjectPtr, PyObjectPtr
        ) thin abi("C") -> c_int
        var fn_ptr: _objobjargproc = _mp_ass_subscript_wrapper[
            self_type, method
        ]
        ptr[]._insert_slot(
            PyType_Slot(
                PySlotIndex.mp_setitem,
                rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
            )
        )

    @staticmethod
    def ssizeargfunc[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], Int
        ) thin raises PySlotError -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `ssizeargfunc` slot into the builder pointed to by `ptr`."""
        comptime _ssizeargfunc = def(PyObjectPtr, Py_ssize_t) thin abi(
            "C"
        ) -> PyObjectPtr
        var fn_ptr: _ssizeargfunc = _ssizeargfunc_wrapper[self_type, method]
        ptr[]._insert_slot(
            PyType_Slot(slot, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr))
        )

    @staticmethod
    def ssizeobjargproc[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin],
            Int,
            Variant[PythonObject, Int],
        ) thin raises PySlotError -> None,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert the `ssizeobjargproc` slot (`sq_ass_item`) into the builder pointed to by `ptr`.
        """
        comptime _ssizeobjargproc = def(
            PyObjectPtr, Py_ssize_t, PyObjectPtr
        ) thin abi("C") -> c_int
        var fn_ptr: _ssizeobjargproc = _ssizeobjargproc_wrapper[
            self_type, method
        ]
        ptr[]._insert_slot(
            PyType_Slot(
                PySlotIndex.sq_ass_item,
                rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
            )
        )

    @staticmethod
    def objobjproc[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], PythonObject
        ) thin raises PySlotError -> Bool,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert an `objobjproc` slot into the builder pointed to by `ptr`."""
        comptime _objobjproc = def(PyObjectPtr, PyObjectPtr) thin abi(
            "C"
        ) -> c_int
        var fn_ptr: _objobjproc = _objobjproc_wrapper[self_type, method]
        ptr[]._insert_slot(
            PyType_Slot(slot, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr))
        )

    @staticmethod
    def unary_nr[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin]
        ) thin -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `unaryfunc` slot from a non-raising method."""
        Self.unary[self_type, _lift_to_obj[self_type, method], slot](ptr)

    @staticmethod
    def unary_val[
        self_type: ImplicitlyDestructible,
        method: def(self_type) thin raises PySlotError -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `unaryfunc` slot from a value-receiver method."""
        Self.unary[self_type, _lift_val_to_obj[self_type, method], slot](ptr)

    @staticmethod
    def unary_conv_r[
        self_type: ImplicitlyDestructible,
        R: _CPython,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin]
        ) thin raises PySlotError -> R,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `unaryfunc` slot from a raising ConvertibleToPython method.
        """
        Self.unary[self_type, _conv_ptr_r_unary[self_type, R, method], slot](
            ptr
        )

    @staticmethod
    def unary_conv_nr[
        self_type: ImplicitlyDestructible,
        R: _CPython,
        method: def(UnsafePointer[self_type, MutAnyOrigin]) thin -> R,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `unaryfunc` slot from a non-raising ConvertibleToPython method.
        """
        Self.unary[self_type, _conv_ptr_nr_unary[self_type, R, method], slot](
            ptr
        )

    @staticmethod
    def unary_conv_val[
        self_type: ImplicitlyDestructible,
        R: _CPython,
        method: def(self_type) thin raises PySlotError -> R,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `unaryfunc` slot from a value-receiver ConvertibleToPython method.
        """
        Self.unary[self_type, _conv_val_r_unary[self_type, R, method], slot](
            ptr
        )

    # Inquiry

    @staticmethod
    def inquiry_nr[
        self_type: ImplicitlyDestructible,
        method: def(UnsafePointer[self_type, MutAnyOrigin]) thin -> Bool,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert an `inquiry` slot from a non-raising method."""
        Self.inquiry[self_type, _lift_to_bool[self_type, method], slot](ptr)

    @staticmethod
    def inquiry_val[
        self_type: ImplicitlyDestructible,
        method: def(self_type) thin raises PySlotError -> Bool,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert an `inquiry` slot from a value-receiver method."""
        Self.inquiry[self_type, _lift_val_to_bool[self_type, method], slot](ptr)

    # Binary

    @staticmethod
    def binary_nr[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `binaryfunc` slot from a non-raising method."""
        Self.binary[self_type, _lift_obj_to_obj[self_type, method], slot](ptr)

    @staticmethod
    def binary_val[
        self_type: ImplicitlyDestructible,
        method: def(
            self_type, PythonObject
        ) thin raises PySlotError -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `binaryfunc` slot from a value-receiver method."""
        Self.binary[self_type, _lift_val_obj_to_obj[self_type, method], slot](
            ptr
        )

    @staticmethod
    def binary_mut[
        self_type: ImplicitlyDestructible,
        method: def(
            mut self_type, PythonObject
        ) thin raises PySlotError -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `binaryfunc` slot from a mut-receiver method."""
        Self.binary[self_type, _lift_mut_obj_to_obj[self_type, method], slot](
            ptr
        )

    @staticmethod
    def binary_conv_r[
        self_type: ImplicitlyDestructible,
        R: _CPython,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], PythonObject
        ) thin raises PySlotError -> R,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `binaryfunc` slot from a raising ConvertibleToPython method.
        """
        Self.binary[self_type, _conv_ptr_r_binary[self_type, R, method], slot](
            ptr
        )

    @staticmethod
    def binary_conv_nr[
        self_type: ImplicitlyDestructible,
        R: _CPython,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `binaryfunc` slot from a non-raising ConvertibleToPython method.
        """
        Self.binary[self_type, _conv_ptr_nr_binary[self_type, R, method], slot](
            ptr
        )

    @staticmethod
    def binary_conv_val[
        self_type: ImplicitlyDestructible,
        R: _CPython,
        method: def(self_type, PythonObject) thin raises PySlotError -> R,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `binaryfunc` slot from a value-receiver ConvertibleToPython method.
        """
        Self.binary[self_type, _conv_val_r_binary[self_type, R, method], slot](
            ptr
        )

    # Ternary

    @staticmethod
    def ternary_nr[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], PythonObject, PythonObject
        ) thin -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `ternaryfunc` slot from a non-raising method."""
        Self.ternary[self_type, _lift_obj_obj_to_obj[self_type, method], slot](
            ptr
        )

    @staticmethod
    def ternary_val[
        self_type: ImplicitlyDestructible,
        method: def(
            self_type, PythonObject, PythonObject
        ) thin raises PySlotError -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `ternaryfunc` slot from a value-receiver method."""
        Self.ternary[
            self_type, _lift_val_obj_obj_to_obj[self_type, method], slot
        ](ptr)

    @staticmethod
    def ternary_mut[
        self_type: ImplicitlyDestructible,
        method: def(
            mut self_type, PythonObject, PythonObject
        ) thin raises PySlotError -> PythonObject,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `ternaryfunc` slot from a mut-receiver method."""
        Self.ternary[
            self_type, _lift_mut_obj_obj_to_obj[self_type, method], slot
        ](ptr)

    @staticmethod
    def ternary_conv_r[
        self_type: ImplicitlyDestructible,
        R: _CPython,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], PythonObject, PythonObject
        ) thin raises PySlotError -> R,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `ternaryfunc` slot from a raising ConvertibleToPython method.
        """
        Self.ternary[
            self_type, _conv_ptr_r_ternary[self_type, R, method], slot
        ](ptr)

    @staticmethod
    def ternary_conv_nr[
        self_type: ImplicitlyDestructible,
        R: _CPython,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], PythonObject, PythonObject
        ) thin -> R,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `ternaryfunc` slot from a non-raising ConvertibleToPython method.
        """
        Self.ternary[
            self_type, _conv_ptr_nr_ternary[self_type, R, method], slot
        ](ptr)

    @staticmethod
    def ternary_conv_val[
        self_type: ImplicitlyDestructible,
        R: _CPython,
        method: def(
            self_type, PythonObject, PythonObject
        ) thin raises PySlotError -> R,
        slot: Int32,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert a `ternaryfunc` slot from a value-receiver ConvertibleToPython method.
        """
        Self.ternary[
            self_type, _conv_val_r_ternary[self_type, R, method], slot
        ](ptr)

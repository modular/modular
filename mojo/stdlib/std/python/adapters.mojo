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
from std.python._cpython import PyObject, PyObjectPtr, Py_ssize_t, PyType_Slot
from std.python.bindings import PythonTypeBuilder
from std.python.conversions import ConvertibleToPython
from std.utils import Variant

from .utils import PySlotError


@always_inline
def _unwrap_self[
    T: ImplicitlyDestructible
](py_self: PyObjectPtr) raises -> UnsafePointer[T, MutAnyOrigin]:
    """Downcast a raw PyObjectPtr to a typed Mojo pointer.
    """
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
# CPython type slot indices — do not renumber; these are part of the stable ABI.
# ref: https://github.com/python/cpython/blob/main/Include/typeslots.h
# ===----------------------------------------------------------------------=== #


struct _PySlotIndex:
    """CPython slot index constants for use with `PythonTypeBuilder._insert_slot`.

    These match the values in CPython's `typeslots.h` and are part of the
    stable ABI — do not renumber them.
    """

    # Buffer protocol
    comptime bf_getbuffer = Int32(1)
    comptime bf_releasebuffer = Int32(2)
    # Mapping protocol
    comptime mp_setitem = Int32(3)  # mp_ass_subscript
    comptime mp_length = Int32(4)
    comptime mp_getitem = Int32(5)  # mp_subscript
    # Number protocol
    comptime nb_absolute = Int32(6)
    comptime nb_add = Int32(7)
    comptime nb_and = Int32(8)
    comptime nb_bool = Int32(9)
    comptime nb_divmod = Int32(10)
    comptime nb_float = Int32(11)
    comptime nb_floor_divide = Int32(12)
    comptime nb_index = Int32(13)
    comptime nb_inplace_add = Int32(14)
    comptime nb_inplace_and = Int32(15)
    comptime nb_inplace_floor_divide = Int32(16)
    comptime nb_inplace_lshift = Int32(17)
    comptime nb_inplace_multiply = Int32(18)
    comptime nb_inplace_or = Int32(19)
    comptime nb_inplace_power = Int32(20)
    comptime nb_inplace_remainder = Int32(21)
    comptime nb_inplace_rshift = Int32(22)
    comptime nb_inplace_subtract = Int32(23)
    comptime nb_inplace_true_divide = Int32(24)
    comptime nb_inplace_xor = Int32(25)
    comptime nb_int = Int32(26)
    comptime nb_invert = Int32(27)
    comptime nb_lshift = Int32(28)
    comptime nb_multiply = Int32(29)
    comptime nb_negative = Int32(30)
    comptime nb_or = Int32(31)
    comptime nb_positive = Int32(32)
    comptime nb_power = Int32(33)
    comptime nb_remainder = Int32(34)
    comptime nb_rshift = Int32(35)
    comptime nb_subtract = Int32(36)
    comptime nb_true_divide = Int32(37)
    comptime nb_xor = Int32(38)
    # Sequence protocol
    comptime sq_ass_item = Int32(39)
    comptime sq_concat = Int32(40)
    comptime sq_contains = Int32(41)
    comptime sq_inplace_concat = Int32(42)
    comptime sq_inplace_repeat = Int32(43)
    comptime sq_item = Int32(44)
    comptime sq_length = Int32(45)
    comptime sq_repeat = Int32(46)
    # Type protocol
    comptime tp_alloc = Int32(47)
    comptime tp_base = Int32(48)
    comptime tp_bases = Int32(49)
    comptime tp_call = Int32(50)
    comptime tp_clear = Int32(51)
    comptime tp_dealloc = Int32(52)
    comptime tp_del = Int32(53)
    comptime tp_descr_get = Int32(54)
    comptime tp_descr_set = Int32(55)
    comptime tp_doc = Int32(56)
    comptime tp_getattr = Int32(57)
    comptime tp_getattro = Int32(58)
    comptime tp_hash = Int32(59)
    comptime tp_init = Int32(60)
    comptime tp_is_gc = Int32(61)
    comptime tp_iter = Int32(62)
    comptime tp_iternext = Int32(63)
    comptime tp_methods = Int32(64)
    comptime tp_new = Int32(65)
    comptime tp_repr = Int32(66)
    comptime tp_richcompare = Int32(67)
    comptime tp_setattr = Int32(68)
    comptime tp_setattro = Int32(69)
    comptime tp_str = Int32(70)
    comptime tp_traverse = Int32(71)
    comptime tp_members = Int32(72)
    comptime tp_getset = Int32(73)
    comptime tp_free = Int32(74)
    comptime nb_matrix_multiply = Int32(75)
    comptime nb_inplace_matrix_multiply = Int32(76)
    # Async protocol (Python 3.5+)
    comptime am_await = Int32(77)
    comptime am_aiter = Int32(78)
    comptime am_anext = Int32(79)
    comptime tp_finalize = Int32(80)  # Python 3.5+
    comptime am_send = Int32(81)  # Python 3.10+
    comptime tp_vectorcall = Int32(82)  # Python 3.14+
    comptime tp_token = Int32(83)  # Python 3.14+


# ===----------------------------------------------------------------------=== #
# Slot-install helpers — insert a typed C function pointer into a builder
# ===----------------------------------------------------------------------=== #


def _install_unary[
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


def _install_binary[
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


def _install_ternary[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject, PythonObject
    ) thin raises PySlotError -> PythonObject,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `ternaryfunc` slot into the builder pointed to by `ptr`."""
    comptime _ternaryfunc = def(PyObjectPtr, PyObjectPtr, PyObjectPtr) thin abi(
        "C"
    ) -> PyObjectPtr
    var fn_ptr: _ternaryfunc = _ternaryfunc_wrapper[self_type, method]
    ptr[]._insert_slot(
        PyType_Slot(slot, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr))
    )


def _install_inquiry[
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


def _install_richcompare[
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
            _PySlotIndex.tp_richcompare,
            rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
        )
    )


def _install_lenfunc[
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
            _PySlotIndex.mp_length, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr)
        )
    )


def _install_mp_getitem[
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
            _PySlotIndex.mp_getitem, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr)
        )
    )


def _install_objobjargproc[
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
    var fn_ptr: _objobjargproc = _mp_ass_subscript_wrapper[self_type, method]
    ptr[]._insert_slot(
        PyType_Slot(
            _PySlotIndex.mp_setitem, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr)
        )
    )


def _install_ssizeargfunc[
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


def _install_ssizeobjargproc[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], Int, Variant[PythonObject, Int]
    ) thin raises PySlotError -> None,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert the `ssizeobjargproc` slot (`sq_ass_item`) into the builder pointed to by `ptr`.
    """
    comptime _ssizeobjargproc = def(
        PyObjectPtr, Py_ssize_t, PyObjectPtr
    ) thin abi("C") -> c_int
    var fn_ptr: _ssizeobjargproc = _ssizeobjargproc_wrapper[self_type, method]
    ptr[]._insert_slot(
        PyType_Slot(
            _PySlotIndex.sq_ass_item,
            rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
        )
    )


def _install_objobjproc[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject
    ) thin raises PySlotError -> Bool,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert an `objobjproc` slot into the builder pointed to by `ptr`."""
    comptime _objobjproc = def(PyObjectPtr, PyObjectPtr) thin abi("C") -> c_int
    var fn_ptr: _objobjproc = _objobjproc_wrapper[self_type, method]
    ptr[]._insert_slot(
        PyType_Slot(slot, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr))
    )


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
    method: def(UnsafePointer[T, MutAnyOrigin], Int) thin raises PySlotError -> R,
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


# ===----------------------------------------------------------------------=== #
# Wrapped slot-install helpers — variants that compose a lift/conv helper
# around the caller's method so per-protocol builder call sites become a
# single line instead of a multi-line `_install_X[..., _lift_Y[...], slot]`.
# ===----------------------------------------------------------------------=== #


# Unary


def _install_unary_nr[
    self_type: ImplicitlyDestructible,
    method: def(UnsafePointer[self_type, MutAnyOrigin]) thin -> PythonObject,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `unaryfunc` slot from a non-raising method."""
    _install_unary[self_type, _lift_to_obj[self_type, method], slot](ptr)


def _install_unary_val[
    self_type: ImplicitlyDestructible,
    method: def(self_type) thin raises PySlotError -> PythonObject,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `unaryfunc` slot from a value-receiver method."""
    _install_unary[self_type, _lift_val_to_obj[self_type, method], slot](ptr)


def _install_unary_conv_r[
    self_type: ImplicitlyDestructible,
    R: _CPython,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin]
    ) thin raises PySlotError -> R,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `unaryfunc` slot from a raising ConvertibleToPython method."""
    _install_unary[self_type, _conv_ptr_r_unary[self_type, R, method], slot](
        ptr
    )


def _install_unary_conv_nr[
    self_type: ImplicitlyDestructible,
    R: _CPython,
    method: def(UnsafePointer[self_type, MutAnyOrigin]) thin -> R,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `unaryfunc` slot from a non-raising ConvertibleToPython method.
    """
    _install_unary[self_type, _conv_ptr_nr_unary[self_type, R, method], slot](
        ptr
    )


def _install_unary_conv_val[
    self_type: ImplicitlyDestructible,
    R: _CPython,
    method: def(self_type) thin raises PySlotError -> R,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `unaryfunc` slot from a value-receiver ConvertibleToPython method.
    """
    _install_unary[self_type, _conv_val_r_unary[self_type, R, method], slot](
        ptr
    )


# Inquiry


def _install_inquiry_nr[
    self_type: ImplicitlyDestructible,
    method: def(UnsafePointer[self_type, MutAnyOrigin]) thin -> Bool,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert an `inquiry` slot from a non-raising method."""
    _install_inquiry[self_type, _lift_to_bool[self_type, method], slot](ptr)


def _install_inquiry_val[
    self_type: ImplicitlyDestructible,
    method: def(self_type) thin raises PySlotError -> Bool,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert an `inquiry` slot from a value-receiver method."""
    _install_inquiry[self_type, _lift_val_to_bool[self_type, method], slot](ptr)


# Binary


def _install_binary_nr[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject
    ) thin -> PythonObject,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `binaryfunc` slot from a non-raising method."""
    _install_binary[self_type, _lift_obj_to_obj[self_type, method], slot](ptr)


def _install_binary_val[
    self_type: ImplicitlyDestructible,
    method: def(self_type, PythonObject) thin raises PySlotError -> PythonObject,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `binaryfunc` slot from a value-receiver method."""
    _install_binary[self_type, _lift_val_obj_to_obj[self_type, method], slot](
        ptr
    )


def _install_binary_mut[
    self_type: ImplicitlyDestructible,
    method: def(
        mut self_type, PythonObject
    ) thin raises PySlotError -> PythonObject,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `binaryfunc` slot from a mut-receiver method."""
    _install_binary[self_type, _lift_mut_obj_to_obj[self_type, method], slot](
        ptr
    )


def _install_binary_conv_r[
    self_type: ImplicitlyDestructible,
    R: _CPython,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject
    ) thin raises PySlotError -> R,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `binaryfunc` slot from a raising ConvertibleToPython method."""
    _install_binary[self_type, _conv_ptr_r_binary[self_type, R, method], slot](
        ptr
    )


def _install_binary_conv_nr[
    self_type: ImplicitlyDestructible,
    R: _CPython,
    method: def(UnsafePointer[self_type, MutAnyOrigin], PythonObject) thin -> R,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `binaryfunc` slot from a non-raising ConvertibleToPython method.
    """
    _install_binary[self_type, _conv_ptr_nr_binary[self_type, R, method], slot](
        ptr
    )


def _install_binary_conv_val[
    self_type: ImplicitlyDestructible,
    R: _CPython,
    method: def(self_type, PythonObject) thin raises PySlotError -> R,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `binaryfunc` slot from a value-receiver ConvertibleToPython method.
    """
    _install_binary[self_type, _conv_val_r_binary[self_type, R, method], slot](
        ptr
    )


# Ternary


def _install_ternary_nr[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject, PythonObject
    ) thin -> PythonObject,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `ternaryfunc` slot from a non-raising method."""
    _install_ternary[self_type, _lift_obj_obj_to_obj[self_type, method], slot](
        ptr
    )


def _install_ternary_val[
    self_type: ImplicitlyDestructible,
    method: def(
        self_type, PythonObject, PythonObject
    ) thin raises PySlotError -> PythonObject,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `ternaryfunc` slot from a value-receiver method."""
    _install_ternary[
        self_type, _lift_val_obj_obj_to_obj[self_type, method], slot
    ](ptr)


def _install_ternary_mut[
    self_type: ImplicitlyDestructible,
    method: def(
        mut self_type, PythonObject, PythonObject
    ) thin raises PySlotError -> PythonObject,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `ternaryfunc` slot from a mut-receiver method."""
    _install_ternary[
        self_type, _lift_mut_obj_obj_to_obj[self_type, method], slot
    ](ptr)


def _install_ternary_conv_r[
    self_type: ImplicitlyDestructible,
    R: _CPython,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject, PythonObject
    ) thin raises PySlotError -> R,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `ternaryfunc` slot from a raising ConvertibleToPython method."""
    _install_ternary[
        self_type, _conv_ptr_r_ternary[self_type, R, method], slot
    ](ptr)


def _install_ternary_conv_nr[
    self_type: ImplicitlyDestructible,
    R: _CPython,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], PythonObject, PythonObject
    ) thin -> R,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `ternaryfunc` slot from a non-raising ConvertibleToPython method.
    """
    _install_ternary[
        self_type, _conv_ptr_nr_ternary[self_type, R, method], slot
    ](ptr)


def _install_ternary_conv_val[
    self_type: ImplicitlyDestructible,
    R: _CPython,
    method: def(
        self_type, PythonObject, PythonObject
    ) thin raises PySlotError -> R,
    slot: Int32,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert a `ternaryfunc` slot from a value-receiver ConvertibleToPython method.
    """
    _install_ternary[
        self_type, _conv_val_r_ternary[self_type, R, method], slot
    ](ptr)

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

from std.memory import OpaquePointer, UnsafePointer
from std.python import PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t, PySlotIndex, PyType_Slot
from std.python.bindings import PythonTypeBuilder
from std.utils import Variant

from .utils import PySlotError

from .adapters import (
    _CPython,
    _SlotInstaller,
    _conv_ptr_nr_int_arg,
    _conv_ptr_r_int_arg,
    _conv_val_r_int_arg,
    _lift_int_to_obj,
    _lift_int_var_to_none,
    _lift_mut_int_var_to_none,
    _lift_obj_to_bool,
    _lift_to_int,
    _lift_val_int_to_obj,
    _lift_val_int_var_to_none,
    _lift_val_obj_to_bool,
    _lift_val_to_int,
    _mp_length_wrapper,
)


struct SequenceProtocolBuilder[self_type: ImplicitlyDestructible]:
    """Installs CPython sequence protocol slots on a `PythonTypeBuilder`.

    Construct directly from a `PythonTypeBuilder`.  Method names follow the
    corresponding Python dunders.
    Handler functions receive `UnsafePointer[T, MutAnyOrigin]` as their first
    argument instead of a raw `PythonObject`.

    `def_getitem`, `def_repeat`, and `def_irepeat` use `ssizeargfunc`
    (integer index/count), unlike the mapping protocol which uses a
    `PythonObject` key.  `def_contains` uses `objobjproc`.

    Usage:
        ```mojo
        var spb = SequenceProtocolBuilder[MyStruct](tb)
        spb.def_len[MyStruct.py__len__]()
           .def_getitem[MyStruct.py__getitem__]()
           .def_contains[MyStruct.py__contains__]()
        ```
    """

    var _ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]

    def __init__(out self, mut inner: PythonTypeBuilder):
        self._ptr = UnsafePointer(to=inner)

    def __init__(
        out self,
        ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin],
    ):
        self._ptr = ptr

    def def_len[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises PySlotError ->Int
    ](mut self) -> ref[self] Self:
        """Install `__len__` via the `sq_length` slot.

        Called by `len(obj)`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_length
        """
        comptime _lenfunc = def(PyObjectPtr) thin abi("C") -> Py_ssize_t
        var fn_ptr: _lenfunc = _mp_length_wrapper[Self.self_type, method]
        self._ptr[]._insert_slot(
            PyType_Slot(
                PySlotIndex.sq_length,
                rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
            )
        )
        return self

    def def_getitem[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], Int
        ) thin raises PySlotError ->PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `sq_item` slot (integer index).

        Called by `obj[index]`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_item
        """
        _SlotInstaller.ssizeargfunc[Self.self_type, method, PySlotIndex.sq_item](
            self._ptr
        )
        return self

    def def_setitem[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            Int,
            Variant[PythonObject, Int],
        ) thin raises PySlotError -> None
    ](mut self) -> ref[self] Self:
        """Install `__setitem__`/`__delitem__` via the `sq_ass_item` slot.

        Called by `obj[index] = value` or `del obj[index]`.

        The third argument to `method` is a `Variant`:
        - `Variant[PythonObject, Int](value)` for assignment.
        - `Variant[PythonObject, Int](Int(0))` for deletion (null C pointer).

        `method` declares `raises PySlotError`; raise a variant such as
        `PySlotError.index_error(...)` or `PySlotError.type_error(...)` and
        the wrapper maps it to the corresponding Python exception.

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_ass_item
        """
        _SlotInstaller.ssizeobjargproc[Self.self_type, method](self._ptr)
        return self

    def def_contains[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises PySlotError ->Bool
    ](mut self) -> ref[self] Self:
        """Install `__contains__` via the `sq_contains` slot.

        Called by `item in obj`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_contains
        """
        _SlotInstaller.objobjproc[Self.self_type, method, PySlotIndex.sq_contains](
            self._ptr
        )
        return self

    def def_concat[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises PySlotError ->PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__add__` (concatenation) via the `sq_concat` slot.

        Called by `obj + other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_concat
        """
        _SlotInstaller.binary[Self.self_type, method, PySlotIndex.sq_concat](
            self._ptr
        )
        return self

    def def_repeat[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], Int
        ) thin raises PySlotError ->PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__mul__` (repetition) via the `sq_repeat` slot.

        Called by `obj * count`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_repeat
        """
        _SlotInstaller.ssizeargfunc[Self.self_type, method, PySlotIndex.sq_repeat](
            self._ptr
        )
        return self

    def def_iconcat[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises PySlotError ->PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` (in-place concatenation) via the `sq_inplace_concat` slot.

        Called by `obj += other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_concat
        """
        _SlotInstaller.binary[Self.self_type, method, PySlotIndex.sq_inplace_concat](
            self._ptr
        )
        return self

    def def_irepeat[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], Int
        ) thin raises PySlotError ->PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imul__` (in-place repetition) via the `sq_inplace_repeat` slot.

        Called by `obj *= count`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_repeat
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type, method, PySlotIndex.sq_inplace_repeat
        ](self._ptr)
        return self

    # Non-raising overloads

    def def_len[
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin]) thin -> Int
    ](mut self) -> ref[self] Self:
        """Install `__len__` via the `sq_length` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_length
        """
        comptime _lenfunc = def(PyObjectPtr) thin abi("C") -> Py_ssize_t
        var fn_ptr: _lenfunc = _mp_length_wrapper[
            Self.self_type, _lift_to_int[Self.self_type, method]
        ]
        self._ptr[]._insert_slot(
            PyType_Slot(
                PySlotIndex.sq_length,
                rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
            )
        )
        return self

    def def_getitem[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], Int
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `sq_item` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_item
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _lift_int_to_obj[Self.self_type, method],
            PySlotIndex.sq_item,
        ](self._ptr)
        return self

    def def_setitem[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            Int,
            Variant[PythonObject, Int],
        ) thin -> None
    ](mut self) -> ref[self] Self:
        """Install `__setitem__`/`__delitem__` via the `sq_ass_item` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_ass_item
        """
        _SlotInstaller.ssizeobjargproc[
            Self.self_type, _lift_int_var_to_none[Self.self_type, method]
        ](self._ptr)
        return self

    def def_contains[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> Bool
    ](mut self) -> ref[self] Self:
        """Install `__contains__` via the `sq_contains` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_contains
        """
        _SlotInstaller.objobjproc[
            Self.self_type,
            _lift_obj_to_bool[Self.self_type, method],
            PySlotIndex.sq_contains,
        ](self._ptr)
        return self

    def def_concat[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__add__` (concatenation) via the `sq_concat` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_concat
        """
        _SlotInstaller.binary_nr[Self.self_type, method, PySlotIndex.sq_concat](
            self._ptr
        )
        return self

    def def_repeat[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], Int
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__mul__` (repetition) via the `sq_repeat` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_repeat
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _lift_int_to_obj[Self.self_type, method],
            PySlotIndex.sq_repeat,
        ](self._ptr)
        return self

    def def_iconcat[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` (in-place concatenation) via the `sq_inplace_concat` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_concat
        """
        _SlotInstaller.binary_nr[
            Self.self_type, method, PySlotIndex.sq_inplace_concat
        ](self._ptr)
        return self

    def def_irepeat[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], Int
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imul__` (in-place repetition) via the `sq_inplace_repeat` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_repeat
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _lift_int_to_obj[Self.self_type, method],
            PySlotIndex.sq_inplace_repeat,
        ](self._ptr)
        return self

    # Value-receiver overloads

    def def_len[
        method: def(Self.self_type) thin raises PySlotError ->Int
    ](mut self) -> ref[self] Self:
        """Install `__len__` via the `sq_length` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_length
        """
        comptime _lenfunc = def(PyObjectPtr) thin abi("C") -> Py_ssize_t
        var fn_ptr: _lenfunc = _mp_length_wrapper[
            Self.self_type, _lift_val_to_int[Self.self_type, method]
        ]
        self._ptr[]._insert_slot(
            PyType_Slot(
                PySlotIndex.sq_length,
                rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
            )
        )
        return self

    def def_getitem[
        method: def(Self.self_type, Int) thin raises PySlotError ->PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `sq_item` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_item
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _lift_val_int_to_obj[Self.self_type, method],
            PySlotIndex.sq_item,
        ](self._ptr)
        return self

    def def_setitem[
        method: def(
            Self.self_type, Int, Variant[PythonObject, Int]
        ) thin raises PySlotError -> None
    ](mut self) -> ref[self] Self:
        """Install `__setitem__`/`__delitem__` via the `sq_ass_item` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_ass_item
        """
        _SlotInstaller.ssizeobjargproc[
            Self.self_type, _lift_val_int_var_to_none[Self.self_type, method]
        ](self._ptr)
        return self

    def def_setitem[
        method: def(
            mut Self.self_type, Int, Variant[PythonObject, Int]
        ) thin raises PySlotError -> None
    ](mut self) -> ref[self] Self:
        """Install `__setitem__`/`__delitem__` via the `sq_ass_item` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_ass_item
        """
        _SlotInstaller.ssizeobjargproc[
            Self.self_type, _lift_mut_int_var_to_none[Self.self_type, method]
        ](self._ptr)
        return self

    def def_contains[
        method: def(Self.self_type, PythonObject) thin raises PySlotError ->Bool
    ](mut self) -> ref[self] Self:
        """Install `__contains__` via the `sq_contains` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_contains
        """
        _SlotInstaller.objobjproc[
            Self.self_type,
            _lift_val_obj_to_bool[Self.self_type, method],
            PySlotIndex.sq_contains,
        ](self._ptr)
        return self

    def def_concat[
        method: def(Self.self_type, PythonObject) thin raises PySlotError ->PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__add__` (concatenation) via the `sq_concat` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_concat
        """
        _SlotInstaller.binary_val[Self.self_type, method, PySlotIndex.sq_concat](
            self._ptr
        )
        return self

    def def_repeat[
        method: def(Self.self_type, Int) thin raises PySlotError ->PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__mul__` (repetition) via the `sq_repeat` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_repeat
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _lift_val_int_to_obj[Self.self_type, method],
            PySlotIndex.sq_repeat,
        ](self._ptr)
        return self

    def def_iconcat[
        method: def(Self.self_type, PythonObject) thin raises PySlotError ->PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` (in-place concatenation) via the `sq_inplace_concat` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_concat
        """
        _SlotInstaller.binary_val[
            Self.self_type, method, PySlotIndex.sq_inplace_concat
        ](self._ptr)
        return self

    def def_irepeat[
        method: def(Self.self_type, Int) thin raises PySlotError ->PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imul__` (in-place repetition) via the `sq_inplace_repeat` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_repeat
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _lift_val_int_to_obj[Self.self_type, method],
            PySlotIndex.sq_inplace_repeat,
        ](self._ptr)
        return self

    # ConvertibleToPython return overloads

    def def_getitem[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], Int
        ) thin raises PySlotError ->R,
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `sq_item` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_item
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _conv_ptr_r_int_arg[Self.self_type, R, method],
            PySlotIndex.sq_item,
        ](self._ptr)
        return self

    def def_getitem[
        R: _CPython,
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin], Int) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `sq_item` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_item
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _conv_ptr_nr_int_arg[Self.self_type, R, method],
            PySlotIndex.sq_item,
        ](self._ptr)
        return self

    def def_getitem[
        R: _CPython,
        method: def(Self.self_type, Int) thin raises PySlotError ->R,
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `sq_item` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_item
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _conv_val_r_int_arg[Self.self_type, R, method],
            PySlotIndex.sq_item,
        ](self._ptr)
        return self

    def def_concat[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises PySlotError ->R,
    ](mut self) -> ref[self] Self:
        """Install `__add__` via the `sq_concat` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_concat
        """
        _SlotInstaller.binary_conv_r[
            Self.self_type, R, method, PySlotIndex.sq_concat
        ](self._ptr)
        return self

    def def_concat[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__add__` via the `sq_concat` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_concat
        """
        _SlotInstaller.binary_conv_nr[
            Self.self_type, R, method, PySlotIndex.sq_concat
        ](self._ptr)
        return self

    def def_concat[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises PySlotError ->R,
    ](mut self) -> ref[self] Self:
        """Install `__add__` via the `sq_concat` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_concat
        """
        _SlotInstaller.binary_conv_val[
            Self.self_type, R, method, PySlotIndex.sq_concat
        ](self._ptr)
        return self

    def def_repeat[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], Int
        ) thin raises PySlotError ->R,
    ](mut self) -> ref[self] Self:
        """Install `__mul__` via the `sq_repeat` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_repeat
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _conv_ptr_r_int_arg[Self.self_type, R, method],
            PySlotIndex.sq_repeat,
        ](self._ptr)
        return self

    def def_repeat[
        R: _CPython,
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin], Int) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__mul__` via the `sq_repeat` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_repeat
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _conv_ptr_nr_int_arg[Self.self_type, R, method],
            PySlotIndex.sq_repeat,
        ](self._ptr)
        return self

    def def_repeat[
        R: _CPython,
        method: def(Self.self_type, Int) thin raises PySlotError ->R,
    ](mut self) -> ref[self] Self:
        """Install `__mul__` via the `sq_repeat` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_repeat
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _conv_val_r_int_arg[Self.self_type, R, method],
            PySlotIndex.sq_repeat,
        ](self._ptr)
        return self

    def def_iconcat[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises PySlotError ->R,
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` via the `sq_inplace_concat` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_concat
        """
        _SlotInstaller.binary_conv_r[
            Self.self_type, R, method, PySlotIndex.sq_inplace_concat
        ](self._ptr)
        return self

    def def_iconcat[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` via the `sq_inplace_concat` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_concat
        """
        _SlotInstaller.binary_conv_nr[
            Self.self_type, R, method, PySlotIndex.sq_inplace_concat
        ](self._ptr)
        return self

    def def_iconcat[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises PySlotError ->R,
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` via the `sq_inplace_concat` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_concat
        """
        _SlotInstaller.binary_conv_val[
            Self.self_type, R, method, PySlotIndex.sq_inplace_concat
        ](self._ptr)
        return self

    def def_irepeat[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], Int
        ) thin raises PySlotError ->R,
    ](mut self) -> ref[self] Self:
        """Install `__imul__` via the `sq_inplace_repeat` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_repeat
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _conv_ptr_r_int_arg[Self.self_type, R, method],
            PySlotIndex.sq_inplace_repeat,
        ](self._ptr)
        return self

    def def_irepeat[
        R: _CPython,
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin], Int) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__imul__` via the `sq_inplace_repeat` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_repeat
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _conv_ptr_nr_int_arg[Self.self_type, R, method],
            PySlotIndex.sq_inplace_repeat,
        ](self._ptr)
        return self

    def def_irepeat[
        R: _CPython,
        method: def(Self.self_type, Int) thin raises PySlotError ->R,
    ](mut self) -> ref[self] Self:
        """Install `__imul__` via the `sq_inplace_repeat` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_repeat
        """
        _SlotInstaller.ssizeargfunc[
            Self.self_type,
            _conv_val_r_int_arg[Self.self_type, R, method],
            PySlotIndex.sq_inplace_repeat,
        ](self._ptr)
        return self

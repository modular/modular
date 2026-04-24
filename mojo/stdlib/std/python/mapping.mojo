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

from std.memory import UnsafePointer
from std.python import PythonObject
from std.python.bindings import PythonTypeBuilder
from std.utils import Variant

from .adapters import (
    _CPython,
    _conv_ptr_nr_binary,
    _conv_ptr_r_binary,
    _conv_val_r_binary,
    _install_lenfunc,
    _install_mp_getitem,
    _install_objobjargproc,
    _lift_mut_obj_var_to_none,
    _lift_obj_to_obj,
    _lift_obj_var_to_none,
    _lift_to_int,
    _lift_val_obj_to_obj,
    _lift_val_to_int,
)


struct MappingProtocolBuilder[self_type: ImplicitlyDestructible]:
    """Installs CPython mapping protocol slots on a `PythonTypeBuilder`.

    Construct directly from a `PythonTypeBuilder`.  The three methods correspond
    to `__len__`, `__getitem__`, and `__setitem__`/`__delitem__`.
    Handler functions receive `UnsafePointer[T, MutAnyOrigin]` as their first
    argument instead of a raw `PythonObject`.

    Usage:
        ```mojo
        var mpb = MappingProtocolBuilder[MyStruct](tb)
        mpb.def_len[MyStruct.py__len__]()
           .def_getitem[MyStruct.py__getitem__]()
           .def_setitem[MyStruct.py__setitem__]()
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
        ) thin raises -> Int
    ](mut self) -> ref[self] Self:
        """Install `__len__` via the `mp_length` slot.

        Called by `len(obj)`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_length
        """
        _install_lenfunc[Self.self_type, method](self._ptr)
        return self

    def def_getitem[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `mp_subscript` slot.

        Called by `obj[key]`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_subscript
        """
        _install_mp_getitem[Self.self_type, method](self._ptr)
        return self

    def def_setitem[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            PythonObject,
            Variant[PythonObject, Int],
        ) thin raises -> None
    ](mut self) -> ref[self] Self:
        """Install `__setitem__`/`__delitem__` via the `mp_ass_subscript` slot.

        Called by `obj[key] = value` or `del obj[key]`.

        The third argument to `method` is a `Variant`:
        - `Variant[PythonObject, Int](value)` for assignment.
        - `Variant[PythonObject, Int](Int(0))` for deletion (null C pointer).
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_ass_subscript
        """
        _install_objobjargproc[Self.self_type, method](self._ptr)
        return self

    # Non-raising overloads

    def def_len[
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin]) thin -> Int
    ](mut self) -> ref[self] Self:
        """Install `__len__` via the `mp_length` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_length
        """
        _install_lenfunc[Self.self_type, _lift_to_int[Self.self_type, method]](
            self._ptr
        )
        return self

    def def_getitem[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `mp_subscript` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_subscript
        """
        _install_mp_getitem[
            Self.self_type, _lift_obj_to_obj[Self.self_type, method]
        ](self._ptr)
        return self

    def def_setitem[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            PythonObject,
            Variant[PythonObject, Int],
        ) thin -> None
    ](mut self) -> ref[self] Self:
        """Install `__setitem__`/`__delitem__` via the `mp_ass_subscript` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_ass_subscript
        """
        _install_objobjargproc[
            Self.self_type, _lift_obj_var_to_none[Self.self_type, method]
        ](self._ptr)
        return self

    # Value-receiver overloads

    def def_len[
        method: def(Self.self_type) thin raises -> Int
    ](mut self) -> ref[self] Self:
        """Install `__len__` via the `mp_length` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_length
        """
        _install_lenfunc[
            Self.self_type, _lift_val_to_int[Self.self_type, method]
        ](self._ptr)
        return self

    def def_getitem[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `mp_subscript` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_subscript
        """
        _install_mp_getitem[
            Self.self_type, _lift_val_obj_to_obj[Self.self_type, method]
        ](self._ptr)
        return self

    def def_setitem[
        method: def(
            mut Self.self_type, PythonObject, Variant[PythonObject, Int]
        ) thin raises -> None
    ](mut self) -> ref[self] Self:
        """Install `__setitem__`/`__delitem__` via the `mp_ass_subscript` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_ass_subscript
        """
        _install_objobjargproc[
            Self.self_type, _lift_mut_obj_var_to_none[Self.self_type, method]
        ](self._ptr)
        return self

    # ConvertibleToPython return overloads

    def def_getitem[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `mp_subscript` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_subscript
        """
        _install_mp_getitem[
            Self.self_type, _conv_ptr_r_binary[Self.self_type, R, method]
        ](self._ptr)
        return self

    def def_getitem[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `mp_subscript` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_subscript
        """
        _install_mp_getitem[
            Self.self_type, _conv_ptr_nr_binary[Self.self_type, R, method]
        ](self._ptr)
        return self

    def def_getitem[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__getitem__` via the `mp_subscript` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_subscript
        """
        _install_mp_getitem[
            Self.self_type, _conv_val_r_binary[Self.self_type, R, method]
        ](self._ptr)
        return self

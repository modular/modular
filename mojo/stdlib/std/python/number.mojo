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

from .adapters import (
    _CPython,
    _PySlotIndex,
    _install_binary,
    _install_binary_conv_nr,
    _install_binary_conv_r,
    _install_binary_conv_val,
    _install_binary_mut,
    _install_binary_nr,
    _install_binary_val,
    _install_inquiry,
    _install_inquiry_nr,
    _install_inquiry_val,
    _install_ternary,
    _install_ternary_conv_nr,
    _install_ternary_conv_r,
    _install_ternary_conv_val,
    _install_ternary_mut,
    _install_ternary_nr,
    _install_ternary_val,
    _install_unary,
    _install_unary_conv_nr,
    _install_unary_conv_r,
    _install_unary_conv_val,
    _install_unary_nr,
    _install_unary_val,
)


struct NumberProtocolBuilder[self_type: ImplicitlyDestructible]:
    """Installs CPython number protocol slots on a `PythonTypeBuilder`.

    Construct directly from a `PythonTypeBuilder`.  Each method is named after the
    corresponding Python dunder and accepts only the matching function signature.
    Handler functions receive `UnsafePointer[T, MutAnyOrigin]` as their first
    argument instead of a raw `PythonObject`.

    Binary methods (`def_add`, `def_mul`, etc.) and ternary methods (`def_pow`,
    `def_ipow`) support `NotImplementedError`: raise it from your handler to
    return `Py_NotImplemented` to Python, triggering the reflected operation.

    Usage:
        ```mojo
        var npb = NumberProtocolBuilder[MyStruct](tb)
        npb.def_neg[MyStruct.py__neg__]()
           .def_bool[MyStruct.py__bool__]()
           .def_add[MyStruct.py__add__]()
           .def_pow[MyStruct.py__pow__]()
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

    # ------------------------------------------------------------------
    # Unary slots — C type: unaryfunc  def(PyObject *) -> PyObject *
    # ------------------------------------------------------------------

    def def_abs[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__abs__` via the `nb_absolute` slot.

        Called by `abs(obj)`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_absolute
        """
        _install_unary[Self.self_type, method, _PySlotIndex.nb_absolute](
            self._ptr
        )
        return self

    def def_float[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__float__` via the `nb_float` slot.

        Called by `float(obj)`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_float
        """
        _install_unary[Self.self_type, method, _PySlotIndex.nb_float](self._ptr)
        return self

    def def_index[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__index__` via the `nb_index` slot.

        Called by `operator.index(obj)`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_index
        """
        _install_unary[Self.self_type, method, _PySlotIndex.nb_index](self._ptr)
        return self

    def def_int[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__int__` via the `nb_int` slot.

        Called by `int(obj)`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_int
        """
        _install_unary[Self.self_type, method, _PySlotIndex.nb_int](self._ptr)
        return self

    def def_invert[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__invert__` via the `nb_invert` slot.

        Called by `~obj`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_invert
        """
        _install_unary[Self.self_type, method, _PySlotIndex.nb_invert](
            self._ptr
        )
        return self

    def def_neg[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__neg__` via the `nb_negative` slot.

        Called by `-obj`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_negative
        """
        _install_unary[Self.self_type, method, _PySlotIndex.nb_negative](
            self._ptr
        )
        return self

    def def_pos[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__pos__` via the `nb_positive` slot.

        Called by `+obj`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_positive
        """
        _install_unary[Self.self_type, method, _PySlotIndex.nb_positive](
            self._ptr
        )
        return self

    # Non-raising unary overloads

    def def_abs[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__abs__` via the `nb_absolute` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_absolute
        """
        _install_unary_nr[Self.self_type, method, _PySlotIndex.nb_absolute](
            self._ptr
        )
        return self

    def def_float[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__float__` via the `nb_float` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_float
        """
        _install_unary_nr[Self.self_type, method, _PySlotIndex.nb_float](
            self._ptr
        )
        return self

    def def_index[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__index__` via the `nb_index` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_index
        """
        _install_unary_nr[Self.self_type, method, _PySlotIndex.nb_index](
            self._ptr
        )
        return self

    def def_int[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__int__` via the `nb_int` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_int
        """
        _install_unary_nr[Self.self_type, method, _PySlotIndex.nb_int](
            self._ptr
        )
        return self

    def def_invert[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__invert__` via the `nb_invert` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_invert
        """
        _install_unary_nr[Self.self_type, method, _PySlotIndex.nb_invert](
            self._ptr
        )
        return self

    def def_neg[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__neg__` via the `nb_negative` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_negative
        """
        _install_unary_nr[Self.self_type, method, _PySlotIndex.nb_negative](
            self._ptr
        )
        return self

    def def_pos[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__pos__` via the `nb_positive` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_positive
        """
        _install_unary_nr[Self.self_type, method, _PySlotIndex.nb_positive](
            self._ptr
        )
        return self

    # Value-receiver unary overloads

    def def_abs[
        method: def(Self.self_type) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__abs__` via the `nb_absolute` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_absolute
        """
        _install_unary_val[Self.self_type, method, _PySlotIndex.nb_absolute](
            self._ptr
        )
        return self

    def def_float[
        method: def(Self.self_type) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__float__` via the `nb_float` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_float
        """

        _install_unary_val[Self.self_type, method, _PySlotIndex.nb_float](
            self._ptr
        )
        return self

    def def_index[
        method: def(Self.self_type) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__index__` via the `nb_index` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_index
        """
        _install_unary_val[Self.self_type, method, _PySlotIndex.nb_index](
            self._ptr
        )
        return self

    def def_int[
        method: def(Self.self_type) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__int__` via the `nb_int` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_int
        """
        _install_unary_val[Self.self_type, method, _PySlotIndex.nb_int](
            self._ptr
        )
        return self

    def def_invert[
        method: def(Self.self_type) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__invert__` via the `nb_invert` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_invert
        """
        _install_unary_val[Self.self_type, method, _PySlotIndex.nb_invert](
            self._ptr
        )
        return self

    def def_neg[
        method: def(Self.self_type) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__neg__` via the `nb_negative` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_negative
        """
        _install_unary_val[Self.self_type, method, _PySlotIndex.nb_negative](
            self._ptr
        )
        return self

    def def_pos[
        method: def(Self.self_type) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__pos__` via the `nb_positive` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_positive
        """
        _install_unary_val[Self.self_type, method, _PySlotIndex.nb_positive](
            self._ptr
        )
        return self

    # ------------------------------------------------------------------
    # Bool slot — C type: inquiry  int(*)(PyObject *)
    # ------------------------------------------------------------------

    def def_bool[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> Bool
    ](mut self) -> ref[self] Self:
        """Install `__bool__` via the `nb_bool` slot.

        Called by `bool(obj)`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_bool
        """
        _install_inquiry[Self.self_type, method, _PySlotIndex.nb_bool](
            self._ptr
        )
        return self

    def def_bool[
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin]) thin -> Bool
    ](mut self) -> ref[self] Self:
        """Install `__bool__` via the `nb_bool` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_bool
        """
        _install_inquiry_nr[Self.self_type, method, _PySlotIndex.nb_bool](
            self._ptr
        )
        return self

    def def_bool[
        method: def(Self.self_type) thin raises -> Bool
    ](mut self) -> ref[self] Self:
        """Install `__bool__` via the `nb_bool` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_bool
        """
        _install_inquiry_val[Self.self_type, method, _PySlotIndex.nb_bool](
            self._ptr
        )
        return self

    # ------------------------------------------------------------------
    # Binary slots — C type: binaryfunc  def(PyObject *, PyObject *) -> PyObject *
    # Raise NotImplementedError() to return Py_NotImplemented.
    # ------------------------------------------------------------------

    def def_add[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__add__` via the `nb_add` slot.

        Called by `obj + other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_add
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_add](self._ptr)
        return self

    def def_and[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__and__` via the `nb_and` slot.

        Called by `obj & other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_and
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_and](self._ptr)
        return self

    def def_divmod[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__divmod__` via the `nb_divmod` slot.

        Called by `divmod(obj, other)`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_divmod
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_divmod](
            self._ptr
        )
        return self

    def def_floordiv[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__floordiv__` via the `nb_floor_divide` slot.

        Called by `obj // other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_floor_divide
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_floor_divide](
            self._ptr
        )
        return self

    def def_lshift[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__lshift__` via the `nb_lshift` slot.

        Called by `obj << other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_lshift
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_lshift](
            self._ptr
        )
        return self

    def def_matmul[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__matmul__` via the `nb_matrix_multiply` slot.

        Called by `obj @ other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_matrix_multiply
        """
        _install_binary[
            Self.self_type, method, _PySlotIndex.nb_matrix_multiply
        ](self._ptr)
        return self

    def def_mod[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__mod__` via the `nb_remainder` slot.

        Called by `obj % other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_remainder
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_remainder](
            self._ptr
        )
        return self

    def def_mul[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__mul__` via the `nb_multiply` slot.

        Called by `obj * other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_multiply
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_multiply](
            self._ptr
        )
        return self

    def def_or[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__or__` via the `nb_or` slot.

        Called by `obj | other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_or
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_or](self._ptr)
        return self

    def def_rshift[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__rshift__` via the `nb_rshift` slot.

        Called by `obj >> other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_rshift
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_rshift](
            self._ptr
        )
        return self

    def def_sub[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__sub__` via the `nb_subtract` slot.

        Called by `obj - other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_subtract
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_subtract](
            self._ptr
        )
        return self

    def def_truediv[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__truediv__` via the `nb_true_divide` slot.

        Called by `obj / other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_true_divide
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_true_divide](
            self._ptr
        )
        return self

    def def_xor[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__xor__` via the `nb_xor` slot.

        Called by `obj ^ other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_xor
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_xor](self._ptr)
        return self

    # In-place binary slots

    def def_iadd[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` via the `nb_inplace_add` slot.

        Called by `obj += other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_add
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_inplace_add](
            self._ptr
        )
        return self

    def def_iand[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__iand__` via the `nb_inplace_and` slot.

        Called by `obj &= other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_and
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_inplace_and](
            self._ptr
        )
        return self

    def def_ifloordiv[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ifloordiv__` via the `nb_inplace_floor_divide` slot.

        Called by `obj //= other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_floor_divide
        """
        _install_binary[
            Self.self_type, method, _PySlotIndex.nb_inplace_floor_divide
        ](self._ptr)
        return self

    def def_ilshift[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ilshift__` via the `nb_inplace_lshift` slot.

        Called by `obj <<= other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_lshift
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_inplace_lshift](
            self._ptr
        )
        return self

    def def_imatmul[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imatmul__` via the `nb_inplace_matrix_multiply` slot.

        Called by `obj @= other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_matrix_multiply
        """
        _install_binary[
            Self.self_type, method, _PySlotIndex.nb_inplace_matrix_multiply
        ](self._ptr)
        return self

    def def_imod[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imod__` via the `nb_inplace_remainder` slot.

        Called by `obj %= other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_remainder
        """
        _install_binary[
            Self.self_type, method, _PySlotIndex.nb_inplace_remainder
        ](self._ptr)
        return self

    def def_imul[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imul__` via the `nb_inplace_multiply` slot.

        Called by `obj *= other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_multiply
        """
        _install_binary[
            Self.self_type, method, _PySlotIndex.nb_inplace_multiply
        ](self._ptr)
        return self

    def def_ior[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ior__` via the `nb_inplace_or` slot.

        Called by `obj |= other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_or
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_inplace_or](
            self._ptr
        )
        return self

    def def_irshift[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__irshift__` via the `nb_inplace_rshift` slot.

        Called by `obj >>= other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_rshift
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_inplace_rshift](
            self._ptr
        )
        return self

    def def_isub[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__isub__` via the `nb_inplace_subtract` slot.

        Called by `obj -= other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_subtract
        """
        _install_binary[
            Self.self_type, method, _PySlotIndex.nb_inplace_subtract
        ](self._ptr)
        return self

    def def_itruediv[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__itruediv__` via the `nb_inplace_true_divide` slot.

        Called by `obj /= other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_true_divide
        """
        _install_binary[
            Self.self_type, method, _PySlotIndex.nb_inplace_true_divide
        ](self._ptr)
        return self

    def def_ixor[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ixor__` via the `nb_inplace_xor` slot.

        Called by `obj ^= other`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_xor
        """
        _install_binary[Self.self_type, method, _PySlotIndex.nb_inplace_xor](
            self._ptr
        )
        return self

    # Non-raising binary overloads

    def def_add[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__add__` via the `nb_add` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_add
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_add](
            self._ptr
        )
        return self

    def def_and[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__and__` via the `nb_and` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_and
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_and](
            self._ptr
        )
        return self

    def def_divmod[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__divmod__` via the `nb_divmod` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_divmod
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_divmod](
            self._ptr
        )
        return self

    def def_floordiv[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__floordiv__` via the `nb_floor_divide` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_floor_divide
        """
        _install_binary_nr[
            Self.self_type, method, _PySlotIndex.nb_floor_divide
        ](self._ptr)
        return self

    def def_lshift[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__lshift__` via the `nb_lshift` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_lshift
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_lshift](
            self._ptr
        )
        return self

    def def_matmul[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__matmul__` via the `nb_matrix_multiply` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_matrix_multiply
        """
        _install_binary_nr[
            Self.self_type, method, _PySlotIndex.nb_matrix_multiply
        ](self._ptr)
        return self

    def def_mod[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__mod__` via the `nb_remainder` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_remainder
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_remainder](
            self._ptr
        )
        return self

    def def_mul[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__mul__` via the `nb_multiply` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_multiply
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_multiply](
            self._ptr
        )
        return self

    def def_or[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__or__` via the `nb_or` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_or
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_or](
            self._ptr
        )
        return self

    def def_rshift[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__rshift__` via the `nb_rshift` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_rshift
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_rshift](
            self._ptr
        )
        return self

    def def_sub[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__sub__` via the `nb_subtract` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_subtract
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_subtract](
            self._ptr
        )
        return self

    def def_truediv[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__truediv__` via the `nb_true_divide` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_true_divide
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_true_divide](
            self._ptr
        )
        return self

    def def_xor[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__xor__` via the `nb_xor` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_xor
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_xor](
            self._ptr
        )
        return self

    def def_iadd[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` via the `nb_inplace_add` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_add
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_inplace_add](
            self._ptr
        )
        return self

    def def_iand[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__iand__` via the `nb_inplace_and` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_and
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_inplace_and](
            self._ptr
        )
        return self

    def def_ifloordiv[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ifloordiv__` via the `nb_inplace_floor_divide` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_floor_divide
        """
        _install_binary_nr[
            Self.self_type, method, _PySlotIndex.nb_inplace_floor_divide
        ](self._ptr)
        return self

    def def_ilshift[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ilshift__` via the `nb_inplace_lshift` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_lshift
        """
        _install_binary_nr[
            Self.self_type, method, _PySlotIndex.nb_inplace_lshift
        ](self._ptr)
        return self

    def def_imatmul[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imatmul__` via the `nb_inplace_matrix_multiply` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_matrix_multiply
        """
        _install_binary_nr[
            Self.self_type, method, _PySlotIndex.nb_inplace_matrix_multiply
        ](self._ptr)
        return self

    def def_imod[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imod__` via the `nb_inplace_remainder` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_remainder
        """
        _install_binary_nr[
            Self.self_type, method, _PySlotIndex.nb_inplace_remainder
        ](self._ptr)
        return self

    def def_imul[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imul__` via the `nb_inplace_multiply` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_multiply
        """
        _install_binary_nr[
            Self.self_type, method, _PySlotIndex.nb_inplace_multiply
        ](self._ptr)
        return self

    def def_ior[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ior__` via the `nb_inplace_or` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_or
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_inplace_or](
            self._ptr
        )
        return self

    def def_irshift[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__irshift__` via the `nb_inplace_rshift` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_rshift
        """
        _install_binary_nr[
            Self.self_type, method, _PySlotIndex.nb_inplace_rshift
        ](self._ptr)
        return self

    def def_isub[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__isub__` via the `nb_inplace_subtract` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_subtract
        """
        _install_binary_nr[
            Self.self_type, method, _PySlotIndex.nb_inplace_subtract
        ](self._ptr)
        return self

    def def_itruediv[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__itruediv__` via the `nb_inplace_true_divide` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_true_divide
        """
        _install_binary_nr[
            Self.self_type, method, _PySlotIndex.nb_inplace_true_divide
        ](self._ptr)
        return self

    def def_ixor[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ixor__` via the `nb_inplace_xor` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_xor
        """
        _install_binary_nr[Self.self_type, method, _PySlotIndex.nb_inplace_xor](
            self._ptr
        )
        return self

    # Value-receiver binary overloads

    def def_add[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__add__` via the `nb_add` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_add
        """
        _install_binary_val[Self.self_type, method, _PySlotIndex.nb_add](
            self._ptr
        )
        return self

    def def_and[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__and__` via the `nb_and` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_and
        """
        _install_binary_val[Self.self_type, method, _PySlotIndex.nb_and](
            self._ptr
        )
        return self

    def def_divmod[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__divmod__` via the `nb_divmod` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_divmod
        """
        _install_binary_val[Self.self_type, method, _PySlotIndex.nb_divmod](
            self._ptr
        )
        return self

    def def_floordiv[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__floordiv__` via the `nb_floor_divide` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_floor_divide
        """
        _install_binary_val[
            Self.self_type, method, _PySlotIndex.nb_floor_divide
        ](self._ptr)
        return self

    def def_lshift[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__lshift__` via the `nb_lshift` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_lshift
        """
        _install_binary_val[Self.self_type, method, _PySlotIndex.nb_lshift](
            self._ptr
        )
        return self

    def def_matmul[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__matmul__` via the `nb_matrix_multiply` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_matrix_multiply
        """
        _install_binary_val[
            Self.self_type, method, _PySlotIndex.nb_matrix_multiply
        ](self._ptr)
        return self

    def def_mod[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__mod__` via the `nb_remainder` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_remainder
        """
        _install_binary_val[Self.self_type, method, _PySlotIndex.nb_remainder](
            self._ptr
        )
        return self

    def def_mul[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__mul__` via the `nb_multiply` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_multiply
        """
        _install_binary_val[Self.self_type, method, _PySlotIndex.nb_multiply](
            self._ptr
        )
        return self

    def def_or[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__or__` via the `nb_or` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_or
        """
        _install_binary_val[Self.self_type, method, _PySlotIndex.nb_or](
            self._ptr
        )
        return self

    def def_rshift[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__rshift__` via the `nb_rshift` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_rshift
        """
        _install_binary_val[Self.self_type, method, _PySlotIndex.nb_rshift](
            self._ptr
        )
        return self

    def def_sub[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__sub__` via the `nb_subtract` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_subtract
        """
        _install_binary_val[Self.self_type, method, _PySlotIndex.nb_subtract](
            self._ptr
        )
        return self

    def def_truediv[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__truediv__` via the `nb_true_divide` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_true_divide
        """
        _install_binary_val[
            Self.self_type, method, _PySlotIndex.nb_true_divide
        ](self._ptr)
        return self

    def def_xor[
        method: def(Self.self_type, PythonObject) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__xor__` via the `nb_xor` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_xor
        """
        _install_binary_val[Self.self_type, method, _PySlotIndex.nb_xor](
            self._ptr
        )
        return self

    # ------------------------------------------------------------------
    # Ternary slots — C type: ternaryfunc  def(PyObject *, PyObject *, PyObject *) -> PyObject *
    # `mod` is None unless pow(base, exp, mod) was called.
    # Raise NotImplementedError() to return Py_NotImplemented.
    # ------------------------------------------------------------------

    def def_pow[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            PythonObject,
            PythonObject,
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__pow__` via the `nb_power` slot.

        Called by `obj ** exp`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_power
        """
        _install_ternary[Self.self_type, method, _PySlotIndex.nb_power](
            self._ptr
        )
        return self

    def def_ipow[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            PythonObject,
            PythonObject,
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ipow__` via the `nb_inplace_power` slot.

        Called by `obj **= exp`.
        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_power
        """
        _install_ternary[Self.self_type, method, _PySlotIndex.nb_inplace_power](
            self._ptr
        )
        return self

    # Non-raising ternary overloads

    def def_pow[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            PythonObject,
            PythonObject,
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__pow__` via the `nb_power` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_power
        """
        _install_ternary_nr[Self.self_type, method, _PySlotIndex.nb_power](
            self._ptr
        )
        return self

    def def_ipow[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            PythonObject,
            PythonObject,
        ) thin -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ipow__` via the `nb_inplace_power` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_power
        """
        _install_ternary_nr[
            Self.self_type, method, _PySlotIndex.nb_inplace_power
        ](self._ptr)
        return self

    def def_pow[
        method: def(
            Self.self_type, PythonObject, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__pow__` via the `nb_power` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_power
        """
        _install_ternary_val[Self.self_type, method, _PySlotIndex.nb_power](
            self._ptr
        )
        return self

    # Mut-receiver overloads

    def def_iadd[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` via the `nb_inplace_add` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_add
        """
        _install_binary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_add
        ](self._ptr)
        return self

    def def_iand[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__iand__` via the `nb_inplace_and` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_and
        """
        _install_binary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_and
        ](self._ptr)
        return self

    def def_ifloordiv[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ifloordiv__` via the `nb_inplace_floor_divide` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_floor_divide
        """
        _install_binary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_floor_divide
        ](self._ptr)
        return self

    def def_ilshift[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ilshift__` via the `nb_inplace_lshift` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_lshift
        """
        _install_binary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_lshift
        ](self._ptr)
        return self

    def def_imatmul[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imatmul__` via the `nb_inplace_matrix_multiply` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_matrix_multiply
        """
        _install_binary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_matrix_multiply
        ](self._ptr)
        return self

    def def_imod[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imod__` via the `nb_inplace_remainder` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_remainder
        """
        _install_binary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_remainder
        ](self._ptr)
        return self

    def def_imul[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__imul__` via the `nb_inplace_multiply` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_multiply
        """
        _install_binary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_multiply
        ](self._ptr)
        return self

    def def_ior[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ior__` via the `nb_inplace_or` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_or
        """
        _install_binary_mut[Self.self_type, method, _PySlotIndex.nb_inplace_or](
            self._ptr
        )
        return self

    def def_irshift[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__irshift__` via the `nb_inplace_rshift` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_rshift
        """
        _install_binary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_rshift
        ](self._ptr)
        return self

    def def_isub[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__isub__` via the `nb_inplace_subtract` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_subtract
        """
        _install_binary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_subtract
        ](self._ptr)
        return self

    def def_itruediv[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__itruediv__` via the `nb_inplace_true_divide` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_true_divide
        """
        _install_binary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_true_divide
        ](self._ptr)
        return self

    def def_ixor[
        method: def(
            mut Self.self_type, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ixor__` via the `nb_inplace_xor` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_xor
        """
        _install_binary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_xor
        ](self._ptr)
        return self

    def def_ipow[
        method: def(
            mut Self.self_type, PythonObject, PythonObject
        ) thin raises -> PythonObject
    ](mut self) -> ref[self] Self:
        """Install `__ipow__` via the `nb_inplace_power` slot (mut-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_power
        """
        _install_ternary_mut[
            Self.self_type, method, _PySlotIndex.nb_inplace_power
        ](self._ptr)
        return self

    # ConvertibleToPython return overloads

    def def_abs[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__abs__` via the `nb_absolute` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_absolute
        """
        _install_unary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_absolute
        ](self._ptr)
        return self

    def def_abs[
        R: _CPython,
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin]) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__abs__` via the `nb_absolute` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_absolute
        """
        _install_unary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_absolute
        ](self._ptr)
        return self

    def def_abs[
        R: _CPython,
        method: def(Self.self_type) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__abs__` via the `nb_absolute` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_absolute
        """
        _install_unary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_absolute
        ](self._ptr)
        return self

    def def_float[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__float__` via the `nb_float` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_float
        """
        _install_unary_conv_r[Self.self_type, R, method, _PySlotIndex.nb_float](
            self._ptr
        )
        return self

    def def_float[
        R: _CPython,
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin]) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__float__` via the `nb_float` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_float
        """
        _install_unary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_float
        ](self._ptr)
        return self

    def def_float[
        R: _CPython,
        method: def(Self.self_type) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__float__` via the `nb_float` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_float
        """
        _install_unary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_float
        ](self._ptr)
        return self

    def def_index[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__index__` via the `nb_index` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_index
        """
        _install_unary_conv_r[Self.self_type, R, method, _PySlotIndex.nb_index](
            self._ptr
        )
        return self

    def def_index[
        R: _CPython,
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin]) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__index__` via the `nb_index` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_index
        """
        _install_unary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_index
        ](self._ptr)
        return self

    def def_index[
        R: _CPython,
        method: def(Self.self_type) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__index__` via the `nb_index` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_index
        """
        _install_unary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_index
        ](self._ptr)
        return self

    def def_int[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__int__` via the `nb_int` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_int
        """
        _install_unary_conv_r[Self.self_type, R, method, _PySlotIndex.nb_int](
            self._ptr
        )
        return self

    def def_int[
        R: _CPython,
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin]) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__int__` via the `nb_int` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_int
        """
        _install_unary_conv_nr[Self.self_type, R, method, _PySlotIndex.nb_int](
            self._ptr
        )
        return self

    def def_int[
        R: _CPython,
        method: def(Self.self_type) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__int__` via the `nb_int` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_int
        """
        _install_unary_conv_val[Self.self_type, R, method, _PySlotIndex.nb_int](
            self._ptr
        )
        return self

    def def_invert[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__invert__` via the `nb_invert` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_invert
        """
        _install_unary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_invert
        ](self._ptr)
        return self

    def def_invert[
        R: _CPython,
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin]) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__invert__` via the `nb_invert` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_invert
        """
        _install_unary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_invert
        ](self._ptr)
        return self

    def def_invert[
        R: _CPython,
        method: def(Self.self_type) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__invert__` via the `nb_invert` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_invert
        """
        _install_unary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_invert
        ](self._ptr)
        return self

    def def_neg[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__neg__` via the `nb_negative` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_negative
        """
        _install_unary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_negative
        ](self._ptr)
        return self

    def def_neg[
        R: _CPython,
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin]) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__neg__` via the `nb_negative` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_negative
        """
        _install_unary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_negative
        ](self._ptr)
        return self

    def def_neg[
        R: _CPython,
        method: def(Self.self_type) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__neg__` via the `nb_negative` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_negative
        """
        _install_unary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_negative
        ](self._ptr)
        return self

    def def_pos[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin]
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__pos__` via the `nb_positive` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_positive
        """
        _install_unary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_positive
        ](self._ptr)
        return self

    def def_pos[
        R: _CPython,
        method: def(UnsafePointer[Self.self_type, MutAnyOrigin]) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__pos__` via the `nb_positive` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_positive
        """
        _install_unary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_positive
        ](self._ptr)
        return self

    def def_pos[
        R: _CPython,
        method: def(Self.self_type) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__pos__` via the `nb_positive` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_positive
        """
        _install_unary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_positive
        ](self._ptr)
        return self

    def def_add[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__add__` via the `nb_add` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_add
        """
        _install_binary_conv_r[Self.self_type, R, method, _PySlotIndex.nb_add](
            self._ptr
        )
        return self

    def def_add[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__add__` via the `nb_add` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_add
        """
        _install_binary_conv_nr[Self.self_type, R, method, _PySlotIndex.nb_add](
            self._ptr
        )
        return self

    def def_add[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__add__` via the `nb_add` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_add
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_add
        ](self._ptr)
        return self

    def def_and[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__and__` via the `nb_and` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_and
        """
        _install_binary_conv_r[Self.self_type, R, method, _PySlotIndex.nb_and](
            self._ptr
        )
        return self

    def def_and[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__and__` via the `nb_and` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_and
        """
        _install_binary_conv_nr[Self.self_type, R, method, _PySlotIndex.nb_and](
            self._ptr
        )
        return self

    def def_and[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__and__` via the `nb_and` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_and
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_and
        ](self._ptr)
        return self

    def def_divmod[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__divmod__` via the `nb_divmod` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_divmod
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_divmod
        ](self._ptr)
        return self

    def def_divmod[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__divmod__` via the `nb_divmod` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_divmod
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_divmod
        ](self._ptr)
        return self

    def def_divmod[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__divmod__` via the `nb_divmod` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_divmod
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_divmod
        ](self._ptr)
        return self

    def def_floordiv[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__floordiv__` via the `nb_floor_divide` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_floor_divide
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_floor_divide
        ](self._ptr)
        return self

    def def_floordiv[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__floordiv__` via the `nb_floor_divide` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_floor_divide
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_floor_divide
        ](self._ptr)
        return self

    def def_floordiv[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__floordiv__` via the `nb_floor_divide` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_floor_divide
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_floor_divide
        ](self._ptr)
        return self

    def def_lshift[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__lshift__` via the `nb_lshift` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_lshift
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_lshift
        ](self._ptr)
        return self

    def def_lshift[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__lshift__` via the `nb_lshift` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_lshift
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_lshift
        ](self._ptr)
        return self

    def def_lshift[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__lshift__` via the `nb_lshift` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_lshift
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_lshift
        ](self._ptr)
        return self

    def def_matmul[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__matmul__` via the `nb_matrix_multiply` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_matrix_multiply
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_matrix_multiply
        ](self._ptr)
        return self

    def def_matmul[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__matmul__` via the `nb_matrix_multiply` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_matrix_multiply
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_matrix_multiply
        ](self._ptr)
        return self

    def def_matmul[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__matmul__` via the `nb_matrix_multiply` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_matrix_multiply
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_matrix_multiply
        ](self._ptr)
        return self

    def def_mod[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__mod__` via the `nb_remainder` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_remainder
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_remainder
        ](self._ptr)
        return self

    def def_mod[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__mod__` via the `nb_remainder` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_remainder
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_remainder
        ](self._ptr)
        return self

    def def_mod[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__mod__` via the `nb_remainder` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_remainder
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_remainder
        ](self._ptr)
        return self

    def def_mul[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__mul__` via the `nb_multiply` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_multiply
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_multiply
        ](self._ptr)
        return self

    def def_mul[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__mul__` via the `nb_multiply` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_multiply
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_multiply
        ](self._ptr)
        return self

    def def_mul[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__mul__` via the `nb_multiply` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_multiply
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_multiply
        ](self._ptr)
        return self

    def def_or[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__or__` via the `nb_or` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_or
        """
        _install_binary_conv_r[Self.self_type, R, method, _PySlotIndex.nb_or](
            self._ptr
        )
        return self

    def def_or[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__or__` via the `nb_or` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_or
        """
        _install_binary_conv_nr[Self.self_type, R, method, _PySlotIndex.nb_or](
            self._ptr
        )
        return self

    def def_or[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__or__` via the `nb_or` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_or
        """
        _install_binary_conv_val[Self.self_type, R, method, _PySlotIndex.nb_or](
            self._ptr
        )
        return self

    def def_rshift[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__rshift__` via the `nb_rshift` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_rshift
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_rshift
        ](self._ptr)
        return self

    def def_rshift[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__rshift__` via the `nb_rshift` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_rshift
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_rshift
        ](self._ptr)
        return self

    def def_rshift[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__rshift__` via the `nb_rshift` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_rshift
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_rshift
        ](self._ptr)
        return self

    def def_sub[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__sub__` via the `nb_subtract` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_subtract
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_subtract
        ](self._ptr)
        return self

    def def_sub[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__sub__` via the `nb_subtract` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_subtract
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_subtract
        ](self._ptr)
        return self

    def def_sub[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__sub__` via the `nb_subtract` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_subtract
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_subtract
        ](self._ptr)
        return self

    def def_truediv[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__truediv__` via the `nb_true_divide` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_true_divide
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_true_divide
        ](self._ptr)
        return self

    def def_truediv[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__truediv__` via the `nb_true_divide` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_true_divide
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_true_divide
        ](self._ptr)
        return self

    def def_truediv[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__truediv__` via the `nb_true_divide` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_true_divide
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_true_divide
        ](self._ptr)
        return self

    def def_xor[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__xor__` via the `nb_xor` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_xor
        """
        _install_binary_conv_r[Self.self_type, R, method, _PySlotIndex.nb_xor](
            self._ptr
        )
        return self

    def def_xor[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__xor__` via the `nb_xor` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_xor
        """
        _install_binary_conv_nr[Self.self_type, R, method, _PySlotIndex.nb_xor](
            self._ptr
        )
        return self

    def def_xor[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__xor__` via the `nb_xor` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_xor
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_xor
        ](self._ptr)
        return self

    def def_iadd[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` via the `nb_inplace_add` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_add
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_add
        ](self._ptr)
        return self

    def def_iadd[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` via the `nb_inplace_add` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_add
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_add
        ](self._ptr)
        return self

    def def_iadd[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__iadd__` via the `nb_inplace_add` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_add
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_add
        ](self._ptr)
        return self

    def def_iand[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__iand__` via the `nb_inplace_and` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_and
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_and
        ](self._ptr)
        return self

    def def_iand[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__iand__` via the `nb_inplace_and` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_and
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_and
        ](self._ptr)
        return self

    def def_iand[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__iand__` via the `nb_inplace_and` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_and
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_and
        ](self._ptr)
        return self

    def def_ifloordiv[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ifloordiv__` via the `nb_inplace_floor_divide` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_floor_divide
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_floor_divide
        ](self._ptr)
        return self

    def def_ifloordiv[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ifloordiv__` via the `nb_inplace_floor_divide` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_floor_divide
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_floor_divide
        ](self._ptr)
        return self

    def def_ifloordiv[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ifloordiv__` via the `nb_inplace_floor_divide` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_floor_divide
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_floor_divide
        ](self._ptr)
        return self

    def def_ilshift[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ilshift__` via the `nb_inplace_lshift` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_lshift
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_lshift
        ](self._ptr)
        return self

    def def_ilshift[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ilshift__` via the `nb_inplace_lshift` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_lshift
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_lshift
        ](self._ptr)
        return self

    def def_ilshift[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ilshift__` via the `nb_inplace_lshift` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_lshift
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_lshift
        ](self._ptr)
        return self

    def def_imatmul[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__imatmul__` via the `nb_inplace_matrix_multiply` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_matrix_multiply
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_matrix_multiply
        ](self._ptr)
        return self

    def def_imatmul[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__imatmul__` via the `nb_inplace_matrix_multiply` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_matrix_multiply
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_matrix_multiply
        ](self._ptr)
        return self

    def def_imatmul[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__imatmul__` via the `nb_inplace_matrix_multiply` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_matrix_multiply
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_matrix_multiply
        ](self._ptr)
        return self

    def def_imod[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__imod__` via the `nb_inplace_remainder` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_remainder
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_remainder
        ](self._ptr)
        return self

    def def_imod[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__imod__` via the `nb_inplace_remainder` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_remainder
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_remainder
        ](self._ptr)
        return self

    def def_imod[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__imod__` via the `nb_inplace_remainder` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_remainder
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_remainder
        ](self._ptr)
        return self

    def def_imul[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__imul__` via the `nb_inplace_multiply` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_multiply
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_multiply
        ](self._ptr)
        return self

    def def_imul[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__imul__` via the `nb_inplace_multiply` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_multiply
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_multiply
        ](self._ptr)
        return self

    def def_imul[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__imul__` via the `nb_inplace_multiply` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_multiply
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_multiply
        ](self._ptr)
        return self

    def def_ior[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ior__` via the `nb_inplace_or` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_or
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_or
        ](self._ptr)
        return self

    def def_ior[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ior__` via the `nb_inplace_or` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_or
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_or
        ](self._ptr)
        return self

    def def_ior[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ior__` via the `nb_inplace_or` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_or
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_or
        ](self._ptr)
        return self

    def def_irshift[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__irshift__` via the `nb_inplace_rshift` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_rshift
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_rshift
        ](self._ptr)
        return self

    def def_irshift[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__irshift__` via the `nb_inplace_rshift` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_rshift
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_rshift
        ](self._ptr)
        return self

    def def_irshift[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__irshift__` via the `nb_inplace_rshift` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_rshift
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_rshift
        ](self._ptr)
        return self

    def def_isub[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__isub__` via the `nb_inplace_subtract` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_subtract
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_subtract
        ](self._ptr)
        return self

    def def_isub[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__isub__` via the `nb_inplace_subtract` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_subtract
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_subtract
        ](self._ptr)
        return self

    def def_isub[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__isub__` via the `nb_inplace_subtract` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_subtract
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_subtract
        ](self._ptr)
        return self

    def def_itruediv[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__itruediv__` via the `nb_inplace_true_divide` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_true_divide
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_true_divide
        ](self._ptr)
        return self

    def def_itruediv[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__itruediv__` via the `nb_inplace_true_divide` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_true_divide
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_true_divide
        ](self._ptr)
        return self

    def def_itruediv[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__itruediv__` via the `nb_inplace_true_divide` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_true_divide
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_true_divide
        ](self._ptr)
        return self

    def def_ixor[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ixor__` via the `nb_inplace_xor` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_xor
        """
        _install_binary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_xor
        ](self._ptr)
        return self

    def def_ixor[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ixor__` via the `nb_inplace_xor` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_xor
        """
        _install_binary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_xor
        ](self._ptr)
        return self

    def def_ixor[
        R: _CPython,
        method: def(Self.self_type, PythonObject) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ixor__` via the `nb_inplace_xor` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_xor
        """
        _install_binary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_xor
        ](self._ptr)
        return self

    def def_pow[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            PythonObject,
            PythonObject,
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__pow__` via the `nb_power` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_power
        """
        _install_ternary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_power
        ](self._ptr)
        return self

    def def_pow[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            PythonObject,
            PythonObject,
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__pow__` via the `nb_power` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_power
        """
        _install_ternary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_power
        ](self._ptr)
        return self

    def def_pow[
        R: _CPython,
        method: def(
            Self.self_type, PythonObject, PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__pow__` via the `nb_power` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_power
        """
        _install_ternary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_power
        ](self._ptr)
        return self

    def def_ipow[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            PythonObject,
            PythonObject,
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ipow__` via the `nb_inplace_power` slot (ConvertibleToPython return overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_power
        """
        _install_ternary_conv_r[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_power
        ](self._ptr)
        return self

    def def_ipow[
        R: _CPython,
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin],
            PythonObject,
            PythonObject,
        ) thin -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ipow__` via the `nb_inplace_power` slot (ConvertibleToPython return, non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_power
        """
        _install_ternary_conv_nr[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_power
        ](self._ptr)
        return self

    def def_ipow[
        R: _CPython,
        method: def(
            Self.self_type, PythonObject, PythonObject
        ) thin raises -> R,
    ](mut self) -> ref[self] Self:
        """Install `__ipow__` via the `nb_inplace_power` slot (ConvertibleToPython return, value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_power
        """
        _install_ternary_conv_val[
            Self.self_type, R, method, _PySlotIndex.nb_inplace_power
        ](self._ptr)
        return self

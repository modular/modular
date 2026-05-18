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

"""Implements `PySlotError` and `RichCompareOps`.

Provides the Mojo-native error type used by Python type-slot handlers and
the rich-comparison op-code constants consumed by `tp_richcompare`
handlers.
"""


struct RichCompareOps:
    """Flags used by the tp_richcompare function.

    Pass the `op` argument from your rich compare handler to these constants
    to determine which comparison is being requested.

    References:
    - https://github.com/python/cpython/blob/main/Include/object.h#L721
    - https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_richcompare
    """

    comptime Py_LT = 0
    """Less-than comparison op (`<`)."""
    comptime Py_LE = 1
    """Less-than-or-equal comparison op (`<=`)."""
    comptime Py_EQ = 2
    """Equality comparison op (`==`)."""
    comptime Py_NE = 3
    """Inequality comparison op (`!=`)."""
    comptime Py_GT = 4
    """Greater-than comparison op (`>`)."""
    comptime Py_GE = 5
    """Greater-than-or-equal comparison op (`>=`)."""


@fieldwise_init
struct PySlotError(Copyable, Movable, Writable):
    """Mojo-native error type for Python type-slot handlers.

    Raise a variant of this type from a Mojo function bound to a CPython type
    slot; adapter wrappers that take `raises PySlotError` translate it into
    the corresponding Python exception (or into `NotImplemented` for the
    `not_implemented` variant).

    Construct via the staticmethod factories rather than the synthesized
    fieldwise initializer:

    ```mojo
    raise PySlotError.index_error("index out of range")
    raise PySlotError.not_implemented()
    ```
    """

    var _variant: Int
    """Variant tag; one of the `_*` codes below."""

    var msg: String
    """Human-readable message; ignored for `not_implemented`."""

    # Variant codes — kept private; users go through the factories.
    comptime _NOT_IMPLEMENTED = 0
    comptime _INDEX_ERROR = 1
    comptime _TYPE_ERROR = 2
    comptime _VALUE_ERROR = 3
    comptime _KEY_ERROR = 4
    comptime _ATTRIBUTE_ERROR = 5
    comptime _OVERFLOW_ERROR = 6
    comptime _RUNTIME_ERROR = 7

    @staticmethod
    def not_implemented() -> Self:
        """Signal Python's `NotImplemented` to the wrapper.

        For binary/ternary/rich-compare slots, this causes the adapter to
        return `Py_NotImplemented`, prompting Python to try the reflected
        operation on the other operand.

        Returns:
            A `PySlotError` with the `not_implemented` variant.
        """
        return Self(_variant=Self._NOT_IMPLEMENTED, msg=String())

    @staticmethod
    def index_error(var msg: String) -> Self:
        """Map to Python's `IndexError`.

        Args:
            msg: Human-readable error message.

        Returns:
            A `PySlotError` configured as an `IndexError`.
        """
        return Self(_variant=Self._INDEX_ERROR, msg=msg^)

    @staticmethod
    def type_error(var msg: String) -> Self:
        """Map to Python's `TypeError`.

        Args:
            msg: Human-readable error message.

        Returns:
            A `PySlotError` configured as a `TypeError`.
        """
        return Self(_variant=Self._TYPE_ERROR, msg=msg^)

    @staticmethod
    def value_error(var msg: String) -> Self:
        """Map to Python's `ValueError`.

        Args:
            msg: Human-readable error message.

        Returns:
            A `PySlotError` configured as a `ValueError`.
        """
        return Self(_variant=Self._VALUE_ERROR, msg=msg^)

    @staticmethod
    def key_error(var msg: String) -> Self:
        """Map to Python's `KeyError`.

        Args:
            msg: Human-readable error message.

        Returns:
            A `PySlotError` configured as a `KeyError`.
        """
        return Self(_variant=Self._KEY_ERROR, msg=msg^)

    @staticmethod
    def attribute_error(var msg: String) -> Self:
        """Map to Python's `AttributeError`.

        Args:
            msg: Human-readable error message.

        Returns:
            A `PySlotError` configured as an `AttributeError`.
        """
        return Self(_variant=Self._ATTRIBUTE_ERROR, msg=msg^)

    @staticmethod
    def overflow_error(var msg: String) -> Self:
        """Map to Python's `OverflowError`.

        Args:
            msg: Human-readable error message.

        Returns:
            A `PySlotError` configured as an `OverflowError`.
        """
        return Self(_variant=Self._OVERFLOW_ERROR, msg=msg^)

    @staticmethod
    def runtime_error(var msg: String) -> Self:
        """Map to Python's `RuntimeError`.

        Args:
            msg: Human-readable error message.

        Returns:
            A `PySlotError` configured as a `RuntimeError`.
        """
        return Self(_variant=Self._RUNTIME_ERROR, msg=msg^)

    def write_to(self, mut writer: Some[Writer]):
        """Write a textual representation of this error to `writer`.

        Args:
            writer: The destination writer.
        """
        if self._variant == Self._NOT_IMPLEMENTED:
            writer.write("PySlotError.not_implemented")
        else:
            writer.write(self.msg)

    def pyexc_global_name(self) -> StaticString:
        """Return the CPython `PyExc_*` symbol name for this variant.

        Used by adapter wrappers to fetch the matching Python exception
        global via `cpython.get_error_global(...)`. The `not_implemented`
        variant has no Python-exception counterpart and maps to
        `PyExc_RuntimeError` — callers that want the `Py_NotImplemented`
        singleton must check `_variant == _NOT_IMPLEMENTED` first.

        Returns:
            The name of the matching `PyExc_*` global as a `StaticString`.
        """
        if self._variant == Self._INDEX_ERROR:
            return "PyExc_IndexError"
        elif self._variant == Self._TYPE_ERROR:
            return "PyExc_TypeError"
        elif self._variant == Self._VALUE_ERROR:
            return "PyExc_ValueError"
        elif self._variant == Self._KEY_ERROR:
            return "PyExc_KeyError"
        elif self._variant == Self._ATTRIBUTE_ERROR:
            return "PyExc_AttributeError"
        elif self._variant == Self._OVERFLOW_ERROR:
            return "PyExc_OverflowError"
        else:
            # _RUNTIME_ERROR or _NOT_IMPLEMENTED
            return "PyExc_RuntimeError"

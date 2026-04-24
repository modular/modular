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
    _install_richcompare,
    _lift_obj_int_to_bool,
    _lift_val_obj_int_to_bool,
)


struct TypeProtocolBuilder[self_type: ImplicitlyDestructible]:
    """Wraps a `PythonTypeBuilder` reference and installs CPython type protocol slots.

    `TypeProtocolBuilder` holds a pointer to a `PythonTypeBuilder` that is
    owned by the enclosing `PythonModuleBuilder`.  The caller must ensure the
    module builder (and its type_builders list) outlives this object, which is
    naturally satisfied when both are used within the same `PyInit_*` function.

    Usage:
        ```mojo
        ref tb = b.add_type[MyStruct]("MyStruct")
            .def_init_defaultable[MyStruct]()
            .def_staticmethod[MyStruct.new]("new")
        TypeProtocolBuilder[MyStruct](tb).def_richcompare[MyStruct.rich_compare]()
        MappingProtocolBuilder[MyStruct](tb)
            .def_len[MyStruct.py__len__]()
            .def_getitem[MyStruct.py__getitem__]()
            .def_setitem[MyStruct.py__setitem__]()
        NumberProtocolBuilder[MyStruct](tb).def_neg[MyStruct.py__neg__]()
        ```
    """

    # Unsafe pointer into the module builder's type_builders list.
    # The pointed-to builder must outlive this TypeProtocolBuilder.
    var _ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]

    def __init__(out self, mut inner: PythonTypeBuilder):
        var ptr = UnsafePointer(to=inner)
        self._ptr = ptr

    # ------------------------------------------------------------------
    # Type Protocol — tp_richcompare (__lt__, __eq__, etc.)
    # ------------------------------------------------------------------

    def def_richcompare[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject, Int
        ) thin raises -> Bool
    ](mut self) -> ref[self] Self:
        """Install rich comparison via the `tp_richcompare` slot.

        Called by `obj < other`, `obj == other`, etc.

        Raise `NotImplementedError()` from `method` to return
        `Py_NotImplemented` to Python (triggering the reflected operation).

        Parameters:
            method: Static method with signature
                `def(self_ptr: UnsafePointer[T, MutAnyOrigin], other: PythonObject, op: Int) raises -> Bool`
                where `op` is one of `RichCompareOps.Py_LT` … `Py_GE`.

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_richcompare
        """
        _install_richcompare[Self.self_type, method](self._ptr)
        return self

    def def_richcompare[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], PythonObject, Int
        ) thin -> Bool
    ](mut self) -> ref[self] Self:
        """Install rich comparison via the `tp_richcompare` slot (non-raising overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_richcompare
        """
        _install_richcompare[
            Self.self_type, _lift_obj_int_to_bool[Self.self_type, method]
        ](self._ptr)
        return self

    def def_richcompare[
        method: def(Self.self_type, PythonObject, Int) thin raises -> Bool
    ](mut self) -> ref[self] Self:
        """Install rich comparison via the `tp_richcompare` slot (value-receiver overload).

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_richcompare
        """
        _install_richcompare[
            Self.self_type, _lift_val_obj_int_to_bool[Self.self_type, method]
        ](self._ptr)
        return self

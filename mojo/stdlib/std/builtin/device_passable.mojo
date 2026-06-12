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
"""Implements the `DevicePassable` trait for types transferable to accelerator devices."""

from std.os import abort
from std.sys import size_of
from std.sys.info import _TargetType
from std.sys.intrinsics import _type_is_eq
from std.builtin.rebind import downcast, trait_downcast
from std.gpu.host.device_context import DeviceBuffer, DevicePointer
from std.reflection import reflect


trait DevicePassable:
    """This trait marks types as passable to accelerator devices."""

    comptime device_type: AnyType
    """Indicate the type being used on accelerator devices."""

    @staticmethod
    def _is_convertible_to_device_type[SrcT: AnyType]() -> Bool:
        comptime if not _type_is_eq[Self, Self.device_type]() and conforms_to(
            Self.device_type, DevicePassable
        ):
            return downcast[
                Self.device_type, DevicePassable
            ]._is_convertible_to_device_type[SrcT]()
        else:
            return _type_is_eq[SrcT, Self.device_type]()

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """
        Convert the host type object to a device_type and store it at the
        target address.

        NOTE: This should only be called by `DeviceContext` during invocation
        of accelerator kernels.
        """
        ...

    @staticmethod
    def get_type_name() -> String:
        """
        Gets the name of the host type (the one implementing this trait).
        For example, Int would return "Int", DeviceBuffer[DType.float32] would
        return "DeviceBuffer[DType.float32]". This is used for error messages
        when passing types to the device.
        TODO: This method will be retired soon when better kernel call error
        messages arrive.

        Returns:
            The host type's name.
        """
        ...


def _contains_device_passable_field[T: AnyType]() -> Bool:
    """Returns whether `T` is a struct transitively containing a
    `DevicePassable` field.

    Used by `encode_fields` to choose between recursing into a composite field
    and bit-copying it wholesale.

    Parameters:
        T: The type to inspect.

    Returns:
        `True` if a `DevicePassable` field exists anywhere in `T`'s field tree,
        `False` otherwise (including for non-struct types).
    """
    comptime if not reflect[T].is_struct():
        return False
    comptime r = reflect[T]
    comptime field_types = r.field_types()
    comptime for i in range(r.field_count()):
        comptime FieldType = field_types[i]
        comptime if conforms_to(FieldType, DevicePassable):
            return True
        elif _contains_device_passable_field[FieldType]():
            return True
    return False


trait DeviceTypeEncoder:
    """This trait marks types as capable of encoding device types.

    Used in `DevicePassable._to_device_type()` to enable target specific
    encoding of device types at the boundary where functions are enqueued for
    execution on an accelerator device.
    """

    @staticmethod
    def target() -> _TargetType:
        """Returns the target architecture this encoder is encoding for.

        Layout-sensitive queries (`size_of`, `align_of`,
        `reflect[T].field_offset[...]`) inside `encode` /
        `encode_device_ptr` and inside `DevicePassable._to_device_type`
        implementations should pass `target=Self.target()` so that the
        device's data layout — not the host's — is consulted.

        Returns:
            The target architecture this encoder is encoding for.
        """
        ...

    def encode[
        ValueType: AnyType
    ](mut self, value: ValueType, dst: MutOpaquePointer[_]):
        """Encodes `value` into `dst` by copying its bits.

        This is the default device encoding for a type whose
        `DevicePassable.device_type` is `Self`.

        Parameters:
            ValueType: The type of `value`, see constraints.

        Args:
            value: The variable to encode.
            dst: The opaque destination pointer to encode into. Must point
                to uninitialized storage at least
                `size_of[ValueType, target=Self.target()]()` bytes wide.

        Constraints:
            - `ValueType` must conform to `DevicePassable` or `RegisterPassable`.
            - `ValueType` must conform to
              `ImplicitlyCopyable & ImplicitlyDestructible` or
              `Copyable & ImplicitlyDestructible`.
            - If `ValueType` is `DevicePassable`, it must be its own leaf
              `device_type`
              (`ValueType._is_convertible_to_device_type[ValueType]()`), since a
              bit-copy only encodes an identity mapping correctly.
        """
        comptime assert conforms_to(ValueType, DevicePassable) or conforms_to(
            ValueType, RegisterPassable
        ), String(
            t"encode: ValueType '{reflect[ValueType].base_name()}' must conform"
            t" to DevicePassable or RegisterPassable"
        )

        comptime if conforms_to(ValueType, DevicePassable):
            comptime DPType = downcast[ValueType, DevicePassable]
            comptime assert DPType._is_convertible_to_device_type[
                ValueType
            ](), String(
                t"encode: ValueType '{reflect[ValueType].base_name()}' being"
                t" DevicePassable must be convertible to it's leaf device_type"
            )
            comptime assert (
                size_of[ValueType]()
                == size_of[ValueType, target=Self.target()]()
            ), String(
                t"encode: ValueType '{reflect[ValueType].base_name()}' mismatch"
                t" between host-type and device-type size"
            )

        comptime if conforms_to(
            ValueType, ImplicitlyCopyable & ImplicitlyDestructible
        ):
            comptime T = downcast[
                ValueType, ImplicitlyCopyable & ImplicitlyDestructible
            ]
            dst.bitcast[T]()[] = rebind[T](value)
        elif conforms_to(ValueType, Copyable & ImplicitlyDestructible):
            comptime T = downcast[ValueType, Copyable & ImplicitlyDestructible]
            dst.bitcast[T]()[] = rebind[T](value).copy()
        else:
            comptime assert False, String(
                t"encode: ValueType '{reflect[ValueType].base_name()}' must"
                t" conform to ImplicitlyCopyable & ImplicitlyDestructible or"
                t" Copyable & ImplicitlyDestructible"
            )

    def encode_fields[
        StructType: AnyType,
    ](mut self, value: StructType, dst: MutOpaquePointer[_]):
        """Encodes each field of `value` into `dst` at its device offset.

        For each field of `StructType`:

        - If it conforms to `DevicePassable`, dispatch to its own
          `_to_device_type()`.
        - Otherwise, if it is a composite transitively containing a
          `DevicePassable` member, recurse into `encode_fields`.
        - Otherwise, delegate to `encode` (a bit-copy for a register-passable
          field; any other type is rejected there at compile time).

        This is the building block composite types use to encode their members,
        including compiler-synthesized unified-closure wrappers whose
        closure-state type does not itself conform to `DevicePassable`. Field
        offsets use the encoder's target data layout (`Self.target()`) rather
        than the host's, so each field lands at the offset the device expects.

        Parameters:
            StructType: The composite host-side type whose fields are being
                encoded.

        Args:
            value: The composite host-side value to encode.
            dst: The opaque destination pointer that receives the
                encoded fields.

        Constraints:
            - `StructType` must conform to `RegisterPassable` and be a Mojo
              struct type.
            - Every field must either conform to `DevicePassable`, be a
              composite transitively containing a `DevicePassable` member,
              conform to `ImplicitlyCopyable & ImplicitlyDestructible`, or
              conform to `Copyable & ImplicitlyDestructible`.
        """
        # NOTE: The trait system does not enforce viral conformance to
        # DevicePassable if a RegisterPassable struct field is DevicePassable.
        # Instead we allow non-DevicePassable types to be encoded when they are
        # RegisterPassable.
        comptime assert conforms_to(StructType, RegisterPassable), String(
            t"encode_fields: StructType '{reflect[StructType].base_name()}'"
            t" must conform to RegisterPassable"
        )
        comptime assert reflect[StructType].is_struct(), String(
            t"encode_fields: StructType '{reflect[StructType].base_name()}'"
            t" must be struct"
        )
        comptime r = reflect[StructType]
        comptime field_types = r.field_types()

        # FIXME: MOCO-4018 We don't properly support field reflection on these
        # types yet.
        if r.base_name() in ("InlineArray", "StaticTuple", "_RegTuple"):
            abort(
                t"encode_fields: StructType '{r.base_name()}' is not currently"
                t" supported"
            )

        comptime for i in range(r.field_count()):
            comptime FieldType = field_types[i]
            # Offset in the device data layout, not the host's.
            comptime offset = r.field_offset[index=i, target=Self.target()]()
            ref field = r.field_ref[i](value)
            var sub = (dst.bitcast[UInt8]() + offset).bitcast[NoneType]()

            comptime if conforms_to(FieldType, DevicePassable):
                trait_downcast[DevicePassable](field)._to_device_type(self, sub)
            elif _contains_device_passable_field[FieldType]():
                # Recurse so the nested `DevicePassable` member runs its own
                # `_to_device_type` instead of being byte-copied.
                self.encode_fields[FieldType](field, sub)
            else:
                # Register-passable field with no `DevicePassable` member:
                # bit-copy. `encode` rejects any other type at compile time.
                self.encode(field, sub)

    def encode_device_ptr(
        mut self, value: DevicePointer, dst: MutOpaquePointer[_]
    ):
        """Encodes a `DevicePointer` into `dst`.

        Args:
            value: The `DevicePointer` instance to encode into `dst`.
            dst: The opaque destination pointer to encode into.
        """
        ...

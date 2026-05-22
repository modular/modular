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
from std.sys.intrinsics import _type_is_eq
from std.builtin.rebind import downcast
from std.gpu.host.device_context import DeviceBuffer, DevicePointer


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


trait DeviceTypeEncoder:
    """This trait marks types as capable of encoding device types.

    Used in `DevicePassable._to_device_type()` to enable target specific
    encoding of device types at the boundary where functions are enqueued for
    execution on an accelerator device.
    """

    def encode[
        ValueType: DevicePassable
    ](mut self, value: ValueType, target: MutOpaquePointer[_]):
        """Encodes `value` into `target` as its device-side representation.

        This is the default implementation for types whose
        `DevicePassable.device_type` is `Self`, it writes the value's bits into
        the storage pointed to by `target`.

        Parameters:
            ValueType: The type of `value`, see constraints.

        Args:
            value: The variable to encode.
            target: The opaque destination pointer to encode into. Must point
                to uninitialized storage at least `size_of[ValueType]()` bytes
                wide.

        Constraints:
            - `ValueType` must conform to `DevicePassable`.
            - `ValueType` must be its own leaf `device_type` (i.e.
              `ValueType._is_convertible_to_device_type[ValueType]()` holds).
            - `ValueType` must conform to `ImplicitlyCopyable` or to
              `Copyable & ImplicitlyDestructible`.
        """
        comptime assert ValueType._is_convertible_to_device_type[
            ValueType
        ](), "encode: ValueType must be its own leaf device_type"

        comptime if conforms_to(ValueType, ImplicitlyCopyable):
            comptime T = downcast[ValueType, ImplicitlyCopyable]
            target.bitcast[T]()[] = rebind[T](value)
        elif conforms_to(ValueType, Copyable & ImplicitlyDestructible):
            comptime T = downcast[ValueType, Copyable & ImplicitlyDestructible]
            target.bitcast[T]()[] = rebind[T](value).copy()
        else:
            abort(
                "encode: Type must conform to ImplicitlyCopyable or Copyable &"
                " ImplicitlyDestructible"
            )

    def encode_device_ptr(
        mut self, value: DevicePointer, target: MutOpaquePointer[_]
    ):
        """Encodes a `DevicePointer` into `target`.

        Args:
            value: The `DevicePointer` instance to encode into `target`.
            target: The opaque destination pointer to encode into.
        """
        ...

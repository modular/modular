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

from sys.intrinsics import _type_is_eq
from builtin.rebind import downcast


trait DevicePassable:
    """This trait marks types as passable to accelerator devices."""

    comptime device_type: AnyType
    """Indicate the type being used on accelerator devices."""

    @staticmethod
    fn _is_convertible_to_device_type[SrcT: AnyType]() -> Bool:
        comptime if not _type_is_eq[Self, Self.device_type]() and conforms_to(
            Self.device_type, DevicePassable
        ):
            return downcast[
                Self.device_type, DevicePassable
            ]._is_convertible_to_device_type[SrcT]()
        else:
            return _type_is_eq[SrcT, Self.device_type]()

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """
        Convert the host type object to a device_type and store it at the
        target address.

        NOTE: This should only be called by `DeviceContext` during invocation
        of accelerator kernels.
        """
        ...

    @staticmethod
    fn get_type_name() -> String:
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

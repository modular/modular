# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


trait DevicePassable:
    """This trait marks types as passable to accelerator devices."""

    alias device_type: AnyTrivialRegType
    """Indicate the type being used on accelerator devices."""

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

    @staticmethod
    fn get_device_type_name() -> String:
        """
        Gets device_type's name. For example, because DeviceBuffer's
        device_type is UnsafePointer, DeviceBuffer[DType.float32]'s
        get_device_type_name() should return something like
        "UnsafePointer[Scalar[DType.float32]]". This is used for error messages
        when passing types to the device.
        TODO: This method will be retired soon when better kernel call error
        messages arrive.

        Returns:
            The device type's name.
        """
        ...

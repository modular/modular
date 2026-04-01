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
"""HAL status codes and error handling."""

from std.ffi import CStringSlice


comptime STATUS_SUCCESS: Int64 = 0
comptime STATUS_UNKNOWN_ERROR: Int64 = -1
comptime STATUS_ARGUMENT_OUT_OF_RANGE: Int64 = -2
comptime STATUS_UNKNOWN_PROPERTY_NAME: Int64 = -3
comptime STATUS_NO_DEVICES: Int64 = -4
comptime STATUS_BAD_DEVICE: Int64 = -5
comptime STATUS_BAD_CONTEXT: Int64 = -6
comptime STATUS_BAD_MEMORY: Int64 = -7
comptime STATUS_OUT_OF_MEMORY: Int64 = -8
comptime STATUS_INVALID_ARG: Int64 = -9

comptime STATUS_UNINIT: Int64 = Int64.MAX_FINITE


struct HALError(Movable, Writable):
    """An error from a HAL plugin operation."""

    var status: Int64
    var message: String

    def __init__[
        O: ImmutOrigin
    ](out self, status: Int64, *, message: CStringSlice[O]) raises:
        """Copies from the null-terminated `CStringSlice`.

        Parameters:
            O: The origin of the `CStringSlice`.

        Args:
            status: The status code.
            message: The status message as a C string.
        """
        self.status = status
        self.message = String(from_utf8=message.as_bytes())

    def __init__(out self, status: Int64, *, message: String):
        """Creates an `HALError` from a status code and message string.

        Args:
            status: The status code.
            message: The status message.
        """
        self.status = status
        self.message = message

    def __init__(out self):
        """
        Default construct a HALError with status=STATUS_UNINIT.
        This is distinct from any other valid success or error status.
        """
        self.status = STATUS_UNINIT
        self.message = "(default constructed HALError)"

    def write_to[W: Writer](self, mut writer: W):
        writer.write("HALError(code=", self.status, "): ", self.message)

    def __str__(self) -> String:
        return String.write(self)

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
"""Small helpers that the legacy and the HAL `DeviceContext` share: the C error
check, launch-dimension validation, the stream priority range and the type of a
dump path.
"""

from max.gpu.host._driver_handles import _CString
from max.gpu.host._legacy.dim import Dim
from std.ffi import external_call
from std.pathlib import Path
from std.reflection import (
    SourceLocation,
    call_location,
)
from std.utils import Variant


comptime _DumpPath = Variant[Bool, Path, StaticString, def() thin -> Path]


def _string_from_owned_charptr(c_str: _CString) -> String:
    var result = String()
    if case .Some(c_str) = c_str:
        result = String(unsafe_from_utf8_ptr=c_str.ptr())
    # void AsyncRT_DeviceContext_strfree(const char* ptr)
    external_call["AsyncRT_DeviceContext_strfree", NoneType](c_str)
    return result^


@inline(.never)
def _raise_checked_impl(
    err_msg: _CString, msg: String, location: SourceLocation
) raises:
    var err = _string_from_owned_charptr(err_msg)
    raise Error(location.prefix(err + ((" " + msg) if msg else "")))


@inline(.always)
def _checked(
    err: _CString,
    *,
    msg: String = "",
    location: Optional[SourceLocation] = None,
) raises:
    if err:
        _raise_checked_impl(err, msg, location.or_else(call_location()))


# Raises unless every component of `dim` is in `[1, Int32.MAX]`: drivers take
# launch dims as 32-bit values, so a larger one would be truncated. Callers skip
# it when the plugin's launcher takes over grid/block launches, since that
# launcher reads the dims its own way (eg, 0 for every core).
def _check_dim_in_range[
    func_name_for_msg: StringLiteral, dim_name_for_msg: StringLiteral
](dim: Dim, *, location: SourceLocation) raises:
    if dim.x() <= 0 or dim.x() > Int(Int32.MAX):
        comptime msg = String(
            func_name_for_msg,
            ": Dim value ",
            dim_name_for_msg,
            ".x must be in [1, Int32.MAX].",
        )
        raise Error(location.prefix(msg))
    if dim.y() <= 0 or dim.y() > Int(Int32.MAX):
        comptime msg = String(
            func_name_for_msg,
            ": Dim value ",
            dim_name_for_msg,
            ".y must be in [1, Int32.MAX].",
        )
        raise Error(location.prefix(msg))
    if dim.z() <= 0 or dim.z() > Int(Int32.MAX):
        comptime msg = String(
            func_name_for_msg,
            ": Dim value ",
            dim_name_for_msg,
            ".z must be in [1, Int32.MAX].",
        )
        raise Error(location.prefix(msg))


@fieldwise_init
struct StreamPriorityRange(TrivialRegisterPassable, Writable):
    """Represents the range of valid stream priorities for a GPU device.

    Stream priorities control the scheduling of GPU operations, with higher
    priority streams being executed preferentially over lower priority streams.
    """

    var least: Int
    """The lowest (numerically smallest) priority value."""

    var greatest: Int
    """The highest (numerically largest) priority value."""

    @inline(.always)
    def write_to(self, mut writer: Some[Writer]):
        """Writes the stream priority range to the given writer.

        Args:
            writer: The writer to output the stream priority range to.
        """
        writer.write(
            "StreamPriorityRange(least=",
            self.least,
            ", greatest=",
            self.greatest,
            ")",
        )

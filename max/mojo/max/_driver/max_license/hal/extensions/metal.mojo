# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from std.collections import OptionalReg
from std.ffi import _DLHandle

from ..plugins import RawPluginComponent
from ..status import HALError
from .completion_flag import (
    FlagCondition,
    FlagConditionSet,
    RawPluginCompletionFlag,
    RawPluginCompletionFlagReset,
    load_optional,
)
from .traits import HALExtensions, SupportsCompletionFlagResetV1


struct MetalExtensionFunctions(RawPluginComponent):
    pass


struct MetalHALExtensions(HALExtensions, SupportsCompletionFlagResetV1):
    """`HALExtensions` implementation for Apple Metal backends."""

    comptime FlagGroup = RawPluginCompletionFlag
    comptime ResetGroup = RawPluginCompletionFlagReset
    # An `MTLSharedEvent` wait only compares greater-or-equal.
    comptime FlagConditions = FlagConditionSet(FlagCondition.geq)
    var _flag: OptionalReg[RawPluginCompletionFlag]
    var _flag_reset: OptionalReg[RawPluginCompletionFlagReset]

    def __init__(out self, lib: _DLHandle, so_path: StringSpan) raises HALError:
        self._flag = load_optional[RawPluginCompletionFlag](lib, so_path)
        self._flag_reset = load_optional[RawPluginCompletionFlagReset](
            lib, so_path
        )

    def completion_flag(self) -> OptionalReg[Self.FlagGroup]:
        return self._flag

    def completion_flag_reset(self) -> OptionalReg[Self.ResetGroup]:
        return self._flag_reset

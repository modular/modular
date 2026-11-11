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
    RawPluginCompletionFlagAddress,
    RawPluginCompletionFlagReset,
    load_optional,
)
from .peer_access import RawPluginPeerAccess
from .traits import (
    HALExtensions,
    SupportsCompletionFlagAddressV1,
    SupportsCompletionFlagResetV1,
    SupportsPeerAccessV1,
)


struct CUDAExtensionFunctions(RawPluginComponent):
    pass


struct CUDAHALExtensions(
    HALExtensions,
    SupportsCompletionFlagAddressV1,
    SupportsCompletionFlagResetV1,
    SupportsPeerAccessV1,
):
    """`HALExtensions` implementation for NVIDIA CUDA backends."""

    comptime PeerGroup = RawPluginPeerAccess
    comptime FlagGroup = RawPluginCompletionFlag
    comptime ResetGroup = RawPluginCompletionFlagReset
    comptime AddressGroup = RawPluginCompletionFlagAddress
    comptime FlagConditions = FlagConditionSet(
        FlagCondition.geq, FlagCondition.eq
    )
    var _peer: OptionalReg[RawPluginPeerAccess]
    var _flag: OptionalReg[RawPluginCompletionFlag]
    var _flag_reset: OptionalReg[RawPluginCompletionFlagReset]
    var _flag_address: OptionalReg[RawPluginCompletionFlagAddress]

    def __init__(out self, lib: _DLHandle, so_path: StringSpan) raises HALError:
        # TODO(DRIV-4): require peer access once a `Device`'s spec follows its
        # loaded plugin.
        self._peer = load_optional[RawPluginPeerAccess](lib, so_path)
        self._flag = load_optional[RawPluginCompletionFlag](lib, so_path)
        self._flag_reset = load_optional[RawPluginCompletionFlagReset](
            lib, so_path
        )
        self._flag_address = load_optional[RawPluginCompletionFlagAddress](
            lib, so_path
        )

    def peer_group(self) -> OptionalReg[Self.PeerGroup]:
        return self._peer

    def completion_flag(self) -> OptionalReg[Self.FlagGroup]:
        return self._flag

    def completion_flag_reset(self) -> OptionalReg[Self.ResetGroup]:
        return self._flag_reset

    def completion_flag_address(self) -> OptionalReg[Self.AddressGroup]:
        return self._flag_address

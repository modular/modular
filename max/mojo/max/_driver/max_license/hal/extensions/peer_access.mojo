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
from std.memory import MaybeUninit

from ..plugin import RawDriver
from ..plugins import (
    ContextHandle,
    DeviceHandle,
    HALFunction,
    OutParam,
    PluginResultCode,
    RawPluginComponent,
)
from ..status import STATUS_SUCCESS, STATUS_UNKNOWN_SYMBOL, HALError
from .traits import PeerAccessV1


struct RawPluginPeerAccess(PeerAccessV1, RawPluginComponent):
    """Peer-access entry points of `M_driver_peer.h`, shared by CUDA and HIP."""

    var device_can_access_peer: HALFunction[
        "M_driver_device_can_access_peer",
        def(
            device: DeviceHandle,
            peer: DeviceHandle,
            out_can_access: OutParam[Bool, _],
        ) thin abi("C") -> PluginResultCode,
        required=True,
    ]
    var context_enable_peer_access: HALFunction[
        "M_driver_context_enable_peer_access",
        def(
            context: ContextHandle,
            peer: ContextHandle,
        ) thin abi("C") -> PluginResultCode,
        required=True,
    ]

    @staticmethod
    def load(
        lib: _DLHandle, so_path: StringSlice
    ) raises HALError -> OptionalReg[Self]:
        """Resolves the group, or None when `lib` exports neither symbol."""
        # TODO(DRIV-4): drop once a `Device`'s spec follows its loaded plugin.
        try:
            return Self(lib, so_path)
        except e:
            if e.status == STATUS_UNKNOWN_SYMBOL:
                return None
            raise e^

    def can_access_peer(
        self, driver: RawDriver, device: DeviceHandle, peer: DeviceHandle
    ) raises HALError -> Bool:
        var value = MaybeUninit[Bool]()
        var status = self.device_can_access_peer.get()(
            device, peer, OutParam[Bool](to=value)
        )
        if status != STATUS_SUCCESS:
            var err = driver.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to query peer access: {err.message}"),
            )
        return value.unsafe_assume_init()

    def enable_peer_access(
        self, driver: RawDriver, context: ContextHandle, peer: ContextHandle
    ) raises HALError:
        var status = self.context_enable_peer_access.get()(context, peer)
        if status != STATUS_SUCCESS:
            var err = driver.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to enable peer access: {err.message}"),
            )

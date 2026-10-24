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
from .peer_access import RawPluginPeerAccess
from .traits import HALExtensions, SupportsPeerAccessV1


struct CUDAExtensionFunctions(RawPluginComponent):
    pass


struct CUDAHALExtensions(HALExtensions, SupportsPeerAccessV1):
    """`HALExtensions` implementation for NVIDIA CUDA backends."""

    comptime PeerAccessTy = RawPluginPeerAccess
    var _peer: OptionalReg[RawPluginPeerAccess]

    def __init__(
        out self, lib: _DLHandle, so_path: StringSlice
    ) raises HALError:
        self._peer = RawPluginPeerAccess.load(lib, so_path)

    def peer_access(self) -> OptionalReg[Self.PeerAccessTy]:
        return self._peer

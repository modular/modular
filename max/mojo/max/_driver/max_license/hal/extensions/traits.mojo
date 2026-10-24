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

from ..plugin import RawDriver
from ..plugins import ContextHandle, DeviceHandle, HALPluginConstructable
from ..status import HALError


trait HALExtensions(Deinitable, HALPluginConstructable, Movable):
    """Compile-time interface for a vendor's HAL extension set.

    Each backend provides a struct conforming to this trait to expose its
    surface of HAL extension functions. A plugin advertises its extension set
    via the `comptime extensions` member on `HalPluginHooks`, and each
    `Device` holds one behind its `extensions()` accessor.

    A capability is a `Supports<Name>V<N>` trait; the vendor set holds its
    symbols as a `RawPluginComponent` field resolved in `__init__`. Callers
    branch on a literal `conforms_to`, which refines `Ext` so the call
    type-checks; the Bool flags below do not refine:

    ```mojo
    comptime Ext = type_of(device.extensions())
    comptime if conforms_to(Ext, SupportsPeerAccessV1):
        var peer_access = device.extensions().peer_access()
    ```
    """

    comptime SupportsGraphCaptureV1: Bool = conforms_to(
        Self, SupportsGraphCaptureV1
    )
    comptime SupportsPeerAccessV1: Bool = conforms_to(
        Self, SupportsPeerAccessV1
    )

    def __init__(
        out self, lib: _DLHandle, so_path: StringSlice
    ) raises HALError:
        ...


trait SupportsGraphCaptureV1(Deinitable, Movable):
    comptime GraphCaptureTy: GraphCaptureV1

    def graph_capture(ref self) -> ref[self] Self.GraphCaptureTy:
        ...


trait GraphCaptureV1:
    def start_capture(mut self) raises:
        ...

    def end_capture(mut self) raises:
        ...


trait SupportsPeerAccessV1(Deinitable, Movable):
    """Capability: device-to-device memory access (`M_driver_peer.h`).

    See `CUDAHALExtensions` in `cuda.mojo` for an implementation.

    `peer_access()` returns `None` for plugins without peer support, such as
    `CPUPlugin.so` in a CUDA build.
    """

    comptime PeerAccessTy: PeerAccessV1

    def peer_access(self) -> OptionalReg[Self.PeerAccessTy]:
        ...


trait PeerAccessV1(TrivialRegisterPassable):
    # Status messages are driver-scoped, hence the `RawDriver` argument.
    def can_access_peer(
        self, driver: RawDriver, device: DeviceHandle, peer: DeviceHandle
    ) raises HALError -> Bool:
        ...

    def enable_peer_access(
        self, driver: RawDriver, context: ContextHandle, peer: ContextHandle
    ) raises HALError:
        ...


struct DefaultHALExtensions(HALExtensions):
    """`HALExtensions` implementation used when no plugin is active."""

    def __init__(
        out self, lib: _DLHandle, so_path: StringSlice
    ) raises HALError:
        pass

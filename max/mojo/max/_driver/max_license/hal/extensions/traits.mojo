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
from ..plugins import (
    ContextHandle,
    DeviceHandle,
    FlagHandle,
    HALPluginConstructable,
)
from ..status import STATUS_UNKNOWN_SYMBOL, HALError
from .completion_flag import (
    CompletionFlagAddressV1,
    CompletionFlagResetV1,
    CompletionFlagV1,
    FlagConditionSet,
)


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
        var ok = device.extensions().can_access_peer(
            device.raw_driver()[], device.handle(), peer.handle()
        )
    ```
    """

    comptime SupportsGraphCaptureV1: Bool = conforms_to(
        Self, SupportsGraphCaptureV1
    )
    comptime SupportsPeerAccessV1: Bool = conforms_to(
        Self, SupportsPeerAccessV1
    )
    comptime SupportsCompletionFlagV1: Bool = conforms_to(
        Self, SupportsCompletionFlagV1
    )
    comptime SupportsCompletionFlagResetV1: Bool = conforms_to(
        Self, SupportsCompletionFlagResetV1
    )
    comptime SupportsCompletionFlagAddressV1: Bool = conforms_to(
        Self, SupportsCompletionFlagAddressV1
    )

    def __init__(out self, lib: _DLHandle, so_path: StringSpan) raises HALError:
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

    Vendors set `PeerGroup` and implement only `peer_group()`, as
    `CUDAHALExtensions` does. It is None for plugins without peer support,
    such as `CPUPlugin.so` in a CUDA build.
    """

    comptime PeerGroup: PeerAccessV1

    def peer_group(self) -> OptionalReg[Self.PeerGroup]:
        """Returns the loaded peer symbols, or None when the plugin has none."""
        ...

    def supports_peer_access(self) -> Bool:
        return Bool(self.peer_group())

    def can_access_peer[
        device_origin: ImmOrigin, peer_origin: ImmOrigin, //
    ](
        self,
        driver: RawDriver,
        device: Pointer[DeviceHandle.T, device_origin],
        peer: Pointer[DeviceHandle.T, peer_origin],
    ) raises HALError -> Bool:
        if case .Some(group) = self.peer_group():
            return group.can_access_peer(
                driver,
                rebind[DeviceHandle](device),
                rebind[DeviceHandle](peer),
            )
        return False

    def enable_peer_access[
        context_origin: ImmOrigin, peer_origin: ImmOrigin, //
    ](
        self,
        driver: RawDriver,
        context: Pointer[ContextHandle.T, context_origin],
        peer: Pointer[ContextHandle.T, peer_origin],
    ) raises HALError:
        if case .Some(group) = self.peer_group():
            group.enable_peer_access(
                driver,
                rebind[ContextHandle](context),
                rebind[ContextHandle](peer),
            )
        else:
            raise HALError(
                STATUS_UNKNOWN_SYMBOL,
                message="plugin does not export peer access",
            )


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


trait SupportsCompletionFlagV1(Deinitable, Movable):
    """Capability: host-signaled flags a queue waits on (`M_driver_flag.h`).

    `FlagConditions` is the in-tree vendor's set (GEQ at minimum), for
    `comptime assert` checks. `M_driver_flag_conditions` reports the loaded
    plugin's fixed set, which can differ for an out-of-tree plugin.
    `completion_flag()` is None for plugins without flags, such as
    `CPUPlugin.so` in a CUDA build.
    """

    comptime FlagGroup: CompletionFlagV1
    comptime FlagConditions: FlagConditionSet

    def completion_flag(self) -> OptionalReg[Self.FlagGroup]:
        """Returns the loaded flag symbols, or None when the plugin has none."""
        ...

    # These take origin-carrying handles, as peer access does, so the context
    # that owns a flag's memory stays alive across each call.
    def create_flag[
        context_origin: ImmOrigin, //
    ](
        self,
        driver: RawDriver,
        context: Pointer[ContextHandle.T, context_origin],
    ) raises HALError -> FlagHandle:
        return _flag_group(self).create(driver, rebind[ContextHandle](context))

    def destroy_flag[
        context_origin: ImmOrigin, //
    ](
        self,
        driver: RawDriver,
        context: Pointer[ContextHandle.T, context_origin],
        flag: FlagHandle,
    ) raises HALError:
        _flag_group(self).destroy(driver, rebind[ContextHandle](context), flag)

    def flag_conditions[
        device_origin: ImmOrigin, //
    ](
        self,
        driver: RawDriver,
        device: Pointer[DeviceHandle.T, device_origin],
    ) raises HALError -> FlagConditionSet:
        return _flag_group(self).conditions(
            driver, rebind[DeviceHandle](device)
        )


def _flag_group[
    Ext: SupportsCompletionFlagV1
](ext: Ext) raises HALError -> Ext.FlagGroup:
    if case .Some(group) = ext.completion_flag():
        return group
    raise HALError(
        STATUS_UNKNOWN_SYMBOL, message="plugin does not export completion flags"
    )


trait SupportsCompletionFlagResetV1(SupportsCompletionFlagV1):
    """Capability: reset a flag to 0, keeping its handle."""

    comptime ResetGroup: CompletionFlagResetV1

    def completion_flag_reset(self) -> OptionalReg[Self.ResetGroup]:
        """Returns the loaded reset symbol, or None when the plugin lacks it."""
        ...


trait SupportsCompletionFlagAddressV1(SupportsCompletionFlagV1):
    """Capability: a flag's device-visible 64-bit slot (CUDA, HIP)."""

    comptime AddressGroup: CompletionFlagAddressV1

    def completion_flag_address(self) -> OptionalReg[Self.AddressGroup]:
        """Returns the loaded address symbol, or None if absent."""
        ...


struct DefaultHALExtensions(HALExtensions):
    """`HALExtensions` implementation used when no plugin is active."""

    def __init__(out self, lib: _DLHandle, so_path: StringSpan) raises HALError:
        pass

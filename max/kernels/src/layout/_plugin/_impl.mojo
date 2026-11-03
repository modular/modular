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
"""Resolves the active `layout` backend and the `TensorEngine` it picks.

`CurrentTensorEngine` is keyed on the target being compiled for, so during
device codegen it is already the device's engine, and no `target` parameter
needs threading to the call sites that consume it. `TensorEngineForTarget`
covers the host-side cases that must name a device other than the current one.
"""

from std.memory import AddressSpace
from std.sys.info import _TargetType, _accelerator_arch, _current_target

from max.gpu.host.info import _gpu_mlir_target

from ._overlay import LAYOUT_PLUGINS
from ._trait import LayoutPluginHooks
from ..tensor_engine import TensorEngine
from ..tile_layout import TensorLayout


def _plugin_index[target: _TargetType, i: Int = 0]() -> Int:
    # Unlike `PluginSelector.current`, a backend with no entry falls back to
    # `DefaultLayoutPlugin` instead of failing to compile: out-of-tree
    # backends register with `STD_PLUGINS` without overlaying this package,
    # and the graph compiler's tensor arguments still need an engine there.
    comptime if i >= LAYOUT_PLUGINS.PluginTypes.length:
        return 0
    elif LAYOUT_PLUGINS._matches[target, i.__mlir_index__()]():
        return i
    else:
        return _plugin_index[target, i + 1]()


comptime LayoutPluginForTarget[target: _TargetType]: LayoutPluginHooks = (
    LAYOUT_PLUGINS.PluginTypes[_plugin_index[target]()]
)
"""The `layout` backend selected for `target`."""

comptime CurrentLayoutPlugin: LayoutPluginHooks = LayoutPluginForTarget[
    _current_target()
]
"""The `layout` backend selected for the current target."""

comptime CurrentTensorEngine[
    L: TensorLayout, A: AddressSpace = AddressSpace.GENERIC
]: TensorEngine = CurrentLayoutPlugin.TensorEngineFor[L, A]
"""The `TensorEngine` backing a tile of layout `L` in address space `A` for the
current target."""

comptime TensorEngineForTarget[
    Target: _TargetType, L: TensorLayout, A: AddressSpace = AddressSpace.GENERIC
]: TensorEngine = LayoutPluginForTarget[Target].TensorEngineFor[L, A]
"""The `TensorEngine` backing a tile of layout `L` in address space `A` for
`Target`."""


comptime _device_target: _TargetType = _gpu_mlir_target[_accelerator_arch()]()


comptime DeviceTensorEngine[
    L: TensorLayout, A: AddressSpace = AddressSpace.GENERIC
]: TensorEngine = TensorEngineForTarget[_device_target, L, A]
"""The `TensorEngine` backing a tile of layout `L` in address space `A` on the
accelerator, for host code that spells a kernel's argument types."""

comptime DeviceGlobalAddressSpace: AddressSpace = LayoutPluginForTarget[
    _device_target
].global_address_space
"""The address space a kernel sees device global memory in, on the
accelerator."""

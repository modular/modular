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
from std.sys.info import _TargetType, _current_target

from ._overlay import LAYOUT_PLUGINS
from ._trait import LayoutPluginHooks
from ..tensor_engine import TensorEngine


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
    A: AddressSpace = AddressSpace.GENERIC
]: TensorEngine = CurrentLayoutPlugin.TensorEngineFor[A]
"""The `TensorEngine` backing tiles in address space `A` for the current target.
"""

comptime TensorEngineForTarget[
    Target: _TargetType, A: AddressSpace = AddressSpace.GENERIC
]: TensorEngine = LayoutPluginForTarget[Target].TensorEngineFor[A]
"""The `TensorEngine` backing tiles in address space `A` for `Target`."""

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

from std.collections.string.string_span import _get_kgen_string
from std._plugin.selector import Plugin
from std.sys.info import CompilationTarget

from .._machine import DeviceRef
from ..execution_config import ExecutionConfig
from ..execution_config.cpu import CPUExecutionConfiguration
from ..extensions.traits import DefaultHALExtensions, HALExtensions


trait HalPlugin(Plugin):
    comptime HookFor[target: CompilationTarget]: HalPluginHooks


trait HalPluginHooks:
    """Compile-time hook interface for pluggable stdlib behavior.

    Most hooks are `comptime OptionalReg[Callable]` fields; call sites invoke
    `comptime if CurrentPlugin.xxx_fn: return comptime(CurrentPlugin.xxx_fn.value())(...)`,
    so implementors that leave a hook at `None` add zero cost.

    A few hooks (`abort_fn`, `debug_assert_emit_fn`) are required
    `@staticmethod` trait methods rather than `OptionalReg` fields, because
    their dispatch sites lie on `OptionalReg.value()`'s own instantiation
    path — an `OptionalReg` field would re-enter that template via its own
    `debug_assert` and deadlock comptime instantiation.
    """

    comptime ExecutionConfigType[device: DeviceRef]: ExecutionConfig

    comptime extensions: HALExtensions = DefaultHALExtensions


# ===-----------------------------------------------------------------------===#
# DefaultPlugin
# ===-----------------------------------------------------------------------===#


struct DefaultHalPlugin(HalPlugin):
    """Default `HalPluginHooks` implementation used when no plugin is active.

    Every hook is left at its `HalPluginHooks` default, so the built-in stdlib
    code paths are preserved.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["default"]()
    comptime HookFor[target: CompilationTarget] = DefaultHalPluginHooks[target]


struct DefaultHalPluginHooks[target: CompilationTarget](HalPluginHooks):
    comptime ExecutionConfigType[device: DeviceRef] = CPUExecutionConfiguration

    comptime extensions = DefaultHALExtensions

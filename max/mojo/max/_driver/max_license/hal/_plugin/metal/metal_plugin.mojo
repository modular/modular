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

from std.sys.info import CompilationTarget
from .._trait import HalPlugin, HalPluginHooks
from ..._machine import DeviceRef
from std.collections.string.string_span import _get_kgen_string
from ...execution_config._apple import ExecutionConfigForTarget
from ...extensions.metal import MetalHALExtensions


struct MetalHalPlugin(HalPlugin):
    """`PluginHooks` implementation for Apple Metal backends.

    Every hook is left at its `PluginHooks` default.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["metal"]()
    comptime HookFor[target: CompilationTarget] = MetalHalPluginHooks[target]


struct MetalHalPluginHooks[target: CompilationTarget](HalPluginHooks):
    comptime ExecutionConfigType[device: DeviceRef] = ExecutionConfigForTarget[
        device.spec.name
    ]
    """Returns the execution configuration for the given device."""

    comptime extensions = MetalHALExtensions

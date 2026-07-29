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

from _hal._plugin._trait import HalPluginHooks
from machine import DeviceRef
from _hal.execution_config import ExecutionConfig
from std.collections.string.string_slice import _get_kgen_string
from _hal.execution_config._apple import ExecutionConfigForTarget


struct MetalHalPlugin(HalPluginHooks):
    """`PluginHooks` implementation for Apple Metal backends.

    Every hook is left at its `PluginHooks` default.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["metal"]()

    comptime ExecutionConfigType[device: DeviceRef] = ExecutionConfigForTarget[
        device.spec.name
    ]
    """Returns the execution configuration for the given device."""

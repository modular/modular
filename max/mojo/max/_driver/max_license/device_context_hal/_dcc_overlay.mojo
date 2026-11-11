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
"""Selects the HAL `DeviceContext` behind `max.gpu.host`.

Replaces `max/gpu/host/_overlay.mojo` under --config=hal-device-context (see
`:dcc_overlay` in this package's BUILD.bazel), and exports the same names as
that file, plus the HAL-only ones. It imports absolutely so it resolves the same
staged there as here, where it also compiles in every configuration.

The HAL structs carry the backend's name, so this file is where they regain the
public ones. The explicit aliases below also shadow the `HAL*` names the star
imports bring in, which is why nothing that imports from `max.gpu.host` names a
backend.
"""

from max._driver.max_license.device_context_hal.device_context_hal import *
from max._driver.max_license.device_context_hal.device_context_hal import (
    HALCompletionFlag as CompletionFlag,
    HALDevice as Device,
    HALDeviceBuffer as DeviceBuffer,
    HALDeviceContext as DeviceContext,
    HALDeviceContextArray as DeviceContextArray,
    HALDeviceEvent as DeviceEvent,
    HALDeviceExternalFunction as DeviceExternalFunction,
    HALDeviceFunction as DeviceFunction,
    HALDeviceMulticastBuffer as DeviceMulticastBuffer,
    HALDevicePointer as DevicePointer,
    HALDeviceStream as DeviceStream,
    HALHostBuffer as HostBuffer,
    _HALDeviceContextScope,
)
from max.gpu.host._driver_handles import (
    _CString,
    _DeviceBufferCpp,
    _DeviceContextPtr,
    _DeviceFunctionPtr,
)
from max.gpu.host._driver_common import (
    _DumpPath,
    _check_dim_in_range,
    _checked,
)
from max._driver.max_license.device_context_hal.device_graph_hal import *
from max._driver.max_license.device_context_hal.device_graph_hal import (
    HALDeviceGraph as DeviceGraph,
    HALDeviceGraphBuilder as DeviceGraphBuilder,
    HALDeviceGraphCache as DeviceGraphCache,
    HALDeviceGraphInput as DeviceGraphInput,
    HALDeviceGraphInputKind as DeviceGraphInputKind,
    HALDeviceGraphNode as DeviceGraphNode,
    HALStableAddr as StableAddr,
)
from max._driver.max_license.device_context_hal._nvidia_cuda_hal import *
from max.gpu.host._vendor_handles import _CUstream_st
from max._driver.max_license.device_context_hal._amdgpu_hip_hal import (
    HIP,
    HIP_MODULE,
)
from max.gpu.host._vendor_handles import (
    hipDevice_t,
    hipModule_t,
    hipStream_t,
)

comptime HAL_DEVICE_CONTEXT = True
"""Whether `max.gpu.host` is the HAL `DeviceContext`."""

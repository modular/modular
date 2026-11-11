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
"""Selects the `DeviceContext` implementation behind `max.gpu.host`.

`max.gpu.host` takes the `device_context`, `device_graph`, `_nvidia_cuda`,
and `_amdgpu_hip` names from here. This variant names the legacy
implementation; an overlay replaces this file to swap in another one that
exports the same names.

The backend's own structs carry the backend's name, so this file is where they
regain the public ones. The explicit aliases below also shadow the `Legacy*`
names the star imports bring in, which is why nothing that imports from
`max.gpu.host` mentions a backend.
"""

from ._legacy.device_context import *
from ._legacy.device_context import (
    LegacyCompletionFlag as CompletionFlag,
    LegacyDevice as Device,
    LegacyDeviceBuffer as DeviceBuffer,
    LegacyDeviceClosureFunction as DeviceClosureFunction,
    LegacyDeviceContext as DeviceContext,
    LegacyDeviceContextArray as DeviceContextArray,
    LegacyDeviceEvent as DeviceEvent,
    LegacyDeviceExternalFunction as DeviceExternalFunction,
    LegacyDeviceFunction as DeviceFunction,
    LegacyDeviceMulticastBuffer as DeviceMulticastBuffer,
    LegacyDevicePointer as DevicePointer,
    LegacyDeviceStream as DeviceStream,
    LegacyHostBuffer as HostBuffer,
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
from ._legacy.device_graph import *
from ._legacy.device_graph import (
    LegacyDeviceGraph as DeviceGraph,
    LegacyDeviceGraphBuilder as DeviceGraphBuilder,
    LegacyDeviceGraphCache as DeviceGraphCache,
    LegacyDeviceGraphInput as DeviceGraphInput,
    LegacyDeviceGraphInputKind as DeviceGraphInputKind,
    LegacyDeviceGraphMemoryPool as DeviceGraphMemoryPool,
    LegacyDeviceGraphNode as DeviceGraphNode,
    LegacyStableAddr as StableAddr,
)
from ._legacy._nvidia_cuda import *
from max.gpu.host._vendor_handles import _CUstream_st
from ._legacy._amdgpu_hip import (
    HIP,
    HIP_MODULE,
)
from max.gpu.host._vendor_handles import (
    hipDevice_t,
    hipStream_t,
)

comptime HAL_DEVICE_CONTEXT = False
"""Whether `max.gpu.host` is the HAL `DeviceContext`."""

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
"""Implements the gpu host package."""


from .info import default_accelerator, get_gpu_target

from ._legacy.constant_memory_mapping import ConstantMemoryMapping
from ._legacy.device_attribute import DeviceAttribute
from ._overlay import (
    CompletionFlag,
    DefaultDeviceTypeEncoder,
    DeviceBuffer,
    DeviceContext,
    DeviceContextArray,
    DeviceContextList,
    DeviceEvent,
    DeviceExternalFunction,
    DeviceFunction,
    DeviceMulticastBuffer,
    DevicePointer,
    DeviceStream,
    DeviceTypeEncoder,
    HostBuffer,
    _CString,
    _DeviceBufferCpp,
    _DeviceContextPtr,
    _DeviceFunctionPtr,
    _DumpPath,
    _checked,
)
from ._legacy.dim import Dim
from ._legacy.func_attribute import Attribute, FuncAttribute
from ._legacy.launch_attribute import (
    LaunchAttribute,
    LaunchAttributeID,
    LaunchAttributeValue,
)
from ._overlay import (
    DeviceGraph,
    DeviceGraphBuilder,
    DeviceGraphCache,
    DeviceGraphInput,
    DeviceGraphInputKind,
    DeviceGraphNode,
    StableAddr,
)
from ._overlay import (
    CUDA,
    CUDA_MODULE,
    CUDA_get_current_context,
    CUcontext,
    CUmodule,
    CUstream,
    _CUstream_st,
)
from ._legacy._amdgpu_hip import HIP, HIP_MODULE, hipDevice_t, hipStream_t
from ._legacy._kernel_load_gate import (
    CollectiveEnqueueScope,
    CollectiveGate,
    KernelLoadScope,
)
from ._legacy._metal_capture import (
    _end_metal_trace_capture,
    _set_metal_gpu_print_enabled,
    _start_metal_trace_capture,
)
from ._legacy._tensormap import SwizzleMode, create_tensormap_im2col
from ._legacy._tracing import (
    LIBRARY_PATHS,
    Color,
    RangeID,
    _end_range,
    _is_enabled,
    _is_enabled_details,
    _load_library,
    _mark,
    _start_range,
)

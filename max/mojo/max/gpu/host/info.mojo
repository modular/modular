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
"""Contains information about GPU architectures and their capabilities.

This module provides detailed specifications for various GPU models including
NVIDIA and AMD GPUs. It includes information about compute capabilities,
memory specifications, thread organization, and performance characteristics.
"""


from std.sys.info import _TargetType, _accelerator_arch, _current_target

from std._gpu.host.info import _get_gpu_target
from std._plugin._overlay import ADDITIONAL_TARGETS


@__doc_inline
from std._gpu.host.info import (
    AcceleratorArchitectureFamily,
    GPUInfo,
    NoGPU,
    get_gpu_target,
    is_accelerator,
    is_cpu,
    is_gpu,
    is_valid_target,
)


@__doc_inline
from std._gpu.host._builtin_targets import (
    A10,
    A100,
    AMDCDNA2Family,
    AMDCDNA3Family,
    AMDCDNA4Family,
    AMDRDNAFamily,
    AppleMetalFamily,
    B100,
    B200,
    B300,
    DGXSpark,
    GTX1060,
    GTX1080Ti,
    GTX970,
    H100,
    JetsonThor,
    L4,
    MI250X,
    MI300A,
    MI300X,
    MI355X,
    MetalM1,
    MetalM1Metal4,
    MetalM2,
    MetalM2Metal4,
    MetalM3,
    MetalM3Metal4,
    MetalM4,
    MetalM4Metal4,
    MetalM5,
    MetalM5Metal4,
    NvidiaAdaFamily,
    NvidiaAmpereDatacenterFamily,
    NvidiaAmpereEmbeddedFamily,
    NvidiaAmpereWorkstationFamily,
    NvidiaBlackwellConsumerFamily,
    NvidiaBlackwellFamily,
    NvidiaHopperFamily,
    NvidiaMaxwellFamily,
    NvidiaPascalFamily,
    NvidiaTuringFamily,
    OrinNano,
    RTX2060,
    RTX3090,
    RTX4090,
    RTX4090m,
    RTX5090,
    Radeon6900,
    Radeon7600,
    Radeon7800,
    Radeon780m,
    Radeon7900,
    Radeon8060s,
    Radeon860m,
    Radeon880m,
    Radeon9060,
    Radeon9070,
    SteamDeck,
    TeslaP100,
)

from std._gpu.host.info import _empty_target

from std._gpu.host._builtin_targets import (
    _a100_target,
    _h100_target,
    _metal_m1_target,
    _metal_m2_target,
    _mi300x_target,
    _mi355x_target,
    _is_sm10x_gpu,
    _is_sm12x_gpu,
)


@inline(.always)
def _device_type_encoder_target() -> _TargetType:
    """Returns the target a `DeviceTypeEncoder` encodes for.

    Encoded field offsets have to match the layout the device reads them with.

    Returns:
        The target to compute device type layout with.
    """
    comptime if _accelerator_arch() == "":
        return _current_target()
    elif ADDITIONAL_TARGETS.encode_device_types_with_host_layout:
        return _current_target()
    else:
        return _get_gpu_target()

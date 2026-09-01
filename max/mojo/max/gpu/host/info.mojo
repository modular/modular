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

import std.gpu.host.info
from std.sys.info import _TargetType, _accelerator_arch

# Structs cannot carry their members across a re-export: `mojo doc` renders an
# aliased struct as a bare `comptime` entry, dropping the fields and methods
# that `std.gpu.host.info` documents on a page of its own. These two keep only
# their summary until the definitions themselves move here.
comptime AcceleratorArchitectureFamily = std.gpu.host.info.AcceleratorArchitectureFamily
"""Defines common defaults for a GPU architecture family.

This struct captures the shared characteristics across GPUs in the same
architecture family, reducing redundancy when defining new GPU models.
"""

comptime GPUInfo = std.gpu.host.info.GPUInfo
"""Comprehensive information about a GPU architecture.

This struct contains detailed specifications about GPU capabilities,
including compute units, memory, thread organization, and performance
characteristics.
"""


@always_inline("nodebug")
def get_gpu_target[
    target_arch: StaticString = _accelerator_arch(),
]() -> _TargetType:
    """Gets the GPU target information for the specified architecture.

    Parameters:
        target_arch: GPU architecture name (defaults to current accelerator architecture).

    Returns:
        Target type information for the specified GPU architecture.
    """
    return std.gpu.host.info.get_gpu_target[target_arch]()


comptime NvidiaMaxwellFamily = std.gpu.host.info.NvidiaMaxwellFamily
"""NVIDIA Maxwell architecture family (sm_50-sm_53)."""

comptime NvidiaPascalFamily = std.gpu.host.info.NvidiaPascalFamily
"""NVIDIA Pascal architecture family (sm_60-sm_62)."""

comptime NvidiaTuringFamily = std.gpu.host.info.NvidiaTuringFamily
"""NVIDIA Turing architecture family (sm_75)."""

comptime NvidiaAmpereDatacenterFamily = std.gpu.host.info.NvidiaAmpereDatacenterFamily
"""NVIDIA Ampere datacenter architecture family (sm_80)."""

comptime NvidiaAmpereWorkstationFamily = std.gpu.host.info.NvidiaAmpereWorkstationFamily
"""NVIDIA Ampere workstation architecture family (sm_86)."""

comptime NvidiaAmpereEmbeddedFamily = std.gpu.host.info.NvidiaAmpereEmbeddedFamily
"""NVIDIA Ampere embedded architecture family (sm_87)."""

comptime NvidiaAdaFamily = std.gpu.host.info.NvidiaAdaFamily
"""NVIDIA Ada Lovelace architecture family (sm_89)."""

comptime NvidiaHopperFamily = std.gpu.host.info.NvidiaHopperFamily
"""NVIDIA Hopper architecture family (sm_90)."""

comptime NvidiaBlackwellFamily = std.gpu.host.info.NvidiaBlackwellFamily
"""NVIDIA Blackwell datacenter architecture family (sm_100)."""

comptime NvidiaBlackwellConsumerFamily = std.gpu.host.info.NvidiaBlackwellConsumerFamily
"""NVIDIA Blackwell consumer architecture family (sm_120)."""

comptime AMDCDNA2Family = std.gpu.host.info.AMDCDNA2Family
"""AMD CDNA2 architecture family (gfx90a)."""

comptime AMDCDNA3Family = std.gpu.host.info.AMDCDNA3Family
"""AMD CDNA3 architecture family (gfx94x)."""

comptime AMDCDNA4Family = std.gpu.host.info.AMDCDNA4Family
"""AMD CDNA4 architecture family (gfx95x)."""

comptime AMDRDNAFamily = std.gpu.host.info.AMDRDNAFamily
"""AMD RDNA architecture family."""

comptime AppleMetalFamily = std.gpu.host.info.AppleMetalFamily
"""Apple Metal GPU architecture family."""

comptime NoGPU = std.gpu.host.info.NoGPU
"""Placeholder for when no GPU is available."""

comptime MetalM1 = std.gpu.host.info.MetalM1
"""Apple M1 GPU configuration."""

comptime MetalM2 = std.gpu.host.info.MetalM2
"""Apple M2 GPU configuration."""

comptime MetalM3 = std.gpu.host.info.MetalM3
"""Apple M3 GPU configuration."""

comptime MetalM4 = std.gpu.host.info.MetalM4
"""Apple M4 GPU configuration."""

comptime MetalM5 = std.gpu.host.info.MetalM5
"""Apple M5 GPU configuration."""

comptime MetalM1Metal4 = std.gpu.host.info.MetalM1Metal4
"""Apple M1 GPU configuration for Metal 4."""

comptime MetalM2Metal4 = std.gpu.host.info.MetalM2Metal4
"""Apple M2 GPU configuration for Metal 4."""

comptime MetalM3Metal4 = std.gpu.host.info.MetalM3Metal4
"""Apple M3 GPU configuration for Metal 4."""

comptime MetalM4Metal4 = std.gpu.host.info.MetalM4Metal4
"""Apple M4 GPU configuration for Metal 4."""

comptime MetalM5Metal4 = std.gpu.host.info.MetalM5Metal4
"""Apple M5 GPU configuration for Metal 4."""

comptime A100 = std.gpu.host.info.A100
"""NVIDIA A100 GPU configuration."""

comptime A10 = std.gpu.host.info.A10
"""NVIDIA A10 GPU configuration."""

comptime OrinNano = std.gpu.host.info.OrinNano
"""NVIDIA Orin Nano GPU configuration."""

comptime JetsonThor = std.gpu.host.info.JetsonThor
"""NVIDIA Jetson Thor GPU configuration."""

comptime DGXSpark = std.gpu.host.info.DGXSpark
"""NVIDIA DGX Spark GPU configuration."""

comptime L4 = std.gpu.host.info.L4
"""NVIDIA L4 GPU configuration."""

comptime RTX4090m = std.gpu.host.info.RTX4090m
"""NVIDIA RTX 4090 Mobile GPU configuration."""

comptime RTX4090 = std.gpu.host.info.RTX4090
"""NVIDIA RTX 4090 GPU configuration."""

comptime H100 = std.gpu.host.info.H100
"""NVIDIA H100 GPU configuration."""

comptime B100 = std.gpu.host.info.B100
"""NVIDIA B100 GPU configuration."""

comptime B200 = std.gpu.host.info.B200
"""NVIDIA B200 GPU configuration."""

comptime B300 = std.gpu.host.info.B300
"""NVIDIA B300 GPU configuration."""

comptime RTX5090 = std.gpu.host.info.RTX5090
"""NVIDIA RTX 5090 GPU configuration."""

comptime RTX3090 = std.gpu.host.info.RTX3090
"""NVIDIA GeForce RTX 3090 GPU configuration."""

comptime GTX1080Ti = std.gpu.host.info.GTX1080Ti
"""NVIDIA GeForce GTX 1080 Ti GPU configuration."""

comptime GTX1060 = std.gpu.host.info.GTX1060
"""NVIDIA GeForce GTX 1060 GPU configuration."""

comptime GTX970 = std.gpu.host.info.GTX970
"""NVIDIA GeForce GTX 970 GPU configuration."""

comptime TeslaP100 = std.gpu.host.info.TeslaP100
"""NVIDIA Tesla P100 GPU configuration."""

comptime RTX2060 = std.gpu.host.info.RTX2060
"""NVIDIA RTX 2060 GPU configuration."""

comptime MI250X = std.gpu.host.info.MI250X
"""AMD MI250X GPU configuration."""

comptime MI300X = std.gpu.host.info.MI300X
"""AMD MI300X GPU configuration."""

comptime MI300A = std.gpu.host.info.MI300A
"""AMD MI300A APU configuration.

The MI300A is an Accelerated Processing Unit (APU) that integrates Zen 4 CPU
cores with CDNA 3 GPU compute units and unified HBM3 memory. It shares the
`gfx942` ISA with the MI300X but has fewer compute units (228 vs 304) and
unified host/device memory. Found in systems such as the CINES Adastra
supercomputer.
"""

comptime MI355X = std.gpu.host.info.MI355X
"""AMD MI355X GPU configuration."""

comptime Radeon9070 = std.gpu.host.info.Radeon9070
"""AMD Radeon 9070 GPU configuration."""

comptime Radeon9060 = std.gpu.host.info.Radeon9060
"""AMD Radeon 9060 GPU configuration."""

comptime Radeon7900 = std.gpu.host.info.Radeon7900
"""AMD Radeon 7900 GPU configuration."""

comptime Radeon7800 = std.gpu.host.info.Radeon7800
"""AMD Radeon 7800/7700 GPU configuration."""

comptime Radeon7600 = std.gpu.host.info.Radeon7600
"""AMD Radeon 7600 GPU configuration."""

comptime Radeon6900 = std.gpu.host.info.Radeon6900
"""AMD Radeon 6900 GPU configuration."""

comptime Radeon780m = std.gpu.host.info.Radeon780m
"""AMD Radeon 780M GPU configuration."""

comptime Radeon880m = std.gpu.host.info.Radeon880m
"""AMD Radeon 880M GPU configuration."""

comptime Radeon8060s = std.gpu.host.info.Radeon8060s
"""AMD Radeon 8060S GPU configuration."""

comptime Radeon860m = std.gpu.host.info.Radeon860m
"""AMD Radeon 860M GPU configuration."""

comptime SteamDeck = std.gpu.host.info.SteamDeck
"""Steam Deck (Van Gogh) APU configuration."""


@always_inline("nodebug")
def is_gpu[target: StringSlice]() -> Bool:
    """Checks if the target is a GPU (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is a GPU, False otherwise.
    """
    return std.gpu.host.info.is_gpu[target]()


@always_inline("nodebug")
def is_gpu(target: StringSlice) -> Bool:
    """Checks if the target is a GPU (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is a GPU, False otherwise.
    """
    return std.gpu.host.info.is_gpu(target)


@always_inline("nodebug")
def is_cpu[target: StringSlice]() -> Bool:
    """Checks if the target is a CPU (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is a CPU, False otherwise.
    """
    return std.gpu.host.info.is_cpu[target]()


@always_inline("nodebug")
def is_cpu(target: StringSlice) -> Bool:
    """Checks if the target is a CPU (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is a CPU, False otherwise.
    """
    return std.gpu.host.info.is_cpu(target)


@always_inline("nodebug")
def is_npu[target: StringSlice]() -> Bool:
    """Checks if the target is an NPU (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is an NPU, False otherwise.
    """
    return std.gpu.host.info.is_npu[target]()


@always_inline("nodebug")
def is_npu(target: StringSlice) -> Bool:
    """Checks if the target is an NPU (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is an NPU, False otherwise.
    """
    return std.gpu.host.info.is_npu(target)


@always_inline("nodebug")
def is_accelerator[target: StringSlice]() -> Bool:
    """Checks if the target is an accelerator (compile-time version).

    True for any non-CPU compute target -- GPUs and NPUs alike.

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is a GPU or NPU, False otherwise.
    """
    return std.gpu.host.info.is_accelerator[target]()


@always_inline("nodebug")
def is_accelerator(target: StringSlice) -> Bool:
    """Checks if the target is an accelerator (runtime version).

    True for any non-CPU compute target -- GPUs and NPUs alike.

    Args:
        target: Target string to check.

    Returns:
        True if the target is a GPU or NPU, False otherwise.
    """
    return std.gpu.host.info.is_accelerator(target)


@always_inline("nodebug")
def is_valid_target[target: StringSlice]() -> Bool:
    """Checks if the target is valid (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is valid (CPU, GPU, or NPU), False otherwise.
    """
    return std.gpu.host.info.is_valid_target[target]()


@always_inline("nodebug")
def is_valid_target(target: StringSlice) -> Bool:
    """Checks if the target is valid (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is valid (CPU, GPU, or NPU), False otherwise.
    """
    return std.gpu.host.info.is_valid_target(target)


from std.gpu.host.info import (
    _all_targets,
    _get_a100_target,
    _get_empty_target,
    _get_h100_target,
    _get_metal_m1_target,
    _get_metal_m2_target,
    _get_mi300x_target,
    _get_mi355x_target,
    _is_sm10x_gpu,
    _is_sm12x_gpu,
)

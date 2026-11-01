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


from std.sys import CompilationTarget
from std.sys.info import _TargetType, _accelerator_arch, _current_target


@__doc_inline
from std._gpu.host.info import (
    AcceleratorArchitectureFamily,
    GPUInfo,
    NoGPU,
    TargetAccelerator,
    is_accelerator,
    is_cpu,
    is_gpu,
    is_valid_target,
)


@__doc_inline
from max._plugin.cuda.targets import (
    A10,
    A100,
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
    TeslaP100,
)


@__doc_inline
from max._plugin.hip.targets import (
    AMDCDNA2Family,
    AMDCDNA3Family,
    AMDCDNA4Family,
    AMDRDNAFamily,
    MI250X,
    MI300A,
    MI300X,
    MI355X,
    MI455X,
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
)


@__doc_inline
from max._plugin.metal.targets import (
    AppleMetalFamily,
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
    MetalM6,
    MetalM6Metal4,
)

from std._gpu.host.info import TargetAcceleratorCollection

from max._plugin.cuda.targets import (
    _a100_target,
    _h100_target,
    _is_sm10x_gpu,
    _is_sm12x_gpu,
)
from max._plugin.hip.targets import _mi300x_target, _mi355x_target
from max._plugin.metal.targets import _metal_m1_target, _metal_m2_target

from max._plugin._overlay import TARGETS

# ===----------------------------------------------------------------------=== #
# Accelerator target lookup
# ===----------------------------------------------------------------------=== #

# TODO(MOLIB-127): Move this out of the `max.gpu` module and somewhere more
#   general. This is not limited to GPUs only.
comptime default_accelerator = get_gpu_target[_accelerator_arch()]
"""Get the default accelerator target.

The default accelerator is determined automatically by the compiler, but
may be overridden by the `--target-accelerator` command line flag.

This function will always return the same value, whether called in host or
accelerator offload (`compile_info`) compilation context.

Constraints:
    An accelerator target must be configured.

Returns:
    A value representing the current accelerator target.
"""


def get_gpu_target[
    # TODO: Ideally this is an Optional[StaticString] but blocked by MOCO-1039
    target_arch: StaticString = _accelerator_arch(),
]() -> CompilationTarget[_mlir_value=_gpu_mlir_target[target_arch]()]:
    """Gets the GPU target information for the specified architecture.

    Parameters:
        target_arch: GPU architecture name (defaults to current accelerator architecture).

    Returns:
        Target type information for the specified GPU architecture.
    """
    comptime current = CompilationTarget.current()

    comptime _default_accelerator = CompilationTarget[
        _mlir_value=_gpu_mlir_target[_accelerator_arch()]()
    ]()

    # Note:
    #   Defensively assert that there aren't multiple different accelerator
    #   targets in play in a program that depends on calling
    #   `default_accelerator()`. In theory, this should never be hit in
    #   Mojo programs doing accelerator programming in the standard way,
    #   without hard-coding targets, using the `--target-accelerator`.
    comptime if current.is_accelerator():
        # This function is being called from non-host code.
        comptime assert current._eq_triple_and_arch[
            _default_accelerator
        ](), String(
            (
                "default_accelerator() called while compiling for accelerator"
                " target '"
            ),
            StaticString(current.__triple()),
            "' (arch: '",
            StaticString(current.__arch()),
            "'), which is not the default accelerator '",
            StaticString(_default_accelerator.__triple()),
            "' (arch: '",
            StaticString(_default_accelerator.__arch()),
            (
                "'). This divergence could mean subtle behavior differences"
                " between host and accelerator code. The default accelerator"
                " comes from `--target-accelerator` or the detected GPU, and"
                " this function assumes a program only compiles for that one"
                " accelerator. This may happens when code is offloaded with an"
                " explicit `target` that differs from the default. In"
                " accelerator-only code, use `current_accelerator()`, or pass"
                " the target explicitly. This error may be overly cautious,"
                " please file a bug if it blocks your progress."
            ),
        )

    return {}


# Must stay an ordinary (not builtin-inlined) function. The default
# `accelerator_arch` is only resolved at elaboration time, so the table lookup
# can't fold while types are parsed; inlining it would embed the whole
# accelerator table in every type and symbol name that mentions the target.
def _gpu_mlir_target[target_arch: StaticString]() -> _TargetType:
    return TargetAccelerator.from_arch[target_arch, TARGETS].mlir_target


def get_gpu_info[target_arch: StaticString = _accelerator_arch()]() -> GPUInfo:
    """Gets `GPUInfo` for a specific target architecture.

    Maps target architecture strings to corresponding `GPUInfo` instances.

    Parameters:
        target_arch: Target architecture name (e.g., "sm_80", "gfx942").
          Defaults to the current accelerator architecture.

    Returns:
        `GPUInfo` instance for the specified target architecture.
    """
    comptime result = TargetAccelerator.from_arch[target_arch, TARGETS]

    return materialize[result.gpu_info]()


# ===----------------------------------------------------------------------=== #
# Target Accelerator Collections
# ===----------------------------------------------------------------------=== #


# All supported target architectures in canonical form.
#
# Normalization: "nvidia:80"     -> "sm_80"
#                "amdgpu:gfx942" -> "gfx942"
#                "metal:4"       -> "apple-m4"
#
# SYNC: This list must stay in sync with the TargetTraits accelerator tables
#       in Mojo/lib/Target/. Run the following test to verify:
#       bazel test //Mojo/test/mojo-tool:build/internal/verify_supported_accelerators_sync.mojo.test
def _all_target_accelerators() -> List[String]:
    var list = List[String]()
    comptime for idx in range(TARGETS.length):
        list.extend(_target_accelerator_values[TARGETS[idx]]())
    list.append("cuda")
    return list^


def _target_accelerator_values[
    C: TargetAcceleratorCollection
]() -> List[String]:
    """Values recognizable by `--target-accelerator` from this target collection.
    """

    var list = List[String]()

    comptime for idx in range(C.TARGETS.length):
        comptime entry = C.TARGETS[idx]

        list.extend(materialize[entry.target_accelerator_values]())

    return list^


# ===----------------------------------------------------------------------=== #
# Other
# ===----------------------------------------------------------------------=== #


comptime _encodes_with_host_layout[C: TargetAcceleratorCollection]: Bool = (
    C.encode_device_types_with_host_layout
)

# TODO(DRIV-154): Decide this per target rather than for the whole build.
comptime _encode_device_types_with_host_layout: Bool = TARGETS.any[
    _encodes_with_host_layout
]()


@inline(.always)
def _device_type_encoder_target() -> _TargetType:
    """Returns the target a `DeviceTypeEncoder` encodes for.

    Encoded field offsets have to match the layout the device reads them with.

    Returns:
        The target to compute device type layout with.
    """
    comptime if _accelerator_arch() == "":
        return _current_target()
    elif _encode_device_types_with_host_layout:
        return _current_target()
    else:
        return get_gpu_target()._mlir_value

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
"""Mojo types and traits for modeling GPU architectures and their capabilities.

This module provides detailed specifications for various GPU models including
NVIDIA and AMD GPUs. It includes information about compute capabilities,
memory specifications, thread organization, and performance characteristics.
"""

# Contributor note: if you're adding support for a new GPU architecture, see
# `Mojo/docs/contributing/stdlib/adding-gpu-targets.md` for a step-by-step guide covering
# the MLIR target configuration, the `data_layout` string format, and the
# locations in this file that need to be updated.

from std.math import ceildiv, floor
from std.sys.info import (
    CompilationTarget,
    _accelerator_arch,
    _TargetType,
)

from std._plugin._overlay import ADDITIONAL_TARGETS

from ._builtin_targets import BuiltinTargets


@inline(.always)
def get_gpu_target[
    # TODO: Ideally this is an Optional[StaticString] but blocked by MOCO-1039
    target_arch: StaticString = _accelerator_arch(),
]() -> CompilationTarget[
    _mlir_value=TargetAccelerator.from_arch[target_arch].mlir_target
]:
    """Gets the GPU target information for the specified architecture.

    Parameters:
        target_arch: GPU architecture name (defaults to current accelerator architecture).

    Returns:
        Target type information for the specified GPU architecture.
    """
    return {}


# ===----------------------------------------------------------------------=== #
# _LookupTargetAccelerator
# ===----------------------------------------------------------------------=== #

comptime _matches_target_arch[
    target_arch: StaticString,
    C: TargetAcceleratorCollection,
    Elt: TargetAcceleratorType,
    Idx: Int,
]: Bool = Elt.target_accelerator_values.__contains__(
    C.normalize_target_arch(target_arch)
)

comptime _matches_gpu_name[
    gpu_name: StaticString,
    __unused: TargetAcceleratorCollection,
    Elt: TargetAcceleratorType,
    Idx: Int,
]: Bool = Elt.gpu_info.name == gpu_name


struct _LookupTargetAccelerator[
    filter_pred: __generator_type[
        C: TargetAcceleratorCollection,
        Elt: TargetAcceleratorType,
        Idx: Int,
    ] Bool,
    Collections: TypeList[Trait=TargetAcceleratorCollection, ...] = TypeList.of[
        Trait=TargetAcceleratorCollection, BuiltinTargets, ADDITIONAL_TARGETS
    ](),
]:
    comptime by_target_arch[target_arch: StaticString] = (
        _LookupTargetAccelerator[
            _matches_target_arch[
                target_arch,
                _,
                _,
                _,
            ],
            Self.Collections,
        ]
    )

    comptime by_gpu_name[gpu_name: StaticString] = (
        _LookupTargetAccelerator[
            _matches_gpu_name[gpu_name, _, _, _], Self.Collections
        ]
    )

    comptime _matching_target_values[C: TargetAcceleratorCollection] = (
        C.TARGETS.filter_idx[Self.filter_pred[C, _, _]]().values
    )

    comptime results = TypeList._concat[
        *Self.Collections.map_to_values[Self._matching_target_values]()
    ]()

    comptime single_result = (
        Self.results[0] if Self.results.length
        == 1 else Self._error_non_single_result()
    )

    @staticmethod
    def _error_non_single_result() -> (
        TargetAcceleratorType
    ) where Self.results.length != 1:
        comptime if Self.results.length == 0:
            # No matching target could be found, so issue a descriptive error.
            # FIXME(MSTDL-3207): This error is not applicable to all the queries
            #       where `single_result` is used.
            comptime assert False, _build_unsupported_arch_error[
                _accelerator_arch()
            ]()
        else:
            comptime assert False, String(
                t"target lookup must match exactly one target entry, got"
                t" {Self.results.length}"
            )


# ===----------------------------------------------------------------------=== #
# Target Accelerator Collections
# ===----------------------------------------------------------------------=== #


# TODO: This trait wouldn't be needed if `type_of(TargetAccelerator)` worked as
#       the `TypeList` metatype.
trait TargetAcceleratorType(Defaultable):
    comptime gpu_info: GPUInfo
    comptime mlir_target: _TargetType
    comptime target_accelerator_values: List[String]

    comptime target = CompilationTarget[_mlir_value=Self.mlir_target]()


struct TargetAccelerator[
    gpu_info_: GPUInfo,
    target_: CompilationTarget,
    # TODO: Should be the same as either GPUInfo.arch_name or GPUInfo.version,
    #       however that field is currently used inconsistently.
    target_accelerator_values_: List[String] = [],
](TargetAcceleratorType):
    # TargetAcceleratorType conformance
    comptime gpu_info = Self.gpu_info_
    comptime mlir_target = Self.target_._mlir_value
    comptime target_accelerator_values = Self.target_accelerator_values_

    # Used when constructing a `TargetAcceleratorCollection` table to add
    # accepted `--target-accelerator` values to a `TargetAccelerator` instance
    # that has the default `[]` list.
    #
    # We could declare these ahead of time on the target constants (e.g. `B200`)
    # but that makes it harder to how where the current ambiguities and
    # conflicts are (e.g. GTX1060 and GTX1060Ti could arguably both match
    # "sm_61", but only is declared with as matching that `--target-accelerator`
    # value).
    comptime _with_cli_values[values: List[String]] = TargetAccelerator[
        Self.gpu_info_, Self.target_, values
    ]

    def __init__(out self):
        pass

    comptime current_accelerator = Self.from_arch[_accelerator_arch()]

    # Note:
    #   These parametric aliases can't instead be a static methods like:
    #       `TargetAccelerator.from_arch[..]()`
    #   because the compiler requires the `Self` type be concrete, otherwise the
    #   call fails due to an 'unbound struct type parameters' error.

    comptime from_arch[name: StaticString] = (
        _LookupTargetAccelerator[].by_target_arch[name].single_result
    )

    comptime from_gpu_name[name: StaticString] = (
        _LookupTargetAccelerator[].by_gpu_name[name].single_result
    )

    # Forwarded for backwards compatibility. Enables e.g. `B200.sm_count` to
    # work, instead of the more verbose `B200.gpu_info.sm_count`.
    # Other fields could be forwarded, but these are the commonly accessed ones.
    # fmt: off
    comptime sm_count                         = Self.gpu_info_.sm_count
    comptime shared_memory_per_multiprocessor = Self.gpu_info_.shared_memory_per_multiprocessor
    comptime compute                          = Self.gpu_info_.compute
    # fmt: on


trait TargetAcceleratorCollection:
    comptime vendor_name: String

    comptime encode_device_types_with_host_layout: Bool = False

    comptime RAW_TARGETS: TypeList[Trait=TargetAcceleratorType]._mlir_type

    comptime TARGETS = TypeList[Trait=TargetAcceleratorType, Self.RAW_TARGETS]()

    @staticmethod
    def normalize_target_arch(target_arch0: StaticString) -> String:
        ...


struct EmptyTargetCollection(TargetAcceleratorCollection):
    comptime vendor_name: String = "<empty>"

    comptime RAW_TARGETS = TypeList.of[Trait=TargetAcceleratorType]().values

    @staticmethod
    def normalize_target_arch(target_arch0: StaticString) -> String:
        return target_arch0


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


def _unsupported_arch_error_additions[
    C: TargetAcceleratorCollection
]() -> String:
    var string = String(t"{C.vendor_name}: ")

    comptime for idx in range(C.TARGETS.length):
        comptime if idx != 0:
            string.write(", ")

        comptime entry = C.TARGETS[idx]

        comptime arch_name = entry.gpu_info.arch_name
        comptime name = entry.gpu_info.name

        string.write(t"{arch_name} ({name})")

    return string^


# ===-----------------------------------------------------------------------===#
# AcceleratorArchitectureFamily
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct AcceleratorArchitectureFamily(TrivialRegisterPassable):
    """Defines common defaults for a GPU architecture family.

    This struct captures the shared characteristics across GPUs in the same
    architecture family, reducing redundancy when defining new GPU models.
    """

    var warp_size: Int
    """Number of threads in a warp/wavefront."""

    var threads_per_multiprocessor: Int
    """Maximum number of threads per streaming multiprocessor."""

    var shared_memory_per_multiprocessor: Int
    """Size of shared memory available per multiprocessor in bytes."""

    var max_registers_per_block: Int
    """Maximum number of registers that can be allocated to a thread block."""

    var max_thread_block_size: Int
    """Maximum number of threads allowed in a thread block."""


# ===-----------------------------------------------------------------------===#
# NoGPU
# ===-----------------------------------------------------------------------===#


comptime _empty_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "", `,
        `arch = "", `,
        `features = "", `,
        `data_layout="",`,
        `index_bit_width = 0,`,
        `simd_bit_width = 0`,
        `> : !kgen.target`,
    ]
]()
"""Empty target configuration for when no GPU is available."""


comptime NoGPU = GPUInfo(
    name="NoGPU",
    api="none",
    arch_name="no_gpu",
    compute=0,
    version="",
    sm_count=0,
    warp_size=0,
    threads_per_multiprocessor=0,
    shared_memory_per_multiprocessor=0,
    max_registers_per_block=0,
    max_thread_block_size=0,
)
"""Placeholder for when no GPU is available."""


# ===-----------------------------------------------------------------------===#
# GPUInfo
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct GPUInfo(Copyable, Equatable, Movable, RegisterPassable, Writable):
    """Comprehensive information about a GPU architecture.

    This struct contains detailed specifications about GPU capabilities,
    including compute units, memory, thread organization, and performance
    characteristics.
    """

    var name: StaticString
    """The model name of the GPU."""

    var api: StaticString
    """The graphics/compute API the GPU is programmed through.

    Each vendor has its own API, so this also identifies the vendor: `"cuda"`
    for NVIDIA, `"hip"` for AMD, `"metal"` for Apple, and `"none"` for `NoGPU`.
    A stdlib plugin contributes its own name for hardware the stdlib has no
    built-in knowledge of. Comparisons must use these exact spellings.
    """

    var arch_name: StaticString
    """The architecture name of the GPU (e.g., sm_80, gfx942)."""

    var compute: Float32
    """Compute capability version number for NVIDIA GPUs."""

    var version: StaticString
    """Version string of the GPU architecture."""

    var sm_count: Int
    """Number of streaming multiprocessors (SMs) on the GPU."""

    var warp_size: Int
    """Number of threads in a warp/wavefront."""

    var threads_per_multiprocessor: Int
    """Maximum number of threads per streaming multiprocessor."""

    var shared_memory_per_multiprocessor: Int
    """Size of shared memory available per multiprocessor in bytes."""

    var max_registers_per_block: Int
    """Maximum number of registers that can be allocated to a thread block."""

    var max_thread_block_size: Int
    """Maximum number of threads allowed in a thread block."""

    @staticmethod
    def from_target[target: CompilationTarget]() -> Self:
        """Creates a `GPUInfo` instance from an MLIR target.

        Parameters:
            target: MLIR target configuration.

        Returns:
            GPU info corresponding to the target.
        """
        return Self.from_name[target._arch()]()

    @staticmethod
    def current_accelerator() -> Self:
        """Gets `GPUInfo` for the default target accelerator.

        Returns:
            `GPUInfo` instance for the default target accelerator.
        """
        return GPUInfo.from_name[_accelerator_arch()]()

    @staticmethod
    def from_name[target_arch0: StaticString, /]() -> Self:
        """Gets `GPUInfo` for a specific target architecture.

        Maps target architecture strings to corresponding `GPUInfo` instances.

        Parameters:
            target_arch0: Target architecture name (e.g., "sm_80", "gfx942").

        Returns:
            `GPUInfo` instance for the specified target architecture.
        """
        comptime result = _LookupTargetAccelerator[].by_target_arch[
            target_arch0
        ].single_result

        return materialize[result.gpu_info]()

    @staticmethod
    def from_family(
        family: AcceleratorArchitectureFamily,
        name: StaticString,
        api: StaticString,
        arch_name: StaticString,
        compute: Float32,
        version: StaticString,
        sm_count: Int,
    ) -> Self:
        """Creates a `GPUInfo` instance using architecture family defaults.

        This constructor simplifies GPU definition by inheriting common
        characteristics from an architecture family while allowing specific
        values to be overridden.

        Args:
            family: Architecture family providing default values.
            name: The model name of the GPU.
            api: The graphics/compute API supported by the GPU.
            arch_name: The architecture name of the GPU.
            compute: Compute capability version number.
            version: Version string of the GPU architecture.
            sm_count: Number of streaming multiprocessors.

        Returns:
            A fully configured GPUInfo instance.
        """
        return Self(
            name=name,
            api=api,
            arch_name=arch_name,
            compute=compute,
            version=version,
            sm_count=sm_count,
            warp_size=family.warp_size,
            threads_per_multiprocessor=family.threads_per_multiprocessor,
            shared_memory_per_multiprocessor=family.shared_memory_per_multiprocessor,
            max_registers_per_block=family.max_registers_per_block,
            max_thread_block_size=family.max_thread_block_size,
        )

    def __eq__(self, other: Self) -> Bool:
        """Checks if two `GPUInfo` instances represent the same GPU model.

        Args:
            other: Another `GPUInfo` instance to compare against.

        Returns:
            True if both instances represent the same GPU model.
        """
        return self.name == other.name

    def __eq__(self, other: TargetAccelerator[_, _, _]) -> Bool:
        """Compares this GPU information with a target accelerator.

        Compares against the accelerator's associated `GPUInfo`, allowing
        device information to be checked directly against a built-in target
        constant.

        Args:
            other: The target accelerator to compare.

        Returns:
            True if both represent the same GPU model.

        Examples:

        ```mojo
        var device_info = materialize[H100.gpu_info]()
        print(device_info == H100)  # True
        ```
        """
        return self == materialize[other.gpu_info]()

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        """Writes GPU information to a writer.

        Outputs all GPU specifications and capabilities to the provided writer
        in a human-readable format.

        Args:
            writer: A Writer instance to output the GPU information.
        """
        writer.write("name: ", self.name, "\n")
        writer.write("api: ", self.api, "\n")
        writer.write("arch_name: ", self.arch_name, "\n")
        writer.write("compute: ", self.compute, "\n")
        writer.write("version: ", self.version, "\n")
        writer.write("sm_count: ", self.sm_count, "\n")
        writer.write("warp_size: ", self.warp_size, "\n")
        writer.write(
            "threads_per_multiprocessor: ",
            self.threads_per_multiprocessor,
            "\n",
        )
        writer.write(
            "shared_memory_per_multiprocessor: ",
            self.shared_memory_per_multiprocessor,
            "\n",
        )
        writer.write(
            "max_registers_per_block: ", self.max_registers_per_block, "\n"
        )
        writer.write(
            "max_thread_block_size: ", self.max_thread_block_size, "\n"
        )


# ===-----------------------------------------------------------------------===#
# _build_unsupported_arch_error
# ===-----------------------------------------------------------------------===#


def _build_unsupported_arch_error[target_arch: StaticString]() -> String:
    """Builds a helpful error message for unsupported GPU architectures.

    Provides a comprehensive list of all supported GPU architectures across
    all vendors with documentation links.

    Parameters:
        target_arch: The unsupported target architecture string.

    Returns:
        A detailed error message with supported architectures and doc links.
    """
    comptime nvidia_archs = (
        "sm_52 (Maxwell), sm_60/sm_61 (Pascal), sm_75 (Turing), sm_80 (Ampere"
        " A100), sm_86 (Ampere A10), sm_87 (Orin), sm_89 (Ada L4/RTX4090),"
        " sm_90/sm_90a (Hopper H100), sm_100/sm_100a (Blackwell B100/B200),"
        " sm_110 (Jetson Thor), sm_120/sm_120a (Blackwell RTX5090), sm_121 (DGX"
        " Spark)"
    )
    comptime amd_archs = (
        "gfx90a (MI250X), gfx942 (MI300X/MI300A), gfx950 (MI355X), gfx1030"
        " (Radeon 6900), gfx1033 (Van Gogh), gfx1100 (Radeon 7900), gfx1101"
        " (Radeon 7800), gfx1102 (Radeon 7600), gfx1103 (Radeon 780M),"
        " gfx1150/gfx1151/gfx1152 (Radeon 8xx), gfx1200 (Radeon 9060), gfx1201"
        " (Radeon 9070)"
    )
    comptime apple_archs = (
        "metal:1 (M1), metal:2 (M2), metal:3 (M3), metal:4 (M4)"
    )

    var prefix: String

    comptime if target_arch == "":
        prefix = "Unknown GPU architecture detected."
    else:
        prefix = String(
            "GPU architecture '", target_arch, "' is not supported."
        )

    return String(
        prefix,
        "\n\nSupported GPU architectures:\n\n",
        "  NVIDIA: ",
        nvidia_archs,
        "\n  See: https://developer.nvidia.com/cuda-gpus\n\n",
        "  AMD: ",
        amd_archs,
        (
            "\n  See:"
            " https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html"
            "\n\n"
        ),
        "  Apple: ",
        apple_archs,
        (
            "\n  See:"
            " https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf"
            "\n\n"
        ),
        comptime (_unsupported_arch_error_additions[ADDITIONAL_TARGETS]()),
    )


# All supported target architectures in canonical form.
#
# Normalization: "nvidia:80"     -> "sm_80"
#                "amdgpu:gfx942" -> "gfx942"
#                "metal:4"       -> "apple-m4"
#
# SYNC: This list must stay in sync with the TargetTraits accelerator tables
#       in Mojo/lib/Target/. Run the following test to verify:
#       bazel test //Mojo/test/mojo-tool:build/internal/verify_supported_accelerators_sync.mojo.test
comptime _all_target_accelerator_values: List[String] = (
    _target_accelerator_values[BuiltinTargets]()
    + ["cuda"]
    + _target_accelerator_values[ADDITIONAL_TARGETS]()
)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


def is_gpu[target: StringSlice]() -> Bool:
    """Checks if the target is a GPU (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is a GPU, False otherwise.
    """
    return is_gpu(target)


def is_gpu(target: StringSlice) -> Bool:
    """Checks if the target is a GPU (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is a GPU, False otherwise.
    """
    return target == "gpu"


def is_cpu[target: StringSlice]() -> Bool:
    """Checks if the target is a CPU (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is a CPU, False otherwise.
    """
    return is_cpu(target)


def is_cpu(target: StringSlice) -> Bool:
    """Checks if the target is a CPU (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is a CPU, False otherwise.
    """
    return target == "cpu"


def is_accelerator[target: StringSlice]() -> Bool:
    """Checks if the target is an accelerator (compile-time version).

    True for any non-CPU compute target.

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is an accelerator, False otherwise.
    """
    return is_accelerator(target)


def is_accelerator(target: StringSlice) -> Bool:
    """Checks if the target is an accelerator (runtime version).

    True for any non-CPU compute target.

    Args:
        target: Target string to check.

    Returns:
        True if the target is an accelerator, False otherwise.
    """
    return is_gpu(target)


def is_valid_target[target: StringSlice]() -> Bool:
    """Checks if the target is valid (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is valid (CPU or GPU), False otherwise.
    """
    return is_valid_target(target)


def is_valid_target(target: StringSlice) -> Bool:
    """Checks if the target is valid (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is valid (CPU or GPU), False otherwise.
    """
    return is_cpu(target) or is_accelerator(target)

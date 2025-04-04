# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Contains information about GPU architectures and their capabilities.

This module provides detailed specifications for various GPU models including
NVIDIA and AMD GPUs. It includes information about compute capabilities,
memory specifications, thread organization, and performance characteristics.
"""

from math import ceildiv, floor
from os import abort
from sys import env_get_string
from sys.info import _accelerator_arch, _get_arch
from collections.string.string_slice import StringSlice, StaticString
from builtin.string_literal import get_string_literal_slice

alias DEFAULT_GPU_ARCH = _accelerator_arch()
alias DEFAULT_GPU = Info.from_name[DEFAULT_GPU_ARCH]()
alias DEFAULT_GPU_TARGET = DEFAULT_GPU.target()

alias _KB = 1024

# ===-----------------------------------------------------------------------===#
# Vendor
# ===-----------------------------------------------------------------------===#


@value
@register_passable
struct Vendor(Writable):
    """Represents GPU vendors.

    This struct provides identifiers for different GPU vendors and utility
    methods for comparison and string representation.

    The Vendor struct defines constants for common GPU vendors (NVIDIA, AMD)
    and includes a NO_GPU option for systems without GPU support. It provides
    comparison operators and string conversion methods for vendor identification.
    """

    var _value: Int8
    """The underlying integer value representing the vendor."""

    alias NO_GPU = Self(0)
    """Represents no GPU or CPU-only execution."""

    alias AMD_GPU = Self(1)
    """Represents AMD GPU vendor."""

    alias NVIDIA_GPU = Self(2)
    """Represents NVIDIA GPU vendor."""

    fn __eq__(self, other: Self) -> Bool:
        """Checks if two `Vendor` instances are equal.

        Args:
            other: The `Vendor` to compare with.

        Returns:
            True if vendors are equal, False otherwise.
        """
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        """Checks if two `Vendor` instances are not equal.

        Args:
            other: The `Vendor` to compare with.

        Returns:
            True if vendors are not equal, False otherwise.
        """
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        """Identity comparison for vendors.

        Args:
            other: The `Vendor` to compare with.

        Returns:
            True if vendors are identical, False otherwise.
        """
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        """Negative identity comparison for vendors.

        Args:
            other: The Vendor to compare with.

        Returns:
            True if vendors are not identical, False otherwise.
        """
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Writes vendor information to a writer.

        Parameters:
            W: The type of writer to use for output. Must implement the Writer trait.

        Args:
            writer: The writer to output vendor information to.
        """
        if self is Vendor.NO_GPU:
            writer.write("no_gpu")
            return
        if self is Vendor.AMD_GPU:
            writer.write("amd_gpu")
            return
        writer.write("nvidia_gpu")

    @no_inline
    fn __str__(self) -> String:
        """Returns a string representation of the vendor.

        Returns:
            String representation of the vendor.
        """
        return String.write(self)


# ===-----------------------------------------------------------------------===#
# NoGPU
# ===-----------------------------------------------------------------------===#


fn _get_empty_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an empty target configuration for when no GPU is available.

    Returns:
        An empty MLIR target configuration.
    """
    return __mlir_attr[
        `#kgen.target<triple = "", `,
        `arch = "", `,
        `features = "", `,
        `data_layout="",`,
        `simd_bit_width = 0,`,
        `index_bit_width = 0`,
        `> : !kgen.target`,
    ]


alias NoGPU = Info(
    name="NoGPU",
    vendor=Vendor.NO_GPU,
    api="none",
    arch_name="no_gpu",
    compile_options="",
    compute=0,
    version="",
    sm_count=0,
    warp_size=0,
    threads_per_sm=0,
    threads_per_warp=0,
    warps_per_multiprocessor=0,
    threads_per_multiprocessor=0,
    thread_blocks_per_multiprocessor=0,
    shared_memory_per_multiprocessor=0,
    register_file_size=0,
    register_allocation_unit_size=0,
    allocation_granularity="none",
    max_registers_per_thread=0,
    max_registers_per_block=0,
    max_blocks_per_multiprocessor=0,
    shared_memory_allocation_unit_size=0,
    warp_allocation_granularity=0,
    max_thread_block_size=0,
)

# ===-----------------------------------------------------------------------===#
# A100
# ===-----------------------------------------------------------------------===#

# Note: features = "+ptx81" means that the kernel should be compiled using
# PTX version 8.1. This must be less than or equal to the installed CUDA
# driver's maximum supported PTX version. Currently we hardcode this to
# PTX version 8.1 which means that you need to have a CUDA driver included with
# CUDA 12.5 toolkit. The mapping from CUDA Driver to PTX version can be found by
# looking at the PTX ISA in the versioned docs
# https://developer.nvidia.com/cuda-toolkit-archive.


fn _get_a100_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA A100 GPU.

    Returns:
        MLIR target configuration for A100.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx81,+sm_80", `,
        `tune_cpu = "sm_80", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


alias A100 = Info(
    name="A100",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ampere",
    compile_options="nvptx-short-ptr=true",
    compute=8.0,
    version="sm_80",
    sm_count=108,
    warp_size=32,
    threads_per_sm=2048,
    threads_per_warp=32,
    warps_per_multiprocessor=64,
    threads_per_multiprocessor=2048,
    thread_blocks_per_multiprocessor=32,
    shared_memory_per_multiprocessor=167936,
    register_file_size=65536,
    register_allocation_unit_size=256,
    allocation_granularity="warp",
    max_registers_per_thread=255,
    max_registers_per_block=65536,
    max_blocks_per_multiprocessor=32,
    shared_memory_allocation_unit_size=128,
    warp_allocation_granularity=4,
    max_thread_block_size=1024,
)

# ===-----------------------------------------------------------------------===#
# A10
# ===-----------------------------------------------------------------------===#


fn _get_a10_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA A10 GPU.

    Returns:
        MLIR target configuration for A10.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx81,+sm_86", `,
        `tune_cpu = "sm_86", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


alias A10 = Info(
    name="A10",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ampere",
    compile_options="nvptx-short-ptr=true",
    compute=8.6,
    version="sm_86",
    sm_count=72,
    warp_size=32,
    threads_per_sm=1536,
    threads_per_warp=32,
    warps_per_multiprocessor=64,
    threads_per_multiprocessor=2048,
    thread_blocks_per_multiprocessor=32,
    shared_memory_per_multiprocessor=102400,
    register_file_size=65536,
    register_allocation_unit_size=256,
    allocation_granularity="warp",
    max_registers_per_thread=255,
    max_registers_per_block=65536,
    max_blocks_per_multiprocessor=16,
    shared_memory_allocation_unit_size=128,
    warp_allocation_granularity=4,
    max_thread_block_size=1024,
)

# ===-----------------------------------------------------------------------===#
# Jetson Orin Nano
# ===-----------------------------------------------------------------------===#


fn _get_orin_nano_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA Jetson Orin Nano GPU.

    Returns:
        MLIR target configuration for Orin Nano.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_87", `,
        `features = "+ptx81,+sm_87", `,
        `tune_cpu = "sm_87", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


alias OrinNano = Info(
    name="Orin Nano",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ampere",
    compile_options="nvptx-short-ptr=true",
    compute=8.7,
    version="sm_87",
    sm_count=8,
    warp_size=32,
    threads_per_sm=1536,
    threads_per_warp=32,
    warps_per_multiprocessor=64,
    threads_per_multiprocessor=2048,
    thread_blocks_per_multiprocessor=32,
    shared_memory_per_multiprocessor=167936,
    register_file_size=65536,
    register_allocation_unit_size=256,
    allocation_granularity="warp",
    max_registers_per_thread=255,
    max_registers_per_block=65536,
    max_blocks_per_multiprocessor=16,
    shared_memory_allocation_unit_size=128,
    warp_allocation_granularity=4,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# L4
# ===-----------------------------------------------------------------------===#


fn _get_l4_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA L4 GPU.

    Returns:
        MLIR target configuration for L4.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_89", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


alias L4 = Info(
    name="L4",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ada",
    compile_options="nvptx-short-ptr=true",
    compute=8.9,
    version="sm_89",
    sm_count=58,
    warp_size=32,
    threads_per_sm=1536,
    threads_per_warp=32,
    warps_per_multiprocessor=64,
    threads_per_multiprocessor=2048,
    thread_blocks_per_multiprocessor=32,
    shared_memory_per_multiprocessor=102400,
    register_file_size=65536,
    register_allocation_unit_size=256,
    allocation_granularity="warp",
    max_registers_per_thread=255,
    max_registers_per_block=65536,
    max_blocks_per_multiprocessor=24,
    shared_memory_allocation_unit_size=128,
    warp_allocation_granularity=4,
    max_thread_block_size=1024,
)

# ===-----------------------------------------------------------------------===#
# H100
# ===-----------------------------------------------------------------------===#


fn _get_h100_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA H100 GPU.

    Returns:
        MLIR target configuration for H100.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90a", `,
        `features = "+ptx85,+sm_90a", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
alias H100 = Info(
    name="H100",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="hopper",
    compile_options="nvptx-short-ptr=true",
    compute=9.0,
    version="sm_90a",
    sm_count=132,
    warp_size=32,
    threads_per_sm=-1,
    threads_per_warp=32,
    warps_per_multiprocessor=64,
    threads_per_multiprocessor=2048,
    thread_blocks_per_multiprocessor=32,
    shared_memory_per_multiprocessor=228 * _KB,
    register_file_size=65536,
    register_allocation_unit_size=256,
    allocation_granularity="warp",
    max_registers_per_thread=255,
    max_registers_per_block=65536,
    max_blocks_per_multiprocessor=32,
    shared_memory_allocation_unit_size=128,
    warp_allocation_granularity=4,
    max_thread_block_size=1024,
)

# ===-----------------------------------------------------------------------===#
# MI300X
# ===-----------------------------------------------------------------------===#


fn _get_mi300x_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for AMD MI300X GPU.

    Returns:
        MLIR target configuration for MI300X.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx942", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


alias MI300X = Info(
    name="MI300X",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx942",
    compile_options=(
        "amdhsa-code-object-version=5,greedy-reverse-local-assignment=true"
    ),
    compute=9.4,
    version="CDNA3",
    sm_count=304,
    warp_size=64,
    threads_per_sm=2048,
    threads_per_warp=64,
    warps_per_multiprocessor=32,  # 2048 threads per sm / 64 threads per warp = 32 warps per sm
    threads_per_multiprocessor=2048,
    thread_blocks_per_multiprocessor=2,
    shared_memory_per_multiprocessor=65536,
    register_file_size=65536,
    register_allocation_unit_size=256,
    allocation_granularity="warp",
    max_registers_per_thread=255,
    max_registers_per_block=65536,
    max_blocks_per_multiprocessor=2,
    shared_memory_allocation_unit_size=128,
    warp_allocation_granularity=4,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# Info
# ===-----------------------------------------------------------------------===#


@value
@register_passable
struct Info(Writable):
    """
    Comprehensive information about a GPU architecture.

    This struct contains detailed specifications about GPU capabilities,
    including compute units, memory, thread organization, and performance
    characteristics.
    """

    var name: StaticString
    """The model name of the GPU."""

    var vendor: Vendor
    """The vendor/manufacturer of the GPU (e.g., NVIDIA, AMD)."""

    var api: StaticString
    """The graphics/compute API supported by the GPU (e.g., CUDA, ROCm)."""

    var arch_name: StaticString
    """The architecture name of the GPU (e.g., sm_80, gfx942)."""

    var compile_options: StaticString
    """Compiler options specific to this GPU architecture."""

    var compute: Float32
    """Compute capability version number for NVIDIA GPUs."""

    var version: StaticString
    """Version string of the GPU architecture."""

    var sm_count: Int
    """Number of streaming multiprocessors (SMs) on the GPU."""

    var warp_size: Int
    """Number of threads in a warp/wavefront."""

    var threads_per_sm: Int
    """Maximum number of threads per streaming multiprocessor."""

    var threads_per_warp: Int
    """Number of threads that execute together in a warp/wavefront."""

    var warps_per_multiprocessor: Int
    """Maximum number of warps that can be active on a multiprocessor."""

    var threads_per_multiprocessor: Int
    """Maximum number of threads that can be active on a multiprocessor."""

    var thread_blocks_per_multiprocessor: Int
    """Maximum number of thread blocks that can be active on a multiprocessor."""

    var shared_memory_per_multiprocessor: Int
    """Size of shared memory available per multiprocessor in bytes."""

    var register_file_size: Int
    """Total size of the register file per multiprocessor in bytes."""

    var register_allocation_unit_size: Int
    """Minimum allocation size for registers in bytes."""

    var allocation_granularity: StaticString
    """Description of how resources are allocated on the GPU."""

    var max_registers_per_thread: Int
    """Maximum number of registers that can be allocated to a single thread."""

    var max_registers_per_block: Int
    """Maximum number of registers that can be allocated to a thread block."""

    var max_blocks_per_multiprocessor: Int
    """Maximum number of blocks that can be scheduled on a multiprocessor."""

    var shared_memory_allocation_unit_size: Int
    """Minimum allocation size for shared memory in bytes."""

    var warp_allocation_granularity: Int
    """Granularity at which warps are allocated resources."""

    var max_thread_block_size: Int
    """Maximum number of threads allowed in a thread block."""

    fn target(self) -> __mlir_type.`!kgen.target`:
        """
        Gets the MLIR target configuration for this GPU.

        Returns:
            MLIR target configuration for the GPU.
        """
        if self.name == "A100":
            return _get_a100_target()
        if self.name == "A10":
            return _get_a10_target()
        if self.name == "L4":
            return _get_l4_target()
        if self.name == "H100":
            return _get_h100_target()
        if self.name == "MI300X":
            return _get_mi300x_target()
        if self.name == "":
            return _get_empty_target()
        return _get_a100_target()

    @staticmethod
    fn from_target[target: __mlir_type.`!kgen.target`]() -> Self:
        """
        Creates an Info instance from an MLIR target.

        Parameters:
            target: MLIR target configuration.

        Returns:
            GPU info corresponding to the target.
        """
        return _get_info_from_target[_get_arch[target]()]()

    @staticmethod
    fn from_name[name: StaticString]() -> Self:
        """
        Creates an Info instance from a GPU architecture name.

        Parameters:
            name: GPU architecture name (e.g., "sm_80", "gfx942").

        Returns:
            GPU info corresponding to the architecture name.
        """
        return _get_info_from_target[name]()

    fn _warps_per_block(self, threads_per_block: Int) -> Int:
        """
        Calculates the number of warps per thread block.

        Args:
            threads_per_block: Number of threads in a block.

        Returns:
            Number of warps needed for the specified threads.
        """
        return ceildiv(threads_per_block, self.threads_per_warp)

    fn _registers_per_warp(self, registers_per_thread: Int) -> Int:
        """
        Calculates the total registers used by a warp.

        Args:
            registers_per_thread: Number of registers per thread.

        Returns:
            Total registers used by a warp, aligned to allocation unit.
        """
        return _quantized_ceil(
            registers_per_thread * self.threads_per_warp,
            self.register_allocation_unit_size,
        )

    fn _registers_per_block(
        self, threads_per_block: Int, registers_per_thread: Int
    ) -> Int:
        """
        Calculates the total registers used by a thread block.

        Args:
            threads_per_block: Number of threads in a block.
            registers_per_thread: Number of registers per thread.

        Returns:
            Total registers used by the thread block.
        """
        return self._registers_per_warp(
            registers_per_thread
        ) * self._warps_per_block(threads_per_block)

    fn _warps_per_multiprocessor_register_limited(
        self, registers_per_thread: Int
    ) -> Int:
        """
        Calculates max warps per SM based on register constraints.

        Args:
            registers_per_thread: Number of registers per thread.

        Returns:
            Maximum number of warps per SM limited by register usage.
        """
        return _quantized_floor(
            self.max_registers_per_block
            / self._registers_per_warp(registers_per_thread),
            self.warp_allocation_granularity,
        )

    fn _blocks_per_multiprocessor_register_limited(
        self, *, threads_per_block: Int, registers_per_thread: Int
    ) -> Int:
        """
        Calculates max blocks per SM based on register constraints.

        Args:
            threads_per_block: Number of threads in a block.
            registers_per_thread: Number of registers per thread.

        Returns:
            Maximum number of blocks per SM limited by register usage.
        """
        return Int(
            self._warps_per_multiprocessor_register_limited(
                registers_per_thread
            )
            / self._warps_per_block(threads_per_block)
        ) * Int(self.register_file_size / self.max_registers_per_block)

    fn _block_runtime_shared_memory(self) -> Int:
        """
        Calculates shared memory used by the CUDA runtime per block.

        Returns:
            Amount of shared memory used by the runtime in bytes.
        """
        if self.compute > 8:
            # starting with Compute Capability 8.x, the CUDA runtime consumes
            # 1KB of shared memory the amount might change depending on the
            # CUDA runtime version in the future.
            return 1024
        return 0

    fn _block_shared_memory(self, *, shared_memory_per_block: Int) -> Int:
        """
        Calculates total shared memory needed per block.

        Args:
            shared_memory_per_block: User-requested shared memory per block.

        Returns:
            Total shared memory needed per block, aligned to allocation unit.
        """
        return ceildiv(
            shared_memory_per_block + self._block_runtime_shared_memory(),
            self.shared_memory_allocation_unit_size,
        )

    fn _thread_blocks_per_multiprocessor_limited_by_warps_or_blocks_per_multiprocessor(
        self, threads_per_block: Int
    ) -> Float64:
        """
        Calculates max blocks per SM based on warp and block limits.

        Args:
            threads_per_block: Number of threads in a block.

        Returns:
            Maximum number of blocks per SM, limited by either warps or blocks.
        """
        return min(
            self.thread_blocks_per_multiprocessor,
            floor(
                self.warps_per_multiprocessor
                / self._warps_per_block(threads_per_block),
            ),
        )

    fn _warps_per_multiprocessor_limited_by_registers(
        self, registers_per_thread: Int
    ) -> Int:
        """
        Calculates maximum warps per multiprocessor limited by register usage.

        Determines how many warps can fit in a multiprocessor based on the
        register requirements, quantized to allocation granularity.

        Args:
            registers_per_thread: Number of registers used by each thread.

        Returns:
            Maximum number of warps per multiprocessor limited by registers.
        """
        return _quantized_floor(
            self.max_registers_per_block
            / self._registers_per_warp(registers_per_thread),
            self.warp_allocation_granularity,
        )

    fn _thread_blocks_per_multiprocessor_limited_by_registers_per_multiprocessor(
        self, *, threads_per_block: Int, registers_per_thread: Int
    ) -> Float64:
        """
        Calculates maximum blocks per SM limited by register availability.

        Determines how many thread blocks can fit in a streaming multiprocessor
        based on register usage constraints.

        Args:
            threads_per_block: Number of threads in each block.
            registers_per_thread: Number of registers used by each thread.

        Returns:
            Maximum number of blocks per SM limited by register constraints.
        """
        if registers_per_thread > self.max_registers_per_thread:
            return 0
        if registers_per_thread > 0:
            return floor(
                self._warps_per_multiprocessor_limited_by_registers(
                    registers_per_thread
                )
                / self._warps_per_block(threads_per_block)
            ) * floor(self.register_file_size / self.max_registers_per_block)
        return self.thread_blocks_per_multiprocessor

    fn occupancy(
        self, *, threads_per_block: Int, registers_per_thread: Int
    ) -> Float64:
        """
        Calculates theoretical occupancy for given thread and register config.

        Occupancy represents the ratio of active warps to the maximum possible
        warps on a streaming multiprocessor.

        Args:
            threads_per_block: Number of threads in each block.
            registers_per_thread: Number of registers used by each thread.

        Returns:
            Occupancy as a ratio between 0.0 and 1.0.

        Note:
            TODO (KERN-795): Add occupancy calculation based on shared memory
            usage and thread block size and take use the minimum value.
        """
        return (
            self._blocks_per_multiprocessor_register_limited(
                threads_per_block=threads_per_block,
                registers_per_thread=registers_per_thread,
            )
            * self._warps_per_block(threads_per_block)
            / self.warps_per_multiprocessor
        )

    fn __lt__(self, other: Self) -> Bool:
        """
        Compares if this GPU has lower compute capability than another.

        Args:
            other: Another GPU Info instance to compare against.

        Returns:
            True if this GPU has lower compute capability, False otherwise.
        """
        debug_assert(
            self.vendor == other.vendor,
            "the vendors must be the same to perform the comparison",
        )
        return self.compute < other.compute

    fn __le__(self, other: Self) -> Bool:
        """
        Compares if this GPU has lower or equal compute capability.

        Args:
            other: Another GPU Info instance to compare against.

        Returns:
            True if this GPU has lower or equal compute capability.
        """
        debug_assert(
            self.vendor == other.vendor,
            "the vendors must be the same to perform the comparison",
        )
        return self.compute <= other.compute

    fn __gt__(self, other: Self) -> Bool:
        """
        Compares if this GPU has higher compute capability than another.

        Args:
            other: Another GPU Info instance to compare against.

        Returns:
            True if this GPU has higher compute capability, False otherwise.
        """
        debug_assert(
            self.vendor == other.vendor,
            "the vendors must be the same to perform the comparison",
        )
        return self.compute > other.compute

    fn __ge__(self, other: Self) -> Bool:
        """
        Compares if this GPU has higher or equal compute capability.

        Args:
            other: Another GPU Info instance to compare against.

        Returns:
            True if this GPU has higher or equal compute capability.
        """
        debug_assert(
            self.vendor == other.vendor,
            "the vendors must be the same to perform the comparison",
        )
        return self.compute >= other.compute

    fn __eq__(self, other: Self) -> Bool:
        """
        Checks if two GPU Info instances represent the same GPU model.

        Args:
            other: Another GPU Info instance to compare against.

        Returns:
            True if both instances represent the same GPU model.
        """
        return self.name == other.name

    fn __ne__(self, other: Self) -> Bool:
        """
        Checks if two GPU Info instances represent different GPU models.

        Args:
            other: Another GPU Info instance to compare against.

        Returns:
            True if instances represent different GPU models.
        """
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        """
        Identity comparison operator for GPU Info instances.

        Args:
            other: Another GPU Info instance to compare against.

        Returns:
            True if both instances represent the same GPU model.
        """
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        """
        Negative identity comparison operator for GPU Info instances.

        Args:
            other: Another GPU Info instance to compare against.

        Returns:
            True if instances represent different GPU models.
        """
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Writes GPU information to a writer.

        Outputs all GPU specifications and capabilities to the provided writer
        in a human-readable format.

        Parameters:
            W: The type of writer to use for output. Must implement the Writer trait.

        Args:
            writer: A Writer instance to output the GPU information.
        """
        writer.write("name: ", self.name, "\n")
        writer.write("vendor: ", self.vendor, "\n")
        writer.write("api: ", self.api, "\n")
        writer.write("arch_name: ", self.arch_name, "\n")
        writer.write("compile_options: ", self.compile_options, "\n")
        writer.write("compute: ", self.compute, "\n")
        writer.write("version: ", self.version, "\n")
        writer.write("sm_count: ", self.sm_count, "\n")
        writer.write("warp_size: ", self.warp_size, "\n")
        writer.write("threads_per_sm: ", self.threads_per_sm, "\n")
        writer.write("threads_per_warp: ", self.threads_per_warp, "\n")
        writer.write(
            "warps_per_multiprocessor: ", self.warps_per_multiprocessor, "\n"
        )
        writer.write(
            "threads_per_multiprocessor: ",
            self.threads_per_multiprocessor,
            "\n",
        )
        writer.write(
            "thread_blocks_per_multiprocessor: ",
            self.thread_blocks_per_multiprocessor,
            "\n",
        )
        writer.write(
            "shared_memory_per_multiprocessor: ",
            self.shared_memory_per_multiprocessor,
            "\n",
        )
        writer.write(
            "register_file_size: ",
            self.register_file_size,
            "\n",
        )
        writer.write(
            "register_allocation_unit_size: ",
            self.register_allocation_unit_size,
            "\n",
        )
        writer.write(
            "allocation_granularity: ", self.allocation_granularity, "\n"
        )
        writer.write(
            "max_registers_per_thread: ", self.max_registers_per_thread, "\n"
        )
        writer.write(
            "max_registers_per_block: ", self.max_registers_per_block, "\n"
        )
        writer.write(
            "max_blocks_per_multiprocessor: ",
            self.max_blocks_per_multiprocessor,
            "\n",
        )
        writer.write(
            "shared_memory_allocation_unit_size: ",
            self.shared_memory_allocation_unit_size,
            "\n",
        )
        writer.write(
            "warp_allocation_granularity: ",
            self.warp_allocation_granularity,
            "\n",
        )
        writer.write(
            "max_thread_block_size: ", self.max_thread_block_size, "\n"
        )

    @no_inline
    fn __str__(self) -> String:
        """
        Returns a string representation of the GPU information.

        Converts all GPU specifications and capabilities to a human-readable
        string format.

        Returns:
            String containing all GPU information.
        """
        return String.write(self)


# ===-----------------------------------------------------------------------===#
# _get_info_from_target
# ===-----------------------------------------------------------------------===#


@always_inline
fn _get_info_from_compute_capability[compute_capability: Int]() -> Info:
    """
    Gets GPU Info for a specific compute capability (compile-time version).

    Maps compute capability numbers to corresponding GPU Info instances at
    compile time.

    Parameters:
        compute_capability: The compute capability as an integer.

    Returns:
        Info instance for the specified compute capability.
    """
    constrained[
        compute_capability in (0, 80, 86, 87, 89, 90, 94),
        "invalid compute capability",
    ]()

    @parameter
    if compute_capability == 0:
        return NoGPU
    if compute_capability == 80:
        return A100
    elif compute_capability == 86:
        return A10
    elif compute_capability == 87:
        return OrinNano
    elif compute_capability == 89:
        return L4
    elif compute_capability == 90:
        return H100
    elif compute_capability == 94:
        return MI300X
    return abort[Info]("invalid compute capability")


@always_inline
fn _get_info_from_compute_capability(compute_capability: Int) raises -> Info:
    """
    Gets GPU Info for a specific compute capability (runtime version).

    Maps compute capability numbers to corresponding GPU Info instances at
    runtime.

    Args:
        compute_capability: The compute capability as an integer.

    Returns:
        Info instance for the specified compute capability.
    """
    if compute_capability == 0:
        return _get_info_from_compute_capability[0]()
    if compute_capability == 80:
        return _get_info_from_compute_capability[80]()
    if compute_capability == 86:
        return _get_info_from_compute_capability[86]()
    if compute_capability == 87:
        return _get_info_from_compute_capability[87]()
    if compute_capability == 89:
        return _get_info_from_compute_capability[89]()
    if compute_capability == 90:
        return _get_info_from_compute_capability[90]()
    if compute_capability == 94:
        return _get_info_from_compute_capability[94]()

    raise "invalid compute capability"


@always_inline
fn _get_info_from_target[target_arch0: StaticString]() -> Info:
    """
    Gets GPU Info for a specific target architecture.

    Maps target architecture strings to corresponding GPU Info instances.

    Parameters:
        target_arch0: Target architecture string (e.g., "sm_80", "gfx942").

    Returns:
        Info instance for the specified target architecture.
    """
    alias target_arch = get_string_literal_slice[
        target_arch0.replace("sm_", "")
    ]()

    constrained[
        target_arch
        in (
            "cuda",
            "80",
            "86",
            "87",
            "89",
            "90",
            "90a",
            "nvidia:80",
            "nvidia:86",
            "nvidia:87",
            "nvidia:89",
            "nvidia:90",
            "nvidia:90a",
            "amdgpu:94",
            "mi300x",
            "gfx942",
            "",
        ),
        "the target architecture '",
        target_arch,
        "' is not valid",
    ]()

    @parameter
    if target_arch in ("80", "nvidia:80"):
        return A100
    elif target_arch in ("86", "nvidia:86"):
        return A10
    elif target_arch in ("87", "nvidia:87"):
        return OrinNano
    elif target_arch in ("89", "nvidia:89"):
        return L4
    elif target_arch in ("90", "90a", "nvidia:90", "nvidia:90a"):
        return H100
    elif target_arch in ("gfx942", "mi300x", "amdgpu:94"):
        return MI300X
    elif DEFAULT_GPU_ARCH == "":
        return NoGPU

    return _get_info_from_target[DEFAULT_GPU_ARCH]()


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


fn _quantized_ceil(a: Float64, b: Int) -> Int:
    """
    Rounds up a value to the nearest multiple of another value.

    Args:
        a: Value to round up.
        b: Quantization factor.

    Returns:
        Rounded up value that is a multiple of b.
    """
    return Int(ceildiv(a, b) * b)


fn _quantized_floor(a: Float64, b: Int) -> Int:
    """
    Rounds down a value to the nearest multiple of another value.

    Args:
        a: Value to round down.
        b: Quantization factor.

    Returns:
        Rounded down value that is a multiple of b.
    """
    return Int(floor(a / b) * b)


fn is_gpu[target: StringSlice]() -> Bool:
    """
    Checks if the target is a GPU (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is a GPU, False otherwise.
    """
    return is_gpu(target)


fn is_gpu(target: StringSlice) -> Bool:
    """
    Checks if the target is a GPU (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is a GPU, False otherwise.
    """
    return target == "gpu"


fn is_cpu[target: StringSlice]() -> Bool:
    """
    Checks if the target is a CPU (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is a CPU, False otherwise.
    """
    return is_cpu(target)


fn is_cpu(target: StringSlice) -> Bool:
    """
    Checks if the target is a CPU (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is a CPU, False otherwise.
    """
    return target == "cpu"


fn is_valid_target[target: StringSlice]() -> Bool:
    """
    Checks if the target is valid (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is valid (CPU or GPU), False otherwise.
    """
    return is_valid_target(target)


fn is_valid_target(target: StringSlice) -> Bool:
    """
    Checks if the target is valid (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is valid (CPU or GPU), False otherwise.
    """
    return is_gpu(target) or is_cpu(target)

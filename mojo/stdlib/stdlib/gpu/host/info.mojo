# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""
Contains information about GPU architectures and their capabilities.

This module provides detailed specifications for various GPU models including
NVIDIA and AMD GPUs. It includes information about compute capabilities,
memory specifications, thread organization, and performance characteristics.
"""

from math import ceildiv, floor
from os import abort
from sys.info import _accelerator_arch, _TargetType, CompilationTarget

alias _KB = 1024

# ===-----------------------------------------------------------------------===#
# Vendor
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable("trivial")
struct Vendor(Identifiable, Writable):
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

    alias APPLE_GPU = Self(3)
    """Represents Apple GPU vendor."""

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

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Writes vendor information to a writer.

        Args:
            writer: The writer to output vendor information to.
        """
        if self is Vendor.NO_GPU:
            writer.write("no_gpu")
            return
        if self is Vendor.AMD_GPU:
            writer.write("amd_gpu")
            return
        if self is Vendor.APPLE_GPU:
            writer.write("apple_gpu")
        if self is Vendor.NVIDIA_GPU:
            writer.write("nvidia_gpu")

        abort("unable to format unrecognized `Vendor` value")

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


fn _get_empty_target() -> _TargetType:
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


alias NoGPU = GPUInfo(
    name="NoGPU",
    vendor=Vendor.NO_GPU,
    api="none",
    arch_name="no_gpu",
    compute=0,
    version="",
    sm_count=0,
    warp_size=0,
    threads_per_sm=0,
    shared_memory_per_multiprocessor=0,
    max_registers_per_block=0,
    max_thread_block_size=0,
)


# ===-----------------------------------------------------------------------===#
# Apple M1
# ===-----------------------------------------------------------------------===#
fn _get_metal_m1_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for M1 Metal GPU.
    Returns:
        MLIR target configuration for M1 Metal.
    """

    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `arch = "apple-m1", `,
        `features = "", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_metal_m2_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for M2 Metal GPU.
    Returns:
        MLIR target configuration for M2 Metal.
    """

    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `arch = "apple-m2", `,
        `features = "", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_metal_m3_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for M3 Metal GPU.
    Returns:
        MLIR target configuration for M3 Metal.
    """

    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `arch = "apple-m3", `,
        `features = "", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_metal_m4_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for M4 Metal GPU.
    Returns:
        MLIR target configuration for M4 Metal.
    """

    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `arch = "apple-m4", `,
        `features = "", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


alias MetalM1 = GPUInfo(
    name="M1",
    vendor=Vendor.APPLE_GPU,
    api="metal",
    arch_name="apple-m1",
    compute=3.0,  # Metal version 3.0
    version="metal_3",
    sm_count=8,  # M1 has 8 GPU cores
    warp_size=32,  # Metal uses 32-thread SIMD groups (like warps)
    threads_per_sm=1024,  # Threads per compute unit
    shared_memory_per_multiprocessor=32768,  # 32KB shared memory per compute unit
    max_registers_per_block=65536,
    max_thread_block_size=1024,  # Max threads per threadgroup
)

alias MetalM2 = GPUInfo(
    name="M2",
    vendor=Vendor.APPLE_GPU,
    api="metal",
    arch_name="apple-m2",
    compute=3.0,  # Metal version 3.0
    version="metal_3",
    sm_count=10,  # M2 has 10 GPU cores
    warp_size=32,  # Metal uses 32-thread SIMD groups (like warps)
    threads_per_sm=1024,  # Threads per compute unit
    shared_memory_per_multiprocessor=32768,  # 32KB shared memory per compute unit
    max_registers_per_block=65536,
    max_thread_block_size=1024,  # Max threads per threadgroup
)

alias MetalM3 = GPUInfo(
    name="M3",
    vendor=Vendor.APPLE_GPU,
    api="metal",
    arch_name="apple-m3",
    compute=3.0,  # Metal version 3.0 for M3
    version="metal_3",
    sm_count=10,  # M3 has 10 GPU cores
    warp_size=32,  # Metal uses 32-thread SIMD groups (like warps)
    threads_per_sm=1024,  # Threads per compute unit
    shared_memory_per_multiprocessor=32768,  # 32KB shared memory per compute unit
    max_registers_per_block=65536,
    max_thread_block_size=1024,  # Max threads per threadgroup
)

alias MetalM4 = GPUInfo(
    name="M4",
    vendor=Vendor.APPLE_GPU,
    api="metal",
    arch_name="apple-m4",
    compute=4.0,  # Metal version 4.0 for M4
    version="metal_4",
    sm_count=10,  # M4 has 10 GPU cores
    warp_size=32,  # Metal uses 32-thread SIMD groups (like warps)
    threads_per_sm=1024,  # Threads per compute unit
    shared_memory_per_multiprocessor=32768,  # 32KB shared memory per compute unit
    max_registers_per_block=65536,
    max_thread_block_size=1024,  # Max threads per threadgroup
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


fn _get_a100_target() -> _TargetType:
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
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


alias A100 = GPUInfo(
    name="A100",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ampere",
    compute=8.0,
    version="sm_80",
    sm_count=108,
    warp_size=32,
    threads_per_sm=2048,
    shared_memory_per_multiprocessor=167936,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)

# ===-----------------------------------------------------------------------===#
# A10
# ===-----------------------------------------------------------------------===#


fn _get_a10_target() -> _TargetType:
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
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


alias A10 = GPUInfo(
    name="A10",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ampere",
    compute=8.6,
    version="sm_86",
    sm_count=72,
    warp_size=32,
    threads_per_sm=1536,
    shared_memory_per_multiprocessor=102400,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)

# ===-----------------------------------------------------------------------===#
# Jetson Orin Nano
# ===-----------------------------------------------------------------------===#


fn _get_orin_nano_target() -> _TargetType:
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
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


alias OrinNano = GPUInfo(
    name="Orin Nano",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ampere",
    compute=8.7,
    version="sm_87",
    sm_count=8,
    warp_size=32,
    threads_per_sm=1536,
    shared_memory_per_multiprocessor=167936,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# L4
# ===-----------------------------------------------------------------------===#


fn _get_l4_target() -> _TargetType:
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
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


alias L4 = GPUInfo(
    name="L4",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ada",
    compute=8.9,
    version="sm_89",
    sm_count=58,
    warp_size=32,
    threads_per_sm=1536,
    shared_memory_per_multiprocessor=102400,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)

# ===-----------------------------------------------------------------------===#
# RTX 4090 M
# ===-----------------------------------------------------------------------===#


fn _get_rtx4090m_target() -> _TargetType:
    """
    Creates an MLIR target configuration for NVIDIA RTX 4090 Mobile GPU.

    Returns:
        MLIR target configuration for H100.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


alias RTX4090m = GPUInfo(
    name="RTX4090m",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ada lovelace",
    compute=8.9,
    version="sm_89",
    sm_count=76,
    warp_size=32,
    threads_per_sm=-1,
    shared_memory_per_multiprocessor=102400,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)

# ===-----------------------------------------------------------------------===#
# RTX 4090
# ===-----------------------------------------------------------------------===#


fn _get_rtx4090_target() -> _TargetType:
    """
    Creates an MLIR target configuration for NVIDIA RTX 4090.

    Returns:
        MLIR target configuration for H100.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


alias RTX4090 = GPUInfo(
    name="RTX4090",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ada lovelace",
    compute=8.9,
    version="sm_89",
    sm_count=128,
    warp_size=32,
    threads_per_sm=-1,
    shared_memory_per_multiprocessor=102400,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# H100
# ===-----------------------------------------------------------------------===#


fn _get_h100_target() -> _TargetType:
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
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
alias H100 = GPUInfo(
    name="H100",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="hopper",
    compute=9.0,
    version="sm_90a",
    sm_count=132,
    warp_size=32,
    threads_per_sm=2048,
    shared_memory_per_multiprocessor=228 * _KB,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# B100
# ===-----------------------------------------------------------------------===#


fn _get_b100_target() -> _TargetType:
    """
    Creates an MLIR target configuration for NVIDIA B100 GPU.

    Returns:
        MLIR target configuration for B100.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_100a", `,
        `features = "+ptx88,+sm_100a", `,
        `tune_cpu = "sm_100a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://resources.nvidia.com/en-us-blackwell-architecture
# TODO: Update once we have B100 access.
alias B100 = GPUInfo(
    name="B100",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="blackwell",
    compute=10.0,
    version="sm_100a",
    sm_count=132,
    warp_size=32,
    threads_per_sm=-1,
    shared_memory_per_multiprocessor=256 * _KB,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)

alias B200 = GPUInfo(
    name="B200",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="blackwell",
    compute=10.0,
    version="sm_100a",
    sm_count=148,
    warp_size=32,
    threads_per_sm=-1,
    shared_memory_per_multiprocessor=228 * _KB,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)

# ===-----------------------------------------------------------------------===#
# RTX5090
# ===-----------------------------------------------------------------------===#


fn _get_rtx5090_target() -> _TargetType:
    """
    Creates an MLIR target configuration for NVIDIA RTX5090 GPU.

    Returns:
        MLIR target configuration for RTX5090.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_120a", `,
        `features = "+ptx86,+sm_120a", `,
        `tune_cpu = "sm_120a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
alias RTX5090 = GPUInfo(
    name="RTX5090",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="blackwell",
    compute=12.0,
    version="sm_120a",
    sm_count=170,
    warp_size=32,
    threads_per_sm=-1,
    shared_memory_per_multiprocessor=58 * _KB,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# RTX3090
# ===-----------------------------------------------------------------------===#


fn _get_rtx3090_target() -> _TargetType:
    """
    Creates an MLIR target configuration for NVIDIA GeForce RTX 3090

    Returns:
        MLIR target configuration for NVIDIA GeForce RTX 3090.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx63,+sm_86", `,
        `tune_cpu = "sm_86", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/
alias RTX3090 = GPUInfo(
    name="NVIDIA GeForce RTX 3090",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ampere",
    compute=8.6,
    version="sm_86",
    sm_count=82,
    warp_size=32,
    threads_per_sm=-1,
    shared_memory_per_multiprocessor=102400,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# GTX1080Ti
# ===-----------------------------------------------------------------------===#


fn _get_gtx1080ti_target() -> _TargetType:
    """
    Creates an MLIR target configuration for NVIDIA GTX 1080 Ti GPU.
    Returns:
        MLIR target configuration for GTX 1080 Ti.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_61", `,
        `features = "+ptx50,+sm_61", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


alias GTX1080Ti = GPUInfo(
    name="NVIDIA GeForce GTX 1080 Ti",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="pascal",
    compute=6.1,
    version="sm_61",
    sm_count=28,
    warp_size=32,
    threads_per_sm=2048,
    shared_memory_per_multiprocessor=98304,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# Tesla P100
# ===-----------------------------------------------------------------------===#


fn _get_teslap100_target() -> _TargetType:
    """
    Creates an MLIR target configuration for NVIDIA Tesla P100 GPU.

    Returns:
        MLIR target configuration for Tesla P100.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_60", `,
        `features = "+ptx50,+sm_60", `,
        `tune_cpu = "sm_60", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


alias TeslaP100 = GPUInfo(
    name="NVIDIA Tesla P100",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="pascal",
    compute=6.0,
    version="sm_60",
    sm_count=56,
    warp_size=32,
    threads_per_sm=2048,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# RTX2060
# ===-----------------------------------------------------------------------===#


fn _get_rtx2060_target() -> _TargetType:
    """
    Creates an MLIR target configuration for NVIDIA RTX 2060 GPU.

    Returns:
        MLIR target configuration for RTX 2060.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_75", `,
        `features = "+ptx63,+sm_75", `,
        `tune_cpu = "sm_75", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


alias RTX2060 = GPUInfo(
    name="RTX2060",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="turing",
    compute=7.5,
    version="sm_75",
    sm_count=30,
    warp_size=32,
    threads_per_sm=2048,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=32768,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# MI300X
# ===-----------------------------------------------------------------------===#


fn _get_mi300x_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD MI300X GPU.

    Returns:
        MLIR target configuration for MI300X.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx942", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


alias MI300X = GPUInfo(
    name="MI300X",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx942",
    compute=9.4,
    version="CDNA3",
    sm_count=304,
    warp_size=64,
    threads_per_sm=2048,
    shared_memory_per_multiprocessor=65536,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# MI355X
# ===-----------------------------------------------------------------------===#


fn _get_mi355x_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD MI355X GPU.

    Returns:
        MLIR target configuration for MI355X.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx950", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


alias MI355X = GPUInfo(
    name="MI355X",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx950",
    compute=9.5,
    version="CDNA4",
    sm_count=256,
    warp_size=64,
    threads_per_sm=2048,
    shared_memory_per_multiprocessor=160 * _KB,
    max_registers_per_block=65536,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# Radeon 7xxx, 9xxx, 780m
# ===-----------------------------------------------------------------------===#


fn _get_9070_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD Radeon 9070 GPU.

    Returns:
        MLIR target configuration for 9070.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1201", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_9060_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD Radeon 9060 GPU.

    Returns:
        MLIR target configuration for 9060.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1200", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_7900_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD Radeon 7900 GPU.

    Returns:
        MLIR target configuration for 7900.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1100", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_7800_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD Radeon 7800/7700 GPU.

    Returns:
        MLIR target configuration for 7800/7700.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1101", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_7600_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD Radeon 7600 GPU.

    Returns:
        MLIR target configuration for 7600.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1102", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_6900_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD Radeon 6900 GPU.

    Returns:
        MLIR target configuration for 6900.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1030", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_780m_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD Radeon 780m GPU.

    Returns:
        MLIR target configuration for 780m.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1103", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_880m_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD Radeon 880M GPU.

    Returns:
        MLIR target configuration for 880M.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1150", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_8060s_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD Radeon 8060S GPU.

    Returns:
        MLIR target configuration for 8060S.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1151", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_860m_target() -> _TargetType:
    """
    Creates an MLIR target configuration for AMD Radeon 860M GPU.

    Returns:
        MLIR target configuration for 860M.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1152", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


alias Radeon9070 = GPUInfo(
    name="Radeon 9070",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1201",
    compute=12.0,
    version="RDNA4",
    sm_count=64,
    warp_size=32,
    threads_per_sm=1024,
    shared_memory_per_multiprocessor=32768,
    max_registers_per_block=32768,
    max_thread_block_size=1024,
)

alias Radeon9060 = GPUInfo(
    name="Radeon 9060",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1200",
    compute=12.0,
    version="RDNA4",
    sm_count=32,
    warp_size=32,
    threads_per_sm=1024,
    shared_memory_per_multiprocessor=32768,
    max_registers_per_block=32768,
    max_thread_block_size=1024,
)

alias Radeon7900 = GPUInfo(
    name="Radeon 7900",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1100",
    compute=11.0,
    version="RDNA3",
    sm_count=96,
    warp_size=32,
    threads_per_sm=1024,
    shared_memory_per_multiprocessor=32768,
    max_registers_per_block=32768,
    max_thread_block_size=1024,
)

alias Radeon7800 = GPUInfo(
    name="Radeon 7800/7700",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1101",
    compute=11.0,
    version="RDNA3",
    sm_count=60,
    warp_size=32,
    threads_per_sm=1024,
    shared_memory_per_multiprocessor=32768,
    max_registers_per_block=32768,
    max_thread_block_size=1024,
)

alias Radeon7600 = GPUInfo(
    name="Radeon 7600",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1102",
    compute=11.0,
    version="RDNA3",
    sm_count=32,
    warp_size=32,
    threads_per_sm=1024,
    shared_memory_per_multiprocessor=32768,
    max_registers_per_block=32768,
    max_thread_block_size=1024,
)

alias Radeon6900 = GPUInfo(
    name="Radeon 6900",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1102",
    compute=10.3,
    version="RDNA2",
    sm_count=60,
    warp_size=32,
    threads_per_sm=1024,
    shared_memory_per_multiprocessor=32768,
    max_registers_per_block=32768,
    max_thread_block_size=1024,
)


alias Radeon780m = GPUInfo(
    name="Radeon 780M",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1103",
    compute=11.0,
    version="RDNA3",
    sm_count=12,
    warp_size=32,
    threads_per_sm=1024,
    shared_memory_per_multiprocessor=32768,
    max_registers_per_block=32768,
    max_thread_block_size=1024,
)

alias Radeon880m = GPUInfo(
    name="Radeon 880M",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1150",
    compute=11.5,
    version="RDNA3.5",
    sm_count=12,
    warp_size=32,
    threads_per_sm=1024,
    shared_memory_per_multiprocessor=32768,
    max_registers_per_block=32768,
    max_thread_block_size=1024,
)

alias Radeon8060s = GPUInfo(
    name="Radeon 8060S",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1151",
    compute=11.5,
    version="RDNA3.5",
    sm_count=40,
    warp_size=32,
    threads_per_sm=1024,
    shared_memory_per_multiprocessor=32768,
    max_registers_per_block=32768,
    max_thread_block_size=1024,
)

alias Radeon860m = GPUInfo(
    name="Radeon 860M",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1152",
    compute=11.5,
    version="RDNA3.5",
    sm_count=8,
    warp_size=32,
    threads_per_sm=1024,
    shared_memory_per_multiprocessor=32768,
    max_registers_per_block=32768,
    max_thread_block_size=1024,
)


# ===-----------------------------------------------------------------------===#
# GPUInfo
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable
struct GPUInfo(Identifiable, Stringable, Writable):
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

    var shared_memory_per_multiprocessor: Int
    """Size of shared memory available per multiprocessor in bytes."""

    var max_registers_per_block: Int
    """Maximum number of registers that can be allocated to a thread block."""

    var max_thread_block_size: Int
    """Maximum number of threads allowed in a thread block."""

    fn target(self) -> _TargetType:
        """
        Gets the MLIR target configuration for this GPU.

        Returns:
            MLIR target configuration for the GPU.
        """
        if self.name == "NVIDIA Tesla P100":
            return _get_teslap100_target()
        if self.name == "NVIDIA GeForce GTX 1080 Ti":
            return _get_gtx1080ti_target()
        if self.name == "RTX2060":
            return _get_rtx2060_target()
        if self.name == "NVIDIA GeForce RTX 3090":
            return _get_rtx3090_target()
        if self.name == "A100":
            return _get_a100_target()
        if self.name == "A10":
            return _get_a10_target()
        if self.name == "L4":
            return _get_l4_target()
        if self.name == "RTX4090m":
            return _get_rtx4090m_target()
        if self.name == "RTX4090":
            return _get_rtx4090_target()
        if self.name == "H100":
            return _get_h100_target()
        if self.name == "B100" or self.name == "B200":
            return _get_b100_target()
        if self.name == "RTX5090":
            return _get_rtx5090_target()
        if self.name == "MI300X":
            return _get_mi300x_target()
        if self.name == "MI355X":
            return _get_mi355x_target()
        if self.name == "Radeon 780M":
            return _get_780m_target()
        if self.name == "Radeon 880M":
            return _get_880m_target()
        if self.name == "Radeon 8060S":
            return _get_8060s_target()
        if self.name == "Radeon 860M":
            return _get_860m_target()
        if self.name == "Radeon 6900":
            return _get_6900_target()
        if self.name == "Radeon 7900":
            return _get_7900_target()
        if self.name == "Radeon 7800/7700":
            return _get_7800_target()
        if self.name == "Radeon 7600":
            return _get_7600_target()
        if self.name == "Radeon 9070":
            return _get_9070_target()
        if self.name == "Radeon 9060":
            return _get_9060_target()
        if self.name == "M1":
            return _get_metal_m1_target()
        if self.name == "M2":
            return _get_metal_m2_target()
        if self.name == "M3":
            return _get_metal_m3_target()
        if self.name == "M4":
            return _get_metal_m4_target()

        if self.name == "":
            return _get_empty_target()
        return _get_a100_target()

    @staticmethod
    fn from_target[target: _TargetType]() -> Self:
        """
        Creates a `GPUInfo` instance from an MLIR target.

        Parameters:
            target: MLIR target configuration.

        Returns:
            GPU info corresponding to the target.
        """
        return _get_info_from_target[CompilationTarget[target]._arch()]()

    @staticmethod
    fn from_name[name: StaticString]() -> Self:
        """
        Creates a `GPUInfo` instance from a GPU architecture name.

        Parameters:
            name: GPU architecture name (e.g., "sm_80", "gfx942").

        Returns:
            GPU info corresponding to the architecture name.
        """
        return _get_info_from_target[name]()

    fn __lt__(self, other: Self) -> Bool:
        """
        Compares if this GPU has lower compute capability than another.

        Args:
            other: Another `GPUInfo` instance to compare against.

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
            other: Another `GPUInfo` instance to compare against.

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
            other: Another `GPUInfo` instance to compare against.

        Returns:
            True if this GPU has higher compute capability, False otherwise.
        """
        if self.vendor != other.vendor:
            return False
        return self.compute > other.compute

    fn __ge__(self, other: Self) -> Bool:
        """
        Compares if this GPU has higher or equal compute capability.

        Args:
            other: Another `GPUInfo` instance to compare against.

        Returns:
            True if this GPU has higher or equal compute capability.
        """
        if self.vendor != other.vendor:
            return False
        return self.compute >= other.compute

    fn __eq__(self, other: Self) -> Bool:
        """
        Checks if two `GPUInfo` instances represent the same GPU model.

        Args:
            other: Another `GPUInfo` instance to compare against.

        Returns:
            True if both instances represent the same GPU model.
        """
        return self.name == other.name

    fn __ne__(self, other: Self) -> Bool:
        """
        Checks if two `GPUInfo` instances represent different GPU models.

        Args:
            other: Another `GPUInfo` instance to compare against.

        Returns:
            True if instances represent different GPU models.
        """
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        """
        Identity comparison operator for `GPUInfo` instances.

        Args:
            other: Another `GPUInfo` instance to compare against.

        Returns:
            True if both instances represent the same GPU model.
        """
        return self == other

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """
        Writes GPU information to a writer.

        Outputs all GPU specifications and capabilities to the provided writer
        in a human-readable format.

        Args:
            writer: A Writer instance to output the GPU information.
        """
        writer.write("name: ", self.name, "\n")
        writer.write("vendor: ", self.vendor, "\n")
        writer.write("api: ", self.api, "\n")
        writer.write("arch_name: ", self.arch_name, "\n")
        writer.write("compute: ", self.compute, "\n")
        writer.write("version: ", self.version, "\n")
        writer.write("sm_count: ", self.sm_count, "\n")
        writer.write("warp_size: ", self.warp_size, "\n")
        writer.write("threads_per_sm: ", self.threads_per_sm, "\n")
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
fn _get_info_from_target[target_arch0: StaticString]() -> GPUInfo:
    """
    Gets `GPUInfo` for a specific target architecture.

    Maps target architecture strings to corresponding `GPUInfo` instances.

    Parameters:
        target_arch0: Target architecture string (e.g., "sm_80", "gfx942").

    Returns:
        `GPUInfo` instance for the specified target architecture.
    """
    alias target_arch = target_arch0.replace("sm_", "").replace(
        "nvidia:", ""
    ).replace("amdgpu:", "").replace("metal:", "apple-m")

    constrained[
        StaticString(target_arch)
        in (
            # NVIDIA
            StaticString("cuda"),
            StaticString("60"),
            StaticString("61"),
            StaticString("75"),
            StaticString("80"),
            StaticString("86"),
            StaticString("87"),
            StaticString("89"),
            StaticString("90"),
            StaticString("90a"),
            StaticString("100"),
            StaticString("100a"),
            StaticString("120"),
            StaticString("120a"),
            # AMD
            StaticString("mi300x"),
            StaticString("mi355x"),
            StaticString("gfx942"),
            StaticString("gfx950"),
            StaticString("gfx1030"),
            StaticString("gfx1100"),
            StaticString("gfx1101"),
            StaticString("gfx1102"),
            StaticString("gfx1103"),
            StaticString("gfx1150"),
            StaticString("gfx1151"),
            StaticString("gfx1152"),
            StaticString("gfx1200"),
            StaticString("gfx1201"),
            # Apple
            StaticString("apple-m1"),
            StaticString("apple-m2"),
            StaticString("apple-m3"),
            StaticString("apple-m4"),
        ),
        "the target architecture '",
        target_arch,
        "' is not valid",
    ]()

    @parameter
    if target_arch == "61":
        return materialize[GTX1080Ti]()
    elif target_arch == "75":
        return materialize[RTX2060]()
    elif target_arch == "80":
        return materialize[A100]()
    elif target_arch == "86":
        return materialize[A10]()
    elif target_arch == "87":
        return materialize[OrinNano]()
    elif target_arch == "89":
        return materialize[L4]()
    elif target_arch == "90" or target_arch == "90a":
        return materialize[H100]()
    elif target_arch == "100" or target_arch == "100a":
        # FIXME (KERN-1814): Unlike H100 and H200, blackwell devices (B100 vs B200)
        # architecture wise are different. We need to differentiate between them here.
        return materialize[B200]()
    elif target_arch == "120" or target_arch == "120a":
        return materialize[RTX5090]()
    elif target_arch == "gfx942" or target_arch == "mi300x":
        return materialize[MI300X]()
    elif target_arch == "gfx950" or target_arch == "mi355x":
        return materialize[MI355X]()
    elif target_arch == "gfx1030":
        return materialize[Radeon6900]()
    elif target_arch == "gfx1100":
        return materialize[Radeon7900]()
    elif target_arch == "gfx1101":
        return materialize[Radeon7800]()
    elif target_arch == "gfx1102":
        return materialize[Radeon7600]()
    elif target_arch == "gfx1103":
        return materialize[Radeon780m]()
    elif target_arch == "gfx1150":
        return materialize[Radeon880m]()
    elif target_arch == "gfx1151":
        return materialize[Radeon8060s]()
    elif target_arch == "gfx1152":
        return materialize[Radeon860m]()
    elif target_arch == "gfx1200":
        return materialize[Radeon9060]()
    elif target_arch == "gfx1201":
        return materialize[Radeon9070]()
    elif target_arch == "apple-m1":
        return materialize[MetalM1]()
    elif target_arch == "apple-m2":
        return materialize[MetalM2]()
    elif target_arch == "apple-m3":
        return materialize[MetalM3]()
    elif target_arch == "apple-m4":
        return materialize[MetalM4]()
    elif _accelerator_arch() == "":
        return materialize[NoGPU]()
    else:
        return _get_info_from_target[_accelerator_arch()]()


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


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

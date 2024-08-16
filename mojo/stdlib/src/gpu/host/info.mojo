# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Contains information about the GPUs."""

# ===----------------------------------------------------------------------===#
# Flops
# ===----------------------------------------------------------------------===#


@value
@register_passable
struct Flops:
    var fp16: Int
    var i8: Int
    var i4: Int


# ===----------------------------------------------------------------------===#
# Info
# ===----------------------------------------------------------------------===#


@value
@register_passable
struct Info:
    var name: StringLiteral
    var version: StringLiteral
    var target: __mlir_type.`!kgen.target`
    var target_32bit: __mlir_type.`!kgen.target`
    var threads_per_warp: Int
    var warps_per_multiprocessor: Int
    var threads_per_multiprocessor: Int
    var thread_blocks_per_multiprocessor: Int
    var shared_memory_per_multiprocessor: Int
    var register_file_size: Int
    var register_allocation_unit_size: Int
    var allocation_granularity: StringLiteral
    var max_registers_per_thread: Int
    var max_registers_per_block: Int
    var max_blocks_per_multiprocessor: Int
    var shared_memory_allocation_unit_size: Int
    var warp_allocation_granularity: Int
    var max_thread_block_size: Int
    var flops: Flops

    fn __lt__(self, other: Self) -> Bool:
        if self.name == "sm_90a":
            return True

        return _get_compute(self.name) < _get_compute(other.name)

    fn __le__(self, other: Self) -> Bool:
        if self == other:
            return True
        return self < other

    fn __gt__(self, other: Self) -> Bool:
        if self.name == "sm_90a":
            return False
        return _get_compute(self.name) == _get_compute(other.name)

    fn __ge__(self, other: Self) -> Bool:
        if self == other:
            return True
        return _get_compute(self.name) >= _get_compute(other.name)

    fn __eq__(self, other: Self) -> Bool:
        return self.name == other.name

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other


# ===----------------------------------------------------------------------===#
# _get_info_from_target
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_info_from_target[target_arch: StringLiteral]() -> Info:
    constrained[target_arch in ("sm_80", "sm_86", "sm_89", "sm_90", "sm_90a")]()

    @parameter
    if target_arch == "sm_80":
        return A100
    elif target_arch == "sm_86":
        return A10
    elif target_arch == "sm_89":
        return L4
    elif target_arch in ("sm_90", "sm_90a"):
        return H100

    return A100


@always_inline("nodebug")
fn _get_compute(name: String) -> Int:
    if name == "sm_50":
        return 50
    elif name == "sm_60":
        return 60
    elif name == "sm_61":
        return 61
    elif name == "sm_70":
        return 70
    elif name == "sm_75":
        return 75
    elif name == "sm_80":
        return 80
    elif name == "sm_86":
        return 86
    elif name == "sm_89":
        return 89
    elif name == "sm_90":
        return 90
    else:
        return -1


# ===----------------------------------------------------------------------===#
# A100
# ===----------------------------------------------------------------------===#

# Note: features = "+ptx85" means that the kernel should be compiled using
# PTX version 8.5. This must be less than or equal to the installed CUDA
# driver's maximum supported PTX version. Currently we hardcode this to
# PTX version 8.5 which means that you need to have a CUDA driver included with
# CUDA 12.5 toolkit. The mapping from CUDA Driver to PTX version can be found by
# looking at the PTX ISA in the versioned docs
# https://developer.nvidia.com/cuda-toolkit-archive.

alias A100 = Info(
    name="A100",
    version="sm_80",
    target=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx85", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ],
    target_32bit=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx85", `,
        `data_layout="e-p32:64:64-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ],
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
    flops=Flops(fp16=312, i8=624, i4=1248),
)

# ===----------------------------------------------------------------------===#
# A10
# ===----------------------------------------------------------------------===#


alias A10 = Info(
    name="A10",
    version="sm_86",
    target=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx85", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ],
    target_32bit=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx85", `,
        `data_layout="e-p32:64:64-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ],
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
    flops=Flops(fp16=125, i8=250, i4=500),
)

# ===----------------------------------------------------------------------===#
# L4
# ===----------------------------------------------------------------------===#


alias L4 = Info(
    name="L4",
    version="sm_89",
    target=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx85", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ],
    target_32bit=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx85", `,
        `data_layout="e-p32:64:64-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ],
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
    flops=Flops(fp16=121, i8=242, i4=485),
)

# ===----------------------------------------------------------------------===#
# H100
# ===----------------------------------------------------------------------===#


alias H100 = Info(
    name="H100",
    version="sm_90",
    target=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90a", `,
        `features = "+ptx85", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ],
    target_32bit=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90a", `,
        `features = "+ptx85", `,
        `data_layout="e-p32:64:64-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ],
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
    flops=Flops(fp16=1979, i8=3958, i4=7916),
)

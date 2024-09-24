# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Contains information about the GPUs."""

from math import ceildiv, floor
from sys import env_get_string

alias DEFAULT_GPU_ARCH = env_get_string["DEFAULT_GPU_ARCH", "sm_80"]()

# ===----------------------------------------------------------------------===#
# A100
# ===----------------------------------------------------------------------===#

# Note: features = "+ptx81" means that the kernel should be compiled using
# PTX version 8.1. This must be less than or equal to the installed CUDA
# driver's maximum supported PTX version. Currently we hardcode this to
# PTX version 8.1 which means that you need to have a CUDA driver included with
# CUDA 12.5 toolkit. The mapping from CUDA Driver to PTX version can be found by
# looking at the PTX ISA in the versioned docs
# https://developer.nvidia.com/cuda-toolkit-archive.

alias A100 = Info(
    name="A100",
    compute=8.0,
    version="sm_80",
    target=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx81", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ],
    target_32bit=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx81", `,
        `data_layout="e-p32:64:64-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 32`,
        `> : !kgen.target`,
    ],
    sm_count=108,
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
    flops=Flops(fp16=312, i8=624, i4=1248),
)

# ===----------------------------------------------------------------------===#
# A10
# ===----------------------------------------------------------------------===#


alias A10 = Info(
    name="A10",
    compute=8.6,
    version="sm_86",
    target=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx81", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ],
    target_32bit=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx81", `,
        `data_layout="e-p32:64:64-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 32`,
        `> : !kgen.target`,
    ],
    sm_count=72,
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
    flops=Flops(fp16=125, i8=250, i4=500),
)

# ===----------------------------------------------------------------------===#
# L4
# ===----------------------------------------------------------------------===#


alias L4 = Info(
    name="L4",
    compute=8.9,
    version="sm_89",
    target=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ],
    target_32bit=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81", `,
        `data_layout="e-p32:64:64-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 32`,
        `> : !kgen.target`,
    ],
    sm_count=58,
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
    flops=Flops(fp16=121, i8=242, i4=485),
)

# ===----------------------------------------------------------------------===#
# H100
# ===----------------------------------------------------------------------===#


alias H100 = Info(
    name="H100",
    compute=9.0,
    version="sm_90a",
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
        `simd_bit_width = 128,`,
        `index_bit_width = 32`,
        `> : !kgen.target`,
    ],
    sm_count=114,
    threads_per_sm=-1,
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

# ===----------------------------------------------------------------------===#
# Flops
# ===----------------------------------------------------------------------===#


@value
@register_passable
struct Flops:
    var fp16: Int
    var i8: Int
    var i4: Int

    @no_inline
    fn format_to(self, inout writer: Formatter):
        writer.write("flops_fp16: ", self.fp16, "\n")
        writer.write("flops_i8: ", self.i8, "\n")
        writer.write("flops_i4: ", self.i4)

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)


# ===----------------------------------------------------------------------===#
# Info
# ===----------------------------------------------------------------------===#


@value
@register_passable
struct Info:
    var name: StringLiteral
    var compute: FloatLiteral
    var version: StringLiteral
    var target: __mlir_type.`!kgen.target`
    var target_32bit: __mlir_type.`!kgen.target`
    var sm_count: Int
    var threads_per_sm: Int
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

    @staticmethod
    fn from_target_name[name: StringLiteral]() -> Self:
        return _get_info_from_target[name]()

    fn __lt__(self, other: Self) -> Bool:
        return self.compute < other.compute

    fn __le__(self, other: Self) -> Bool:
        return self.compute <= other.compute

    fn __gt__(self, other: Self) -> Bool:
        return self.compute > other.compute

    fn __ge__(self, other: Self) -> Bool:
        return self.compute >= other.compute

    fn __eq__(self, other: Self) -> Bool:
        return self.name == other.name

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn format_to(self, inout writer: Formatter):
        writer.write("name: ", self.name, "\n")
        writer.write("compute: ", self.compute, "\n")
        writer.write("version: ", self.version, "\n")
        writer.write("sm_count: ", self.sm_count, "\n")
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
        writer.write(self.flops)

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)


# ===----------------------------------------------------------------------===#
# _get_info_from_target
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_info_from_target[target_arch: StringLiteral]() -> Info:
    constrained[
        target_arch
        in (
            "cuda",
            "cuda-sm_80",
            "cuda-sm_86",
            "cuda-sm_89",
            "cuda-sm_90",
            "cuda-sm_90a",
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
            "sm_90a",
        )
    ]()

    @parameter
    if target_arch in ("cuda-sm_80", "sm_80"):
        return A100
    elif target_arch in ("cuda-sm_86", "sm_86"):
        return A10
    elif target_arch in ("cuda-sm_89", "sm_89"):
        return L4
    elif target_arch in ("cuda-sm_90", "cuda-sm_90a", "sm_90", "sm_90a"):
        return H100

    return _get_info_from_target[DEFAULT_GPU_ARCH]()


@always_inline("nodebug")
fn _get_compute(target_arch: String) -> Float32:
    if target_arch in ("cuda-sm_80", "sm_80"):
        return A100.compute
    elif target_arch in ("cuda-sm_86", "sm_86"):
        return A10.compute
    elif target_arch in ("cuda-sm_89", "sm_89"):
        return L4.compute
    elif target_arch in ("cuda-sm_90", "cuda-sm_90a", "sm_90", "sm_90a"):
        return H100.compute

    return _get_info_from_target[DEFAULT_GPU_ARCH]().compute

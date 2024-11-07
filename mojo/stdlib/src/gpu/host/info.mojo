# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Contains information about the GPUs."""

from math import ceildiv, floor
from os import abort
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


fn _get_a100_target[index_bit_width: Int]() -> __mlir_type.`!kgen.target`:
    @parameter
    if index_bit_width == 64:
        return __mlir_attr[
            `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
            `arch = "sm_80", `,
            `features = "+ptx81", `,
            `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
            `simd_bit_width = 128,`,
            `index_bit_width = 64,`,
            `warp_size = 32`,
            `> : !kgen.target`,
        ]
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx81", `,
        `data_layout="e-p:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 32,`,
        `warp_size = 32`,
        `> : !kgen.target`,
    ]


alias A100 = Info(
    name="A100",
    arch_name="ampere",
    compute=8.0,
    version="sm_80",
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
    flops=Flops(fp16=312, tf32=156, fp64=19.5, i8=624, i4=1248),
)

# ===----------------------------------------------------------------------===#
# A10
# ===----------------------------------------------------------------------===#


fn _get_a10_target[index_bit_width: Int]() -> __mlir_type.`!kgen.target`:
    @parameter
    if index_bit_width == 64:
        return __mlir_attr[
            `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
            `arch = "sm_86", `,
            `features = "+ptx81", `,
            `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
            `simd_bit_width = 128,`,
            `index_bit_width = 64,`,
            `warp_size = 32`,
            `> : !kgen.target`,
        ]
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx81", `,
        `data_layout="e-p:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 32,`,
        `warp_size = 32`,
        `> : !kgen.target`,
    ]


alias A10 = Info(
    name="A10",
    arch_name="ampere",
    compute=8.6,
    version="sm_86",
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
    flops=Flops(fp16=125, tf32=62.5, i8=250, i4=500),
)

# ===----------------------------------------------------------------------===#
# L4
# ===----------------------------------------------------------------------===#


fn _get_l4_target[index_bit_width: Int]() -> __mlir_type.`!kgen.target`:
    @parameter
    if index_bit_width == 64:
        return __mlir_attr[
            `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
            `arch = "sm_89", `,
            `features = "+ptx81", `,
            `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
            `simd_bit_width = 128,`,
            `index_bit_width = 64,`,
            `warp_size = 32`,
            `> : !kgen.target`,
        ]
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81", `,
        `data_layout="e-p:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 32,`,
        `warp_size = 32`,
        `> : !kgen.target`,
    ]


alias L4 = Info(
    name="L4",
    arch_name="ada",
    compute=8.9,
    version="sm_89",
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


fn _get_h100_target[index_bit_width: Int]() -> __mlir_type.`!kgen.target`:
    @parameter
    if index_bit_width == 64:
        return __mlir_attr[
            `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
            `arch = "sm_90a", `,
            `features = "+ptx85", `,
            `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
            `index_bit_width = 64,`,
            `simd_bit_width = 128,`,
            `warp_size = 32`,
            `> : !kgen.target`,
        ]
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90a", `,
        `features = "+ptx85", `,
        `data_layout="e-p:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 32,`,
        `warp_size = 32`,
        `> : !kgen.target`,
    ]


alias H100 = Info(
    name="H100",
    arch_name="hopper",
    compute=9.0,
    version="sm_90a",
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
    flops=Flops(fp8=3958, fp16=1979, tf32=989, fp64=67, i8=3958, i4=7916),
)

# ===----------------------------------------------------------------------===#
# MI300X
# ===----------------------------------------------------------------------===#


fn _get_mi300x_target[index_bit_width: Int]() -> __mlir_type.`!kgen.target`:
    @parameter
    if index_bit_width == 64:
        return __mlir_attr[
            `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
            `arch = "gfx942", `,
            `features = "", `,
            `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
            `index_bit_width = 64,`,
            `simd_bit_width = 128,`,
            `warp_size = 64`,
            `> : !kgen.target`,
        ]
    debug_assert(False, "mi300x with 32bit config is not currently supported")
    return _get_mi300x_target[64]()


alias MI300X = Info(
    name="MI300X",
    arch_name="gfx942",
    compute=9.4,
    version="CDNA3",
    sm_count=304,
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
    # From https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
    flops=Flops(tf32=653.7, fp16=1307.4, fp8=1307.4, i8=2614.9, i4=0),
)

# ===----------------------------------------------------------------------===#
# Flops
# ===----------------------------------------------------------------------===#


@value
@register_passable
struct Flops:
    var fp8: Float64
    var fp16: Float64
    var tf32: Float64
    var fp64: Float64
    var i8: Float64
    var i4: Float64

    fn __init__(
        inout self,
        *,
        fp16: Float64,
        i8: Float64,
        i4: Float64,
        fp8: Float64 = 0,
        tf32: Float64 = 0,
        fp64: Float64 = 0,
    ):
        self.fp8 = fp8
        self.fp16 = fp16
        self.tf32 = tf32
        self.fp64 = fp64
        self.i8 = i8
        self.i4 = i4

    @no_inline
    fn write_to[W: Writer](self, inout writer: W):
        if self.fp8:
            writer.write("flops_fp8: ", self.fp8, "\n")
            writer.write("flops_fp16: ", self.fp16, "\n")
        if self.tf32:
            writer.write("flops_tf32: ", self.tf32, "\n")
        if self.fp64:
            writer.write("flops_fp64: ", self.fp64, "\n")
        writer.write("flops_i8: ", self.i8, "\n")
        writer.write("flops_i4: ", self.i4)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)


# ===----------------------------------------------------------------------===#
# Info
# ===----------------------------------------------------------------------===#


@value
@register_passable
struct Info:
    var name: StringLiteral
    var arch_name: StringLiteral
    var compute: Float32
    var version: StringLiteral
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

    fn target[index_bit_width: Int = 64](self) -> __mlir_type.`!kgen.target`:
        if self.name == "A100":
            return _get_a100_target[index_bit_width]()
        if self.name == "A10":
            return _get_a10_target[index_bit_width]()
        if self.name == "L4":
            return _get_l4_target[index_bit_width]()
        if self.name == "H100":
            return _get_h100_target[index_bit_width]()
        if self.name == "MI300X":
            return _get_mi300x_target[index_bit_width]()
        return _get_a100_target[index_bit_width]()

    @staticmethod
    fn from_target_name[name: StringLiteral]() -> Self:
        return _get_info_from_target[name]()

    fn _warps_per_block(self, threads_per_block: Int) -> Int:
        return ceildiv(threads_per_block, self.threads_per_warp)

    fn _registers_per_warp(self, registers_per_thread: Int) -> Int:
        return _quantized_ceil(
            registers_per_thread * self.threads_per_warp,
            self.register_allocation_unit_size,
        )

    fn _registers_per_block(
        self, threads_per_block: Int, registers_per_thread: Int
    ) -> Int:
        return self._registers_per_warp(
            registers_per_thread
        ) * self._warps_per_block(threads_per_block)

    fn _warps_per_multiprocessor_register_limited(
        self, registers_per_thread: Int
    ) -> Int:
        return _quantized_floor(
            self.max_registers_per_block
            / self._registers_per_warp(registers_per_thread),
            self.warp_allocation_granularity,
        )

    fn _blocks_per_multiprocessor_register_limited(
        self, *, threads_per_block: Int, registers_per_thread: Int
    ) -> Int:
        return int(
            self._warps_per_multiprocessor_register_limited(
                registers_per_thread
            )
            / self._warps_per_block(threads_per_block)
        ) * int(self.register_file_size / self.max_registers_per_block)

    fn _block_runtime_shared_memory(self) -> Int:
        if self.compute > 8:
            # starting with Compute Capability 8.x, the CUDA runtime consumes
            # 1KB of shared memory the amount might change depending on the
            # CUDA runtime version in the future.
            return 1024
        return 0

    fn _block_shared_memory(self, *, shared_memory_per_block: Int) -> Int:
        """shared memory per thread block."""
        return ceildiv(
            shared_memory_per_block + self._block_runtime_shared_memory(),
            self.shared_memory_allocation_unit_size,
        )

    fn _thread_blocks_per_multiprocessor_limited_by_warps_or_blocks_per_multiprocessor(
        self, threads_per_block: Int
    ) -> Float64:
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
        return _quantized_floor(
            self.max_registers_per_block
            / self._registers_per_warp(registers_per_thread),
            self.warp_allocation_granularity,
        )

    fn _thread_blocks_per_multiprocessor_limited_by_registers_per_multiprocessor(
        self, *, threads_per_block: Int, registers_per_thread: Int
    ) -> Float64:
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
        # TODO (KERN-795): Add occupancy calculation based on shared memory
        # usage and thread block size and take use the minimum value
        return (
            self._blocks_per_multiprocessor_register_limited(
                threads_per_block=threads_per_block,
                registers_per_thread=registers_per_thread,
            )
            * self._warps_per_block(threads_per_block)
            / self.warps_per_multiprocessor
        )

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
    fn write_to[W: Writer](self, inout writer: W):
        writer.write("name: ", self.name, "\n")
        writer.write("arch_name: ", self.arch_name, "\n")
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
        return String.write(self)


# ===----------------------------------------------------------------------===#
# _get_info_from_target
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_info_from_compute_capability[compute_capability: Int]() -> Info:
    constrained[
        compute_capability in (80, 86, 89, 90, 94), "invalid compute capability"
    ]()

    @parameter
    if compute_capability == 80:
        return A100
    elif compute_capability == 86:
        return A10
    elif compute_capability == 89:
        return L4
    elif compute_capability == 90:
        return H100
    elif compute_capability == 94:
        return MI300X
    return abort[Info]("invalid compute capability")


@always_inline
fn _get_info_from_compute_capability(compute_capability: Int) raises -> Info:
    if compute_capability == 80:
        return _get_info_from_compute_capability[80]()
    if compute_capability == 86:
        return _get_info_from_compute_capability[86]()
    if compute_capability == 89:
        return _get_info_from_compute_capability[89]()
    if compute_capability == 90:
        return _get_info_from_compute_capability[90]()
    if compute_capability == 94:
        return _get_info_from_compute_capability[94]()

    raise "invalid compute capability"


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
            "mi300x",
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
    elif target_arch in ("mi300x"):
        return MI300X

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
    elif target_arch in ("mi300x"):
        return MI300X.compute

    return _get_info_from_target[DEFAULT_GPU_ARCH]().compute


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


fn _quantized_ceil(a: Float64, b: Int) -> Int:
    return int(ceildiv(a, b) * b)


fn _quantized_floor(a: Float64, b: Int) -> Int:
    return int(floor(a / b) * b)

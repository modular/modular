# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Contains information about the GPUs."""


struct Info:
    var name: String
    var version: String
    var target: __mlir_type.`!kgen.target`
    var threads_per_warp: Int
    var warps_per_multiprocessor: Int
    var threads_per_multiprocessor: Int
    var thread_blocks_per_multiprocessor: Int
    var shared_memory_per_multiprocessor: Int
    var register_file_size: Int
    var register_allocation_unit_size: Int
    var allocation_granularity: String
    var max_registers_per_thread: Int
    var max_registers_per_block: Int
    var shared_memory_allocation_unit_size: Int
    var warp_allocation_granularity: Int
    var max_thread_block_size: Int
    var peak_fp16_tflops: Int
    var peak_i8_tflops: Int
    var peak_i4_tflops: Int

    fn __init__(
        inout self,
        *,
        name: String,
        version: String,
        target: __mlir_type.`!kgen.target`,
        threads_per_warp: Int,
        warps_per_multiprocessor: Int,
        threads_per_multiprocessor: Int,
        thread_blocks_per_multiprocessor: Int,
        shared_memory_per_multiprocessor: Int,
        register_file_size: Int,
        register_allocation_unit_size: Int,
        allocation_granularity: String,
        max_registers_per_thread: Int,
        max_registers_per_block: Int,
        shared_memory_allocation_unit_size: Int,
        warp_allocation_granularity: Int,
        max_thread_block_size: Int,
        peak_fp16_tflops: Int,
        peak_i8_tflops: Int,
        peak_i4_tflops: Int,
    ):
        self.name = name
        self.version = version
        self.target = target
        self.threads_per_warp = threads_per_warp
        self.warps_per_multiprocessor = warps_per_multiprocessor
        self.threads_per_multiprocessor = threads_per_multiprocessor
        self.thread_blocks_per_multiprocessor = thread_blocks_per_multiprocessor
        self.shared_memory_per_multiprocessor = shared_memory_per_multiprocessor
        self.register_file_size = register_file_size
        self.register_allocation_unit_size = register_allocation_unit_size
        self.allocation_granularity = allocation_granularity
        self.max_registers_per_thread = max_registers_per_thread
        self.max_registers_per_block = max_registers_per_block
        self.shared_memory_allocation_unit_size = (
            shared_memory_allocation_unit_size
        )
        self.warp_allocation_granularity = warp_allocation_granularity
        self.max_thread_block_size = max_thread_block_size
        self.peak_fp16_tflops = peak_fp16_tflops
        self.peak_i8_tflops = peak_i8_tflops
        self.peak_i4_tflops = peak_i4_tflops


# ===----------------------------------------------------------------------===#
# A100
# ===----------------------------------------------------------------------===#

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
    shared_memory_allocation_unit_size=128,
    warp_allocation_granularity=4,
    max_thread_block_size=1024,
    peak_fp16_tflops=312,
    peak_i8_tflops=624,
    peak_i4_tflops=1248,
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
    shared_memory_allocation_unit_size=128,
    warp_allocation_granularity=4,
    max_thread_block_size=1024,
    peak_fp16_tflops=125,
    peak_i8_tflops=250,
    peak_i4_tflops=500,
)

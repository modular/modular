# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# RUN: mojo doc %s | FileCheck %s


from gpu.host.info import GPUInfo, Vendor

# CHECK: "name": "Radeon7600",
# CHECK: "value": "GPUInfo(\"Radeon 7600\", Vendor(1), \"hip\", \"gfx1102\", \"\", 11, \"RDNA3\", 32, 32, 1024, 32, 32, 1024, 2, 32768, 32768, 256, \"warp\", 255, 32768, 2, 128, 4, 1024)"
alias Radeon7600 = GPUInfo(
    name="Radeon 7600",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1102",
    compile_options="",
    compute=11.0,
    version="RDNA3",
    sm_count=32,
    warp_size=32,
    threads_per_sm=1024,
    threads_per_warp=32,
    warps_per_multiprocessor=32,
    threads_per_multiprocessor=1024,
    thread_blocks_per_multiprocessor=2,
    shared_memory_per_multiprocessor=32768,
    register_file_size=32768,
    register_allocation_unit_size=256,
    allocation_granularity="warp",
    max_registers_per_thread=255,
    max_registers_per_block=32768,
    max_blocks_per_multiprocessor=2,
    shared_memory_allocation_unit_size=128,
    warp_allocation_granularity=4,
    max_thread_block_size=1024,
)

# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# RUN: mojo doc %s | FileCheck %s


from gpu.host.info import GPUInfo, Vendor

# CHECK: "name": "Radeon7600",
# CHECK: "value": "GPUInfo(\"Radeon 7600\", Vendor(1), \"hip\", \"gfx1102\", 11, \"RDNA3\", 32, 32, 1024, 32768, 32768, 1024)"
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

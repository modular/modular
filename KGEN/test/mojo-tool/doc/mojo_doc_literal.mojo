# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# RUN: mojo doc %s | FileCheck %s


from gpu.host.info import GPUInfo, Vendor, AMDRDNAFamily

# CHECK: "name": "Radeon7600",
# CHECK: "value": "GPUInfo.from_family(AMDRDNAFamily, \"Radeon 7600\", Vendor.AMD_GPU, \"hip\", \"gfx1102\", 11, \"RDNA3\", 32)"
alias Radeon7600 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7600",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1102",
    compute=11.0,
    version="RDNA3",
    sm_count=32,
)

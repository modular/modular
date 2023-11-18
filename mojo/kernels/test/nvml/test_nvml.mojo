# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device, has_nvml
# RUN: %mojo %s | FileCheck %s

from gpu.host.nvml import Device


def main():
    let dev = Device(0)
    let clocks = dev.mem_clocks()
    for i in range(len(clocks)):
        # CHECK: Clock =
        print("Clock =", clocks[i])
    clocks._del_old()

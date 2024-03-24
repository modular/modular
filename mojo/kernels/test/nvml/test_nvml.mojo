# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device, has_nvml
# RUN: %mojo %s | FileCheck %s

from nvml import Device


def main():
    var dev = Device(0)
    var clocks = dev.mem_clocks()
    for i in range(len(clocks)):
        # CHECK: Clock =
        print("Clock =", clocks[i])

# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu.host import Context
from gpu.host.memory import (
    _mem_get_allocation_granularity,
    MemAllocationProp,
    MemAllocationType,
    MemLocationType,
    MemLocation,
    MemAllocationGranularityFlags,
)

from testing import assert_equal


# CHECK-LABEL: test_mem_get_allocation_granularity
fn test_mem_get_allocation_granularity() raises:
    print("== test_mem_get_allocation_granularity")
    alias min_and_recommended_size = 2 * 1024 * 1024
    var prop = MemAllocationProp(
        MemAllocationType.PINNED, MemLocation(MemLocationType.DEVICE, 0)
    )
    assert_equal(
        _mem_get_allocation_granularity(
            prop,
            MemAllocationGranularityFlags.MINIMUM,
        ),
        min_and_recommended_size,
    )
    assert_equal(
        _mem_get_allocation_granularity(
            prop,
            MemAllocationGranularityFlags.RECOMMENDED,
        ),
        min_and_recommended_size,
    )


fn main():
    try:
        with Context():
            test_mem_get_allocation_granularity()

    except e:
        print("CUDA error", e)

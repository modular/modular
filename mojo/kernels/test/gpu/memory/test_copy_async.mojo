# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -disable-prebuilt-packages -O0 -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" %s | FileCheck %s

from memory.unsafe import Pointer, DTypePointer
from gpu.memory import AddressSpace
from gpu.sync import mbarrier


@export
fn test_mbarrier(
    addr0: Pointer[Int8],
    addr1: DTypePointer[DType.uint8],
    addr2: Pointer[Float32, AddressSpace.GLOBAL],
    addr3: Pointer[Float32, AddressSpace.SHARED],
    addr4: DTypePointer[DType.float64, AddressSpace.GLOBAL],
    addr5: DTypePointer[DType.float64, AddressSpace.SHARED],
):
    # CHECK: cp.async.mbarrier.arrive.b64
    mbarrier(addr0)
    # CHECK: cp.async.mbarrier.arrive.b64
    mbarrier(addr1)
    # mbarrier(addr2) # TODO (24115) comment in once fixed.
    # CHECK: cp.async.mbarrier.arrive.shared.b64
    mbarrier(addr3)
    # mbarrier(addr4) # TODO (24115) comment in once fixed.
    # CHECK: cp.async.mbarrier.arrive.shared.b64
    mbarrier(addr5)

# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -disable-prebuilt-packages -O0 -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" %s | FileCheck %s

from memory.unsafe import Pointer, DTypePointer
from gpu.memory import AddressSpace, async_copy
from gpu.sync import mbarrier, mbarrier_init, mbarrier_test_wait
from memory import stack_allocation


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


@export
fn test_mbarrier_init(
    shared_mem: DTypePointer[DType.int32, AddressSpace.SHARED],
):
    # CHECK: ld.param.u64 %[[ADDRS_REG:.+]], [test_mbarrier_init_param_0]
    # CHECK: mov.b32 %[[REG:.+]], 4
    # CHECK: mbarrier.init.shared.b64 [%[[ADDRS_REG]]], %[[REG]]
    mbarrier_init(shared_mem, 4)


@export
fn test_mbarrier_test_wait(
    shared_mem: DTypePointer[DType.int32, AddressSpace.SHARED], state: Int
):
    var done = False
    # CHECK: mbarrier.test_wait.shared.b64
    while not done:
        done = mbarrier_test_wait(shared_mem, state)


@export
fn test_async_copy(src: DTypePointer[DType.float32, AddressSpace.GLOBAL]):
    let barrier = stack_allocation[
        sizeof[DType.int32](), DType.int32, address_space = AddressSpace.SHARED
    ]()
    let shared_mem = stack_allocation[
        4, DType.float32, address_space = AddressSpace.SHARED
    ]()
    # CHECK: cp.async.ca.shared.global
    async_copy[4](src, shared_mem)

# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from gpu.host import Context, Function

from gpu.memory import dynamic_shared_memory
from gpu.id import ThreadIdx
from gpu.sync import barrier
from gpu.host.memory import _malloc_managed


# CHECK-LABEL: test_dynamic_shared_mem
fn test_dynamic_shared_mem() raises:
    print("== test_dynamic_shared_mem")

    fn dynamc_smem_kernel(data: DTypePointer[DType.float32]):
        var dynamic_sram = dynamic_shared_memory[Float32, alignment=4]()
        dynamic_sram[ThreadIdx.x()] = ThreadIdx.x()
        barrier()
        data[ThreadIdx.x()] = dynamic_sram[ThreadIdx.x()]

    var func = Function[__type_of(dynamc_smem_kernel), dynamc_smem_kernel]()

    var res = _malloc_managed[DType.float32](16)
    for i in range(16):
        res[i] = 0

    func(res, grid_dim=(1), block_dim=(16), shared_mem_bytes=64)
    # CHECK: 0.0
    # CHECK: 1.0
    # CHECK: 2.0
    # CHECK: 3.0
    # CHECK: 4.0
    # CHECK: 5.0
    # CHECK: 6.0
    # CHECK: 7.0
    # CHECK: 8.0
    # CHECK: 9.0
    # CHECK: 10.0
    # CHECK: 11.0
    # CHECK: 12.0
    # CHECK: 13.0
    # CHECK: 14.0
    # CHECK: 15.0
    for i in range(16):
        print(res[i])


fn main() raises:
    with Context() as ctx:
        test_dynamic_shared_mem()

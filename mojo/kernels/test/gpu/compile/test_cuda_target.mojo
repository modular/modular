# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: nvptx_backend
# RUN: kgen -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_75 %s | FileCheck %s

from Activations import gelu
from Assert import assert_param
from DType import DType
from Functional import elementwise
from Index import StaticIntTuple
from IO import print
from NvidiaGPU import *
from Pointer import DTypePointer
from Range import range
from TargetInfo import triple_is_nvidia_cuda, simdwidthof
from LLCL import OutputChainPtr


# ===----------------------------------------------------------------------===#
# Check parameterization
# ===----------------------------------------------------------------------===#


# COM: Checks if we can do parameterization on the triple_is_nvidia_cuda check.
# COM: In this case the code that would run on CUDA would return 42 and the
# COM: one that does not would return -1.
@adaptive
@always_inline
fn parameterized_on_cuda_impl() -> Int:
    assert_param[triple_is_nvidia_cuda()]()
    return 42


@adaptive
@always_inline
fn parameterized_on_cuda_impl() -> Int:
    assert_param[not triple_is_nvidia_cuda()]()
    return -1


# CHECK-LABEL: parameterized_on_cuda()
# CHECK: mov.u64 {{.*}}, 42;
@export
fn parameterized_on_cuda() -> Int:
    return parameterized_on_cuda_impl()


# ===----------------------------------------------------------------------===#
# Check elementwise kernel
# ===----------------------------------------------------------------------===#


# CHECK-LABEL: gelu_elementwise
# CHECK-DAG: tid.x
# CHECK-DAG: ntid.x
# CHECK-DAG: ctaid.x
@export
fn gelu_elementwise(buf: DTypePointer[DType.float32], len: Int):
    # Each thread will process 4 * simd_width elements.
    alias granularity = 4 * simdwidthof[DType.float32]()

    let tid = granularity * (ThreadIdx.x() + BlockDim.x() * BlockIdx.x())

    @always_inline
    @parameter
    fn func[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        let offset = tid + idx[0]
        if offset >= len:
            return
        buf.store(offset, gelu(buf.load(offset)))

    elementwise[1, simdwidthof[DType.float32](), func](
        StaticIntTuple[1](granularity), OutputChainPtr()
    )


# ===----------------------------------------------------------------------===#
# Check full kernel
# ===----------------------------------------------------------------------===#


# CHECK-LABEL: gelu_kernel
# CHECK-DAG: tid.x
# CHECK-DAG: ntid.y
# CHECK-DAG: ctaid.y
@export
fn gelu_kernel(buf: DTypePointer[DType.float32], len: Int):
    let tid = ThreadIdx.x() + BlockDim.y() * BlockIdx.y()

    if tid >= len:
        return

    buf.store(tid, gelu(buf.load(tid)))


# ===----------------------------------------------------------------------===#
# Check barrier
# ===----------------------------------------------------------------------===#

# CHECK-LABEL: barrier
# CHECK: bar.sync 0
@export
fn test_barrier():
    barrier()

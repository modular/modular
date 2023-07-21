# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: nvptx_backend
# RUN: kgen -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_75 %s | FileCheck %s

from Assert import assert_param
from Activations import gelu
from DType import DType
from Range import range
from Pointer import DTypePointer
from NvidiaGPU import *
from IO import print
from TargetInfo import triple_is_nvidia_cuda


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

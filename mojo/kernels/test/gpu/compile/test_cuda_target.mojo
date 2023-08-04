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
from Pointer import DTypePointer
from Index import StaticIntTuple
from IO import print
from NvidiaGPU import (
    ThreadIdx,
    BlockIdx,
    BlockDim,
    GridDim,
    AddressSpace,
    stack_allocation,
    barrier,
)
from Memory import memset_zero
from Pointer import DTypePointer
from Range import range
from TargetInfo import triple_is_nvidia_cuda, simdwidthof
from LLCL import OutputChainPtr
from SIMD import Float32


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
# Check stack allocation
# ===----------------------------------------------------------------------===#


# CHECK-LABEL: test_shared_stack_allocation
# CHECK-DAG: cvta.local.u64
@export
fn test_shared_stack_allocation() -> DTypePointer[DType.float32]:
    return stack_allocation[5, DType.float32, AddressSpace.SHARED]()


# ===----------------------------------------------------------------------===#
# Check barrier
# ===----------------------------------------------------------------------===#

# CHECK-LABEL: barrier
# CHECK: bar.sync 0
@export
fn test_barrier():
    barrier()


# ===----------------------------------------------------------------------===#
# SGEMM with register coarsening
# ===----------------------------------------------------------------------===#


@export
fn gemm(
    c: DTypePointer[DType.float32],
    a: DTypePointer[DType.float32],
    b: DTypePointer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
):
    # Compute C = A x B
    #   where A is a (m x k) matrix
    #   where B is a (k x n) matrix
    #   where C is a (m x n) matrix
    #
    # Use register and shared memory tiling and thread coarsening
    #
    # NOTE: A and C are column major, B is row major.

    alias TILE_SZ_A = 128
    alias TILE_SZ_B = 16
    alias TILE_SZ_RATIO = TILE_SZ_A // TILE_SZ_B

    # Utilities for accessing flattened matrices.
    @always_inline
    fn get_a(row: Int, col: Int) -> Float32:
        return a.load(row + m * col)

    @always_inline
    fn get_b(row: Int, col: Int) -> Float32:
        return b.load(row * n + col)

    @always_inline
    fn set_c(row: Int, col: Int, val: Float32):
        c.store(row + col * m, val)

    # Allocate B array into shared memory for tiling.
    let b_shared = stack_allocation[
        TILE_SZ_RATIO * TILE_SZ_B, DType.float32, AddressSpace.SHARED
    ]()

    # Thread indexing offsets.
    let row = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    let col = BlockIdx.y() * TILE_SZ_B

    # Privatization of the C matrix.
    let c_reg = stack_allocation[TILE_SZ_B, DType.float32]()

    memset_zero(c_reg, TILE_SZ_B)

    # Loop over each input tile.
    for tile_idx in range((k - 1) // TILE_SZ_RATIO + 1):
        let i = ThreadIdx.x() // TILE_SZ_B
        let j = ThreadIdx.x() % TILE_SZ_B

        # Load the B matrix into shared memory.
        let b_val: Float32
        if tile_idx * TILE_SZ_RATIO + i < k and col + j < n:
            b_val = get_b(tile_idx * TILE_SZ_RATIO + i, col + j)
        else:
            b_val = 0
        b_shared.store(i * TILE_SZ_B + j, b_val)

        barrier()

        # Loop within the tile.
        for idx in range(TILE_SZ_RATIO):
            # Load the A tile into the register.
            let a_reg: Float32
            if row < m and tile_idx * TILE_SZ_RATIO + idx < k:
                a_reg = get_a(row, tile_idx * TILE_SZ_RATIO + idx)
            else:
                a_reg = 0

            # Compute the output element for each thread.
            for out_idx in range(TILE_SZ_B):
                c_reg.store(
                    out_idx,
                    c_reg.load(out_idx)
                    + a_reg * b_shared.load(idx * TILE_SZ_RATIO + out_idx),
                )
        barrier()

    # Store the values into the output matrix.
    for out_idx in range(TILE_SZ_B):
        if row < m and col + out_idx < n:
            set_c(row, col + out_idx, c_reg.load(out_idx))

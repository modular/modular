# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO(#22563): Remove the use of `-disable-prebuilt-packages`.
# RUN: kgen -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" -disable-prebuilt-packages %s | FileCheck %s

from sys.info import simdwidthof, triple_is_nvidia_cuda

from NN.Activations import gelu
from algorithm import elementwise
from memory import memset_zero, stack_allocation
from memory.unsafe import DTypePointer

from utils.index import StaticIntTuple
from gpu import (
    ThreadIdx,
    BlockIdx,
    BlockDim,
    barrier,
    shuffle_xor,
    shuffle_down,
    shuffle_up,
    WARP_SIZE,
    lane_id,
)
from gpu.memory import AddressSpace

# ===----------------------------------------------------------------------===#
# Check parameterization
# ===----------------------------------------------------------------------===#


# COM: Checks if we can do parameterization on the triple_is_nvidia_cuda check.
# COM: In this case the code that would run on CUDA would return 42 and the
# COM: one that does not would return -1.


# CHECK-LABEL: parameterized_on_cuda()
# CHECK: mov.u64 {{.*}}, 42;
@export
fn parameterized_on_cuda() -> Int:
    @parameter
    if triple_is_nvidia_cuda():
        return 42
    else:
        return -1


# ===----------------------------------------------------------------------===#
# Check print
# ===----------------------------------------------------------------------===#


# CHECK: hello_mojo
@export
fn hello_mojo():
    # CHECK: print
    print("Hello")


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
        buf[offset] = gelu(buf[offset])

    elementwise[1, simdwidthof[DType.float32](), func](
        StaticIntTuple[1](granularity)
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

    buf[tid] = gelu(buf[tid])


# ===----------------------------------------------------------------------===#
# Check stack allocation
# ===----------------------------------------------------------------------===#


# CHECK-LABEL: test_shared_stack_allocation
# CHECK: .shared .align 8 .b8 [[SHM0:.*]][999];
@export
fn test_shared_stack_allocation() -> (
    DTypePointer[DType.int8, AddressSpace.SHARED]
):
    return stack_allocation[
        999, DType.int8, 8, address_space = AddressSpace.SHARED
    ]()


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


# CHECK-LABEL: .visible .entry	gemm
# CHECK: .shared .align 1 .b8 [[SHM1:.*]][512];
# CHECK: st.shared.f32
# CHECK: ld.shared.f32
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
        c[row + col * m] = val

    # Allocate B array into shared memory for tiling.
    let b_shared = stack_allocation[
        TILE_SZ_RATIO * TILE_SZ_B,
        DType.float32,
        address_space = AddressSpace.SHARED,
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
        b_shared[i * TILE_SZ_B + j] = b_val

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
                c_reg[out_idx] += (
                    a_reg * b_shared[idx * TILE_SZ_RATIO + out_idx]
                )
        barrier()

    # Store the values into the output matrix.
    for out_idx in range(TILE_SZ_B):
        if row < m and col + out_idx < n:
            set_c(row, col + out_idx, c_reg.load(out_idx))


# ===----------------------------------------------------------------------===#
# shuffle ops
# ===----------------------------------------------------------------------===#


@always_inline
fn _floorlog2[n: Int]() -> Int:
    return 0 if n <= 1 else 1 + _floorlog2[n >> 1]()


@always_inline
fn _static_log2[n: Int]() -> Int:
    return 0 if n <= 1 else _floorlog2[n - 1]() + 1


# CHECK-LABEL: test_shuffle_up
@export
fn test_shuffle_up(val: Float32) -> Float32:
    var res = val

    alias limit = _static_log2[WARP_SIZE]()

    # CHECK: shfl.sync.up.b32
    @unroll
    for mask in range(limit - 1, -1, -1):
        res += shuffle_up[DType.float32](res, 1 << mask)
    return res


# CHECK-LABEL: test_shuffle_down
@export
fn test_shuffle_down(val: Int32) -> Int32:
    var res = val

    alias limit = _static_log2[WARP_SIZE]()

    # CHECK: shfl.sync.down.b32
    @unroll
    for mask in range(limit - 1, -1, -1):
        res += shuffle_down[DType.int32](res, 1 << mask)
    return res


# ===----------------------------------------------------------------------===#
# Reduction
# ===----------------------------------------------------------------------===#


# CHECK-LABEL: warp_sum_reduce
@export
fn warp_sum_reduce(val: Float32) -> Float32:
    var res = val

    alias limit = _static_log2[WARP_SIZE]()

    # CHECK: shfl.sync.bfly.b32
    @unroll
    for mask in range(limit - 1, -1, -1):
        res += shuffle_xor[DType.float32](res, 1 << mask)
    return res


# CHECK-LABEL: block_reduce
@export
fn block_reduce(val: Float32) -> Float32:
    let shared = stack_allocation[
        WARP_SIZE, DType.float32, address_space = AddressSpace.SHARED
    ]()

    alias warp_shift = _static_log2[WARP_SIZE]()

    # CHECK-DAG: mov.u32         %r{{.*}}, %laneid;
    let lane = lane_id()
    let warp = ThreadIdx.x() // 32

    let warp_sum = warp_sum_reduce(val)

    if lane == 0:
        shared[warp] = warp_sum

    barrier()

    return warp_sum_reduce(
        shared.load(lane) if ThreadIdx.x() < BlockDim.x() // 32 else 0
    )

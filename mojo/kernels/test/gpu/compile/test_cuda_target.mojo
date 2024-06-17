# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import erf
from sys.info import simdwidthof, triple_is_nvidia_cuda

from algorithm import elementwise
from builtin.io import _printf
from gpu import (
    WARP_SIZE,
    BlockDim,
    BlockIdx,
    ThreadIdx,
    barrier,
    lane_id,
    shuffle_down,
    shuffle_up,
    shuffle_xor,
)
from gpu.host._compile import _compile_code, _get_nvptx_target
from gpu.memory import AddressSpace
from memory import memset_zero, stack_allocation
from memory.unsafe import DTypePointer
from testing import *

from utils.index import StaticIntTuple

# ===----------------------------------------------------------------------===#
# Check parameterization
# ===----------------------------------------------------------------------===#


# COM: Checks if we can do parameterization on the triple_is_nvidia_cuda check.
# COM: In this case the code that would run on CUDA would return 42 and the
# COM: one that does not would return -1.


@always_inline
fn _get_nvptx_target_sm90() -> __mlir_type.`!kgen.target`:
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90", `,
        `features = "+ptx81", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ]


fn parameterized_on_cuda() -> Int:
    @parameter
    if triple_is_nvidia_cuda():
        return 42
    else:
        return -1


@always_inline
fn _verify_parameterized_on_cuda(asm: String) raises -> None:
    assert_true("test_cuda_target_parameterized" in asm)
    assert_true("mov.u64" in asm)


def test_parameterized_on_cuda_sm80():
    alias asm = str(
        _compile_code[parameterized_on_cuda, target = _get_nvptx_target()]().asm
    )
    _verify_parameterized_on_cuda(asm)


def test_parameterized_on_cuda_sm90():
    alias asm = str(
        _compile_code[
            parameterized_on_cuda, target = _get_nvptx_target_sm90()
        ]().asm
    )
    _verify_parameterized_on_cuda(asm)


# ===----------------------------------------------------------------------===#
# Check print
# ===----------------------------------------------------------------------===#


fn hello_mojo():
    _printf["Hello"]()


@always_inline
fn _verify_hello(asm: String) raises -> None:
    assert_true("test_cuda_target_hello_mojo" in asm)
    assert_true("vprintf" in asm)


def test_hello_mojo_sm80():
    alias asm = str(
        _compile_code[hello_mojo, target = _get_nvptx_target()]().asm
    )
    _verify_hello(asm)


def test_hello_mojo_sm90():
    alias asm = str(
        _compile_code[hello_mojo, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_hello(asm)


# ===----------------------------------------------------------------------===#
# Check elementwise kernel
# ===----------------------------------------------------------------------===#


fn erf_elementwise(buf: DTypePointer[DType.float32], len: Int):
    # Each thread will process 4 * simd_width elements.
    alias granularity = 4 * simdwidthof[DType.float32]()

    var tid = granularity * (ThreadIdx.x() + BlockDim.x() * BlockIdx.x())

    @always_inline
    @__copy_capture(tid)
    @parameter
    fn func[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        var offset = tid + idx[0]
        if offset >= len:
            return
        buf[offset] = erf(buf[offset])

    elementwise[func, simdwidthof[DType.float32]()](
        StaticIntTuple[1](granularity)
    )


@always_inline
fn _verify_erf_elementwise(asm: String) raises -> None:
    assert_true("test_cuda_target_erf_elementwis" in asm)
    assert_true("tid.x" in asm)
    assert_true("ntid.x" in asm)
    assert_true("ctaid.x" in asm)


def test_erf_elementwise_sm80():
    alias asm = str(
        _compile_code[erf_elementwise, target = _get_nvptx_target()]().asm
    )
    _verify_erf_elementwise(asm)


def test_erf_elementwise_sm90():
    alias asm = str(
        _compile_code[erf_elementwise, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_erf_elementwise(asm)


# ===----------------------------------------------------------------------===#
# Check full kernel
# ===----------------------------------------------------------------------===#


fn erf_kernel(buf: DTypePointer[DType.float32], len: Int):
    var tid = ThreadIdx.x() + BlockDim.y() * BlockIdx.y()

    if tid >= len:
        return

    buf[tid] = erf(buf[tid])


@always_inline
fn _verify_erf_kernel(asm: String) raises -> None:
    assert_true("erf_kernel" in asm)
    assert_true("tid.x" in asm)
    assert_true("ntid.y" in asm)
    assert_true("ctaid.y" in asm)


def test_erf_kernel_sm80():
    alias asm = str(
        _compile_code[erf_kernel, target = _get_nvptx_target()]().asm
    )
    _verify_erf_kernel(asm)


def test_erf_kernel_sm90():
    alias asm = str(
        _compile_code[erf_kernel, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_erf_kernel(asm)


# ===----------------------------------------------------------------------===#
# Check stack allocation
# ===----------------------------------------------------------------------===#


fn test_shared_stack_allocation() -> (
    DTypePointer[DType.int8, AddressSpace.SHARED]
):
    return stack_allocation[
        999, DType.int8, 8, address_space = AddressSpace.SHARED
    ]()


@always_inline
fn _verify_shared_stack_allocation(asm: String) raises -> None:
    assert_true("test_cuda_target_test_shared_" in asm)
    assert_true(".shared .align 8 .b8" in asm)


def test_shared_stack_allocation_sm80():
    alias asm = str(
        _compile_code[
            test_shared_stack_allocation, target = _get_nvptx_target()
        ]().asm
    )
    _verify_shared_stack_allocation(asm)


def test_shared_stack_allocation_sm90():
    alias asm = str(
        _compile_code[
            test_shared_stack_allocation, target = _get_nvptx_target_sm90()
        ]().asm
    )
    _verify_shared_stack_allocation(asm)


# ===----------------------------------------------------------------------===#
# Check barrier
# ===----------------------------------------------------------------------===#


fn test_barrier():
    barrier()


@always_inline
fn _verify_barrier(asm: String) raises -> None:
    assert_true("barrier" in asm)
    assert_true("bar.sync 	0" in asm)


def test_barrier_sm80():
    alias asm = str(
        _compile_code[test_barrier, target = _get_nvptx_target()]().asm
    )
    _verify_barrier(asm)


def test_barrier_sm90():
    alias asm = str(
        _compile_code[test_barrier, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_barrier(asm)


# ===----------------------------------------------------------------------===#
# SGEMM with register coarsening
# ===----------------------------------------------------------------------===#


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
        return Scalar.load(a, row + m * col)

    @always_inline
    fn get_b(row: Int, col: Int) -> Float32:
        return Scalar.load(b, row * n + col)

    @always_inline
    fn set_c(row: Int, col: Int, val: Float32):
        c[row + col * m] = val

    # Allocate B array into shared memory for tiling.
    var b_shared = stack_allocation[
        TILE_SZ_RATIO * TILE_SZ_B,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    # Thread indexing offsets.
    var row = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var col = BlockIdx.y() * TILE_SZ_B

    # Privatization of the C matrix.
    var c_reg = stack_allocation[TILE_SZ_B, DType.float32]()

    memset_zero(c_reg, TILE_SZ_B)

    # Loop over each input tile.
    for tile_idx in range((k - 1) // TILE_SZ_RATIO + 1):
        var i = ThreadIdx.x() // TILE_SZ_B
        var j = ThreadIdx.x() % TILE_SZ_B

        # Load the B matrix into shared memory.
        var b_val: Float32
        if tile_idx * TILE_SZ_RATIO + i < k and col + j < n:
            b_val = get_b(tile_idx * TILE_SZ_RATIO + i, col + j)
        else:
            b_val = 0
        b_shared[i * TILE_SZ_B + j] = b_val

        barrier()

        # Loop within the tile.
        @parameter
        for idx in range(TILE_SZ_RATIO):
            # Load the A tile into the register.
            var a_reg: Float32
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
            set_c(row, col + out_idx, Scalar.load(c_reg, out_idx))


@always_inline
fn _verify_gemm(asm: String) raises -> None:
    assert_true("gemm" in asm)
    assert_true(".shared .align 1 .b8" in asm)
    assert_true("st.shared.f32" in asm)
    assert_true("ld.shared.f32" in asm)


def test_gemm_sm80():
    alias asm = str(_compile_code[gemm, target = _get_nvptx_target()]().asm)
    _verify_gemm(asm)


def test_gemm_sm90():
    alias asm = str(
        _compile_code[gemm, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_gemm(asm)


# ===----------------------------------------------------------------------===#
# shuffle ops
# ===----------------------------------------------------------------------===#


@always_inline
fn _floorlog2[n: Int]() -> Int:
    return 0 if n <= 1 else 1 + _floorlog2[n >> 1]()


@always_inline
fn _static_log2[n: Int]() -> Int:
    return 0 if n <= 1 else _floorlog2[n - 1]() + 1


fn test_shuffle_up(val: Float32) -> Float32:
    var res = val

    alias limit = _static_log2[WARP_SIZE]()

    @parameter
    for mask in reversed(range(limit)):
        res += shuffle_up[DType.float32](res, 1 << mask)
    return res


@always_inline
fn _verify_shuffle_up(asm: String) raises -> None:
    assert_true("test_shuffle_" in asm)
    assert_true("shfl.sync.up.b32" in asm)


def test_shuffle_up_sm80():
    alias asm = str(
        _compile_code[test_shuffle_up, target = _get_nvptx_target()]().asm
    )
    _verify_shuffle_up(asm)


def test_shuffle_up_sm90():
    alias asm = str(
        _compile_code[test_shuffle_up, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_shuffle_up(asm)


fn test_shuffle_down(val: Int32) -> Int32:
    var res = val

    alias limit = _static_log2[WARP_SIZE]()

    @parameter
    for mask in reversed(range(limit)):
        res += shuffle_down[DType.int32](res, 1 << mask)
    return res


@always_inline
fn _verify_shuffle_down(asm: String) raises -> None:
    assert_true("test_shuffle_" in asm)
    assert_true("shfl.sync.down.b32" in asm)


def test_shuffle_down_sm80():
    alias asm = str(
        _compile_code[test_shuffle_down, target = _get_nvptx_target()]().asm
    )
    _verify_shuffle_down(asm)


def test_shuffle_down_sm90():
    alias asm = str(
        _compile_code[
            test_shuffle_down, target = _get_nvptx_target_sm90()
        ]().asm
    )
    _verify_shuffle_down(asm)


# ===----------------------------------------------------------------------===#
# Reduction
# ===----------------------------------------------------------------------===#


fn warp_sum_reduce(val: Float32) -> Float32:
    var res = val

    alias limit = _static_log2[WARP_SIZE]()

    @parameter
    for mask in reversed(range(limit)):
        res += shuffle_xor[DType.float32](res, 1 << mask)
    return res


@always_inline
fn _verify_warp_sum_reduce(asm: String) raises -> None:
    assert_true("warp_sum_" in asm)
    assert_true("shfl.sync.bfly.b32" in asm)


def test_warp_sum_reduce_sm80():
    alias asm = str(
        _compile_code[warp_sum_reduce, target = _get_nvptx_target()]().asm
    )
    _verify_warp_sum_reduce(asm)


def test_warp_sum_reduce_sm90():
    alias asm = str(
        _compile_code[warp_sum_reduce, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_warp_sum_reduce(asm)


fn block_reduce(val: Float32) -> Float32:
    var shared = stack_allocation[
        WARP_SIZE, DType.float32, address_space = AddressSpace.SHARED
    ]()

    alias warp_shift = _static_log2[WARP_SIZE]()

    var lane = lane_id()
    var warp = ThreadIdx.x() // 32

    var warp_sum = warp_sum_reduce(val)

    if lane == 0:
        shared[warp] = warp_sum

    barrier()

    return warp_sum_reduce(
        Scalar.load(shared, lane) if ThreadIdx.x() < BlockDim.x() // 32 else 0
    )


@always_inline
fn _verify_block_reduce(asm: String) raises -> None:
    assert_true("block_reduce" in asm)
    assert_true("mov.u32" in asm)


def test_block_reduce_sm80():
    alias asm = str(
        _compile_code[block_reduce, target = _get_nvptx_target()]().asm
    )
    _verify_block_reduce(asm)


def test_block_reduce_sm90():
    alias asm = str(
        _compile_code[block_reduce, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_block_reduce(asm)


def main():
    @parameter
    if not is_defined["MODULAR_PRODUCTION"]():
        test_parameterized_on_cuda_sm80()
        test_parameterized_on_cuda_sm90()
        test_hello_mojo_sm80()
        test_hello_mojo_sm90()
        test_erf_elementwise_sm80()
        test_erf_elementwise_sm90()
        test_erf_kernel_sm80()
        test_erf_kernel_sm90()
        test_shared_stack_allocation_sm80()
        test_shared_stack_allocation_sm90()
        test_barrier_sm80()
        test_barrier_sm90()
        test_gemm_sm80()
        test_gemm_sm90()
        test_shuffle_up_sm80()
        test_shuffle_up_sm90()
        test_shuffle_down_sm80()
        test_shuffle_down_sm90()
        test_warp_sum_reduce_sm80()
        test_warp_sum_reduce_sm90()
        test_block_reduce_sm80()
        test_block_reduce_sm90()

# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil, min, abs, rsqrt, max, add, neginf, exp
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer
from random import rand
from runtime.llcl import OwningOutputChainPtr, OutputChainPtr, Runtime
from utils.index import Index
from utils.list import DimList
from BatchedMatmul import batched_matmul
from Softmax import softmax
from gpu.host.event import time_function
from sys import argv
from memory.unsafe import AddressSpace as _AddressSpace

from gpu import *
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.memory import AddressSpace
from memory import stack_allocation
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)

from MultiHeadAttention import (
    _naive_attention_with_transpose,
)

# alias type = DType.float32


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


# Using 32 bits index for GPU kernel.
alias _uint32 = Scalar[DType.uint32]


@parameter
@closure
@always_inline
fn _add_capturing[
    type: DType,
    width: Int,
](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
    return x + y


@parameter
@closure
@always_inline
fn _max_capturing[
    type: DType,
    width: Int,
](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
    return max(x, y)


# Helper function for the gemm in attention block.
@always_inline
fn _mm[
    M: _uint32,
    N: _uint32,
    K: _uint32,
    leading_dim_a: _uint32,
    TM: _uint32,
    TN: _uint32,
    transpose_a: Bool,
](
    a: DTypePointer[DType.float32, AddressSpace.SHARED],
    b: DTypePointer[DType.float32, AddressSpace.SHARED],
    row: _uint32,
    col: _uint32,
    reg_m: DTypePointer[DType.float32],
    reg_n: DTypePointer[DType.float32],
    reg_res: DTypePointer[DType.float32],
):
    """Helper function for flash attention to do gemm with inputs from
    shared memory and results in registers."""

    alias simd_size = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_size]]()

    @unroll
    for k in range(int(K)):
        # load a element starting from (row, k) or (k, row) if transposed.
        @parameter
        if transpose_a:
            # vector load
            @unroll
            for offset in range(0, TM.to_int(), simd_size):
                reg_m.simd_store[simd_size](
                    offset,
                    a.aligned_simd_load[simd_size, alignment](
                        int(k * M + row + offset)
                    ),
                )
        else:
            # scalar load
            @unroll
            for i in range(int(TM)):
                reg_m.store(i, a.load(((row + i) * leading_dim_a + k).to_int()))

        @unroll
        for offset in range(0, TN.to_int(), simd_size):
            let vec = b.aligned_simd_load[simd_size, alignment](
                (k * N + col + offset).to_int()
            )
            reg_n.simd_store(offset, vec)

        @unroll
        for i in range(TM.to_int()):

            @unroll
            for j in range(TN.to_int()):
                reg_res.store(
                    (i * TN + j).to_int(),
                    reg_res.load((i * TN + j).to_int())
                    + reg_m.load(i) * reg_n.load(j),
                )


@always_inline
fn _fill[
    len: Int, type: DType, address_space: _AddressSpace
](ptr: DTypePointer[type, address_space], val: Scalar[type]):
    alias simd_width = simdwidthof[val.type]()
    alias vector_end = (len // simd_width) * simd_width

    @unroll
    for i in range(0, vector_end, simd_width):
        ptr.simd_store(i, SIMD[type, simd_width].splat(val))

    @unroll
    for i in range(vector_end, len, 1):
        ptr.store(i, val)


@always_inline
fn _slice_ndbuffer(
    dst: NDBuffer,
    src: NDBuffer,
    offset: StaticIntTuple,  # BSHD
    stride: StaticIntTuple,  # unused for now
):
    alias simd_size = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_size]]()

    alias BM = dst.shape.at[0]().get()
    alias depth = dst.shape.at[1]().get()

    let seq_len = src.dim[1]()
    let num_heads = src.dim[2]()
    let batch_idx = offset[0]
    let seq_idx = offset[1]
    let head_idx = offset[2]
    # Offset in global Q buffer, BSHD layout
    let global_q_offset: _uint32 = depth * (
        head_idx + num_heads * (seq_idx + seq_len * batch_idx)
    )

    let tid = ThreadIdx.x()
    let num_threads = BlockDim.x()
    let loadq_num_rows_per_iter = (num_threads * simd_size) // depth
    let loadq_num_iters = BM // loadq_num_rows_per_iter

    # alias num_threads = 128
    # alias loadq_num_rows_per_iter = (num_threads * simd_size) // depth
    # alias loadq_num_iters = BM // loadq_num_rows_per_iter
    # We transpose Q BSHD -> BHSD. 2 subsequenet rows in q tile have stride
    # != depth in global Q array because the stride is based on BSHD.
    let row_stride = num_heads * depth
    # Index of the 1st row and col loaded by current thread.
    let loadq_row: _uint32 = (tid * simd_size) // depth
    let loadq_col: _uint32 = (tid * simd_size) % depth

    # @unroll
    for i in range(loadq_num_iters):
        let row_in_tile: _uint32 = loadq_row + i * loadq_num_rows_per_iter
        let global_q_idx: _uint32 = global_q_offset + row_in_tile * row_stride + loadq_col
        let vec = src.data.aligned_simd_load[simd_size, alignment](
            global_q_idx.to_int(),
        )
        dst.data.aligned_simd_store[simd_size, alignment](
            (row_in_tile * depth + loadq_col).to_int(), vec.cast[dst.type]()
        )


@always_inline
fn _mma[
    transpose_b: Bool,
    a_addr_space: _AddressSpace,
    a_shape: DimList,
    BK: _uint32 = 16,  #  Blocking factor.
](
    a: NDBuffer[2, a_shape, DType.float32, a_addr_space],  # shared memory
    b: NDBuffer,  # global memory view
    c: NDBuffer,  # thread register tile
):
    alias simd_size = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_size]]()
    alias BM: _uint32 = 32  # a.shape.at[0]().get()
    alias BN: _uint32 = 128
    alias depth: _uint32 = 128  # a.shape.at[1]().get()
    alias num_threads: _uint32 = 128

    alias TM: _uint32 = c.shape.at[0]().get()
    alias TN: _uint32 = c.shape.at[1]().get()
    let reg_m = NDBuffer[1, DimList(int(TM)), DType.float32].stack_allocation()

    let reg_n = NDBuffer[1, DimList(int(TN)), DType.float32].stack_allocation()

    alias num_warps: _uint32 = num_threads // WARP_SIZE

    let tid: _uint32 = ThreadIdx.x()
    let lane: _uint32 = lane_id()
    let warpid: _uint32 = tid // WARP_SIZE

    # Warp index mapping for 2nd gemm.
    alias warp_dim_x: _uint32 = 32
    alias warp_dim_y: _uint32 = 1
    alias num_warps_m: _uint32 = BM // (warp_dim_y * TM)
    alias num_warps_n: _uint32 = depth // (warp_dim_x * TN)
    let warpx: _uint32 = warpid % num_warps_n
    let warpy: _uint32 = warpid // num_warps_n
    # Thread index mapping in MxN matrix.
    # Each warp handles TM rows of output matrix, applicable to both bmms.
    let tx_in_warp: _uint32 = lane % warp_dim_x
    let ty_in_warp: _uint32 = lane // warp_dim_x
    # Thread tile's start row and column in output matrix.
    let mm_row: _uint32 = (ty_in_warp + warpy * warp_dim_y) * TM
    let mm_col: _uint32 = (tx_in_warp + warpx * warp_dim_x) * TN

    alias smem_pad: _uint32 = 0
    alias BN_padded: _uint32 = BN + smem_pad
    let kv_tile = NDBuffer[
        2,
        DimList(int(BN + smem_pad), int(BK)),
        DType.float32,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    @parameter
    if transpose_b:
        let loadk_row: _uint32 = (tid * simd_size) // BK
        let loadk_col: _uint32 = (tid * simd_size) % BK
        let row_stride = b.stride(0)

        alias loadk_num_rows_per_iter: _uint32 = (num_threads * simd_size) // BK
        alias loadk_num_iters: _uint32 = BN // loadk_num_rows_per_iter

        for subtile_start_col in range(0, int(depth), int(BK)):

            @unroll
            for i in range(int(loadk_num_iters)):
                let row_in_tile: _uint32 = loadk_row + i * loadk_num_rows_per_iter
                let global_idx: _uint32 = row_in_tile * row_stride + subtile_start_col + loadk_col
                let vec = b.data.aligned_simd_load[simd_size, alignment](
                    global_idx.to_int()
                )

                # Transpose k tile.
                @unroll
                for j in range(4):
                    kv_tile.data.store(
                        ((loadk_col + j) * BN_padded + row_in_tile).to_int(),
                        vec.cast[DType.float32]()[j],
                    )
            # Gaurd write of q_tile and kv_tile.
            barrier()

            let q_ptr = bitcast[DType.float32](
                a.data.offset(subtile_start_col)
            ).address.address_space_cast[AddressSpace.SHARED]()
            _mm[BM, BN_padded, BK, depth, TM, TN, transpose_a=False](
                q_ptr,
                kv_tile.data,
                mm_row,
                mm_col,
                reg_m.data,
                reg_n.data,
                bitcast[DType.float32](c.data).address.address_space_cast[
                    AddressSpace.GENERIC
                ](),
            )

            # Guard read of kv_tile.
            barrier()

    else:
        let loadv_row: _uint32 = (tid * simd_size) // depth
        let loadv_col: _uint32 = (tid * simd_size) % depth

        let row_stride = b.stride(0)

        @parameter
        if a_addr_space == AddressSpace.GENERIC:
            let p_tile = NDBuffer[
                2,
                DimList(int(BM), int(BK)),
                DType.float32,
                address_space = AddressSpace.SHARED,
            ].stack_allocation()

            alias loadv_num_rows_per_iter = (num_threads * simd_size) // depth
            alias loadv_num_iters = BK // loadv_num_rows_per_iter
            var storep_col_start: _uint32 = 0

            for subtile_start_row in range(0, BN.to_int(), BK.to_int()):
                # Store thread register tile to p sub-tile.
                if (
                    mm_col >= storep_col_start
                    and mm_col < storep_col_start + BK
                ):

                    @unroll
                    for i in range(TM.to_int()):

                        @unroll
                        for j in range(0, TN.to_int(), simd_size):
                            let p_idx = int(
                                (mm_row + i) * BK
                                + mm_col
                                - storep_col_start
                                + j
                            )
                            let vec = a.data.simd_load[simd_size](
                                int(i * TN + j)
                            )
                            p_tile.data.aligned_simd_store[
                                simd_size, alignment
                            ](p_idx, vec.cast[DType.float32]())

                storep_col_start += BK

                # Load v sub-tile.
                @unroll
                for i in range(loadv_num_iters.to_int()):
                    let row_in_tile: _uint32 = loadv_row + i * loadv_num_rows_per_iter
                    let global_idx: _uint32 = (
                        subtile_start_row + row_in_tile
                    ) * row_stride + loadv_col
                    let vec = b.data.aligned_simd_load[simd_size, alignment](
                        global_idx.to_int()
                    )
                    kv_tile.data.aligned_simd_store[simd_size, alignment](
                        (row_in_tile * depth + loadv_col).to_int(),
                        vec.cast[DType.float32](),
                    )
                # Guard writing to p_tile and kv_tile.
                barrier()

                # let p_ptr = p_tile.offset(subtile_start_row)
                _mm[BM, depth, BK, BK, TM, TN, transpose_a=False](
                    p_tile.data,
                    kv_tile.data,
                    mm_row,
                    mm_col,
                    reg_m.data,
                    reg_n.data,
                    bitcast[DType.float32](c.data).address.address_space_cast[
                        AddressSpace.GENERIC
                    ](),
                )
                # Guard reading kv_tile.
                barrier()


@always_inline
fn _rowmax[
    b_shape: DimList
](a: NDBuffer) -> NDBuffer[1, b_shape, DType.float32]:
    alias TM = a.shape.at[0]().get()
    alias TN = a.shape.at[1]().get()

    let rowmax = NDBuffer[1, b_shape, DType.float32].stack_allocation()

    @unroll
    for i in range(TM):
        var val = neginf[DType.float32]()

        @unroll
        for j in range(TN):
            val = max(val, a.data.load(i * TN + j).cast[DType.float32]())

        val = warp_reduce[shuffle_xor, _max_capturing](val)

        rowmax.data.store(i, val.cast[rowmax.type]())

    return rowmax


@always_inline
fn _rowsum[
    b_shape: DimList
](a: NDBuffer) -> NDBuffer[1, b_shape, DType.float32]:
    alias TM = a.shape.at[0]().get()
    alias TN = a.shape.at[1]().get()

    let rowsum = NDBuffer[1, b_shape, DType.float32].stack_allocation()

    @unroll
    for i in range(TM):
        var val = neginf[DType.float32]()

        @unroll
        for j in range(TN):
            val += a.data.load(i * TN + j).cast[DType.float32]()

        val = warp_reduce[shuffle_xor, _add_capturing](val)

        rowsum.data.store(i, val.cast[rowsum.type]())

    return rowsum


@always_inline
fn _exp[
    rank: Int, shape: DimList, type: DType
](a: NDBuffer[rank, shape, type]) -> NDBuffer[rank, shape, type]:
    let res = NDBuffer[rank, shape, type].stack_allocation()

    @parameter
    if rank == 2:
        alias TM = a.shape.at[0]().get()
        alias TN = a.shape.at[1]().get()
        alias simd_size = simdwidthof[type]()

        @unroll
        for i in range(TM):

            @unroll
            for j in range(0, TN, simd_size):
                let idx = i * TN + j
                let vec = a.data.simd_load[simd_size](idx)
                res.data.simd_store[simd_size](exp(vec))

        return res
    else:
        constrained[rank == 1]()
        alias TM = a.shape.at[0]().get()

        @unroll
        for i in range(TM):
            res.data.store(i, exp(res.data.load(i)))

        return res


@always_inline
fn scatter_update(
    a: NDBuffer, offset: StaticIntTuple, stride: StaticIntTuple, b: NDBuffer
):
    alias simd_size = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_size]]()

    alias BM: _uint32 = 32  # a.shape.at[0]().get()
    alias BN: _uint32 = 128
    alias depth: _uint32 = 128  # a.shape.at[1]().get()
    alias num_threads: _uint32 = 128
    alias num_heads = 32

    alias TM: _uint32 = b.shape.at[0]().get()
    alias TN: _uint32 = b.shape.at[1]().get()

    alias num_warps: _uint32 = num_threads // WARP_SIZE

    let tid: _uint32 = ThreadIdx.x()
    let lane: _uint32 = lane_id()
    let warpid: _uint32 = tid // WARP_SIZE

    # Warp index mapping for 2nd gemm.
    alias warp_dim_x: _uint32 = 32
    alias warp_dim_y: _uint32 = 1
    alias num_warps_m: _uint32 = BM // (warp_dim_y * TM)
    alias num_warps_n: _uint32 = depth // (warp_dim_x * TN)
    let warpx: _uint32 = warpid % num_warps_n
    let warpy: _uint32 = warpid // num_warps_n
    # Thread index mapping in MxN matrix.
    # Each warp handles TM rows of output matrix, applicable to both bmms.
    let tx_in_warp: _uint32 = lane % warp_dim_x
    let ty_in_warp: _uint32 = lane // warp_dim_x
    # Thread tile's start row and column in output matrix.
    let mm_row: _uint32 = (ty_in_warp + warpy * warp_dim_y) * TM
    let mm_col: _uint32 = (tx_in_warp + warpx * warp_dim_x) * TN

    let batch_idx: _uint32 = BlockIdx.z()
    let head_idx: _uint32 = BlockIdx.y()
    let seq_idx: _uint32 = BlockIdx.x() * BM
    let seq_len: _uint32 = a.dim[1]()

    let row_stride = num_heads * depth

    let global_q_offset: _uint32 = depth * (
        head_idx + num_heads * (seq_idx + seq_len * batch_idx)
    )
    var o_global_row_offset = global_q_offset + mm_row * row_stride

    @unroll
    for i in range(TM.to_int()):

        @unroll
        for offset in range(0, TN.to_int(), simd_size):
            # Apply the denominator of softmax.
            let vec = b.data.simd_load[simd_size](int(i * TN + offset))

            a.data.aligned_simd_store[simd_size, alignment](
                int(o_global_row_offset + mm_col + offset), vec.cast[a.type]()
            )
        o_global_row_offset += row_stride


@__llvm_metadata(`nvvm.maxntid`=[int(num_threads)])
fn flash_attention_kernel[
    BM: _uint32,  # number of queries per block
    BN: _uint32,  # number of keys per block
    BK: _uint32,  # tile size in depth dimension
    depth: _uint32,
    num_heads: _uint32,
    TM: _uint32,
    TN: _uint32,
    num_threads: _uint32,
    q_shape: DimList,
    k_shape: DimList,
    v_shape: DimList,
    mask_shape: DimList,
    output_shape: DimList,
](
    query: NDBuffer[4, q_shape, DType.float32],
    key: NDBuffer[4, k_shape, DType.float32],
    value: NDBuffer[4, v_shape, DType.float32],
    mask: NDBuffer[3, mask_shape, DType.float32],
    output: NDBuffer[4, output_shape, DType.float32],
    scale: Float32,
    batch_size: _uint32,
    seq_len: _uint32,
):
    constrained[TN == 4, "Only support TN=4 for llama2 shape"]()
    constrained[TM % 4 == 0, "TM should be multiple of 4"]()
    constrained[
        BM == TM * (num_threads // WARP_SIZE), "Incompatible block size"
    ]()
    constrained[BN == TN * WARP_SIZE, "Incompatible block size"]()

    let q_tile = NDBuffer[
        2,
        DimList(int(BM), int(depth)),
        DType.float32,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    let rowmax = NDBuffer[1, DimList(int(TM)), DType.float32].stack_allocation()

    var rowsum = NDBuffer[1, DimList(int(TM)), DType.float32].stack_allocation()
    rowsum.zero()

    var reg_result = NDBuffer[
        2, DimList(int(TM), int(TN)), DType.float32
    ].stack_allocation()

    var o_thread_tile = NDBuffer[
        2, DimList(int(TM), int(TN)), DType.float32
    ].stack_allocation()
    o_thread_tile.zero()

    var correction = NDBuffer[
        1, DimList(int(TM)), DType.float32
    ].stack_allocation()

    let batch_idx: _uint32 = BlockIdx.z()
    let head_idx: _uint32 = BlockIdx.y()
    let q_tile_idx: _uint32 = BlockIdx.x()

    _slice_ndbuffer(
        q_tile,
        query,
        Index(0, int(q_tile_idx * BM), int(head_idx), 0),
        Index(0, 1, 0, 1),
    )

    _fill[int(TM)](rowmax.data, neginf[DType.float32]())

    # Offset of K/V tile in global K/V buffer, i.e., 1st element of current head.
    var global_kv_offset: _uint32 = depth * (
        head_idx + num_heads * seq_len * batch_idx
    )

    # Idealy the main loop is as follow (with names changed to match the paper)
    #
    # let k_view = slice_view(key, ...)
    # mma(q_tile, k_view, reg_result, ...)
    #
    # let curr_rowmax = max(rowmax, reg_result.reduce_max(axis=1))
    # let correction = exp(rowmax - curr_rowmax)
    # reg_result = exp(reg_result - curr_rowmax)
    # rowsum = rowsum * correction + reg_result.reduce_add(axis=1)
    # o_thread_tile *= correction
    #
    # let v_view = slice_view(value, ...)
    # mma(reg_result, v_view, o_thread_tile, ...)

    for kv_tile_start_row in range(0, int(seq_len), int(BN)):
        # Clear thread tile results.
        reg_result.zero()

        let k_view = NDBuffer[2, DimList(int(BN), int(depth)), DType.float32](
            key.data.offset(int(global_kv_offset)),
            dynamic_shape=Index(int(BN), int(depth)),
            dynamic_stride=Index(int(num_heads * depth), 1),
        )
        _mma[transpose_b=True](q_tile, k_view, reg_result)

        reg_result *= scale

        var curr_rowmax: NDBuffer[1, DimList(int(TM)), DType.float32] = _rowmax[
            DimList(int(TM))
        ](reg_result)

        curr_rowmax.max(rowmax)

        correction = _exp(rowmax - curr_rowmax)

        reg_result = _exp(reg_result - curr_rowmax)

        let curr_rowsum: NDBuffer[1, DimList(int(TM)), DType.float32] = _rowsum[
            DimList(int(TM))
        ](reg_result)

        rowsum = rowsum * correction + curr_rowsum

        o_thread_tile *= correction

        let v_view = NDBuffer[2, DimList(int(BN), int(depth)), DType.float32](
            value.data.offset(int(global_kv_offset)),
            dynamic_shape=Index(int(BN), int(depth)),
            dynamic_stride=Index(int(num_heads * depth), 1),
        )
        _mma[transpose_b=False](reg_result, v_view, o_thread_tile)

        # Point to  next tile
        global_kv_offset += BN * num_heads * depth

    o_thread_tile /= rowsum

    scatter_update(
        output,
        Index(0, int(q_tile_idx * BM), int(head_idx), 0),
        Index(0, 1, 0, 1),
        o_thread_tile,
    )


# CHECK-LABEL: test_flash_attention
fn test(seq_len: Int, num_keys: Int, is_benchmark: Bool = False) raises:
    print("test_flash_attention")

    # Query, key, value dimensions.
    alias batch_size = 1
    alias num_heads = 32
    alias depth = 128
    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))

    # Q, K, V shapes.
    let q_size = batch_size * num_heads * seq_len * depth
    let k_size = batch_size * num_heads * num_keys * depth
    let v_size = k_size
    let o_size = q_size

    # Allocate memory for all variables.
    let q_ptr = DTypePointer[DType.float32].alloc(q_size)
    let k_ptr = DTypePointer[DType.float32].alloc(k_size)
    let v_ptr = DTypePointer[DType.float32].alloc(v_size)
    let mask_ptr = DTypePointer[DType.float32].alloc(seq_len * num_keys)
    let output_ptr = DTypePointer[DType.float32].alloc(o_size)
    let flash_output_ptr = DTypePointer[DType.float32].alloc(o_size)

    # Q, K, V are randomly initalized.
    rand[DType.float32](q_ptr, q_size)
    rand[DType.float32](k_ptr, k_size)
    rand[DType.float32](v_ptr, v_size)
    # rand[DType.float32](mask_ptr, seq_len * num_keys)

    # Contruct buffers.
    let q = NDBuffer[4, DimList.create_unknown[4](), DType.float32](
        q_ptr, Index(batch_size, seq_len, num_heads, depth)
    )
    let k = NDBuffer[4, DimList.create_unknown[4](), DType.float32](
        k_ptr, Index(batch_size, num_keys, num_heads, depth)
    )
    let v = NDBuffer[4, DimList.create_unknown[4](), DType.float32](
        v_ptr, Index(batch_size, num_keys, num_heads, depth)
    )
    let mask = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        mask_ptr, Index(seq_len, num_keys)
    )
    let output = NDBuffer[4, DimList.create_unknown[4](), DType.float32](
        output_ptr, Index(batch_size, seq_len, num_heads, depth)
    )

    mask.zero()

    _naive_attention_with_transpose[DType.float32](
        rebind[NDBuffer[4, DimList.create_unknown[4](), DType.float32]](output),
        rebind[NDBuffer[4, DimList.create_unknown[4](), DType.float32]](q),
        rebind[NDBuffer[4, DimList.create_unknown[4](), DType.float32]](k),
        rebind[NDBuffer[4, DimList.create_unknown[4](), DType.float32]](v),
        rebind[NDBuffer[2, DimList.create_unknown[2](), DType.float32]](mask),
        scale,
    )

    let stream = Stream()

    # Device pointers
    let q_device_ptr = _malloc[DType.float32](q_size)
    let k_device_ptr = _malloc[DType.float32](k_size)
    let v_device_ptr = _malloc[DType.float32](v_size)
    let mask_device_ptr = _malloc[DType.float32](seq_len * num_keys)
    let output_device_ptr = _malloc[DType.float32](o_size)

    # Copy from host to device
    _copy_host_to_device(q_device_ptr, q_ptr, q_size)
    _copy_host_to_device(k_device_ptr, k_ptr, k_size)
    _copy_host_to_device(v_device_ptr, v_ptr, v_size)
    _copy_host_to_device(mask_device_ptr, mask_ptr, seq_len * num_keys)

    let q_device = NDBuffer[4, DimList.create_unknown[4](), DType.float32](
        q_device_ptr, Index(batch_size, seq_len, num_heads, depth)
    )
    let k_device = NDBuffer[4, DimList.create_unknown[4](), DType.float32](
        k_device_ptr, Index(batch_size, seq_len, num_heads, depth)
    )
    let v_device = NDBuffer[4, DimList.create_unknown[4](), DType.float32](
        v_device_ptr, Index(batch_size, seq_len, num_heads, depth)
    )
    let mask_device = NDBuffer[3, DimList.create_unknown[3](), DType.float32](
        mask_device_ptr, Index(1, seq_len, seq_len)
    )
    let output_device = NDBuffer[4, DimList.create_unknown[4](), DType.float32](
        output_device_ptr, Index(batch_size, seq_len, num_heads, depth)
    )

    alias q_tile_num_rows = 32

    if seq_len == num_keys and seq_len % 128 == 0:
        let func = Function[
            fn (
                NDBuffer[4, DimList.create_unknown[4](), DType.float32],
                NDBuffer[4, DimList.create_unknown[4](), DType.float32],
                NDBuffer[4, DimList.create_unknown[4](), DType.float32],
                NDBuffer[3, DimList.create_unknown[3](), DType.float32],
                NDBuffer[4, DimList.create_unknown[4](), DType.float32],
                Float32,
                Scalar[DType.uint32],
                Scalar[DType.uint32],
            ) -> None, flash_attention_kernel[
                BM=32,  # q_tile_num_rows,
                BN=128,  # kv_tile_num_rows,
                BK=16,
                depth=128,
                num_heads=num_heads,
                TM=8,
                TN=4,
                num_threads=128,  # q_tile_num_rows * kv_tile_num_rows,
                q_shape = DimList.create_unknown[4](),
                k_shape = DimList.create_unknown[4](),
                v_shape = DimList.create_unknown[4](),
                mask_shape = DimList.create_unknown[3](),
                output_shape = DimList.create_unknown[4](),
            ]
        ]()

        if is_benchmark:
            alias nrun = 1000

            @always_inline
            @parameter
            fn run_func(stream: Stream) raises:
                for i in range(nrun):
                    func(
                        # grid
                        (
                            div_ceil(seq_len, q_tile_num_rows),
                            num_heads,
                            batch_size,
                        ),
                        # block
                        (128, 1, 1),
                        q_device,
                        k_device,
                        v_device,
                        mask_device,
                        output_device,
                        scale,
                        batch_size,
                        seq_len,
                        stream=stream,
                    )

            # Warmup
            run_func(stream)

            var nstime = time_function[run_func](stream) / nrun
            let sectime = nstime / 1000000
            print(nrun, "runs avg", sectime, "ms")

        else:
            func(
                # grid
                (div_ceil(seq_len, q_tile_num_rows), num_heads, batch_size),
                # block
                (128, 1, 1),
                q_device,
                k_device,
                v_device,
                mask_device,
                output_device,
                scale,
                batch_size,
                seq_len,
                stream=stream,
            )

    synchronize()

    _copy_device_to_host(flash_output_ptr, output_device_ptr, q_size)

    var succeed = True
    for h in range(num_heads):
        for s in range(seq_len):
            for d in range(depth):
                let expect = output_ptr.load(d + depth * (h + s * num_heads))
                let actual = flash_output_ptr.load(
                    d + depth * (h + s * num_heads)
                )
                if abs(expect - actual) > 1e-4 * abs(expect):
                    print(d, expect, actual)
                    succeed = False
                    break

    _free(q_device_ptr)
    _free(k_device_ptr)
    _free(v_device_ptr)
    _free(mask_device_ptr)
    _free(output_device_ptr)

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    mask_ptr.free()
    output_ptr.free()
    flash_output_ptr.free()

    # CHECK: Succeed
    if succeed:
        print("Succeed")

    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            test(1024, 1024, is_benchmark())  # only benchmark a large shape

    except e:
        print("CUDA_ERROR:", e)

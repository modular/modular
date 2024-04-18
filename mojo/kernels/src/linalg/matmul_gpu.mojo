# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import align_down, div_ceil
from sys.info import alignof

from algorithm.functional import tile_and_unswitch
from buffer.buffer import (
    Buffer,
    NDBuffer,
)
from buffer.list import Dim, DimList
from .Gemv import gemv
from gpu import WARP_SIZE, BlockDim, BlockIdx, ThreadIdx, barrier, lane_id
from gpu.host import Function, Stream
from gpu.host.memory import _memset_async
from gpu.memory import AddressSpace
from gpu.shuffle import shuffle_down, shuffle_idx, warp_reduce
from gpu.tensor_ops import tc_reduce
from .MatmulUtils import (
    MatmulConfig,
    GemmShape,
    elementwise_epilogue_type,
)
from memory import stack_allocation
from memory.unsafe import DTypePointer, bitcast

from collections import OptionalReg as Optional
from utils.index import Index, StaticIntTuple
from utils.static_tuple import StaticTuple


@always_inline
fn __nvvm_ldg_f4[type: DType](x: DTypePointer[type]) -> SIMD[type, 4]:
    # Load a register variable from global state space via non-coherent cache.

    alias alignment = Int32(alignof[SIMD[type, 4]]())

    @parameter
    if type == DType.float32:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4f32.p0v4f32", SIMD[DType.float32, 4]
            ](x.bitcast[DType.float32](), alignment)
        )
    elif type == DType.bfloat16:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4bf16.p0v4bf16",
                SIMD[DType.bfloat16, 4],
            ](x.bitcast[DType.bfloat16](), alignment)
        )
    elif type == DType.float16:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4f16.p0v4f16",
                SIMD[DType.float16, 4],
            ](x.bitcast[DType.float16](), alignment)
        )
    else:
        constrained[False, "Unhandled DType"]()
        return 0


# BM: The threadblock size for M dimension SMEM caching.
# BN: The threadblock size for N dimension SMEM caching.
# BK: The threadblock size for K dimension SMEM caching.
# WM: M dim of continuous tile computed by each warp.
# WN: N dim of continuous tile computed by each warp.
# WMITER: The number of subwarp tiling steps in M dimension.
# WNITER: The number of subwarp tiling steps in N dimension.
# TM: The per-thread tile size for M dimension.
# TN: The per-thread tile size for N dimension.
@__llvm_metadata(
    `nvvm.maxntid`=StaticTuple[Int32, 1](NUM_THREADS.cast[DType.int32]())
)
fn sgemm_warp_tiling_kernel[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    indexing_integral_dtype: DType,
    BM: Scalar[indexing_integral_dtype],
    BN: Scalar[indexing_integral_dtype],
    BK: Scalar[indexing_integral_dtype],
    WM: Scalar[indexing_integral_dtype],
    WN: Scalar[indexing_integral_dtype],
    WMITER: Scalar[indexing_integral_dtype],
    WNITER: Scalar[indexing_integral_dtype],
    TM: Scalar[indexing_integral_dtype],
    TN: Scalar[indexing_integral_dtype],
    NUM_THREADS: Scalar[indexing_integral_dtype],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    mat_c: NDBuffer[c_type, 2, c_shape],
    mat_a: NDBuffer[a_type, 2, a_shape],
    mat_b: NDBuffer[b_type, 2, b_shape],
    alpha: Scalar[c_type],
    beta: Scalar[c_type],
):
    var M: Scalar[indexing_integral_dtype] = mat_c.dim(0)
    var K: Scalar[indexing_integral_dtype] = mat_a.dim(1)
    var N: Scalar[indexing_integral_dtype] = mat_c.dim(1)

    var c_row: Scalar[indexing_integral_dtype] = BlockIdx.y()
    var c_col: Scalar[indexing_integral_dtype] = BlockIdx.x()

    # Placement of the warp in the threadblock tile.
    var warp_idx = Scalar[indexing_integral_dtype](
        ThreadIdx.x()
    ) // WARP_SIZE  # the warp this thread is in
    var warp_col = warp_idx % (BN // WN)
    var warp_row = warp_idx // (BN // WN)

    # Size of the warp subtile.
    alias w_sub_m = WM // WMITER  # 64/2=32
    alias w_sub_n = WN // WNITER  # 32/2=16

    # Placement of the thread in the warp subtile.
    var thread_Idx_In_warp = Scalar[indexing_integral_dtype](
        ThreadIdx.x()
    ) % WARP_SIZE  # [0, 31]
    var thread_col_in_warp = thread_Idx_In_warp % (w_sub_n // TN)  # i%(16/4)
    var thread_row_in_warp = thread_Idx_In_warp // (w_sub_n // TN)  # i/4

    # Allocate space for the current blocktile in SMEM.
    # Pad the A tile in share memory to avoid bank conflicts.
    # Use 4 to comply with f4 alignment used in accumulation.
    alias sram_bank_padding_size = 4
    alias BM_padded = BM + sram_bank_padding_size
    var a_sram = NDBuffer[
        a_type,
        1,
        DimList(int(BK * BM_padded)),
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var b_sram = NDBuffer[
        b_type,
        1,
        DimList(int(BK * BN)),
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Move blocktile to beginning of A's row and B's column.
    var aa_ptr = mat_a._offset(Index(c_row * BM, 0))
    var bb_ptr = mat_b._offset(Index(0, c_col * BN))
    # Move C_ptr to warp's output tile
    var M_offset_warp = c_row * BM + warp_row * WM
    var N_offset_warp = c_col * BN + warp_col * WN
    var cc_ptr = mat_c._offset(Index(M_offset_warp, N_offset_warp))

    # Calculate the indices that this thread will load into SMEM.
    # We load 128bit / 32bit = 4 elements per thread at each step.
    var inner_row_a = Scalar[indexing_integral_dtype](ThreadIdx.x()) // (
        BK // 4
    )
    var inner_col_a = Scalar[indexing_integral_dtype](ThreadIdx.x()) % (BK // 4)
    alias row_stride_a = (NUM_THREADS * 4) // BK
    var inner_row_b = Scalar[indexing_integral_dtype](ThreadIdx.x()) // (
        BN // 4
    )
    var inner_co_ib = Scalar[indexing_integral_dtype](ThreadIdx.x()) % (BN // 4)
    alias row_stride_b = NUM_THREADS // (BN // 4)

    # TODO: We want these to be register-allocated!
    # Allocate thread-local cache for results in register file.
    var thread_results = NDBuffer[
        c_type,
        4,
        DimList(int(WMITER), int(WNITER), int(TM), int(TN)),
    ]().stack_allocation()
    thread_results.zero()

    # We cache into registers on the warptile level.
    var reg_m = NDBuffer[
        a_type, 2, DimList(int(WMITER), int(TM))
    ]().stack_allocation()
    reg_m.zero()

    var reg_n = NDBuffer[
        b_type, 2, DimList(int(WNITER), int(TN))
    ]().stack_allocation()
    reg_n.zero()

    # Outer-most loop over block tiles.
    for bk_idx in range(0, int(K), int(BK)):
        for offset in range(0, int(BM - row_stride_a + 1), int(row_stride_a)):
            # Load 4 elements at a time and store to shared memory.
            var tmp = __nvvm_ldg_f4[a_type](
                aa_ptr.offset(int((inner_row_a + offset) * K + inner_col_a * 4))
            )

            @unroll
            for i in range(4):
                a_sram[
                    int(
                        (inner_col_a * 4 + i) * BM_padded + inner_row_a + offset
                    )
                ] = tmp[i]

        for offset in range(0, int(BK - row_stride_b + 1), int(row_stride_b)):
            # Load 4 elements at a time and store to shared memory.
            var tmp = __nvvm_ldg_f4[b_type](
                bb_ptr.offset(int((inner_row_b + offset) * N + inner_co_ib * 4))
            )
            b_sram.store[width=4, alignment=16](
                Index((inner_row_b + offset) * BN + inner_co_ib * 4),
                tmp,
            )

        barrier()

        for dot_idx in range(BK):
            # Populate registers for whole warptile.
            @unroll
            for w_sub_row_idx in range(WMITER):

                @unroll
                for i in range(0, int(TM), 4):
                    var vec = a_sram.load[width=4, alignment=16](
                        int(
                            (dot_idx * BM_padded)
                            + warp_row * WM
                            + w_sub_row_idx * w_sub_m
                            + thread_row_in_warp * TM
                            + i
                        )
                    )
                    reg_m.store(Index(w_sub_row_idx, i), vec)

            @unroll
            for w_sub_col_idx in range(WNITER):

                @unroll
                for i in range(0, int(TN), 4):
                    var vec = b_sram.load[width=4, alignment=16](
                        int(
                            (dot_idx * BN)
                            + warp_col * WN
                            + w_sub_col_idx * w_sub_n
                            + thread_col_in_warp * TN
                        )
                    )
                    reg_n.store(Index(w_sub_col_idx, i), vec)

            # Execute warptile matmul.
            @unroll
            for w_sub_row_idx in range(WMITER):

                @unroll
                for w_sub_col_idx in range(WNITER):
                    # Calculate per-thread results.
                    @unroll
                    for res_idx_m in range(TM):

                        @unroll
                        for res_idx_n in range(TN):
                            thread_results[
                                Index(
                                    w_sub_row_idx,
                                    w_sub_col_idx,
                                    res_idx_m,
                                    res_idx_n,
                                )
                            ] += (
                                reg_m[w_sub_row_idx, res_idx_m].cast[c_type]()
                                * reg_n[w_sub_col_idx, res_idx_n].cast[c_type]()
                            )
        aa_ptr = aa_ptr.offset(int(BK))  # move BK columns to right
        bb_ptr = bb_ptr.offset(int(BK * N))  # move BK rows down
        barrier()

    # Write out the results.
    @unroll
    for w_sub_row_idx in range(WMITER):

        @unroll
        for w_sub_col_idx in range(WNITER):
            # Move C pointer to current warp subtile.
            var M_offset_subtile = w_sub_row_idx * w_sub_m
            var N_offset_subtile = w_sub_col_idx * w_sub_n
            var C_interim = cc_ptr.offset(
                int((M_offset_subtile) * N + N_offset_subtile)
            )

            @unroll
            for res_idx_m in range(TM):

                @unroll
                for res_idx_n in range(0, int(TN), 4):
                    var M_offset_val = thread_row_in_warp * TM + res_idx_m
                    var N_offset_val = thread_col_in_warp * TN + res_idx_n
                    var c_idx = M_offset_val * N + N_offset_val
                    var result_vec = thread_results.load[width=4](
                        Index(
                            w_sub_row_idx,
                            w_sub_col_idx,
                            res_idx_m,
                            res_idx_n,
                        )
                    )

                    var vec = alpha * result_vec + beta * C_interim.load[
                        width=4, alignment=16
                    ](int(c_idx))

                    @parameter
                    if elementwise_lambda_fn:
                        alias elementwise_lambda = elementwise_lambda_fn.value()
                        elementwise_lambda[c_type, 4](
                            Index(
                                M_offset_warp + M_offset_subtile + M_offset_val,
                                N_offset_warp + N_offset_subtile + N_offset_val,
                            ),
                            vec,
                        )
                    else:
                        C_interim.store[width=4, alignment=16](int(c_idx), vec)


# Matrix-Column Vector Multiplication
fn gemv_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    s_type: DType = c_type,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: DTypePointer[c_type],
    a: DTypePointer[a_type],
    b: DTypePointer[b_type],
    m: Int,
    n: Int,
    k: Int,
):
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var warpId = x // WARP_SIZE
    var accum = SIMD[s_type, 1]()

    if warpId < m:
        # Every warp processes a single row of the resultant vector
        for i in range(div_ceil(k, WARP_SIZE)):
            var idx = i * WARP_SIZE + int(lane_id())
            var val = SIMD[s_type, 1]()
            if idx < k:
                val = (
                    a.load(warpId * k + idx).cast[s_type]()
                    * b.load(idx).cast[s_type]()
                )

            @parameter
            fn reduce_add[
                type: DType,
                width: Int,
            ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
                return x + y

            val = warp_reduce[shuffle_down, reduce_add](val)

            if lane_id() == 0:
                accum += val

        if lane_id() == 0:

            @parameter
            if elementwise_lambda_fn:
                alias elementwise_lambda = elementwise_lambda_fn.value()
                elementwise_lambda[c_type, 1](
                    Index(warpId, 0), accum.cast[c_type]()
                )
            else:
                c[warpId] = accum.cast[c_type]()


# Matrix-Column Vector Multiplication utilizing Tensor Cores
fn gemv_tc_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    s_type: DType = c_type,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: DTypePointer[c_type],
    a: DTypePointer[a_type],
    b: DTypePointer[b_type],
    m: Int,
    n: Int,
    k: Int,
):
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var warpId = x // WARP_SIZE
    var accum = Scalar[s_type]()

    if warpId < m:
        # Every warp processes a single row of the resultant vector
        for i in range(div_ceil(k, WARP_SIZE)):
            var idx = i * WARP_SIZE + int(lane_id())
            var val = Scalar[a_type]()
            if idx < k:
                val = a.load(warpId * k + idx) * b.load(idx).cast[a_type]()

            var out_val = Scalar[s_type]()
            out_val = tc_reduce[s_type, a_type](val)

            if lane_id() == 0:
                accum += out_val

        if lane_id() == 0:

            @parameter
            if elementwise_lambda_fn:
                alias elementwise_lambda = elementwise_lambda_fn.value()
                elementwise_lambda[c_type, 1](
                    Index(warpId, 0), accum.cast[c_type]()
                )
            else:
                c[warpId] = accum.cast[c_type]()


# Row Vector-Matrix multiplication
fn gevm_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    tile_size: Int,
    s_type: DType = c_type,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: DTypePointer[c_type],
    a: DTypePointer[a_type],
    b: DTypePointer[b_type],
    m: Int,
    n: Int,
    k: Int,
):
    var warpsPerBlock = BlockDim.x() // WARP_SIZE
    var warpId = ThreadIdx.x() // WARP_SIZE
    var accum = SIMD[s_type, 1]()
    var col = BlockIdx.x() * WARP_SIZE + int(lane_id())
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var globalWarpId = x // WARP_SIZE

    var x_shared = stack_allocation[
        tile_size,
        s_type,
        address_space = AddressSpace.SHARED,
    ]()

    # Every block computes warp size length of output values
    for i in range(div_ceil(k, warpsPerBlock)):
        var val = SIMD[c_type, 1]()
        var row = i * warpsPerBlock + warpId
        if lane_id() == 0:
            val = a.load(row).cast[c_type]()
        val = shuffle_idx(val, 0)
        accum += val.cast[s_type]() * b.load(row * n + col).cast[s_type]()

    x_shared[int(lane_id()) * WARP_SIZE + warpId] = accum
    barrier()

    @parameter
    fn reduce_add[
        type: DType,
        width: Int,
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return x + y

    var total = SIMD[s_type, 1]()
    total = x_shared.load(ThreadIdx.x()).cast[s_type]()
    total = warp_reduce[shuffle_down, reduce_add](total)

    if lane_id() == 0:

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                Index(0, globalWarpId), total.cast[c_type]()
            )
        else:
            c[globalWarpId] = total.cast[c_type]()


fn matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    tile_size: Int,
    s_type: DType = c_type,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_ptr: DTypePointer[c_type],
    a_ptr: DTypePointer[a_type],
    b_ptr: DTypePointer[b_type],
    m: Int,
    n: Int,
    k: Int,
):
    """Matrix Multiplication using shared memory.
    This version loads blocks of size tile_size x tile_size from A and B
    and updates a tile_size x tile_size in C.

    The thread block should have shape (tile_size, tile_size, 1). Each
    thread is mapped one element in C. The grid should have shape
    (N/tile_size, M/tile_size, 1). N is the first dimension for coalesced
    access.
    """
    var a = NDBuffer[a_type, 2](a_ptr, Index(m, k))
    var b = NDBuffer[b_type, 2](b_ptr, Index(k, n))
    var c = NDBuffer[c_type, 2](c_ptr, Index(m, n))

    # Allocate A, B tile in shared memory.
    var a_shared = stack_allocation[
        tile_size * tile_size,
        a_type,
        address_space = AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        tile_size * tile_size,
        b_type,
        address_space = AddressSpace.SHARED,
    ]()

    # Global index in C.
    # These are the same indices in A and B when loading to SRAM.
    # Map thread x to column for coalesced access in B.
    var col = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var row = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    # Local index in the c sub-matrix updated by current block.
    var localCol = ThreadIdx.x()
    var localRow = ThreadIdx.y()

    # Result of current thread in C.
    var result = SIMD[s_type, 1](0.0)

    var K_roundbytile = align_down(k, tile_size)
    # Can't use 0 as tile size so set to 1 when the remainder is 0.
    var K_remainder = k - K_roundbytile if k - K_roundbytile > 0 else 1

    @parameter
    @__copy_capture(row, localCol, a, b, localRow, col, a_shared, b_shared)
    @always_inline
    fn update_tile[full_tile: Bool](offset: Int, end: Int, tile_size: Int):
        # If K is not multiple of tile_size, the last tile contains less than
        # tile_size elements. The thread block needs to take addition bound check
        # when loading elements into shared memory.

        # Load A tile into shared memory.
        var a_val: SIMD[a_type, 1]

        @parameter
        if not full_tile:
            a_val = a[row, offset + localCol] if (
                row < m and offset + localCol < k
            ) else 0.0
        else:
            a_val = a[row, offset + localCol] if row < m else 0.0
        a_shared[localRow * tile_size + localCol] = a_val

        # Load B tile into shared memory.
        var b_val: SIMD[b_type, 1]

        @parameter
        if not full_tile:
            b_val = b[offset + localRow, col] if (
                col < n and offset + localRow < k
            ) else 0.0
        else:
            b_val = b[offset + localRow, col] if col < n else 0.0
        b_shared[localRow * tile_size + localCol] = b_val

        barrier()

        for kk in range(tile_size):
            result += (
                a_shared[localRow * tile_size + kk].cast[s_type]()
                * b_shared[kk * tile_size + localCol].cast[s_type]()
            )

        barrier()

    tile_and_unswitch[update_tile](
        0, k, VariadicList[Int](tile_size, K_remainder)
    )

    if row < m and col < n:

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                Index(row, col), result.cast[c_type]()
            )
        else:
            c[Index(row, col)] = result.cast[c_type]()


fn matmul_kernel_naive[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    BLOCK_DIM: Int,
    s_type: DType = c_type,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_ptr: DTypePointer[c_type],
    a_ptr: DTypePointer[a_type],
    b_ptr: DTypePointer[b_type],
    m: Int,
    n: Int,
    k: Int,
):
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var y = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    if x >= m or y >= n:
        return

    var a = NDBuffer[a_type, 2](a_ptr, Index(m, k))
    var b = NDBuffer[b_type, 2](b_ptr, Index(k, n))
    var c = NDBuffer[c_type, 2](c_ptr, Index(m, n))

    var accum = SIMD[s_type, 1]()
    for i in range(k):
        accum = a[x, i].cast[s_type]() * b[i, y].cast[s_type]() + accum

    @parameter
    if elementwise_lambda_fn:
        alias elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](Index(x, y), accum.cast[c_type]())
    else:
        c[Index(x, y)] = accum.cast[c_type]()


@always_inline
fn _matmul_gpu[
    config: MatmulConfig,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    single_thread_blocking_override: Bool = False,
](
    c: NDBuffer[config.c_type, 2, config.c_shape],
    a: NDBuffer[config.a_type, 2, config.a_shape],
    b: NDBuffer[config.b_type, 2, config.b_shape],
    kernel_type_m: Int,
    num_threads: Int = -1,
):
    # HACK HACK HACK https://github.com/modularml/modular/issues/22959
    # single_thread_blocking_override should not be allowed, but the graph
    # compiler has a special case that does not insert the
    # on the GPU
    # constrained[
    #     not single_thread_blocking_override,
    #     "single_thread_blocking_override not applicable",
    # ]()
    constrained[config.transpose_a == False, "only NN matmul is supported"]()
    constrained[config.transpose_b == False, "only NN matmul is supported"]()
    constrained[not config.b_packed, "pre-packing not yet supported"]()
    constrained[
        not config.saturated_vnni, "saturated_vnni_flag not applicable"
    ]()

    var shape = GemmShape.get[False, False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    # TODO: #25898, use max_finite
    alias max_uint32 = Int(0xFFFFFFFF)
    var use_32bit_indexing = m * n < max_uint32 and m * k < max_uint32 and n * k < max_uint32

    @parameter
    if elementwise_lambda_fn:
        if use_32bit_indexing:
            _matmul_gpu_dispatch[
                config.a_type,
                config.a_shape,
                config.b_type,
                config.b_shape,
                config.c_type,
                config.c_shape,
                indexing_integral_dtype = DType.uint32,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](c, a, b)
        else:
            _matmul_gpu_dispatch[
                config.a_type,
                config.a_shape,
                config.b_type,
                config.b_shape,
                config.c_type,
                config.c_shape,
                indexing_integral_dtype = DType.uint64,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](c, a, b)

    else:
        if use_32bit_indexing:
            _matmul_gpu_dispatch[
                config.a_type,
                config.a_shape,
                config.b_type,
                config.b_shape,
                config.c_type,
                config.c_shape,
                indexing_integral_dtype = DType.uint32,
            ](c, a, b)
        else:
            _matmul_gpu_dispatch[
                config.a_type,
                config.a_shape,
                config.b_type,
                config.b_shape,
                config.c_type,
                config.c_shape,
                indexing_integral_dtype = DType.uint64,
            ](c, a, b)


@always_inline
fn _matmul_gpu_dispatch[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    indexing_integral_dtype: DType,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
):
    var shape = GemmShape.get[False, False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K
    try:
        var stream = Stream.get_current_stream()

        alias s_type = DType.float32 if (
            a_type == DType.bfloat16 or a_type == DType.float16
        ) else c_type

        # Currently sgemm_warp_tiling_kernel is supportred only for float32 and
        # no elementwise_epilogue, fallback to generic matmul_kernel.
        var warp_tiled_matmul_suppoered_shape = (
            m % 128 == 0 and n % 128 == 0 and k % 128 == 0
        )
        var warp_tiled_matmul_supported_format = (
            a_type == DType.float32
            and b_type == DType.float32
            and c_type == DType.float32
        )
        if (
            warp_tiled_matmul_suppoered_shape
            and warp_tiled_matmul_supported_format
        ):
            # TODO: Auto tune these for A100.
            # TODO: NUM_THREADS need to vary as M, N varies.
            alias NUM_THREADS = 128
            alias BN = 128
            alias BM = 128
            alias BK = 16
            alias WN = 64
            alias WM = 64
            alias WNITER = 4
            alias TN = 4
            alias TM = 8
            alias WMITER = (WM * WN) // (WARP_SIZE * TM * TN * WNITER)
            alias mm = sgemm_warp_tiling_kernel[
                c_type,
                c_shape,
                a_type,
                a_shape,
                b_type,
                b_shape,
                indexing_integral_dtype=indexing_integral_dtype,
                BM=BM,
                BN=BN,
                BK=BK,
                WM=WM,
                WN=WN,
                WMITER=WMITER,
                WNITER=WNITER,
                TM=TM,
                TN=TN,
                NUM_THREADS=NUM_THREADS,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ]
            var gpu_func = Function[__type_of(mm), mm](
                threads_per_block=NUM_THREADS
            )
            gpu_func(
                c,
                a,
                b,
                Scalar[c_type](1),
                Scalar[c_type](0),
                grid_dim=(div_ceil(n, BN), div_ceil(m, BM)),
                block_dim=(NUM_THREADS),
                stream=stream,
            )
        elif n == 1:
            alias WARPS_PER_BLOCK = 32
            var gpu_func = Function[
                fn (
                    DTypePointer[c_type],
                    DTypePointer[a_type],
                    DTypePointer[b_type],
                    Int,
                    Int,
                    Int,
                ) capturing -> None, gemv_kernel[
                    c_type,
                    a_type,
                    b_type,
                    s_type,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ]
            ]()
            gpu_func(
                c.data,
                a.data,
                b.data,
                m,
                n,
                k,
                grid_dim=div_ceil(m, WARPS_PER_BLOCK),
                block_dim=WARP_SIZE * WARPS_PER_BLOCK,
                stream=stream,
            )
        elif m == 1 and n % WARP_SIZE == 0 and k % 32 == 0:
            # k should be a multiple of warps per block
            alias WARPS_PER_BLOCK = 32
            var gpu_func = Function[
                fn (
                    DTypePointer[c_type],
                    DTypePointer[a_type],
                    DTypePointer[b_type],
                    Int,
                    Int,
                    Int,
                ) capturing -> None, gevm_kernel[
                    c_type,
                    a_type,
                    b_type,
                    WARP_SIZE * WARPS_PER_BLOCK,
                    s_type,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ]
            ]()
            gpu_func(
                c.data,
                a.data,
                b.data,
                m,
                n,
                k,
                grid_dim=div_ceil(n, WARPS_PER_BLOCK),
                block_dim=WARP_SIZE * WARPS_PER_BLOCK,
                stream=stream,
            )
        else:
            # Tile size for tiling in shared memory.
            # Thread block would have shape (tile_size, tile_size, 1)
            # If k < tile_size use naive version.
            alias tile_size = 16
            if k >= tile_size:
                var gpu_func = Function[
                    fn (
                        DTypePointer[c_type],
                        DTypePointer[a_type],
                        DTypePointer[b_type],
                        Int,
                        Int,
                        Int,
                    ) capturing -> None, matmul_kernel[
                        c_type,
                        a_type,
                        b_type,
                        tile_size,
                        s_type,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                    ]
                ]()
                gpu_func(
                    c.data,
                    a.data,
                    b.data,
                    m,
                    n,
                    k,
                    grid_dim=(div_ceil(n, tile_size), div_ceil(m, tile_size)),
                    block_dim=(tile_size, tile_size),
                    stream=stream,
                )
            else:
                alias BLOCK_DIM = 16
                var gpu_func = Function[
                    fn (
                        DTypePointer[a_type],
                        DTypePointer[b_type],
                        DTypePointer[c_type],
                        Int,
                        Int,
                        Int,
                    ) capturing -> None, matmul_kernel_naive[
                        a_type,
                        b_type,
                        c_type,
                        BLOCK_DIM,
                        s_type,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                    ]
                ]()
                gpu_func(
                    c.data,
                    a.data,
                    b.data,
                    m,
                    n,
                    k,
                    grid_dim=(div_ceil(m, BLOCK_DIM), div_ceil(n, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM, BLOCK_DIM),
                    stream=stream,
                )
    except e:
        abort(e)

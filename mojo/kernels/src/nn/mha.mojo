# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from math import align_down, div_ceil, exp, iota, max, min, sqrt
from math.limit import neginf

from algorithm import elementwise, unswitch
from LinAlg.BatchedMatmul import batched_matmul
from buffer import Buffer, NDBuffer
from buffer.list import Dim, DimList
from gpu import (
    WARP_SIZE,
    BlockDim,
    BlockIdx,
    ThreadIdx,
    barrier,
    lane_id,
    shuffle_down,
    shuffle_xor,
    warp_reduce,
)
from gpu.host import Function, Stream
from gpu.memory import AddressSpace
from LinAlg.Matmul import matmul
from memory import stack_allocation
from memory.reference import AddressSpace as _AddressSpace
from memory.unsafe import DTypePointer, bitcast
from LinAlg.transpose import transpose

from utils.index import Index, StaticIntTuple
from utils.static_tuple import StaticTuple

from .softmax import softmax, softmax_3_pass

from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import LayoutTensor

# ===----------------------------------------------------------------------===#
# Multi-Head Attention
# ===----------------------------------------------------------------------===#


fn fused_attention[
    rank: Int,
    q_shape: DimList,
    k_shape: DimList,
    v_shape: DimList,
    mask_shape: DimList,
    output_shape: DimList,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    mask_type: DType,
    output_type: DType,
    transpose_k: Bool = False,
    add_attn_mask: Bool = True,
    add_causal_mask: Bool = False,
](
    output: NDBuffer[output_type, rank, output_shape],
    q: NDBuffer[q_type, rank, q_shape],
    k: NDBuffer[k_type, rank, k_shape],
    v: NDBuffer[v_type, rank, v_shape],
    mask: NDBuffer[mask_type, rank, mask_shape],
    scale: Float32,
    causal_mask_value: Float32,
) raises:
    """Multi-head Attention with fusion.
    Compute:
        (1) P = Bmm(Q, K), P is also called "score";
        (2) P = P * scale + attention_mask + causal_mask;
        (3) P = softmax(P);
        (4) output = Bmm(P, V).

    Q, V, and the output have shape BHSD. K has shape BHDS if transposed=false
    and  otherwise BHSD. B, S, H, D denote batch size, sequence length, head
    count and depth, respectively.

    (2) and (3) can be fused into (1) as elementwise and row-wise epilogue.

    The causal mask is implicitly set as (j <= i ? 0.0 : mask_value). Some
    models do the same thing but in various patterns, making it tricky to match.

    """

    constrained[rank == 3 or rank == 4, "Only support rank 3 and 4."]()

    alias simd_size = simdwidthof[output_type]()

    var score_size: Int
    var M: Int
    var N: Int
    var K: Int
    var flatten_batch_size: Int

    @parameter
    if rank == 4:
        # q shape is [batch size, # heads, seq_len, depth]
        M = q.dim[2]()
        N = k.dim[2]() if transpose_k else k.dim[3]()
        K = q.dim[3]()
        score_size = q.dim[0]() * q.dim[1]() * M * N
        flatten_batch_size = q.dim[0]() * q.dim[1]()
    else:
        # q shape is [batch size * # heads, seq_len, depth]
        M = q.dim[1]()
        N = k.dim[1]() if transpose_k else k.dim[2]()
        K = q.dim[2]()
        flatten_batch_size = q.dim[0]()
        score_size = q.dim[0]() * M * N

    alias score_type = output_type
    var score_ptr = DTypePointer[score_type].alloc(score_size)

    var score_shape: StaticIntTuple[rank]

    @parameter
    if rank == 4:
        score_shape = rebind[StaticIntTuple[rank]](
            Index(q.dim[0](), q.dim[1](), M, N)
        )
    else:
        score_shape = rebind[StaticIntTuple[rank]](Index(q.dim[0](), M, N))
    # fmt: on
    var score = NDBuffer[score_type, rank](score_ptr, score_shape)

    @__copy_capture(M, N, score)
    @parameter
    @always_inline
    fn fuse_elementwise_fn[
        inner_type: DType, width: Int, _rank: Int
    ](_out_coords: StaticIntTuple[_rank], out_val: SIMD[inner_type, width]):
        var seq_offset = M - N
        var fused_val = out_val

        fused_val *= rebind[SIMD[inner_type, 1]](scale)

        @parameter
        if add_causal_mask:
            var vec_indices = iota[inner_type, width](_out_coords[_rank - 1])
            var vec_mask = vec_indices <= (_out_coords[_rank - 2] - seq_offset)
            fused_val = vec_mask.select(
                fused_val,
                rebind[SIMD[inner_type, width]](
                    SIMD[DType.float32, width](causal_mask_value),
                ),
            )

        @parameter
        if add_attn_mask:
            var idx = rebind[StaticIntTuple[rank]](_out_coords)
            fused_val += mask.load[width=width](idx).cast[inner_type]()

        score.store[width=width](
            rebind[StaticIntTuple[rank]](_out_coords),
            fused_val.cast[score_type](),
        )

    fn softmax_closure(
        start_row: Int,
        num_rows: Int,
        c: NDBuffer[score_type, 2],
    ):
        var row_size = c.dim(1)
        for i in range(start_row, start_row + num_rows):
            var row_view = Buffer[DType.float32](
                c.data.offset(i * row_size).bitcast[DType.float32](), row_size
            )

            @__copy_capture(row_view)
            @parameter
            @always_inline
            fn input_fn_1d[
                _width: Int
            ](idx: Int) -> SIMD[DType.float32, _width]:
                return rebind[SIMD[DType.float32, _width]](
                    row_view.load[width=_width](idx)
                )

            softmax_3_pass[simd_size, Dim(), DType.float32, input_fn_1d](
                row_view
            )

    # Fuse softmax when matmul is only partitioned in M.
    # TODO: use portition function instead of copying heuristic.
    # TODO(#26198) Disabled for now. Should be partition aware and has a req
    # of `(M > N) or (M == N and K <= M)`.
    var softmax_fusable = False

    # The transpose of Q K V swaps batch and matmul dimensions,
    # e.x. 1x128x12x64 -> 1x12x128x64, which batched_matmul can't handle.
    # They are properly transposed before this kernel.
    @always_inline
    @__copy_capture(score)
    @parameter
    fn bmm_query_key[fuse_softmax: Bool]():
        batched_matmul[
            rank,
            q_type,
            k_type,
            score_type,
            False,
            transpose_k,
            fuse_elementwise_fn,
            fuse_softmax,
        ](
            score.make_dims_unknown(),
            q.make_dims_unknown(),
            k.make_dims_unknown(),
            softmax_closure,
        )

    unswitch[bmm_query_key](softmax_fusable)

    if not softmax_fusable:
        softmax[score_type, simd_size, rank](score, score, rank - 1)

    fn bmm_null_rowwise_epilogue(
        start_row: Int,
        num_rows: Int,
        c: NDBuffer[output_type, 2],
    ):
        pass

    # NOTE: synchronous, so the stack allocated score_mem is safe.
    batched_matmul[
        rank,
        score_type,  # score type, TODO: quantization.
        v_type,
        output_type,
        False,
        False,
    ](
        output.make_dims_unknown(),
        score.make_dims_unknown(),
        v.make_dims_unknown(),
        bmm_null_rowwise_epilogue,
    )

    # We did not reuse the output buffer, so we have to free the allocate
    # intermediate buffer.
    if score_ptr != output.data.bitcast[score_type]():
        score_ptr.free()


# ===----------------------------------------------------------------------===#
# Flash attention
# ===----------------------------------------------------------------------===#

# Using 32 bits index for GPU kernel.


fn flash_attention[
    rank: Int,
    q_shape: DimList,
    k_shape: DimList,
    v_shape: DimList,
    mask_shape: DimList,
    output_shape: DimList,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    mask_type: DType,
    output_type: DType,
    # llama 2 has attention mask but not causal mask.
    add_attn_mask: Bool = True,
    target: StringLiteral = "cpu",
](
    output: NDBuffer[output_type, rank, output_shape],
    q: NDBuffer[q_type, rank, q_shape],
    k: NDBuffer[k_type, rank, k_shape],
    v: NDBuffer[v_type, rank, v_shape],
    mask: NDBuffer[mask_type, 3, mask_shape],
    scale: Float32,
) raises:
    """Flash attention 2 algorithm.
    Compute:
        (1) Transpose (Q) BSHD -> BHSD;
        (2) Transpose (K) BSHD -> BHSD;
        (3) Transpose (V) BSHD -> BHSD;
        (4) P = Bmm(Q, K), P is also called "score";
        (5) P = P * scale + mask;
        (6) P = softmax(P);
        (7) O = Bmm(P, V)
        (8) Output = Transpose(O).

    B, S, H, D denote batch size, sequence length, head count and depth, respectively.
    (1), (2), (3) happens while loading the data into shared memory.
    (8) happens when writing output to global memory.

    Assumptions:
        (1) depth per head is 128 (or 256, set TN=8).
        (2) seqlen is multiple of 32 and 128.
    """
    constrained[target == "cuda", "only valid on CUDA GPUs"]()
    constrained[rank == 4, "only support rank 4 used in llama 2."]()
    constrained[
        q_type == DType.float32
        and k_type == DType.float32
        and v_type == DType.float32
        and mask_type == DType.float32
        and output_type == DType.float32,
        "only support float32 in llama 2.",
    ]()

    # If propagate static shapes.
    # constrained[q_shape.at[2]().get() == 128, "Only support 32 heads."]()
    # constrained[q_shape.at[3]().get() == 128, "Only support depth = 128."]()

    # q shape [batch_size, seq_len, # heads, depth]
    var batch_size = q.dim[0]()
    var seq_len = q.dim[1]()
    var num_keys = k.dim[1]()
    var num_heads = q.dim[2]()
    var depth = q.dim[3]()

    alias qtile_num_rows = 32
    alias ktile_num_rows = 128
    # TODO: #25898, use max_finite
    alias max_uint32 = Int(0xFFFFFFFF)
    var use_32bit_indexing = qtile_num_rows * depth < max_uint32 and ktile_num_rows * depth < max_uint32 and qtile_num_rows * ktile_num_rows < max_uint32 and batch_size * seq_len * seq_len < max_uint32

    if not use_32bit_indexing:
        raise Error("32bits index overflow.")

    try:
        var stream = Stream.get_current_stream()

        # Use fast kernel for context encoding benchmark.
        if seq_len == num_keys and seq_len % 128 == 0:
            var func = Function[
                fn (
                    DTypePointer[DType.float32],
                    DTypePointer[DType.float32],
                    DTypePointer[DType.float32],
                    DTypePointer[DType.float32],
                    DTypePointer[DType.float32],
                    Float32,
                    Int,
                    Int,
                ) -> None, flash_attention_kernel[
                    BM=qtile_num_rows,
                    BN=ktile_num_rows,
                    BK=16,
                    depth=128,  # llama2 shape
                    num_heads=32,  # llama2 shape
                    TM=8,
                    TN=4,
                    num_threads=128,
                ]
            ]()

            func(
                q.data,
                k.data,
                v.data,
                mask.data,
                output.data,
                scale,
                batch_size,
                seq_len,
                grid_dim=(
                    div_ceil(seq_len, 32),
                    num_heads,
                    batch_size,
                ),
                block_dim=(128, 1, 1),
                stream=stream,
            )
        # Slow path for token generation for now and context encoding with
        # seq_len % 128 != 0.
        else:
            var func = Function[
                fn (
                    DTypePointer[DType.float32],
                    DTypePointer[DType.float32],
                    DTypePointer[DType.float32],
                    DTypePointer[DType.float32],
                    DTypePointer[DType.float32],
                    Float32,
                    Int,
                    Int,
                    Int,
                ) -> None, flash_attention_kernel_flexible_seqlen[
                    BM=qtile_num_rows,
                    BN=ktile_num_rows,
                    BK=16,
                    depth=128,  # llama2 shape
                    num_heads=32,  # llama2 shape
                    TM=8,
                    TN=4,
                    num_threads=128,
                ]
            ]()

            func(
                q.data,
                k.data,
                v.data,
                mask.data,
                output.data,
                scale,
                batch_size,
                seq_len,
                num_keys,
                grid_dim=(
                    div_ceil(seq_len, 32),
                    num_heads,
                    batch_size,
                ),
                block_dim=(128, 1, 1),
                stream=stream,
            )

    except e:
        abort(e)


@parameter
@always_inline
fn _add_capturing[
    type: DType,
    width: Int,
](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
    return x + y


@parameter
@always_inline
fn _max_capturing[
    type: DType,
    width: Int,
](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
    return max(x, y)


# Helper function for the gemm in attention block.
@always_inline
fn _mm[
    M: Int,
    N: Int,
    K: Int,
    leading_dim_a: Int,
    TM: Int,
    TN: Int,
    transpose_a: Bool,
](
    a: DTypePointer[DType.float32, AddressSpace.SHARED],
    b: DTypePointer[DType.float32, AddressSpace.SHARED],
    row: Int,
    col: Int,
    reg_m: DTypePointer[DType.float32],
    reg_n: DTypePointer[DType.float32],
    reg_res: DTypePointer[DType.float32],
):
    """Helper function for flash attention to do gemm with inputs from
    shared memory and results in registers."""

    alias simd_size = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_size]]()

    @unroll
    for k in range(K):
        # load a element starting from (row, k) or (k, row) if transposed.
        @parameter
        if transpose_a:
            # vector load
            @unroll
            for offset in range(0, TM, simd_size):
                reg_m.store[width=simd_size](
                    offset,
                    a.load[width=simd_size, alignment=alignment](
                        k * M + row + offset
                    ),
                )
        else:
            # scalar load
            @unroll
            for i in range(TM):
                reg_m[i] = a[(row + i) * leading_dim_a + k]

        @unroll
        for offset in range(0, TN, simd_size):
            var vec = b.load[width=simd_size, alignment=alignment](
                k * N + col + offset
            )
            reg_n.store(offset, vec)

        @unroll
        for i in range(TM):

            @unroll
            for j in range(TN):
                reg_res[i * TN + j] = reg_res[i * TN + j] + reg_m[i] * reg_n[j]


@always_inline
fn _fill[
    len: Int, type: DType, address_space: _AddressSpace
](ptr: DTypePointer[type, address_space], val: Scalar[type]):
    alias simd_width = simdwidthof[val.type]()
    alias vector_end = align_down(len, simd_width)

    @unroll
    for i in range(0, vector_end, simd_width):
        ptr.store(i, SIMD[type, simd_width].splat(val))

    @unroll
    for i in range(vector_end, len, 1):
        ptr[i] = val


@__llvm_metadata(`nvvm.maxntid`=StaticTuple[Int32, 1](num_threads))
fn flash_attention_kernel[
    BM: Int,  # number of queries per block
    BN: Int,  # number of keys per block
    BK: Int,  # tile size in depth dimension
    depth: Int,
    num_heads: Int,
    TM: Int,
    TN: Int,
    num_threads: Int,
](
    q_ptr: DTypePointer[DType.float32],
    k_ptr: DTypePointer[DType.float32],
    v_ptr: DTypePointer[DType.float32],
    mask_ptr: DTypePointer[DType.float32],
    output_ptr: DTypePointer[DType.float32],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
):
    # To resemble CUDA float4.
    alias simd_size = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_size]]()
    alias float_alignment = alignof[DType.float32]()

    alias num_warps = num_threads // WARP_SIZE

    var tid = ThreadIdx.x()
    var lane = int(lane_id())
    var warpid = tid // WARP_SIZE

    # Warp index mapping for 2nd gemm.
    alias warp_dim_x = 32
    alias warp_dim_y = 1
    alias num_warps_m = BM // (warp_dim_y * TM)
    alias num_warps_n = depth // (warp_dim_x * TN)
    var warpx = warpid % num_warps_n
    var warpy = warpid // num_warps_n
    # Thread index mapping in MxN matrix.
    # Each warp handles TM rows of output matrix, applicable to both bmms.
    var tx_in_warp = lane % warp_dim_x
    var ty_in_warp = lane // warp_dim_x
    # Thread tile's start row and column in output matrix.
    var mm_row = (ty_in_warp + warpy * warp_dim_y) * TM
    var mm_col = (tx_in_warp + warpx * warp_dim_x) * TN

    var q_tile = stack_allocation[
        BM * depth,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    alias smem_pad = 4
    var kv_tile = stack_allocation[
        (BN + smem_pad) * BK,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    var p_tile = stack_allocation[
        BM * BK,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    var rowmax = stack_allocation[
        TM, DType.float32, alignment=float_alignment
    ]()

    var rowsum = stack_allocation[
        TM, DType.float32, alignment=float_alignment
    ]()

    var reg_result = stack_allocation[
        TM * TN,
        DType.float32,
        alignment=float_alignment,
    ]()

    var o_thread_tile = stack_allocation[
        TM * TN,
        DType.float32,
        alignment=float_alignment,
    ]()

    var reg_m = stack_allocation[
        TM,
        DType.float32,
        alignment=float_alignment,
    ]()

    var reg_n = stack_allocation[
        TN,
        DType.float32,
        alignment=float_alignment,
    ]()

    var correction = stack_allocation[
        TM,
        DType.float32,
        alignment=float_alignment,
    ]()

    var batch_idx = BlockIdx.z()
    var head_idx = BlockIdx.y()
    var q_tile_idx = BlockIdx.x()

    var global_mask_offset = batch_idx * seq_len * seq_len

    # Load Q.
    # Offset in global Q buffer, BSHD layout
    var global_q_offset = depth * (
        head_idx + num_heads * (q_tile_idx * BM + seq_len * batch_idx)
    )
    # We transpose Q BSHD -> BHSD. 2 subsequenet rows in q tile have stride
    # != depth in global Q array because the stride is based on BSHD.
    alias row_stride = num_heads * depth
    # Index of the 1st row and col loaded by current thread.
    var loadq_row = (tid * simd_size) // depth
    var loadq_col = (tid * simd_size) % depth

    var q_gmem_base = LayoutTensor[
        Layout(IntTuple(BM, depth), IntTuple(row_stride, 1)), DType.float32
    ](q_ptr.offset(global_q_offset))

    var q_smem_base = LayoutTensor[
        Layout.row_major(BM, depth),
        DType.float32,
        address_space = AddressSpace.SHARED,
    ](q_tile)

    alias thread_xmem_layout = Layout.row_major(
        (num_threads // depth) * simd_size, depth // simd_size
    )
    var thread_loadq_gmem_frags = q_gmem_base.vectorize[
        1, simd_size
    ]().distribute[thread_xmem_layout](ThreadIdx.x())
    var thread_storeq_smem_frags = q_smem_base.vectorize[
        1, simd_size
    ]().distribute[thread_xmem_layout](ThreadIdx.x())
    thread_storeq_smem_frags.copy_from_numa(thread_loadq_gmem_frags)

    # Clear thread's register tile for output.
    _fill[TM * TN](o_thread_tile, 0)

    _fill[TM](rowmax, neginf[DType.float32]())
    _fill[TM](rowsum, 0)

    # Offset of K/V tile in global K/V buffer, i.e., 1st element of current head.
    var global_kv_offset = depth * (head_idx + num_heads * seq_len * batch_idx)

    # K tile has shape [BN, depth] and is divided sub-tiles [BN, BK].
    # 1st row and col in k sub-tile loaded by current thread.
    var loadk_row = (tid * simd_size) // BK
    var loadk_col = (tid * simd_size) % BK

    # V tile has shape [BN, depth] and is divided sub-tiles [BK, depth].
    # 1st row and col in v sub-tile loaded by current thread.
    var loadv_row = (tid * simd_size) // depth
    var loadv_col = (tid * simd_size) % depth

    for kv_tile_start_row in range(0, seq_len, BN):
        # Clear thread tile results.
        _fill[TM * TN](reg_result, 0)

        # K tile has shape [BN, depth]. Load sub-tile [BN, BK] each time and
        # multiply with the corresponding Q slice of shape [BM, BK].
        alias loadk_num_rows_per_iter = (num_threads * simd_size) // BK
        alias loadk_num_iters = BN // loadk_num_rows_per_iter
        alias BN_padded = BN + smem_pad
        for subtile_start_col in range(0, depth, BK):

            @unroll
            for i in range(loadk_num_iters):
                var row_in_tile = loadk_row + i * loadk_num_rows_per_iter
                var global_idx = global_kv_offset + row_in_tile * row_stride + subtile_start_col + loadk_col
                var vec = k_ptr.load[width=simd_size, alignment=alignment](
                    global_idx
                )

                # Transpose k tile.
                @unroll
                for j in range(4):
                    kv_tile[(loadk_col + j) * BN_padded + row_in_tile] = vec[j]

            # Gaurd write of q_tile and kv_tile.
            barrier()

            var q_ptr = q_tile.offset(subtile_start_col)
            _mm[BM, BN_padded, BK, depth, TM, TN, transpose_a=False](
                q_ptr, kv_tile, mm_row, mm_col, reg_m, reg_n, reg_result
            )
            # Guard read of kv_tile.
            barrier()

        # We have the output P [BM, BN] divided in each thread's TMxTN registers.
        # Current thread's tile starts at (mm_row, mm_col).

        # Scale and add mask.
        # Mask has shape [seq_len, seq_len]. p_tile correlates to a mask tile
        # starting at (q_tile_idx * BM, kv_tile_start_row).
        var mask_offset = global_mask_offset + (
            q_tile_idx * BM + mm_row
        ) * seq_len + kv_tile_start_row + mm_col

        @unroll
        for i in range(TM):

            @unroll
            for j in range(0, TN, simd_size):
                var idx = i * TN + j
                var vec = reg_result.load[width=simd_size](idx)
                var mask_idx = mask_offset + i * seq_len + j
                var mask_vec = mask_ptr.load[
                    width=simd_size, alignment=alignment
                ](mask_idx)
                reg_result.store(idx, vec * scale + mask_vec)

        # Online Softmax
        @unroll
        for i in range(TM):
            var curr_rowmax = rowmax[i]

            # Find thread register tile's max at i-th row.
            @unroll
            for j in range(TN):
                curr_rowmax = max(reg_result[i * TN + j], curr_rowmax)
            # Reduce the max of block tile's row.
            curr_rowmax = warp_reduce[shuffle_xor, _max_capturing](curr_rowmax)

            correction[i] = exp(rowmax[i] - curr_rowmax)

            @unroll
            for j in range(TN):
                reg_result[i * TN + j] = exp(
                    reg_result[i * TN + j] - curr_rowmax
                )

            var curr_rowsum = Float32(0.0)

            # Sum thread register tile at the i-th row.
            @unroll
            for j in range(TN):
                curr_rowsum += reg_result[i * TN + j]
            # Reduce the sum of block tile's row.
            curr_rowsum = warp_reduce[shuffle_xor, _add_capturing](curr_rowsum)

            rowmax[i] = curr_rowmax
            rowsum[i] = rowsum[i] * correction[i] + curr_rowsum

        # Correct previous output.
        @unroll
        for i in range(TM):

            @unroll
            for j in range(TN):
                o_thread_tile[i * TN + j] *= correction[i]

        # V tile has shape [BN, depth]. P tile has shape [BM, BN]. Each itertion
        # loads V sub-tile [BK, depth] from global memory to shared memory and
        # stages p sub-tile [BM, BK] from thread register tile to shared memory.
        alias loadv_num_rows_per_iter = (num_threads * simd_size) // depth
        alias loadv_num_iters = BK // loadv_num_rows_per_iter
        alias loadv_iter_stride = loadv_num_rows_per_iter * row_stride
        var storep_col_start = 0
        for subtile_start_row in range(0, BN, BK):
            # Store thread register tile to p sub-tile.
            if mm_col >= storep_col_start and mm_col < storep_col_start + BK:

                @unroll
                for i in range(TM):

                    @unroll
                    for j in range(0, TN, simd_size):
                        p_tile.store[width=simd_size, alignment=alignment](
                            (mm_row + i) * BK + mm_col - storep_col_start + j,
                            reg_result.load[width=simd_size](i * TN + j),
                        )
            storep_col_start += BK

            # Load v sub-tile.
            @unroll
            for i in range(loadv_num_iters):
                var row_in_tile = loadv_row + i * loadv_num_rows_per_iter
                var global_idx = global_kv_offset + (
                    subtile_start_row + row_in_tile
                ) * row_stride + loadv_col
                var vec = v_ptr.load[width=simd_size, alignment=alignment](
                    global_idx
                )
                kv_tile.store[width=simd_size, alignment=alignment](
                    row_in_tile * depth + loadv_col, vec
                )
            # Guard writing to p_tile and kv_tile.
            barrier()

            # var p_ptr = p_tile.offset(subtile_start_row)
            _mm[BM, depth, BK, BK, TM, TN, transpose_a=False](
                p_tile,
                kv_tile,
                mm_row,
                mm_col,
                reg_m,
                reg_n,
                o_thread_tile,
            )
            # Guard reading kv_tile.
            barrier()

        # Point to  next tile
        global_kv_offset += BN * num_heads * depth

    # Write the output from register to global memory.
    # The output tile [BM, depth] is divided into each thread's TMxTN registers.
    # Current thread's tile starts at (mm_row, mm_col).
    var o_global_row_offset = global_q_offset + mm_row * row_stride

    @unroll
    for i in range(TM):

        @unroll
        for offset in range(0, TN, simd_size):
            # Apply the denominator of softmax.
            var vec = o_thread_tile.load[width=simd_size](
                i * TN + offset
            ) / rowsum.load(i)

            output_ptr.store[width=simd_size, alignment=alignment](
                o_global_row_offset + mm_col + offset, vec
            )
        o_global_row_offset += row_stride


@__llvm_metadata(`nvvm.maxntid`=StaticTuple[Int32, 1](num_threads))
fn flash_attention_kernel_flexible_seqlen[
    BM: Int,  # number of queries per block
    BN: Int,  # number of keys per block
    BK: Int,  # tile size in depth dimension
    depth: Int,
    num_heads: Int,
    TM: Int,
    TN: Int,
    num_threads: Int,
](
    q_ptr: DTypePointer[DType.float32],
    k_ptr: DTypePointer[DType.float32],
    v_ptr: DTypePointer[DType.float32],
    mask_ptr: DTypePointer[DType.float32],
    output_ptr: DTypePointer[DType.float32],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
):
    # To resemble CUDA float4.
    alias simd_size = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_size]]()
    alias float_alignment = alignof[DType.float32]()

    alias num_warps = num_threads // WARP_SIZE

    var tid = ThreadIdx.x()
    var lane = int(lane_id())
    var warpid = tid // WARP_SIZE

    # Warp index mapping for 2nd gemm.
    alias warp_dim_x = 32
    alias warp_dim_y = 1
    alias num_warps_m = BM // (warp_dim_y * TM)
    alias num_warps_n = depth // (warp_dim_x * TN)
    var warpx = warpid % num_warps_n
    var warpy = warpid // num_warps_n
    # Thread index mapping in MxN matrix.
    # Each warp handles TM rows of output matrix, applicable to both bmms.
    var tx_in_warp = lane % warp_dim_x
    var ty_in_warp = lane // warp_dim_x
    # Thread tile's start row and column in output matrix.
    var mm_row = (ty_in_warp + warpy * warp_dim_y) * TM
    var mm_col = (tx_in_warp + warpx * warp_dim_x) * TN

    var q_tile = stack_allocation[
        BM * depth,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    alias smem_pad = 4
    var kv_tile = stack_allocation[
        (BN + smem_pad) * BK,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    var p_tile = stack_allocation[
        BM * BN,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    var rowmax = stack_allocation[
        TM, DType.float32, alignment=float_alignment
    ]()

    var rowsum = stack_allocation[
        TM, DType.float32, alignment=float_alignment
    ]()

    var reg_result = stack_allocation[
        TM * TN,
        DType.float32,
        alignment=float_alignment,
    ]()

    var o_thread_tile = stack_allocation[
        TM * TN,
        DType.float32,
        alignment=float_alignment,
    ]()

    var reg_m = stack_allocation[
        TM,
        DType.float32,
        alignment=float_alignment,
    ]()

    var reg_n = stack_allocation[
        TN,
        DType.float32,
        alignment=float_alignment,
    ]()

    var correction = stack_allocation[
        TM,
        DType.float32,
        alignment=float_alignment,
    ]()

    var batch_idx = BlockIdx.z()
    var head_idx = BlockIdx.y()
    var q_tile_idx = BlockIdx.x()

    var global_mask_offset = batch_idx * seq_len * seq_len

    # Load Q.
    # Offset in global Q buffer, BSHD layout
    var global_q_start_row = q_tile_idx * BM
    var global_q_offset = depth * (
        head_idx + num_heads * (q_tile_idx * BM + seq_len * batch_idx)
    )
    alias loadq_num_rows_per_iter = (num_threads * simd_size) // depth
    var loadq_num_rows = min(BM, seq_len - global_q_start_row)
    var loadq_num_iters = div_ceil(loadq_num_rows, loadq_num_rows_per_iter)
    # alias loadq_num_iters = BM // loadq_num_rows_per_iter
    # We transpose Q BSHD -> BHSD. 2 subsequenet rows in q tile have stride
    # != depth in global Q array because the stride is based on BSHD.
    alias row_stride = num_heads * depth
    # Index of the 1st row and col loaded by current thread.
    var loadq_row = (tid * simd_size) // depth
    var loadq_col = (tid * simd_size) % depth

    ##
    for i in range(loadq_num_iters):
        var row_in_tile = loadq_row + i * loadq_num_rows_per_iter
        # The a row from Q in global memory.
        if row_in_tile + global_q_start_row < seq_len:
            var global_q_idx = global_q_offset + row_in_tile * row_stride + loadq_col
            var vec = q_ptr.load[width=simd_size, alignment=alignment](
                global_q_idx,
            )
            q_tile.store[width=simd_size, alignment=alignment](
                row_in_tile * depth + loadq_col, vec
            )
        # The Q tile exceeds global Q buffer, pad with zeros.
        else:
            q_tile.store[width=simd_size, alignment=alignment](
                row_in_tile * depth + loadq_col,
                SIMD[DType.float32, simd_size](0.0),
            )
    ##

    # Clear thread's register tile for output.
    _fill[TM * TN](o_thread_tile, 0)

    _fill[TM](rowmax, neginf[DType.float32]())
    _fill[TM](rowsum, 0)

    # Offset of K/V tile in global K/V buffer, i.e., 1st element of current head.
    var global_kv_offset = depth * (head_idx + num_heads * seq_len * batch_idx)

    # K tile has shape [BN, depth] and is divided sub-tiles [BN, BK].
    # 1st row and col in k sub-tile loaded by current thread.
    var loadk_row = (tid * simd_size) // BK
    var loadk_col = (tid * simd_size) % BK

    # V tile has shape [BN, depth] and is divided sub-tiles [BK, depth].
    # 1st row and col in v sub-tile loaded by current thread.
    var loadv_row = (tid * simd_size) // depth
    var loadv_col = (tid * simd_size) % depth

    for kv_tile_start_row in range(0, num_keys, BN):
        # Clear thread tile results.
        _fill[TM * TN](reg_result, 0)

        # K tile has shape [BN, depth]. Load sub-tile [BN, BK] each time and
        # multiply with the corresponding Q slice of shape [BM, BK].
        alias loadk_num_rows_per_iter = (num_threads * simd_size) // BK
        var loadk_num_rows = min(BN, num_keys - kv_tile_start_row)
        var loadk_num_iters = div_ceil(loadk_num_rows, loadk_num_rows_per_iter)
        alias BN_padded = BN + smem_pad
        for subtile_start_col in range(0, depth, BK):
            ##
            for i in range(loadk_num_iters):
                var row_in_tile = loadk_row + i * loadk_num_rows_per_iter
                if row_in_tile + kv_tile_start_row < num_keys:
                    var global_idx = global_kv_offset + row_in_tile * row_stride + subtile_start_col + loadk_col
                    var vec = k_ptr.load[width=simd_size, alignment=alignment](
                        global_idx
                    )

                    # Transpose k tile.
                    @unroll
                    for j in range(4):
                        kv_tile[
                            (loadk_col + j) * BN_padded + row_in_tile
                        ] = vec[j]
                else:

                    @unroll
                    for j in range(4):
                        kv_tile[(loadk_col + j) * BN_padded + row_in_tile] = 0

            # Gaurd write of q_tile and kv_tile.
            barrier()

            var q_ptr = q_tile.offset(subtile_start_col)
            _mm[BM, BN_padded, BK, depth, TM, TN, transpose_a=False](
                q_ptr, kv_tile, mm_row, mm_col, reg_m, reg_n, reg_result
            )
            # Guard read of kv_tile.
            barrier()

        # We have the output P [BM, BN] divided in each thread's TMxTN registers.
        # Current thread's tile starts at (mm_row, mm_col).

        # Scale and add mask.
        # Mask has shape [seq_len, seq_len]. p_tile correlates to a mask tile
        # starting at (q_tile_idx * BM, kv_tile_start_row).
        # Caution: Assume the mask is large enought so even if the q, kv tile
        # exceeds the global Q, KV buffer, the intermediate output still fits
        # within the mask.
        ##
        var mask_offset = global_mask_offset + (
            q_tile_idx * BM + mm_row
        ) * seq_len + kv_tile_start_row + mm_col

        var mask_row = q_tile_idx * BM + mm_row
        var mask_col = kv_tile_start_row + mm_col
        if mask_row < seq_len and mask_col < num_keys:

            @unroll
            for i in range(TM):
                # Scalar load in case mask dimension is not multiple of simd_size.
                if mask_row + i < seq_len:

                    @unroll
                    for j in range(TN):
                        if mask_col + j < num_keys:
                            var idx = i * TN + j
                            var val = reg_result[idx]
                            var mask_idx = global_mask_offset + (
                                mask_row + i
                            ) * num_keys + mask_col + j
                            var mask_val = mask_ptr[mask_idx]
                            reg_result[idx] = val * scale + mask_val
        ##

        # Online Softmax
        @unroll
        for i in range(TM):
            var curr_rowmax = rowmax[i]

            # Reset result that exceeds num_keys
            var exceed = kv_tile_start_row + mm_col + TN - num_keys
            if exceed > 0:
                for j in range(TN - exceed, TN):
                    reg_result[i * TN + j] = neginf[DType.float32]()

            # Shuffle TN elemnents per thread and choose the max among them.
            @unroll
            for j in range(TN):
                curr_rowmax = max(
                    warp_reduce[shuffle_xor, _max_capturing](
                        reg_result[i * TN + j]
                    ),
                    curr_rowmax,
                )
            correction[i] = exp(rowmax[i] - curr_rowmax)

            @unroll
            for j in range(TN):
                var idx = i * TN + j
                reg_result[idx] = exp(reg_result[idx] - curr_rowmax)

            if exceed > 0:
                for j in range(TN - exceed, TN):
                    reg_result[i * TN + j] = 0.0

            var curr_rowsum = Float32(0.0)

            @unroll
            for j in range(TN):
                curr_rowsum += warp_reduce[shuffle_xor, _add_capturing](
                    reg_result[i * TN + j]
                )

            rowmax[i] = curr_rowmax
            rowsum[i] = rowsum[i] * correction[i] + curr_rowsum

        @unroll
        for i in range(TM):

            @unroll
            for j in range(0, TN, simd_size):
                p_tile.store[width=simd_size, alignment=alignment](
                    ((mm_row + i) * BN + mm_col + j),
                    reg_result.load[width=simd_size]((i * TN + j)),
                )

        # Clear thread register results for P * V.
        _fill[TM * TN](reg_result, 0)

        # V tile has shape [BN, depth]. Load sub-tile [BK, depth] each time and
        # multiply with the corresponding P slice of shape [BM, BK].
        alias loadv_num_rows_per_iter = (num_threads * simd_size) // depth
        alias loadv_num_iters = BK // loadv_num_rows_per_iter
        for subtile_start_row in range(0, BN, BK):
            ##
            @unroll
            for i in range(loadv_num_iters):
                var row_in_tile = loadv_row + i * loadv_num_rows_per_iter
                if (
                    row_in_tile + kv_tile_start_row + subtile_start_row
                    < num_keys
                ):
                    var global_idx = global_kv_offset + (
                        subtile_start_row + row_in_tile
                    ) * row_stride + loadv_col
                    var vec = v_ptr.load[width=simd_size, alignment=alignment](
                        global_idx
                    )
                    kv_tile.store[width=simd_size, alignment=alignment](
                        row_in_tile * depth + loadv_col, vec
                    )
                else:
                    kv_tile.store[width=simd_size, alignment=alignment](
                        row_in_tile * depth + loadv_col,
                        SIMD[DType.float32, simd_size](0.0),
                    )
            ##

            # Guard writing to p_tile and kv_tile.
            barrier()

            var p_ptr = p_tile.offset(subtile_start_row)
            _mm[BM, depth, BK, BN, TM, TN, transpose_a=False](
                p_ptr,
                kv_tile,
                mm_row,
                mm_col,
                reg_m,
                reg_n,
                reg_result,
            )
            # Guard reading kv_tile.
            barrier()

        # Update output tile
        @unroll
        for i in range(TM):

            @unroll
            for j in range(TN):
                var idx = i * TN + j
                o_thread_tile[idx] = (
                    o_thread_tile[idx] * correction[i] + reg_result[idx]
                )

        # Point to  next tile
        global_kv_offset += BN * num_heads * depth

    # Write the output from register to global memory.
    # The output tile [BM, depth] is divided into each thread's TMxTN registers.
    # Current thread's tile starts at (mm_row, mm_col).
    var o_global_row_offset = global_q_offset + mm_row * row_stride

    @unroll
    for i in range(TM):
        if global_q_start_row + mm_row + i < seq_len:

            @unroll
            for offset in range(0, TN, simd_size):
                # Apply the denominator of softmax.
                var vec = o_thread_tile.load[width=simd_size](
                    i * TN + offset
                ) / rowsum.load(i)

                output_ptr.store[width=simd_size, alignment=alignment](
                    o_global_row_offset + mm_col + offset, vec
                )
        o_global_row_offset += row_stride


fn _naive_attention_with_transpose[
    type: DType,
    transpose_k: Bool = False,
](
    output: NDBuffer[type, 4],
    q: NDBuffer[type, 4],
    k: NDBuffer[type, 4],
    v: NDBuffer[type, 4],
    mask: NDBuffer[type, 2],
    scale: Float32,
):
    """This kernel provides reference values for flash attention in llama 2.
    It can't be used in any model.
    Layouts:
        q: BSHD
        k, v: BKHD
        output: BSHD
        mask: SK
    B, S, K, H, D stand for batch size, sequence length, number of keys,
    number of heads, and depth per head, respectively.
    """
    alias simd_size = simdwidthof[type]()

    var batch_size = q.dim[0]()
    var seq_len = q.dim[1]()
    var num_keys = k.dim[1]()
    var num_heads = q.dim[2]()
    var depth = q.dim[3]()

    # Q, K, V transposed
    var qt_ptr = DTypePointer[type].alloc(q.num_elements())
    var kt_ptr = DTypePointer[type].alloc(k.num_elements())
    var vt_ptr = DTypePointer[type].alloc(v.num_elements())
    # Score = softmax(Q * K)
    var score_size = batch_size * num_heads * seq_len * num_keys
    var score_ptr = DTypePointer[type].alloc(score_size)
    # O = Score * V. It's transposed and will be transposed back to output.
    var ot_ptr = DTypePointer[type].alloc(output.num_elements())

    var qt = NDBuffer[type, 4](
        qt_ptr, Index(batch_size, num_heads, seq_len, depth)
    )
    var kt = NDBuffer[type, 4](
        kt_ptr, Index(batch_size, num_heads, depth, num_keys)
    )
    var vt = NDBuffer[type, 4](
        vt_ptr, Index(batch_size, num_heads, num_keys, depth)
    )
    var score = NDBuffer[type, 4](
        score_ptr, Index(batch_size, num_heads, seq_len, num_keys)
    )
    var ot = NDBuffer[type, 4](
        ot_ptr, Index(batch_size, num_heads, seq_len, depth)
    )

    # BSHD -> BHSD
    var q_perm = Buffer[DType.index, 4].stack_allocation()
    q_perm[0] = 0
    q_perm[1] = 2
    q_perm[2] = 1
    q_perm[3] = 3

    # BSHD -> BHDS
    var k_perm = Buffer[DType.index, 4].stack_allocation()
    k_perm[0] = 0
    k_perm[1] = 2
    k_perm[2] = 3
    k_perm[3] = 1

    # BHSD -> BSHD
    var o_perm = Buffer[DType.index, 4].stack_allocation()
    o_perm[0] = 0
    o_perm[1] = 2
    o_perm[2] = 1
    o_perm[3] = 3

    try:
        transpose(qt, q, q_perm.data)
    except e:
        abort(e)

    try:
        transpose(kt, k, k_perm.data)
    except e:
        abort(e)

    try:
        transpose(vt, v, q_perm.data)
    except e:
        abort(e)

    _naive_attention[type, transpose_k](ot, qt, kt, vt, mask, scale)

    try:
        transpose(output, ot, o_perm.data)
    except e:
        abort(e)

    qt_ptr.free()
    kt_ptr.free()
    vt_ptr.free()
    score_ptr.free()
    ot_ptr.free()


fn _naive_attention[
    type: DType,
    transpose_k: Bool = False,
](
    output: NDBuffer[type, 4],
    q: NDBuffer[type, 4],
    k: NDBuffer[type, 4],
    v: NDBuffer[type, 4],
    mask: NDBuffer[type, 2],
    scale: Float32,
):
    """This kernel provides reference values for flash attention in llama 2.
    It can't be used in any model.
    """
    alias simd_size = simdwidthof[type]()

    var batch_size = q.dim[0]()
    var num_heads = q.dim[1]()
    var seq_len = q.dim[2]()
    var num_keys = v.dim[2]()
    var depth = q.dim[3]()

    # Allocate intermediate memory buffer.
    var score_size = batch_size * num_heads * seq_len * num_keys
    var score_ptr = DTypePointer[type].alloc(score_size)
    var score = NDBuffer[type, 4](
        score_ptr, Index(batch_size, num_heads, seq_len, num_keys)
    )

    batched_matmul[4, type, type, type, False, transpose_k](score, q, k)

    @__copy_capture(score)
    @parameter
    @always_inline
    fn scale_and_mask[width: Int, _rank: Int](coords: StaticIntTuple[_rank]):
        var vec = score.load[width=width](rebind[StaticIntTuple[4]](coords))
        vec = vec * scale.cast[type]()
        vec = vec + mask.load[width=width](
            Index(coords[_rank - 2], coords[_rank - 1])
        )
        score.store[width=width](rebind[StaticIntTuple[4]](coords), vec)

    elementwise[scale_and_mask, simd_size, 4](score.dynamic_shape)

    try:
        softmax[type, simd_size, 4](
            score,
            score,
            axis=3,
        )
    except e:
        abort(e)

    batched_matmul[4, type, type, type, False, False](output, score, v)

    score_ptr.free()

# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from math import div_ceil, iota, max, min, sqrt, neginf, exp
from algorithm import (
    elementwise,
    unroll,
    vectorize,
    unswitch,
)
from BatchedMatmul import batched_matmul
from math.limit import max_finite
from Matmul import matmul
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer, bitcast
from memory import stack_allocation
from runtime.llcl import OwningOutputChainPtr, Runtime, OutputChainPtr
from Softmax import softmax, softmax_3_pass
from Transpose import transpose

from utils.index import Index, StaticIntTuple
from utils.list import DimList

from gpu import (
    ThreadIdx,
    BlockIdx,
    BlockDim,
    barrier,
    lane_id,
    WARP_SIZE,
    shuffle_down,
    shuffle_xor,
    warp_reduce,
)
from gpu.host import Function, Stream
from gpu.memory import AddressSpace


# ===----------------------------------------------------------------------===#
# Multi-Head Attention
# ===----------------------------------------------------------------------===#


@parameter
@closure
fn null_bmm_lambda[
    type: DType, width: Int, rank: Int
](out_coords: StaticIntTuple[rank], out_val: SIMD[type, width]):
    pass


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
    output: NDBuffer[rank, output_shape, output_type],
    q: NDBuffer[rank, q_shape, q_type],
    k: NDBuffer[rank, k_shape, k_type],
    v: NDBuffer[rank, v_shape, v_type],
    mask: NDBuffer[2, mask_shape, mask_type],
    scale: Float32,
    causal_mask_value: Float32,
    out_chain: OutputChainPtr,
):
    """Multi-head Attention with fusion.
    Compute:
        (1) P = Bmm(Q, K), P is also called "score";
        (2) P = P * scale + attention_mask + causal_mask;
        (3) P = softmax(P);
        (4) output = Bmm(P, V).

    Q, V, and the output have shape BHSD. K has shape BHDS if transposed otherwise
    BHSD. B, S, H, D denote batch size, sequence length, head count and depth,
    respectively.

    (2) and (3) can be fused into (1) as elementwise and row-wise epilogue.

    The causal mask is implicitly set as (j <= i ? 0.0 : mask_value). Some
    models do the same thing but in various patterns, making it tricky to match.

    """

    constrained[rank == 3 or rank == 4, "Only support rank 3 and 4."]()

    alias simd_size = simdwidthof[output_type]()

    let rt = out_chain.get_runtime()

    let score_size: Int
    let M: Int
    let N: Int
    let K: Int
    let flatten_batch_size: Int

    @parameter
    if rank == 4:
        # q shape is [batch size, # heads, seq_len, depth]
        score_size = q.dim[0]() * q.dim[1]() * q.dim[2]() * k.dim[3]()
        M = q.dim[2]()
        N = k.dim[3]() if transpose_k else k.dim[2]()
        K = q.dim[3]()
        flatten_batch_size = q.dim[0]() * q.dim[1]()
    else:
        # q shape is [batch size * # heads, seq_len, depth]
        score_size = q.dim[0]() * q.dim[1]() * k.dim[2]()
        M = q.dim[1]()
        N = k.dim[2]() if transpose_k else k.dim[1]()
        K = q.dim[2]()
        flatten_batch_size = q.dim[0]()

    # If the size of the score is less than the output, then we can reuse
    # the output buffer, otherwise we have to allocate an intermediate buffer.
    alias score_type = output_type
    let score_ptr: DTypePointer[score_type]
    if score_size <= output.num_elements():
        score_ptr = bitcast[score_type](output.data)
    else:
        score_ptr = DTypePointer[score_type].alloc(score_size)

    let score_shape: StaticIntTuple[rank]

    @parameter
    if rank == 4:
        score_shape = rebind[StaticIntTuple[rank]](
            Index(q.dim[0](), q.dim[1](), q.dim[2](), k.dim[3]())
        )
    else:
        score_shape = rebind[StaticIntTuple[rank]](
            Index(q.dim[0](), q.dim[1](), k.dim[2]())
        )
    # fmt: on
    let score = NDBuffer[rank, DimList.create_unknown[rank](), score_type](
        score_ptr, score_shape
    )

    @parameter
    @always_inline
    fn fuse_elementwise_fn[
        inner_type: DType, width: Int, _rank: Int
    ](_out_coords: StaticIntTuple[_rank], out_val: SIMD[inner_type, width]):
        let seq_offset = M - N
        var fused_val = out_val

        fused_val *= rebind[SIMD[inner_type, 1]](scale)

        @parameter
        if add_causal_mask:
            let vec_indices = iota[inner_type, width](_out_coords[_rank - 1])
            let vec_mask = vec_indices <= (_out_coords[_rank - 2] - seq_offset)
            fused_val = vec_mask.select(
                fused_val,
                rebind[SIMD[inner_type, width]](
                    SIMD[DType.float32, width](causal_mask_value),
                ),
            )

        @parameter
        if add_attn_mask:
            fused_val += rebind[SIMD[inner_type, width]](
                mask.simd_load[width](
                    Index(_out_coords[_rank - 2], _out_coords[_rank - 1])
                )
            )

        score.simd_store[width](
            rebind[StaticIntTuple[rank]](_out_coords),
            fused_val.cast[score_type](),
        )

    @parameter
    @always_inline
    fn softmax_closure(
        start_row: Int,
        num_rows: Int,
        c: NDBuffer[2, DimList.create_unknown[2](), score_type],
    ):
        let row_size = c.dim(1)
        for i in range(start_row, start_row + num_rows):
            let row_view = Buffer[Dim(), DType.float32](
                bitcast[DType.float32](c.data.offset(i * row_size)), row_size
            )

            @parameter
            @always_inline
            fn input_fn_1d[
                _width: Int
            ](idx: Int) -> SIMD[DType.float32, _width]:
                return rebind[SIMD[DType.float32, _width]](
                    row_view.simd_load[_width](idx)
                )

            softmax_3_pass[simd_size, Dim(), DType.float32, input_fn_1d](
                row_view, out_chain
            )

    # Fuse softmax when matmul is only partitioned in M.
    # TODO: use  portition function instead of copying heuristic.
    let softmax_fusable = (M > N) or (M == N and K <= M)

    # The transpose of Q K V swaps batch and matmul dimensions,
    # e.x. 1x128x12x64 -> 1x12x128x64, which batched_matmul can't handle.
    # They are properly transposed before this kernel.
    @always_inline
    @parameter
    fn bmm_query_key[fuse_softmax: Bool]():
        let score_chain = OwningOutputChainPtr(rt)
        batched_matmul[
            rank,
            q_type,
            k_type,
            score_type,
            False,
            transpose_k,
            True,
            fuse_elementwise_fn,
            fuse_softmax,
        ](
            rebind[NDBuffer[rank, DimList.create_unknown[rank](), score_type]](
                score
            ),
            rebind[NDBuffer[rank, DimList.create_unknown[rank](), q_type]](q),
            rebind[NDBuffer[rank, DimList.create_unknown[rank](), k_type]](k),
            softmax_closure,
            score_chain.borrow(),
        )
        score_chain.wait()

    unswitch[bmm_query_key](softmax_fusable)

    if not softmax_fusable:
        let softmax_chain = OwningOutputChainPtr(rt)
        softmax[score_type, simd_size, rank, DimList.create_unknown[rank]()](
            score, score, rank - 1, softmax_chain.borrow()
        )
        softmax_chain.wait()

    @closure
    @always_inline
    fn bmm_null_rowwise_epilogue(
        start_row: Int,
        num_rows: Int,
        c: NDBuffer[2, DimList.create_unknown[2](), output_type],
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
        False,
        null_bmm_lambda,
        False,
    ](
        rebind[NDBuffer[rank, DimList.create_unknown[rank](), output_type]](
            output
        ),
        rebind[NDBuffer[rank, DimList.create_unknown[rank](), score_type]](
            score
        ),
        rebind[NDBuffer[rank, DimList.create_unknown[rank](), v_type]](v),
        bmm_null_rowwise_epilogue,
        out_chain,
    )

    # We did not reuse the output buffer, so we have to free the allocate
    # intermediate buffer.
    if score_ptr != bitcast[score_type](output.data):
        score_ptr.free()


# ===----------------------------------------------------------------------===#
# Flash attention
# ===----------------------------------------------------------------------===#

# Using 32 bits index for GPU kernel.
alias _int32 = Scalar[DType.uint32]


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
    output: NDBuffer[rank, output_shape, output_type],
    q: NDBuffer[rank, q_shape, q_type],
    k: NDBuffer[rank, k_shape, k_type],
    v: NDBuffer[rank, v_shape, v_type],
    mask: NDBuffer[3, mask_shape, mask_type],
    scale: Float32,
    out_chain: OutputChainPtr,
):
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
    let batch_size: _int32 = q.dim[0]()
    let seq_len: _int32 = q.dim[1]()
    let num_heads: _int32 = q.dim[2]()
    let depth: _int32 = q.dim[3]()

    alias qtile_num_rows = 32
    alias ktile_num_rows = 128
    let use_32bit_indexing = qtile_num_rows * depth < max_finite[
        DType.uint32
    ]().to_int() and ktile_num_rows * depth < max_finite[
        DType.uint32
    ]().to_int() and qtile_num_rows * ktile_num_rows < max_finite[
        DType.uint32
    ]().to_int()

    if not use_32bit_indexing:
        return out_chain.mark_error("32bits index overflow.")

    try:
        let func = Function[
            fn (
                DTypePointer[DType.float32],
                DTypePointer[DType.float32],
                DTypePointer[DType.float32],
                DTypePointer[DType.float32],
                DTypePointer[DType.float32],
                Float32,
                _int32,
                _int32,
            ) -> None, flash_attention_kernel[
                BM=qtile_num_rows,
                BN=ktile_num_rows,
                BK=16,
                depth=128,
                num_heads=32,
                TM=8,
                TN=4,
                num_threads=128,
            ]
        ]()

        func(
            # grid
            (
                div_ceil(seq_len.to_int(), 32),
                num_heads.to_int(),
                batch_size.to_int(),
            ),
            # block
            (128, 1, 1),
            q.data,
            k.data,
            v.data,
            mask.data,
            output.data,
            scale,
            batch_size,
            seq_len,
            stream=out_chain.get_cuda_stream(),
        )
    except e:
        out_chain.mark_error(e)


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
    M: _int32,
    N: _int32,
    K: _int32,
    leading_dim_a: _int32,
    TM: _int32,
    TN: _int32,
    transpose_a: Bool,
](
    a: DTypePointer[DType.float32, AddressSpace.SHARED],
    b: DTypePointer[DType.float32, AddressSpace.SHARED],
    row: _int32,
    col: _int32,
    reg_m: DTypePointer[DType.float32],
    reg_n: DTypePointer[DType.float32],
    reg_res: DTypePointer[DType.float32],
):
    """Helper function for flash attention to do gemm with inputs from
    shared memory and results in registers."""

    alias simd_size = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_size]]()

    @unroll
    for k in range(K.to_int()):
        # load a element starting from (row, k) or (k, row) if transposed.
        @parameter
        if transpose_a:
            # vector load
            @unroll
            for offset in range(0, TM.to_int(), simd_size):
                reg_m.simd_store[simd_size](
                    offset,
                    a.aligned_simd_load[simd_size, alignment](
                        (k * M + row + offset).to_int()
                    ),
                )
        else:
            # scalar load
            @unroll
            for i in range(TM.to_int()):
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
fn _fill[len: Int, type: DType](ptr: DTypePointer[type], val: SIMD[type, 1]):
    alias simd_width = simdwidthof[val.type]()
    alias vector_end = (len // simd_width) * simd_width

    @unroll
    for i in range(0, vector_end, simd_width):
        ptr.simd_store(i, SIMD[type, simd_width].splat(val))

    @unroll
    for i in range(vector_end, len, 1):
        ptr.store(i, val)


fn flash_attention_kernel[
    BM: _int32,  # number of queries per block
    BN: _int32,  # number of keys per block
    BK: _int32,  # tile size in depth dimension
    depth: _int32,
    num_heads: _int32,
    TM: _int32,
    TN: _int32,
    num_threads: _int32,
](
    q_ptr: DTypePointer[DType.float32],
    k_ptr: DTypePointer[DType.float32],
    v_ptr: DTypePointer[DType.float32],
    mask_ptr: DTypePointer[DType.float32],
    output_ptr: DTypePointer[DType.float32],
    scale: Float32,
    batch_size: _int32,
    seq_len: _int32,
):
    # To ressemble cuda float4.
    alias simd_size = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_size]]()
    alias float_alignment = alignof[DType.float32]()

    alias num_warps: _int32 = num_threads // WARP_SIZE

    let tid: _int32 = ThreadIdx.x()
    let lane: _int32 = lane_id()
    let warpid: _int32 = tid // WARP_SIZE

    # Warp index mapping for 2nd gemm.
    alias warp_dim_x: _int32 = 32
    alias warp_dim_y: _int32 = 1
    alias num_warps_m: _int32 = BM // (warp_dim_y * TM)
    alias num_warps_n: _int32 = depth // (warp_dim_x * TN)
    let warpx: _int32 = warpid % num_warps_n
    let warpy: _int32 = warpid // num_warps_n
    # Thread index mapping in MxN matrix.
    # Each warp handles TM rows of output matrix, applicable to both bmms.
    let tx_in_warp: _int32 = lane % warp_dim_x
    let ty_in_warp: _int32 = lane // warp_dim_x
    # Thread tile's start row and column in output matrix.
    let mm_row: _int32 = (ty_in_warp + warpy * warp_dim_y) * TM
    let mm_col: _int32 = (tx_in_warp + warpx * warp_dim_x) * TN

    let q_tile = stack_allocation[
        (BM * depth).to_int(),
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    alias smem_pad = 4
    let kv_tile = stack_allocation[
        ((BN + smem_pad) * BK).to_int(),
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    let p_tile = stack_allocation[
        (BM * BN).to_int(),
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    let rowmax = stack_allocation[
        TM.to_int(), DType.float32, alignment=float_alignment
    ]()

    let rowsum = stack_allocation[
        TM.to_int(), DType.float32, alignment=float_alignment
    ]()

    let reg_result = stack_allocation[
        (TM * TN).to_int(),
        DType.float32,
        alignment=float_alignment,
    ]()

    let o_thread_tile = stack_allocation[
        (TM * TN).to_int(),
        DType.float32,
        alignment=float_alignment,
    ]()

    let reg_m = stack_allocation[
        TM.to_int(),
        DType.float32,
        alignment=float_alignment,
    ]()

    let reg_n = stack_allocation[
        TN.to_int(),
        DType.float32,
        alignment=float_alignment,
    ]()

    let correction = stack_allocation[
        TM.to_int(),
        DType.float32,
        alignment=float_alignment,
    ]()

    let batch_idx: _int32 = BlockIdx.z()
    let head_idx: _int32 = BlockIdx.y()
    let q_tile_idx: _int32 = BlockIdx.x()

    let global_mask_offset: _int32 = batch_idx * seq_len * seq_len

    # Load Q.
    # Offset in global Q buffer, BSHD layout
    let global_q_offset: _int32 = depth * (
        head_idx + num_heads * (q_tile_idx * BM + seq_len * batch_idx)
    )
    alias loadq_num_rows_per_iter = (num_threads * simd_size) // depth
    alias loadq_num_iters = BM // loadq_num_rows_per_iter
    # We transpose Q BSHD -> BHSD. 2 subsequenet rows in q tile have stride
    # != depth in global Q array because the stride is based on BSHD.
    alias row_stride = num_heads * depth
    # Index of the 1st row and col loaded by current thread.
    let loadq_row: _int32 = (tid * simd_size) // depth
    let loadq_col: _int32 = (tid * simd_size) % depth

    @unroll
    for i in range(loadq_num_iters.to_int()):
        let row_in_tile: _int32 = loadq_row + i * loadq_num_rows_per_iter
        let global_q_idx: _int32 = global_q_offset + row_in_tile * row_stride + loadq_col
        let vec = q_ptr.aligned_simd_load[simd_size, alignment](
            global_q_idx.to_int(),
        )
        q_tile.aligned_simd_store[simd_size, alignment](
            (row_in_tile * depth + loadq_col).to_int(), vec
        )

    # Clear thread's register tile for output.
    _fill[(TM * TN).to_int()](o_thread_tile, 0)

    _fill[TM.to_int()](rowmax, neginf[DType.float32]())
    _fill[TM.to_int()](rowsum, 0)

    # Offset of K/V tile in global K/V buffer, i.e., 1st element of current head.
    var global_kv_offset: _int32 = depth * (
        head_idx + num_heads * seq_len * batch_idx
    )

    # K tile has shape [BN, depth] and is divided sub-tiles [BN, BK].
    # 1st row and col in k sub-tile loaded by current thread.
    let loadk_row: _int32 = (tid * simd_size) // BK
    let loadk_col: _int32 = (tid * simd_size) % BK

    # V tile has shape [BN, depth] and is divided sub-tiles [BK, depth].
    # 1st row and col in v sub-tile loaded by current thread.
    let loadv_row: _int32 = (tid * simd_size) // depth
    let loadv_col: _int32 = (tid * simd_size) % depth

    for kv_tile_start_row in range(0, seq_len.to_int(), BN.to_int()):
        # Clear thread tile results.
        _fill[(TM * TN).to_int()](reg_result, 0)

        # K tile has shape [BN, depth]. Load sub-tile [BN, BK] each time and
        # multiply with the corresponding Q slice of shape [BM, BK].
        alias loadk_num_rows_per_iter = (num_threads * simd_size) // BK
        alias loadk_num_iters = BN // loadk_num_rows_per_iter
        alias BN_padded = BN + smem_pad
        for subtile_start_col in range(0, depth.to_int(), BK.to_int()):

            @unroll
            for i in range(loadk_num_iters.to_int()):
                let row_in_tile: _int32 = loadk_row + i * loadk_num_rows_per_iter
                let global_idx: _int32 = global_kv_offset + row_in_tile * row_stride + subtile_start_col + loadk_col
                let vec = k_ptr.aligned_simd_load[simd_size, alignment](
                    global_idx.to_int()
                )

                # Transpose k tile.
                @unroll
                for j in range(4):
                    kv_tile.store(
                        ((loadk_col + j) * BN_padded + row_in_tile).to_int(),
                        vec[j],
                    )
            # Gaurd write of q_tile and kv_tile.
            barrier()

            let q_ptr = q_tile.offset(subtile_start_col)
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
        let mask_offset = global_mask_offset + (
            q_tile_idx * BM + mm_row
        ) * seq_len + kv_tile_start_row + mm_col

        @unroll
        for i in range(TM.to_int()):

            @unroll
            for j in range(0, TN.to_int(), simd_size):
                let idx = (i * TN + j).to_int()
                let vec = reg_result.simd_load[simd_size](idx)
                let mask_idx = (mask_offset + i * seq_len + j).to_int()
                let mask_vec = mask_ptr.aligned_simd_load[simd_size, alignment](
                    mask_idx
                )
                reg_result.simd_store(idx, vec * scale + mask_vec)

        # Online Softmax
        @unroll
        for i in range(TM.to_int()):
            var curr_rowmax = rowmax.load(i)

            # Shuffle TN elemnents per thread and choose the max among them.
            @unroll
            for j in range(TN.to_int()):
                curr_rowmax = max(
                    warp_reduce[DType.float32, shuffle_xor, _max_capturing](
                        reg_result.load((i * TN + j).to_int())
                    ),
                    curr_rowmax,
                )
            correction.store(i, exp(rowmax.load(i) - curr_rowmax))

            @unroll
            for j in range(TN.to_int()):
                let idx = (i * TN + j).to_int()
                reg_result.store(idx, exp(reg_result.load(idx) - curr_rowmax))

            var curr_rowsum = Float32(0.0)

            @unroll
            for j in range(TN.to_int()):
                curr_rowsum += warp_reduce[
                    DType.float32, shuffle_xor, _add_capturing
                ](reg_result.load((i * TN + j).to_int()))

            rowmax.store(i, curr_rowmax)
            rowsum.store(i, rowsum.load(i) * correction.load(i) + curr_rowsum)

        @unroll
        for i in range(TM.to_int()):

            @unroll
            for j in range(0, TN.to_int(), simd_size):
                p_tile.aligned_simd_store[simd_size, alignment](
                    ((mm_row + i) * BN + mm_col + j).to_int(),
                    reg_result.simd_load[simd_size]((i * TN + j).to_int()),
                )

        # Clear thread register results for P * V.
        @unroll
        for i in range((TM * TN).to_int()):
            reg_result.store(i, 0.0)

        # V tile has shape [BN, depth]. Load sub-tile [BK, depth] each time and
        # multiply with the corresponding P slice of shape [BM, BK].
        alias loadv_num_rows_per_iter = (num_threads * simd_size) // depth
        alias loadv_num_iters = BK // loadv_num_rows_per_iter
        alias loadv_iter_stride = loadv_num_rows_per_iter * row_stride
        for subtile_start_row in range(0, BN.to_int(), BK.to_int()):

            @unroll
            for i in range(loadv_num_iters.to_int()):
                let row_in_tile: _int32 = loadv_row + i * loadv_num_rows_per_iter
                let global_idx: _int32 = global_kv_offset + (
                    subtile_start_row + row_in_tile
                ) * row_stride + loadv_col
                let vec = v_ptr.aligned_simd_load[simd_size, alignment](
                    global_idx.to_int()
                )
                kv_tile.aligned_simd_store[simd_size, alignment](
                    (row_in_tile * depth + loadv_col).to_int(), vec
                )
            # Guard writing to p_tile and kv_tile.
            barrier()

            let p_ptr = p_tile.offset(subtile_start_row)
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
        for i in range(TM.to_int()):

            @unroll
            for j in range(TN.to_int()):
                let idx = (i * TN + j).to_int()
                o_thread_tile.store(
                    idx,
                    o_thread_tile.load(idx) * correction.load(i)
                    + reg_result.load(idx),
                )

        # Point to  next tile
        global_kv_offset += BN * num_heads * depth

    # Write the output from register to global memory.
    # The output tile [BM, depth] is divided into each thread's TMxTN registers.
    # Current thread's tile starts at (mm_row, mm_col).
    var o_global_row_offset = global_q_offset + mm_row * row_stride

    @unroll
    for i in range(TM.to_int()):

        @unroll
        for offset in range(0, TN.to_int(), simd_size):
            # Apply the denominator of softmax.
            let vec = o_thread_tile.simd_load[simd_size](
                (i * TN + offset).to_int()
            ) / rowsum.load(i)

            output_ptr.aligned_simd_store[simd_size, alignment](
                (o_global_row_offset + mm_col + offset).to_int(), vec
            )
        o_global_row_offset += row_stride


fn _naive_attention_with_transpose[
    type: DType,
    BSHD: DimList,
    BHSD: DimList,
    BHDS: DimList,
    transpose_k: Bool = False,
](
    output: NDBuffer[4, BSHD, type],
    q: NDBuffer[4, BSHD, type],
    k: NDBuffer[4, BSHD, type],
    v: NDBuffer[4, BSHD, type],
    mask: NDBuffer[2, DimList.create_unknown[2](), type],
    scale: Float32,
):
    """This kernel provides reference values for flash attention in llama 2.
    It can't be used in any model.
    """
    alias simd_size = simdwidthof[type]()

    let qkv_size = q.num_elements()
    let batch_size = q.dim[0]()
    let seq_len = q.dim[1]()
    let num_heads = q.dim[2]()
    let depth = q.dim[3]()
    let score_size = batch_size * num_heads * seq_len * seq_len

    # Q, K, V transposed
    let qt_ptr = DTypePointer[type].alloc(qkv_size)
    let kt_ptr = DTypePointer[type].alloc(qkv_size)
    let vt_ptr = DTypePointer[type].alloc(qkv_size)
    # Score = softmax(Q * K)
    let score_ptr = DTypePointer[type].alloc(score_size)
    # O = Score * V. It's transposed and will be transposed back to output.
    let ot_ptr = DTypePointer[type].alloc(qkv_size)

    let qt = NDBuffer[4, BHSD, type](qt_ptr)
    let kt = NDBuffer[4, BHDS, type](kt_ptr)
    let vt = NDBuffer[4, BHSD, type](vt_ptr)
    let score = NDBuffer[4, DimList.create_unknown[4](), type](
        score_ptr, Index(batch_size, num_heads, seq_len, seq_len)
    )
    let ot = NDBuffer[4, BHSD, type](ot_ptr)

    # BSHD -> BHSD
    let q_perm = Buffer[4, DType.index].stack_allocation()
    q_perm[0] = 0
    q_perm[1] = 2
    q_perm[2] = 1
    q_perm[3] = 3

    # BSHD -> BHDS
    let k_perm = Buffer[4, DType.index].stack_allocation()
    k_perm[0] = 0
    k_perm[1] = 2
    k_perm[2] = 3
    k_perm[3] = 1

    # BHSD -> BSHD
    let o_perm = Buffer[4, DType.index].stack_allocation()
    o_perm[0] = 0
    o_perm[1] = 2
    o_perm[2] = 1
    o_perm[3] = 3

    with Runtime() as rt:
        var chain = OwningOutputChainPtr(rt)
        transpose[4, BHSD, BSHD, type](qt, q, q_perm.data, chain.borrow())
        chain.wait()

        chain = OwningOutputChainPtr(rt)
        transpose[4, BHDS, BSHD, type](kt, k, k_perm.data, chain.borrow())
        chain.wait()

        chain = OwningOutputChainPtr(rt)
        transpose[4, BHSD, BSHD, type](vt, v, q_perm.data, chain.borrow())
        chain.wait()

        chain = OwningOutputChainPtr(rt)
        _naive_attention[type, BSHD, BHSD, BHDS, transpose_k](
            ot,
            qt,
            rebind[NDBuffer[4, DimList.create_unknown[4](), type]](kt),
            vt,
            mask,
            scale,
            chain.borrow(),
        )
        chain.wait()

        chain = OwningOutputChainPtr(rt)
        transpose[4, BSHD, BHSD, type](output, ot, o_perm.data, chain.borrow())
        chain.wait()

    qt_ptr.free()
    kt_ptr.free()
    vt_ptr.free()
    score_ptr.free()
    ot_ptr.free()


fn _naive_attention[
    type: DType,
    BSHD: DimList,
    BHSD: DimList,
    BHDS: DimList,
    transpose_k: Bool = False,
](
    output: NDBuffer[4, BHSD, type],
    q: NDBuffer[4, BHSD, type],
    k: NDBuffer[4, DimList.create_unknown[4](), type],
    v: NDBuffer[4, BHSD, type],
    mask: NDBuffer[2, DimList.create_unknown[2](), type],
    scale: Float32,
    out_chain: OutputChainPtr,
):
    """This kernel provides reference values for flash attention in llama 2.
    It can't be used in any model.
    """
    alias simd_size = simdwidthof[type]()

    let batch_size = q.dim[0]()
    let num_heads = q.dim[1]()
    let seq_len = q.dim[2]()
    let depth = q.dim[3]()
    let score_size = batch_size * num_heads * seq_len * seq_len
    let score_ptr = DTypePointer[type].alloc(score_size)
    let score = NDBuffer[4, DimList.create_unknown[4](), type](
        score_ptr, Index(batch_size, num_heads, seq_len, seq_len)
    )

    let rt = out_chain.get_runtime()

    var chain = OwningOutputChainPtr(rt)
    batched_matmul[4, type, type, type, False, transpose_k](
        score,
        rebind[NDBuffer[4, DimList.create_unknown[4](), type]](q),
        rebind[NDBuffer[4, DimList.create_unknown[4](), type]](k),
        chain.borrow(),
    )
    chain.wait()

    @parameter
    @always_inline
    fn scale_and_mask[width: Int, _rank: Int](coords: StaticIntTuple[_rank]):
        var vec = score.simd_load[width](rebind[StaticIntTuple[4]](coords))
        vec = vec * scale.cast[type]()
        vec = vec + mask.simd_load[width](
            Index(coords[_rank - 2], coords[_rank - 1])
        )
        score.simd_store[width](rebind[StaticIntTuple[4]](coords), vec)

    chain = OwningOutputChainPtr(rt)
    elementwise[4, simd_size, scale_and_mask](
        score.dynamic_shape, chain.borrow()
    )
    chain.wait()

    chain = OwningOutputChainPtr(rt)
    softmax[type, simd_size, 4, DimList.create_unknown[4]()](
        score,
        score,
        3,
        chain.borrow(),
    )
    chain.wait()

    batched_matmul[4, type, type, type, False, False](
        rebind[NDBuffer[4, DimList.create_unknown[4](), type]](output),
        score,
        rebind[NDBuffer[4, DimList.create_unknown[4](), type]](v),
        out_chain,
    )

    score_ptr.free()

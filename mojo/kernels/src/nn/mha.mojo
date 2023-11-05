# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from math import add, div_ceil, iota, max, min, sqrt, neginf, exp
from algorithm import (
    elementwise,
    unroll,
    vectorize,
    unswitch,
)
from BatchedMatmul import batched_matmul
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
    AddressSpace,
    lane_id,
    WARP_SIZE,
    shuffle_down,
    shuffle_xor,
    warp_reduce,
)
from gpu.host import Function, Stream


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


# Global settings for GPU flash attention kernel.
alias _gpu_qtile_nrows = 8
alias _gpu_kvtile_nrows = WARP_SIZE
alias _depth = 128


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
    mask: NDBuffer[2, mask_shape, mask_type],
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
        (7) Output = Transpose(P).

    B, S, H, D denote batch size, sequence length, head count and depth, respectively.
    (1), (2), (3) happens while loading the data into shared memory.
    (4) happens when writing output to global memory.

    Assumptions:
        (1) maximum depth per head is 128.
        (2) seqlen is multiple of _gpu_qtile_nrows and _gpu_kvtile_nrows.
        (3) depth is multiple of _gpu_kvtile_nrows.
        (4) _gpu_qtile_nrows * _depth is multiple of threads per block.

    P has shape [_gpu_qtile_nrows,  _gpu_kvtile_nrows]. We for now set
    _gpu_kvtile_nrows to warp_size to use wrap reduction in softmax and each
    thread handles one element. The thread count per block is same as P's size.
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

    # q shape [batch_size, seq_len, # heads, depth]
    let batch_size = q.dim[0]()
    let seq_len = q.dim[1]()
    let num_heads = q.dim[2]()
    let depth = q.dim[3]()

    try:
        let func = Function[
            # fmt: off
            fn (DTypePointer[DType.float32],
                DTypePointer[DType.float32],
                DTypePointer[DType.float32],
                DTypePointer[DType.float32],
                DTypePointer[DType.float32],
                Float32, Int, Int, Int, Int) -> None,
            # fmt: on
            flash_attention_kernel,
        ]()

        func(
            # grid
            (div_ceil(seq_len, _gpu_qtile_nrows), num_heads, batch_size),
            # block
            (min(1024, _gpu_qtile_nrows * _gpu_kvtile_nrows), 1, 1),
            q.data,
            k.data,
            v.data,
            mask.data,
            output.data,
            scale,
            batch_size,
            seq_len,
            num_heads,
            depth,
            stream=out_chain.get_cuda_stream(),
        )
    except e:
        out_chain.mark_error(e)


fn flash_attention_kernel(
    q_ptr: DTypePointer[DType.float32],
    k_ptr: DTypePointer[DType.float32],
    v_ptr: DTypePointer[DType.float32],
    mask_ptr: DTypePointer[DType.float32],
    output_ptr: DTypePointer[DType.float32],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    # head_dim is the same as depth
    depth: Int,
):
    let q_tile = stack_allocation[
        _gpu_qtile_nrows * _depth,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    let o_tile = stack_allocation[
        _gpu_qtile_nrows * _depth,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    let kv_tile = stack_allocation[
        _gpu_kvtile_nrows * _depth,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    let p_tile = stack_allocation[
        _gpu_qtile_nrows * _gpu_kvtile_nrows,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    let rowmax = stack_allocation[
        _gpu_qtile_nrows,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    let rowsum = stack_allocation[
        _gpu_qtile_nrows,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    # q has shape [batch, seq_len * num_heads * depth]. Each thread block loads
    # one tile [_gpu_qtile_nrows, depth]. The block shapes are
    # [num_tiles, num_heads, batch_size].
    let batch_idx = BlockIdx.z()
    let head_idx = BlockIdx.y()
    let q_tile_idx = BlockIdx.x()

    # Load q to shared memory.
    let global_q_offset = batch_idx * seq_len * num_heads * depth + q_tile_idx * _gpu_qtile_nrows * num_heads * depth + head_idx * depth
    let tid = ThreadIdx.x()
    let num_threads = BlockDim.x()
    # Number elments loaded per iteration by all threads.
    let num_rows_per_iter = num_threads // depth

    # Index of the row loaded by current thread
    let local_row_idx = tid // depth
    # Index of the element loaded by current thread.
    let row_offset = tid % depth
    # row_stride between two rows in Q array. Q is transposed to [B, H, S, D] while loading.
    # so row_stride != row size.
    let row_stride = num_heads * depth
    # Index of element loaded by curernt thread in global Q array.
    let global_q_idx = global_q_offset + local_row_idx * row_stride + row_offset
    # Number iterations in loading Q
    var num_iters = _gpu_qtile_nrows // num_rows_per_iter

    # Load q into shared memory
    for i in range(num_iters):
        q_tile.store(
            tid + i * num_threads,
            q_ptr.offset(
                global_q_idx + i * num_rows_per_iter * row_stride
            ).load(),
        )

    # Initialize output to zero
    for i in range(num_iters):
        o_tile.store(tid + i * num_threads, 0.0)

    # Initialize rowsum and rowmax.
    if tid < _gpu_qtile_nrows:
        rowsum.store(tid, 0.0)
        rowmax.store(tid, neginf[DType.float32]())

    barrier()

    # Offset of K/V tile in global K/V buffer, i.e., first element of current head.
    var global_kv_offset = batch_idx * seq_len * num_heads * depth + head_idx * depth
    var global_kv_idx = global_kv_offset + local_row_idx * row_stride + row_offset

    for kv_tile_start_row in range(0, seq_len, _gpu_kvtile_nrows):
        # Load K tile.
        num_iters = _gpu_kvtile_nrows // num_rows_per_iter
        for i in range(num_iters):
            kv_tile.store(
                tid + i * num_threads,
                k_ptr.offset(
                    global_kv_idx + i * num_rows_per_iter * row_stride
                ).load(),
            )
        barrier()

        # P = Q * K^t * scale, has shape [_gpu_qtile_nrows, _gpu_kvtile_nrows = warp_size].
        # It's mapped to _gpu_qtile_nrows warps of threads and each warp covers
        # one row with one element per thread.
        # !!!!!!!!!!!!!! Loading kv_tile has severe bank conflicts !!!!!!!!!!!!!
        # Thread updates element at (p_tile_row,  p_tile_col).
        let p_tile_col = lane_id()
        let p_tile_row = tid // _gpu_kvtile_nrows
        var acc: SIMD[DType.float32, 1] = 0.0
        for d in range(depth):
            acc += q_tile.load(p_tile_row * depth + d) * kv_tile.load(
                p_tile_col * depth + d
            )
        acc = acc * scale
        # Add attention mask
        # Mask has shape [seq_len, seq_len]. p_tile' correlates to a mask tile
        # starting at (q_tile_idx * _gpu_qtile_nrows, kv_tile_start_row).
        let mask_idx = (
            q_tile_idx * _gpu_qtile_nrows + p_tile_row
        ) * seq_len + kv_tile_start_row + p_tile_col
        acc = acc + mask_ptr.offset(mask_idx).load()

        # Online Softmax for P
        let pre_rowmax = rowmax.load(p_tile_row)
        let curr_rowmax = max(
            warp_reduce[DType.float32, shuffle_xor, max](acc), pre_rowmax
        )
        let correction = exp(pre_rowmax - curr_rowmax)
        # Apply the softmax nominator to score (p_tile).
        acc = exp(acc - curr_rowmax)
        p_tile.store(tid, acc)
        # Keep record of running sum and max.
        let curr_rowsum = warp_reduce[DType.float32, shuffle_down, add](acc)
        if p_tile_col == 0:
            rowmax.store(p_tile_row, curr_rowmax)
            rowsum.store(
                p_tile_row, rowsum.load(p_tile_row) * correction + curr_rowsum
            )
        # TODO: Probably can remove the barrier.
        barrier()

        # Load V tile, reuse the buffer for K.
        for i in range(num_iters):
            kv_tile.store(
                tid + i * num_threads,
                v_ptr.offset(
                    global_kv_idx + i * num_rows_per_iter * row_stride
                ).load(),
            )
        barrier()

        # O = O * correction + online_softmax(P) * V
        # Assume depth is multiple of _gpu_kvtile_nrows (= warp_size).
        # Partition the output [_gpu_qtile_nrows, depth] into tiles of shape
        # [_gpu_qtile_nrows, _gpu_kvtile_nrows].
        for i in range(depth // _gpu_kvtile_nrows):
            # Each thread accumulates (o_tile_row, o_tile_col) in the output tile.
            let v_tile_col = p_tile_col + i * _gpu_kvtile_nrows
            acc = 0.0
            for dot_idx in range(_gpu_kvtile_nrows):
                acc += p_tile.load(
                    p_tile_row * _gpu_kvtile_nrows + dot_idx
                ) * kv_tile.load(dot_idx * depth + v_tile_col)
            let o_idx = p_tile_row * depth + v_tile_col
            # o_tile.store(o_idx, o_tile.load(o_idx) + acc)
            o_tile.store(o_idx, o_tile.load(o_idx) * correction + acc)
        # Guard Reading p_tile before writing to it.
        barrier()

        # Point to  next tile
        global_kv_offset += _gpu_kvtile_nrows * num_heads * depth
        global_kv_idx += _gpu_kvtile_nrows * num_heads * depth

    # Sync the writing to o_tile.
    barrier()

    # write output to global memory
    # Output and Q have the same layout. Reuse the index mapping.
    for i in range(_gpu_qtile_nrows // num_rows_per_iter):
        let q_row_idx = local_row_idx + i * num_rows_per_iter
        output_ptr.offset(
            global_q_idx + i * num_rows_per_iter * row_stride
        ).store(o_tile.load(tid + i * num_threads) / rowsum.load(q_row_idx))


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

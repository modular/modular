# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from math import add, div_ceil, max, min, sqrt, neginf, exp
from algorithm import (
    elementwise,
    unroll,
    vectorize,
)
from BatchedMatmul import batched_matmul
from Matmul import matmul
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer
from memory import stack_allocation
from runtime.llcl import OwningOutputChainPtr, Runtime
from Softmax import softmax
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

    P has shape [_gpu_qtile_nrows,  _gpu_kvtile_nrows]. We for now let set
    _gpu_kvtile_nrows to warp_size to use wrap reduction in softmax and each
    thread handle one element. The thread count per block is same as P's size.
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


fn _naive_attention[
    type: DType,
    BSHD: DimList,
    BHSD: DimList,
    BHDS: DimList,
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
        batched_matmul[4, type, type, type, False, False](
            score,
            rebind[NDBuffer[4, DimList.create_unknown[4](), type]](qt),
            rebind[NDBuffer[4, DimList.create_unknown[4](), type]](kt),
            chain.borrow(),
        )
        chain.wait()

        @parameter
        @always_inline
        fn scale_and_mask[
            width: Int, _rank: Int
        ](coords: StaticIntTuple[_rank]):
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

        chain = OwningOutputChainPtr(rt)
        batched_matmul[4, type, type, type, False, False](
            rebind[NDBuffer[4, DimList.create_unknown[4](), type]](ot),
            score,
            rebind[NDBuffer[4, DimList.create_unknown[4](), type]](vt),
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

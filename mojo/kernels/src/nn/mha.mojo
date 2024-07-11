# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from math import align_down, ceildiv, exp, iota

from algorithm import elementwise
from buffer import Buffer, NDBuffer
from buffer.dimlist import DimList
from gpu import (
    WARP_SIZE,
    BlockIdx,
    BlockDim,
    ThreadIdx,
    barrier,
    lane_id,
    shuffle_xor,
    warp_reduce,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.memory import AddressSpace, dynamic_shared_memory
from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy_dram_to_sram,
    copy_local_to_sram,
    copy_local_to_dram,
    copy_sram_to_dram,
)
from layout.tensor_core import (
    get_accum_type,
    get_fragment_size,
    get_mma_shape,
)
from linalg.bmm import batched_matmul
from linalg.matmul import matmul
from linalg.transpose import transpose
from linalg._multistage_gemm_gpu import multistage_mma
from memory import stack_allocation
from memory.reference import AddressSpace as _AddressSpace
from memory.unsafe import DTypePointer, bitcast
from runtime.llcl import MojoCallContextPtr

from utils.index import Index, StaticIntTuple
from utils.numerics import neg_inf, min_or_neg_inf
from utils.static_tuple import StaticTuple

from .softmax import softmax, _online_softmax_iter_for_mma_output, _softmax_gpu

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

    # The transpose of Q K V swaps batch and matmul dimensions,
    # e.x. 1x128x12x64 -> 1x12x128x64, which batched_matmul can't handle.
    # They are properly transposed before this kernel.
    batched_matmul[
        rank,
        q_type,
        k_type,
        score_type,
        transpose_k,
        fuse_elementwise_fn,
    ](
        score.make_dims_unknown(),
        q.make_dims_unknown(),
        k.make_dims_unknown(),
    )

    softmax[score_type, simd_size, rank](score, score, rank - 1)

    # NOTE: synchronous, so the stack allocated score_mem is safe.
    batched_matmul[rank, score_type, v_type, output_type, transpose_b=False](
        output.make_dims_unknown(),
        score.make_dims_unknown(),
        v.make_dims_unknown(),
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
    mask_rank: Int,
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
    use_tensor_core: Bool = False,
](
    output: NDBuffer[output_type, rank, output_shape],
    q: NDBuffer[q_type, rank, q_shape],
    k: NDBuffer[k_type, rank, k_shape],
    v: NDBuffer[v_type, rank, v_shape],
    mask: NDBuffer[mask_type, mask_rank, mask_shape],
    scale: Float32,
    context: MojoCallContextPtr = MojoCallContextPtr(),
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

    All inputs (query, key, and value) must have BSHD layout. The mask can be
    BSS or BHSS.

    This kernel also handles grouped attention optimization. In this case the shape of
    K and V are BShD where h = H / num_groups.
    """
    constrained[target == "cuda", "only valid on Nvidia GPUs"]()
    constrained[rank == 4, "only support rank 4 inputs."]()
    constrained[mask_rank in (3, 4), "only support rank 3 or 4 mask."]()
    constrained[
        q_type == k_type == v_type == output_type,
        "Q, K, V, output should have same type.",
    ]()
    constrained[
        q_type == DType.float32 or q_type.is_half_float(),
        "Only support single and half precision.",
    ]()

    var ctx = context.get_cuda_device()

    # Runtime dimensions.
    var batch_size = q.dim[0]()
    var seq_len = q.dim[1]()
    var num_keys = k.dim[1]()

    @parameter
    if q_shape.all_known[2, 4]() and k_shape.has_value[2]():
        alias num_heads = q_shape.get[2]()
        alias depth = q_shape.get[3]()
        alias k_num_heads = k_shape.get[2]()
        alias group = num_heads // k_num_heads

        flash_attention_impl[
            rank,
            mask_rank,
            q_type,
            k_type,
            v_type,
            mask_type,
            output_type,
            depth,
            num_heads,
            group,
            add_attn_mask,
            target,
            use_tensor_core,
        ](
            output.data,
            q.data,
            k.data,
            v.data,
            mask.data,
            scale,
            batch_size,
            seq_len,
            num_keys,
            ctx,
        )

    else:
        var num_heads = q.dim[2]()
        var depth = q.dim[3]()
        var group = q.dim[2]() // k.dim[2]()

        mha_gpu_naive[
            mask_rank, q_type, k_type, v_type, mask_type, output_type
        ](
            q.data,
            k.data,
            v.data,
            mask.data,
            output.data,
            scale,
            batch_size,
            seq_len,
            num_keys,
            num_heads,
            depth,
            group,
            ctx,
        )


fn flash_attention_impl[
    rank: Int,
    mask_rank: Int,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    mask_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    add_attn_mask: Bool = True,
    target: StringLiteral = "cpu",
    use_tensor_core: Bool = False,
](
    output: DTypePointer[output_type],
    q: DTypePointer[q_type],
    k: DTypePointer[k_type],
    v: DTypePointer[v_type],
    mask: DTypePointer[mask_type],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
) raises:
    alias qtile_num_rows = 32
    alias ktile_num_rows = 128
    # TODO: #25898, use max_finite
    alias max_uint32 = Int(0xFFFFFFFF)
    var use_32bit_indexing = qtile_num_rows * depth < max_uint32 and ktile_num_rows * depth < max_uint32 and qtile_num_rows * ktile_num_rows < max_uint32 and batch_size * seq_len * seq_len < max_uint32

    if not use_32bit_indexing:
        raise Error("32bits index overflow.")

    try:
        # Legacy impl for FP32
        @parameter
        if (
            q_type == DType.float32
            and depth == 128
            and mask_rank == 3
            and not use_tensor_core
        ):
            if seq_len == num_keys and seq_len % 128 == 0:
                var func = ctx.compile_function[
                    flash_attention_kernel[
                        BM=qtile_num_rows,
                        BN=ktile_num_rows,
                        BK=16,
                        depth=depth,
                        num_heads=num_heads,
                        TM=8,
                        TN=4,
                        num_threads=128,
                    ]
                ]()

                ctx.enqueue_function(
                    func,
                    q,
                    k,
                    v,
                    mask,
                    output,
                    scale,
                    batch_size,
                    seq_len,
                    grid_dim=(
                        ceildiv(seq_len, 32),
                        num_heads,
                        batch_size,
                    ),
                    block_dim=(128, 1, 1),
                )

                return

            else:
                var func = ctx.compile_function[
                    flash_attention_kernel_flexible_seqlen[
                        BM=qtile_num_rows,
                        BN=ktile_num_rows,
                        BK=16,
                        depth=depth,
                        num_heads=num_heads,
                        TM=8,
                        TN=4,
                        num_threads=128,
                    ]
                ]()

                ctx.enqueue_function(
                    func,
                    q,
                    k,
                    v,
                    mask,
                    output,
                    scale,
                    batch_size,
                    seq_len,
                    num_keys,
                    grid_dim=(
                        ceildiv(seq_len, 32),
                        num_heads,
                        batch_size,
                    ),
                    block_dim=(128, 1, 1),
                )

                return

        @parameter
        if use_tensor_core:
            if seq_len == num_keys and seq_len % 128 == 0 and depth % 32 == 0:
                var func = ctx.compile_function[
                    mha[
                        mask_rank,
                        q_type,
                        k_type,
                        v_type,
                        mask_type,
                        output_type,
                        BM=qtile_num_rows,
                        BN=ktile_num_rows,
                        BK = 16 if q_type == DType.float32 else 32,
                        WM=qtile_num_rows,
                        WN=32,  # got from cutlass
                        depth=depth,
                        num_heads=num_heads,
                        num_threads=128,  # (depth // WN) * WARP_SIZE,
                        num_pipeline_stages=4,
                        group=group,
                    ]
                ](
                    # TODO: Avoid hard coding shared memory needed.
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        80 * 1024
                    )
                )

                ctx.enqueue_function(
                    func,
                    q,
                    k,
                    v,
                    mask,
                    output,
                    scale,
                    batch_size,
                    seq_len,
                    grid_dim=(
                        ceildiv(seq_len, qtile_num_rows),
                        num_heads,
                        batch_size,
                    ),
                    # block_dim=(depth // WN * WARP_SIZE, 1, 1),
                    block_dim=(128, 1, 1),
                    shared_mem_bytes=80 * 1024,
                )

                return

        mha_gpu_naive[mask_rank](
            q,
            k,
            v,
            mask,
            output,
            scale,
            batch_size,
            seq_len,
            num_keys,
            num_heads,
            depth,
            group,
            ctx,
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

    @parameter
    for k in range(K):
        # load a element starting from (row, k) or (k, row) if transposed.
        @parameter
        if transpose_a:
            # vector load
            @parameter
            for offset in range(0, TM, simd_size):
                SIMD[size=simd_size].store(
                    reg_m,
                    offset,
                    SIMD[size=simd_size].load[alignment=alignment](
                        a, k * M + row + offset
                    ),
                )
        else:
            # scalar load
            @parameter
            for i in range(TM):
                reg_m[i] = a[(row + i) * leading_dim_a + k]

        @parameter
        for offset in range(0, TN, simd_size):
            var vec = SIMD[size=simd_size].load[alignment=alignment](
                b, k * N + col + offset
            )
            SIMD.store(reg_n, offset, vec)

        @parameter
        for i in range(TM):

            @parameter
            for j in range(TN):
                reg_res[i * TN + j] = reg_res[i * TN + j] + reg_m[i] * reg_n[j]


@always_inline
fn _fill[
    len: Int, type: DType, address_space: _AddressSpace
](ptr: DTypePointer[type, address_space], val: Scalar[type]):
    alias simd_width = simdwidthof[val.type]()
    alias vector_end = align_down(len, simd_width)

    @parameter
    for i in range(0, vector_end, simd_width):
        SIMD.store(ptr, i, SIMD[type, simd_width](val))

    @parameter
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
    var lane = lane_id()
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

    var q_gmem_base = LayoutTensor[
        DType.float32, Layout(IntTuple(BM, depth), IntTuple(row_stride, 1))
    ](q_ptr.offset(global_q_offset))

    var q_smem_base = LayoutTensor[
        DType.float32,
        Layout.row_major(BM, depth),
        address_space = AddressSpace.SHARED,
    ](q_tile)

    alias thread_xmem_layout = Layout.row_major(
        (num_threads // depth) * simd_size, depth // simd_size
    )
    copy_dram_to_sram[
        src_thread_layout=thread_xmem_layout,
        dst_thread_layout=thread_xmem_layout,
    ](
        q_smem_base.vectorize[1, simd_size](),
        q_gmem_base.vectorize[1, simd_size](),
    )

    # Clear thread's register tile for output.
    _fill[TM * TN](o_thread_tile, 0)

    _fill[TM](rowmax, neg_inf[DType.float32]())
    _fill[TM](rowsum, 0)

    # Offset of K/V tile in global K/V buffer, i.e., 1st element of current head.
    var global_kv_offset = depth * (head_idx + num_heads * seq_len * batch_idx)

    # K tile has shape [BN, depth] and is divided sub-tiles [BN, BK].
    # 1st row and col in k sub-tile loaded by current thread.

    # V tile has shape [BN, depth] and is divided sub-tiles [BK, depth].
    # 1st row and col in v sub-tile loaded by current thread.

    for kv_tile_start_row in range(0, seq_len, BN):
        # Clear thread tile results.
        _fill[TM * TN](reg_result, 0)

        alias BN_padded = BN + smem_pad

        # K tile has shape [BN, depth]. Load sub-tile [BN, BK] each time and
        # multiply with the corresponding Q slice of shape [BM, BK].
        var k_gmem_base = LayoutTensor[
            DType.float32, Layout(IntTuple(BN, depth), IntTuple(row_stride, 1))
        ](k_ptr.offset(global_kv_offset))

        var k_smem_base = LayoutTensor[
            DType.float32,
            Layout.row_major(BK, BN_padded),
            address_space = AddressSpace.SHARED,
        ](kv_tile)
        var k_smem_tile = k_smem_base.slice[:, :BN]()

        for subtile_start_col in range(0, depth, BK):
            var k_gmem_tile = k_gmem_base.tile[BN, BK](subtile_start_col // BK)

            copy_dram_to_sram[
                src_thread_layout = Layout.row_major(num_threads // BK, BK),
                dst_thread_layout = Layout.col_major(BK, num_threads // BK),
            ](k_smem_tile, k_gmem_tile)

            # Gaurd write of q_tile and kv_tile.
            barrier()

            var q_ptr = q_tile.offset(subtile_start_col)
            _mm[BM, BN_padded, BK, depth, TM, TN, transpose_a=False](
                q_ptr,
                kv_tile,
                mm_row,
                mm_col,
                reg_m,
                reg_n,
                reg_result,
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

        @parameter
        for i in range(TM):

            @parameter
            for j in range(0, TN, simd_size):
                var idx = i * TN + j
                var vec = SIMD[size=simd_size].load(reg_result, idx)
                var mask_idx = mask_offset + i * seq_len + j
                var mask_vec = SIMD[size=simd_size].load[alignment=alignment](
                    mask_ptr, mask_idx
                )
                SIMD.store(reg_result, idx, vec * scale + mask_vec)

        # Online Softmax
        @parameter
        for i in range(TM):
            var curr_rowmax = rowmax[i]

            # Find thread register tile's max at i-th row.
            @parameter
            for j in range(TN):
                curr_rowmax = max(reg_result[i * TN + j], curr_rowmax)
            # Reduce the max of block tile's row.
            curr_rowmax = warp_reduce[shuffle_xor, _max_capturing](curr_rowmax)

            correction[i] = exp(rowmax[i] - curr_rowmax)

            @parameter
            for j in range(TN):
                reg_result[i * TN + j] = exp(
                    reg_result[i * TN + j] - curr_rowmax
                )

            var curr_rowsum = Float32(0.0)

            # Sum thread register tile at the i-th row.
            @parameter
            for j in range(TN):
                curr_rowsum += reg_result[i * TN + j]
            # Reduce the sum of block tile's row.
            curr_rowsum = warp_reduce[shuffle_xor, _add_capturing](curr_rowsum)

            rowmax[i] = curr_rowmax
            rowsum[i] = rowsum[i] * correction[i] + curr_rowsum

        # Correct previous output.
        @parameter
        for i in range(TM):

            @parameter
            for j in range(TN):
                o_thread_tile[i * TN + j] *= correction[i]

        # V tile has shape [BN, depth]. P tile has shape [BM, BN]. Each itertion
        # loads V sub-tile [BK, depth] from global memory to shared memory and
        # stages p sub-tile [BM, BK] from thread register tile to shared memory.

        var v_gmem_base = LayoutTensor[
            DType.float32, Layout(IntTuple(BN, depth), IntTuple(row_stride, 1))
        ](v_ptr.offset(global_kv_offset))

        var v_smem_base = LayoutTensor[
            DType.float32,
            Layout.row_major(BK, BN),
            address_space = AddressSpace.SHARED,
        ](kv_tile)
        var storep_col_start = 0
        for subtile_start_row in range(0, BN, BK):
            # Store thread register tile to p sub-tile.
            if mm_col >= storep_col_start and mm_col < storep_col_start + BK:

                @parameter
                for i in range(TM):

                    @parameter
                    for j in range(0, TN, simd_size):
                        SIMD[size=simd_size].store[alignment=alignment](
                            p_tile,
                            (mm_row + i) * BK + mm_col - storep_col_start + j,
                            SIMD[size=simd_size].load(reg_result, i * TN + j),
                        )
            storep_col_start += BK

            # Load v sub-tile.
            var v_gmem_tile = v_gmem_base.tile[BK, depth](
                subtile_start_row // BK
            )
            alias thread_v_xmem_layout = Layout.row_major(
                (num_threads // BN) * simd_size, BN // simd_size
            )
            copy_dram_to_sram[
                src_thread_layout=thread_v_xmem_layout,
                dst_thread_layout=thread_v_xmem_layout,
            ](
                v_smem_base.vectorize[1, simd_size](),
                v_gmem_tile.vectorize[1, simd_size](),
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

    @parameter
    for i in range(TM):

        @parameter
        for offset in range(0, TN, simd_size):
            # Apply the denominator of softmax.
            var vec = SIMD[size=simd_size].load(
                o_thread_tile, i * TN + offset
            ) / Scalar.load(rowsum, i)

            SIMD[size=simd_size].store[alignment=alignment](
                output_ptr,
                o_global_row_offset + mm_col + offset,
                vec,
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
    var lane = lane_id()
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
    alias loadq_num_rows_per_iter = ((num_threads * simd_size) // depth)
    var loadq_num_rows = min(BM, seq_len - global_q_start_row)
    var loadq_num_iters = ceildiv(loadq_num_rows, loadq_num_rows_per_iter)
    # alias loadq_num_iters = BM // loadq_num_rows_per_iter
    # We transpose Q BSHD -> BHSD. 2 subsequenet rows in q tile have stride
    # != depth in global Q array because the stride is based on BSHD.
    alias row_stride = (num_heads * depth)
    # Index of the 1st row and col loaded by current thread.
    var loadq_row = (tid * simd_size) // depth
    var loadq_col = (tid * simd_size) % depth

    ##
    for i in range(loadq_num_iters):
        var row_in_tile = loadq_row + i * loadq_num_rows_per_iter
        # The a row from Q in global memory.
        if row_in_tile + global_q_start_row < seq_len:
            var global_q_idx = global_q_offset + row_in_tile * row_stride + loadq_col
            var vec = SIMD[size=simd_size].load[alignment=alignment](
                q_ptr, global_q_idx
            )
            SIMD[size=simd_size].store[alignment=alignment](
                q_tile, row_in_tile * depth + loadq_col, vec
            )
        # The Q tile exceeds global Q buffer, pad with zeros.
        else:
            SIMD[size=simd_size].store[alignment=alignment](
                q_tile,
                row_in_tile * depth + loadq_col,
                SIMD[DType.float32, simd_size](0.0),
            )
    ##

    # Clear thread's register tile for output.
    _fill[TM * TN](o_thread_tile, 0)

    _fill[TM](rowmax, neg_inf[DType.float32]())
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
        var loadk_num_iters = ceildiv(loadk_num_rows, loadk_num_rows_per_iter)
        alias BN_padded = BN + smem_pad
        for subtile_start_col in range(0, depth, BK):
            ##
            for i in range(loadk_num_iters):
                var row_in_tile = loadk_row + i * loadk_num_rows_per_iter
                if row_in_tile + kv_tile_start_row < num_keys:
                    var global_idx = global_kv_offset + row_in_tile * row_stride + subtile_start_col + loadk_col
                    var vec = SIMD[size=simd_size].load[alignment=alignment](
                        k_ptr, global_idx
                    )

                    # Transpose k tile.
                    @parameter
                    for j in range(4):
                        kv_tile[
                            (loadk_col + j) * BN_padded + row_in_tile
                        ] = vec[j]
                else:

                    @parameter
                    for j in range(4):
                        kv_tile[(loadk_col + j) * BN_padded + row_in_tile] = 0

            # Gaurd write of q_tile and kv_tile.
            barrier()

            var q_ptr = q_tile.offset(subtile_start_col)
            _mm[BM, BN_padded, BK, depth, TM, TN, transpose_a=False](
                q_ptr,
                kv_tile,
                mm_row,
                mm_col,
                reg_m,
                reg_n,
                reg_result,
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

            @parameter
            for i in range(TM):
                # Scalar load in case mask dimension is not multiple of simd_size.
                if mask_row + i < seq_len:

                    @parameter
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
        @parameter
        for i in range(TM):
            var curr_rowmax = rowmax[i]

            # Reset result that exceeds num_keys
            var exceed = kv_tile_start_row + mm_col + TN - num_keys
            if exceed > 0:
                for j in range(TN - exceed, TN):
                    reg_result[i * TN + j] = neg_inf[DType.float32]()

            # Shuffle TN elemnents per thread and choose the max among them.
            @parameter
            for j in range(TN):
                curr_rowmax = max(
                    warp_reduce[shuffle_xor, _max_capturing](
                        reg_result[i * TN + j]
                    ),
                    curr_rowmax,
                )
            correction[i] = exp(rowmax[i] - curr_rowmax)

            @parameter
            for j in range(TN):
                var idx = i * TN + j
                reg_result[idx] = exp(reg_result[idx] - curr_rowmax)

            if exceed > 0:
                for j in range(TN - exceed, TN):
                    reg_result[i * TN + j] = 0.0

            var curr_rowsum = Float32(0.0)

            @parameter
            for j in range(TN):
                curr_rowsum += warp_reduce[shuffle_xor, _add_capturing](
                    reg_result[i * TN + j]
                )

            rowmax[i] = curr_rowmax
            rowsum[i] = rowsum[i] * correction[i] + curr_rowsum

        @parameter
        for i in range(TM):

            @parameter
            for j in range(0, TN, simd_size):
                SIMD[size=simd_size].store[alignment=alignment](
                    p_tile,
                    (mm_row + i) * BN + mm_col + j,
                    SIMD[size=simd_size].load(reg_result, i * TN + j),
                )

        # Clear thread register results for P * V.
        _fill[TM * TN](reg_result, 0)

        # V tile has shape [BN, depth]. Load sub-tile [BK, depth] each time and
        # multiply with the corresponding P slice of shape [BM, BK].
        alias loadv_num_rows_per_iter = (num_threads * simd_size) // depth
        alias loadv_num_iters = BK // loadv_num_rows_per_iter
        for subtile_start_row in range(0, BN, BK):
            ##
            @parameter
            for i in range(loadv_num_iters):
                var row_in_tile = loadv_row + i * loadv_num_rows_per_iter
                if (
                    row_in_tile + kv_tile_start_row + subtile_start_row
                    < num_keys
                ):
                    var global_idx = global_kv_offset + int(
                        subtile_start_row + row_in_tile
                    ) * row_stride + loadv_col
                    var vec = SIMD[size=simd_size].load[alignment=alignment](
                        v_ptr, global_idx
                    )
                    SIMD[size=simd_size].store[alignment=alignment](
                        kv_tile, row_in_tile * depth + loadv_col, vec
                    )
                else:
                    SIMD[size=simd_size].store[alignment=alignment](
                        kv_tile,
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
        @parameter
        for i in range(TM):

            @parameter
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

    @parameter
    for i in range(TM):
        if global_q_start_row + mm_row + i < seq_len:

            @parameter
            for offset in range(0, TN, simd_size):
                # Apply the denominator of softmax.
                var vec = SIMD[size=simd_size].load(
                    o_thread_tile, i * TN + offset
                ) / Scalar.load(rowsum, i)

                SIMD[size=simd_size].store[alignment=alignment](
                    output_ptr, o_global_row_offset + mm_col + offset, vec
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

    batched_matmul[4, type, type, type, transpose_k](score, q, k)

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

    elementwise[scale_and_mask, simd_size](score.dynamic_shape)

    try:
        softmax[type, simd_size, 4](
            score,
            score,
            axis=3,
        )
    except e:
        abort(e)

    batched_matmul[4, type, type, type, transpose_b=False](output, score, v)

    score_ptr.free()


# ===----------------------------------------------------------------------===#
# Flash attention for tensor core
# ===----------------------------------------------------------------------===#


fn mha[
    mask_rank: Int,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    mask_type: DType,
    output_type: DType,
    BM: Int,  # number of queries per block
    BN: Int,  # number of keys per block
    BK: Int,  # tile size in depth dimension
    WM: Int,
    WN: Int,
    depth: Int,
    num_heads: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    group: Int = 1,
](
    q_ptr: DTypePointer[q_type],
    k_ptr: DTypePointer[k_type],
    v_ptr: DTypePointer[v_type],
    mask_ptr: DTypePointer[mask_type],
    output_ptr: DTypePointer[output_type],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
):
    var batch_idx = BlockIdx.z()
    var q_batch_offset = depth * num_heads * seq_len * batch_idx
    var kv_batch_offset = depth * (num_heads // group) * seq_len * batch_idx
    var mask_batch_offset = batch_idx * seq_len * seq_len * (
        num_heads if mask_rank == 4 else 1
    )

    mha_single_batch[
        mask_rank,
        BM=BM,
        BN=BN,
        BK=BK,
        WM=WM,
        WN=WN,
        depth=depth,
        num_heads=num_heads,
        num_threads=num_threads,
        num_pipeline_stages=num_pipeline_stages,
        group=group,
    ](
        q_ptr.offset(q_batch_offset),
        k_ptr.offset(kv_batch_offset),
        v_ptr.offset(kv_batch_offset),
        mask_ptr.offset(mask_batch_offset),
        output_ptr.offset(q_batch_offset),
        scale,
        seq_len,
    )


fn mha_single_batch[
    mask_rank: Int,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    mask_type: DType,
    output_type: DType,
    *,
    BM: Int,  # number of queries per block
    BN: Int,  # number of keys per block
    BK: Int,  # tile size in depth dimension
    WM: Int,
    WN: Int,
    depth: Int,
    num_heads: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    group: Int = 1,
](
    q_ptr: DTypePointer[q_type],
    k_ptr: DTypePointer[k_type],
    v_ptr: DTypePointer[v_type],
    mask_ptr: DTypePointer[mask_type],
    output_ptr: DTypePointer[output_type],
    scale: Float32,
    seq_len: Int,
):
    """Flash attention v2 algorithm."""
    constrained[q_type == k_type and k_type == v_type]()

    alias simd_size = simdwidthof[q_type]()

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    constrained[
        num_warps_m * num_warps_n == (num_threads // WARP_SIZE),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var tid = ThreadIdx.x()
    var warp_id = (tid // WARP_SIZE)
    var lane = lane_id()

    # Coordinates of the current warp.
    var warp_x: UInt
    var warp_y: UInt
    warp_y, warp_x = divmod(warp_id, UInt(num_warps_n))

    # The entire query block (BM x depth) is tiled in shared memory.
    alias q_smem_size = BM * depth
    var q_smem = dynamic_shared_memory[
        Scalar[q_type], alignment = alignof[SIMD[q_type, simd_size]]()
    ]()
    var q_smem_iter = LayoutTensorIter[
        q_type, Layout.row_major(BM, BK), AddressSpace.SHARED
    ](q_smem, q_smem_size)

    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    alias k_smem_size = num_pipeline_stages * BN * BK
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    var k_smem_iter = LayoutTensorIter[
        k_type, Layout.row_major(BN, BK), AddressSpace.SHARED, circular=True
    ](k_smem, k_smem_size)

    var head_idx = BlockIdx.y()
    var q_tile_idx = BlockIdx.x()

    # Query global memory iterator
    var q_offset = depth * (head_idx + num_heads * q_tile_idx * BM)
    var q_gmem_block = LayoutTensor[
        q_type, Layout(IntTuple(BM, depth), IntTuple(num_heads * depth, 1))
    ](q_ptr + q_offset)
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)

    alias mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    alias accum_type = get_accum_type[q_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = frag_size[2]
    alias p_frag_simdwidth = p_frag_size // 2

    var p_reg_tile = LayoutTensor[
        accum_type, Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size)
    ].stack_allocation()

    var output_reg_tile = LayoutTensor[
        accum_type, Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size)
    ].stack_allocation()
    output_reg_tile.fill(0.0)

    # Rowwise max and sum for online softmax
    var rowmax = stack_allocation[WM, accum_type]()
    var rowsum = stack_allocation[WM, accum_type]()

    @parameter
    for i in range(WM):
        rowmax[i] = min_or_neg_inf[accum_type]()
        rowsum[i] = 0.0

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(num_warps_n, BM),
        address_space = AddressSpace.SHARED,
    ]((k_smem + k_smem_size).bitcast[Scalar[accum_type]]())

    # Shared memory for P = Q * K^t
    # This overlaps key tile but are used at the same time i.e. no race condition.
    var p_smem = (q_smem + q_smem_size).bitcast[Scalar[v_type]]()
    alias p_smem_size = BM * BN
    var p_smem_tile = LayoutTensor[
        v_type,
        Layout.row_major(BM, BN),
        address_space = AddressSpace.SHARED,
    ](p_smem)
    var p_smem_warp_tile = p_smem_tile.tile[WM, WN](warp_y, warp_x)
    var p_smem_iter = p_smem_tile.tiled_iterator[BM, BK, axis=1](0, 0)

    # Share memory tile for Value.
    alias v_smem_size = num_pipeline_stages * BN * BK
    var v_smem = (p_smem + p_smem_size).bitcast[Scalar[v_type]]()
    var v_smem_iter = LayoutTensorIter[
        v_type, Layout.row_major(BK, BN), AddressSpace.SHARED, circular=True
    ](v_smem, v_smem_size)

    # Mask global memory iterator.
    var mask_offset = q_tile_idx * BM * seq_len + (
        Int(head_idx * seq_len * seq_len) if mask_rank == 4 else 0
    )
    var warp_offset = warp_y * WM * seq_len + warp_x * WN
    var mask_warp_ptr = mask_ptr + Int(mask_offset) + Int(warp_offset)

    # Key global memory iterator
    # For group query
    alias kv_num_heads = num_heads // group
    var kv_offset = depth * (head_idx // group)

    for kv_tile_start_row in range(0, seq_len, BN):
        var k_gmem_block = LayoutTensor[
            k_type,
            Layout(IntTuple(BN, depth), IntTuple(kv_num_heads * depth, 1)),
        ](k_ptr + kv_offset + kv_tile_start_row * kv_num_heads * depth)
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        p_reg_tile.fill(0)

        # First iteration load q from global memory to shared memory.
        if kv_tile_start_row == 0:
            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                True,  # transpose_b
            ](
                p_reg_tile,
                q_gmem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                depth // BK,
            )
        # Subsequent iterations just use q in share memory.
        # TODO: Figure out a better function interface instead of passing in
        # shared memory iterator twice.
        else:
            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                True,  # transpose_b
            ](
                p_reg_tile,
                # Pass shared memory iterator to hint not loading from global memory.
                q_smem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                depth // BK,
            )

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        # The dimension of mask are assumed dynamic here so still using index calculation.
        # TODO: check if the explicit index calculation can be avoided.
        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                var frag_offset = m_mma * MMA_M * seq_len + n_mma * MMA_N
                var mask_frag_ptr = mask_warp_ptr + frag_offset

                var frag_lane_row = int(lane // (MMA_N // p_frag_simdwidth))
                var frag_lane_col = int(lane * p_frag_simdwidth % MMA_N)

                alias mask_align = alignof[SIMD[mask_type, p_frag_simdwidth]]()

                @parameter
                for i in range(2):
                    var mask_vec = SIMD[size=p_frag_simdwidth].load[
                        alignment=mask_align
                    ](
                        mask_frag_ptr
                        + (frag_lane_row + i * MMA_M // 2) * seq_len
                        + frag_lane_col
                    )

                    p_reg_vec2[n_mma * num_m_mmas + m_mma, i] = p_reg_vec2[
                        n_mma * num_m_mmas + m_mma, i
                    ] * scale.cast[accum_type]() + rebind[
                        p_reg_vec2.element_type
                    ](
                        mask_vec.cast[accum_type]()
                    )
        # Increment mask to next BM x BN block.
        mask_warp_ptr += BN

        _online_softmax_iter_for_mma_output[
            num_m_mmas, num_n_mmas, num_warps_n, mma_shape
        ](
            output_reg_tile,
            p_reg_tile,
            warp_scratch.tile[num_warps_n, WM](0, warp_y),
            rowmax,
            rowsum,
        )

        # Write p register tiles into shared memory.
        # TODO: support bypass shared memory.
        copy_local_to_sram[thread_layout = Layout.row_major(8, 4)](
            p_smem_warp_tile.vectorize[1, 2](),
            p_reg_tile.vectorize[1, 2]().transpose(),
        )
        barrier()

        var v_gmem_block = LayoutTensor[
            v_type,
            Layout(IntTuple(BN, depth), IntTuple(kv_num_heads * depth, 1)),
        ](v_ptr + kv_offset + kv_tile_start_row * kv_num_heads * depth)
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            False,  # transpose_b
            swizzle_a=False,
        ](
            output_reg_tile,
            p_smem_iter,
            v_gmem_iter,
            p_smem_iter,
            v_smem_iter,
            BN // BK,
        )

    # Apply softmax denumerator.
    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] /= rowsum[
                    2 * m_mma
                ]
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] /= rowsum[2 * m_mma + 1]

    var output_gmem_tile = LayoutTensor[
        output_type,
        Layout(IntTuple(BM, depth), IntTuple(num_heads * depth, 1)),
    ](output_ptr + q_offset)
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN](warp_y, warp_x)

    # Write to global memory.
    @parameter
    if output_type.is_half_float():
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            accum_type,
            Layout.row_major(BM, depth),
            address_space = AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[accum_type]]())
        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](warp_y, warp_x)
        copy_local_to_sram[thread_layout = Layout.row_major(8, 4)](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        copy_sram_to_dram[
            thread_layout = Layout.row_major(
                num_threads * simd_size // depth, depth // simd_size
            )
        ](
            output_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )
    else:
        copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.bitcast[output_type]()
            .vectorize[1, 2]()
            .transpose(),
        )


# ===----------------------------------------------------------------------===#
# Naive GPU multihead attention supporting flexible dimensions.
# ===----------------------------------------------------------------------===#


fn mha_gpu_naive[
    mask_rank: Int,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    mask_type: DType,
    output_type: DType,
](
    q_ptr: DTypePointer[q_type],
    k_ptr: DTypePointer[k_type],
    v_ptr: DTypePointer[v_type],
    mask_ptr: DTypePointer[mask_type],
    output_ptr: DTypePointer[output_type],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
) raises:
    alias p_type = get_accum_type[q_type]()
    var p_device = ctx.create_buffer[p_type](
        batch_size * num_heads * seq_len * num_keys
    )
    var p_ptr = p_device.ptr
    var p_buffer = NDBuffer[p_type, 3](
        p_ptr, Index(batch_size * num_heads, seq_len, num_keys)
    )

    var bmm0_func = ctx.compile_function[
        _bmm0[mask_rank, q_type, k_type, p_type, mask_type]
    ]()
    ctx.enqueue_function(
        bmm0_func,
        p_ptr,
        q_ptr,
        k_ptr,
        mask_ptr,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        grid_dim=(
            ceildiv(num_keys, 32),
            ceildiv(seq_len, 16),
            num_heads * batch_size,
        ),
        block_dim=(32, 16, 1),
    )

    @parameter
    @__copy_capture(p_buffer)
    fn input_fn_device[
        _simd_width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[p_type, _simd_width]:
        return p_buffer.load[width=_simd_width](
            rebind[StaticIntTuple[3]](coords)
        )

    _softmax_gpu[p_type, 1, 3, DimList.create_unknown[3](), input_fn_device](
        Index(batch_size * num_heads, seq_len, num_keys),
        p_buffer,
        2,
        ctx,
    )

    var bmm1_func = ctx.compile_function[_bmm1[p_type, v_type, output_type]]()

    ctx.enqueue_function(
        bmm1_func,
        output_ptr,
        p_ptr,
        v_ptr,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        grid_dim=(
            ceildiv(depth, 32),
            ceildiv(seq_len, 16),
            num_heads * batch_size,
        ),
        block_dim=(32, 16, 1),
    )

    _ = p_device


@always_inline
fn _bmm0[
    mask_rank: Int,
    q_type: DType,
    k_type: DType,
    p_type: DType,
    mask_type: DType,
](
    p_ptr: DTypePointer[p_type],
    q_ptr: DTypePointer[q_type],
    k_ptr: DTypePointer[k_type],
    mask_ptr: DTypePointer[mask_type],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
):
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var y = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()
    if x >= num_keys or y >= seq_len:
        return

    var batch_head = BlockIdx.z()
    var batch: UInt
    var head: UInt
    batch, head = divmod(batch_head, UInt(num_heads))

    var q_offset = int(depth * (head + num_heads * seq_len * batch))
    var q = q_ptr + q_offset

    var kv_num_heads = num_heads // group
    var kv_offset = int(
        depth * (head // group + kv_num_heads * num_keys * batch)
    )
    var k = k_ptr + kv_offset

    var p_offset = batch_head * seq_len * num_keys
    var p = p_ptr + Int(p_offset)

    var mask_offset = (
        batch if mask_rank == 3 else batch_head
    ) * seq_len * num_keys
    var mask = mask_ptr + Int(mask_offset)

    var accum = SIMD[p_type, 1](0.0)

    for d in range(UInt(depth)):
        accum += (
            q[y * num_heads * depth + d].cast[k_type]()
            * k[x * kv_num_heads * depth + d]
        ).cast[p_type]()

    p[y * num_keys + x] = (
        accum * scale.cast[p_type]() + mask[y * num_keys + x].cast[p_type]()
    )


@always_inline
fn _bmm1[
    p_type: DType,
    v_type: DType,
    output_type: DType,
](
    output_ptr: DTypePointer[output_type],
    p_ptr: DTypePointer[p_type],
    v_ptr: DTypePointer[v_type],
    seq_len: Int,
    num_keys: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
):
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var y = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()
    if x >= depth or y >= seq_len:
        return

    var batch_head = BlockIdx.z()
    var batch: UInt
    var head: UInt
    batch, head = divmod(batch_head, UInt(num_heads))

    var p_offset = batch_head * seq_len * num_keys
    var p = p_ptr + Int(p_offset)

    var kv_num_heads = num_heads // group
    var kv_offset = int(
        depth * (head // group + kv_num_heads * num_keys * batch)
    )
    var v = v_ptr + kv_offset

    var output_offset = depth * (head + num_heads * seq_len * batch)
    var output = output_ptr + Int(output_offset)

    var accum = SIMD[DType.float32, 1](0.0)

    for i in range(num_keys):
        accum += (
            p[y * num_keys + i].cast[v_type]() * v[i * kv_num_heads * depth + x]
        ).cast[DType.float32]()

    output[y * num_heads * depth + x] = accum.cast[output_type]()

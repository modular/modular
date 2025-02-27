# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import ceildiv
from sys import alignof, simdwidthof

from algorithm.functional import vectorize
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import block_idx, global_idx
from gpu.host import DeviceContext
from kv_cache.types import KVCacheT
from memory import UnsafePointer
from nn.mha import MHAConfig, _kernel_mask
from nn.mha_mask import MHAMask
from nn.softmax import _softmax_gpu
from utils.numerics import get_accum_type
from utils.index import Index, IndexList


@always_inline
fn _bmm0_bs[
    cache_t: KVCacheT,
    mask_t: MHAMask,
    q_type: DType,
    p_type: DType,
](
    p_ptr: UnsafePointer[Scalar[p_type]],
    q_ptr: UnsafePointer[Scalar[q_type]],
    k_cache: cache_t,
    q_input_row_offsets: NDBuffer[DType.uint32, 1],
    kv_input_row_offsets: NDBuffer[DType.uint32, 1],
    scale: Float32,
    batch_size: Int,
    q_max_seq_len: Int,
    # The maximum current sequence length in the KV cache.
    kv_max_seq_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    mask_functor: mask_t,
):
    # total_context_length
    var x = global_idx.x
    # prompt_length
    var y = global_idx.y

    alias k_type = cache_t.type
    alias kv_num_heads = cache_t.kv_params.num_heads

    var batch_head = block_idx.z
    var batch: UInt
    var head: UInt
    batch, head = divmod(batch_head, UInt(num_heads))

    var cur_query_len: Int
    var cur_kv_len: Int
    var q_offset: Int
    var num_keys: Int
    var padded_num_keys = kv_max_seq_len + max_cache_size
    var p_offset = batch_head * q_max_seq_len * padded_num_keys

    q_seq_start = Int(q_input_row_offsets[batch])
    q_seq_end = Int(q_input_row_offsets[batch + 1])
    cur_query_len = q_seq_end - q_seq_start
    q_offset = Int((q_seq_start * num_heads + head) * depth)

    kv_seq_start = Int(kv_input_row_offsets[batch])
    kv_seq_end = Int(kv_input_row_offsets[batch + 1])
    cur_kv_len = kv_seq_end - kv_seq_start
    # num_heads * kv_max_seq_len * batch * depth + depth * head
    num_keys = cur_kv_len + k_cache.cache_length(batch)

    debug_assert(cur_kv_len <= kv_max_seq_len, "Invalid cur_kv_len")
    debug_assert(num_keys <= padded_num_keys, "Invalid max_cache_size")

    if x >= kv_max_seq_len + max_cache_size or y >= q_max_seq_len:
        return

    var q = q_ptr + q_offset

    var kv_head = Int(head // group)

    var p = p_ptr + Int(p_offset)

    var accum = SIMD[p_type, 1](0.0)

    # Set total KV length: KV written previous to and during this forward.
    if x < num_keys and y < cur_query_len:
        var accum_vec = SIMD[p_type, simdwidthof[p_type]()](0)
        var k_ptr = k_cache.block_paged_ptr[tile_size=1](batch, x, kv_head, 0)

        @parameter
        fn accum_fn[width: Int](offset: Int):
            alias alignment = alignof[SIMD[p_type, width]]()
            var q_val = q.load[width=width, alignment=alignment](
                y * num_heads * depth + offset
            ).cast[k_type]()
            var k_val = k_ptr.load[width=width, alignment=alignment](offset)
            var qk_val = (q_val * k_val).cast[p_type]()

            @parameter
            if width == 1:
                accum += rebind[__type_of(accum)](qk_val)
            else:
                accum_vec += rebind[__type_of(accum_vec)](qk_val)

        vectorize[accum_fn, simdwidthof[p_type]()](depth)
        accum += accum_vec.reduce_add()

    var score_row = y
    var score_col = x
    p[y * padded_num_keys + x] = mask_functor.mask(
        Index(Int(batch), Int(head), Int(score_row), Int(score_col)),
        accum * scale.cast[p_type](),
    )
    p[y * padded_num_keys + x] = _kernel_mask(
        Index(score_row, score_col),
        Index(cur_query_len, num_keys),
        p[y * padded_num_keys + x],
    )


@always_inline
fn _bmm1_bs[
    cache_t: KVCacheT,
    p_type: DType,
    output_type: DType,
](
    output_ptr: UnsafePointer[Scalar[output_type]],
    p_ptr: UnsafePointer[Scalar[p_type]],
    v_cache: cache_t,
    q_input_row_offsets: NDBuffer[DType.uint32, 1],
    kv_input_row_offsets: NDBuffer[DType.uint32, 1],
    q_max_seq_len: Int,
    kv_max_seq_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
):
    alias v_type = cache_t.type
    alias kv_num_heads = cache_t.kv_params.num_heads

    # head_size
    var x = global_idx.x
    # query seq_len
    var y = global_idx.y

    var batch_head = block_idx.z
    var batch: UInt
    var head: UInt
    batch, head = divmod(batch_head, UInt(num_heads))

    var cur_query_len: Int
    var cur_kv_len: Int
    var output_offset: Int
    var padded_num_keys = kv_max_seq_len + max_cache_size
    var p_offset = batch_head * q_max_seq_len * padded_num_keys

    q_seq_start = Int(q_input_row_offsets[batch])
    q_seq_end = Int(q_input_row_offsets[batch + 1])
    cur_query_len = q_seq_end - q_seq_start

    output_offset = Int((q_seq_start * num_heads + head) * depth)

    kv_seq_start = Int(kv_input_row_offsets[batch])
    kv_seq_end = Int(kv_input_row_offsets[batch + 1])
    cur_kv_len = kv_seq_end - kv_seq_start

    debug_assert(cur_query_len <= q_max_seq_len, "Invalid cur_query_len")
    debug_assert(cur_kv_len <= kv_max_seq_len, "Invalid cur_kv_len")

    if x >= depth or y >= cur_query_len:
        return

    var p = p_ptr + p_offset

    var kv_head = Int(head // group)
    var output = output_ptr + Int(output_offset)

    var accum = SIMD[DType.float32, 1](0.0)

    for i in range(cur_kv_len + v_cache.cache_length(batch)):
        var v_ptr = v_cache.block_paged_ptr[tile_size=1](batch, i, kv_head, x)
        accum += (p[y * padded_num_keys + i].cast[v_type]() * v_ptr[0]).cast[
            DType.float32
        ]()

    output[y * num_heads * depth + x] = accum.cast[output_type]()


# ===-----------------------------------------------------------------------===#
# Naive GPU multihead cross attention supporting flexible dimensions and
# batch_size > 1.
# ===-----------------------------------------------------------------------===#


fn mha_cross_gpu_naive[
    cache_t: KVCacheT,
    mask_t: MHAMask,
    type: DType,
    q_shape: DimList, //,
    rank: Int,
](
    output: NDBuffer[_, rank, *_],
    q: NDBuffer[type, rank, q_shape, *_],
    q_input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    q_max_seq_len: Int,
    k: cache_t,
    v: cache_t,
    kv_input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    mask_functor: mask_t,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    """Naive cross attention on GPU.

    Note that this assumes ragged tensor inputs and uses a mask functor.

    Computes:
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
    constrained[rank == 3, "only support rank 3 inputs for ragged inputs."]()
    constrained[
        q.type == cache_t.type == cache_t.type == output.type,
        "Q, K, V, output should have same type.",
    ]()
    constrained[
        q.type == DType.float32 or q.type.is_half_float(),
        "Only support single and half precision.",
    ]()

    alias config = MHAConfig(
        type, q_shape.get[rank - 2](), q_shape.get[rank - 1]()
    )

    alias num_heads = Int(config.num_heads)
    alias depth = Int(config.depth)
    alias kv_num_heads = cache_t.kv_params.num_heads
    alias group = config.num_heads // kv_num_heads
    var kv_max_seq_len = Int(k.max_prompt_length())
    var batch_size = q_input_row_offsets.dim[0]() - 1
    var max_cache_size = Int(k.max_context_length())

    alias q_type = q.type
    alias k_type = cache_t.type
    alias v_type = cache_t.type

    # Assume self attention if the query sequence length isn't passed.
    var num_keys = kv_max_seq_len + max_cache_size
    alias p_type = get_accum_type[q_type]()
    var p_device = ctx.enqueue_create_buffer[p_type](
        batch_size * num_heads * q_max_seq_len * num_keys
    )

    # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
    var p_ptr = p_device.unsafe_ptr()
    var p_buffer = NDBuffer[p_type, 3](
        p_ptr, Index(batch_size * num_heads, q_max_seq_len, num_keys)
    )

    ctx.enqueue_function[_bmm0_bs[__type_of(k), mask_t, q_type, p_type]](
        p_ptr,
        q.data,
        k,
        q_input_row_offsets,
        kv_input_row_offsets,
        scale,
        batch_size,
        q_max_seq_len,
        kv_max_seq_len,
        max_cache_size,
        num_heads,
        depth,
        group,
        mask_functor,
        grid_dim=(
            ceildiv(num_keys, 32),
            ceildiv(q_max_seq_len, 16),
            num_heads * batch_size,
        ),
        block_dim=(32, 16, 1),
    )

    @parameter
    @__copy_capture(p_buffer)
    fn input_fn_device[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[p_type, _simd_width]:
        return p_buffer.load[width=_simd_width](rebind[IndexList[3]](coords))

    _softmax_gpu[p_type, 1, 3, DimList.create_unknown[3](), input_fn_device](
        Index(batch_size * num_heads, q_max_seq_len, num_keys), p_buffer, 2, ctx
    )

    ctx.enqueue_function[_bmm1_bs[__type_of(v), p_type, output.type]](
        output.data,
        p_ptr,
        v,
        q_input_row_offsets,
        kv_input_row_offsets,
        q_max_seq_len,
        kv_max_seq_len,
        max_cache_size,
        num_heads,
        depth,
        group,
        grid_dim=(
            ceildiv(depth, 32),
            ceildiv(q_max_seq_len, 16),
            num_heads * batch_size,
        ),
        block_dim=(32, 16, 1),
    )

    _ = p_device

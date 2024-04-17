# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from sys import has_neon
from buffer.list import DimList
from random import rand
from buffer.buffer import NDBuffer, _compute_nd_index
from math import ceildiv, isclose
from nn.softmax import softmax
from utils.index import Index
from testing import assert_true
from nn.mha import _naive_attention
from algorithm import elementwise
from nn.flash_attention import flash_attention
from benchmark import run


# TODO specialized to block size, pass as parameter and include as known
# dimension in k/v_block_list
# TODO pass in position token to handle unrounded inputs
fn _naive_block_list_attention[
    type: DType, transpose_k: Bool
](
    output: NDBuffer[type, 4],
    q: NDBuffer[type, 4],
    k_block_list: List[List[NDBuffer[type, 3]]],
    v_block_list: List[List[NDBuffer[type, 3]]],
    mask: NDBuffer[type, 2],
    kv_seq_len: Int,
    scale: Float32,
):
    """This kernel provides a simple block-list attention implementation for
    transformer models. It's intended as a POC and shouldn't be used in prod.
    """

    alias simd_size = simdwidthof[type]()

    var batch_size = q.dim[0]()
    var num_heads = q.dim[1]()
    var seq_len = q.dim[2]()
    var block_size = k_block_list[0][0].dim[1]()
    var head_dim = q.dim[3]()

    # Allocate intermediate memory buffer.
    var score_size = batch_size * num_heads * seq_len * kv_seq_len
    var score_ptr = DTypePointer[type].alloc(score_size)
    var score = NDBuffer[type, 4](
        score_ptr, Index(batch_size, num_heads, seq_len, kv_seq_len)
    )

    batched_matmul[4, type, type, type, True, transpose_k](
        score, q, k_block_list
    )

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

    batched_matmul[4, type, type, type, False, False](
        output, score, v_block_list
    )
    score_ptr.free()


# TODO this is pretty specialized, let's give it a different name
@always_inline
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    trans_b: Bool,
    trans_blocks: Bool,
](
    c_buf: NDBuffer[c_type, rank],
    a_buf: NDBuffer[a_type, rank],
    b_buf: List[
        List[NDBuffer[b_type, rank - 1]]
    ],  # TODO different params for rank and block_rank
):
    var outer_batch_size = c_buf.dim[0]()
    var inner_batch_size = c_buf.dim[1]()
    var M = c_buf.dim[2]()
    var N = c_buf.dim[3]()
    var K = a_buf.dim[3]()
    alias block_size_dim = 2 if trans_blocks else 1
    var block_size = b_buf[0][0].dim[block_size_dim]()

    fn _b_input(obs: Int, ibs: Int, n: Int, k: Int) -> Scalar[b_type]:
        var block_idx: Int
        var idx_in_block: Int
        var i: Int

        @parameter
        if trans_b:
            block_idx = n // block_size
            idx_in_block = n % block_size
            i = k
        else:
            block_idx = k // block_size
            idx_in_block = k % block_size
            i = n

        @parameter
        if trans_blocks:
            return b_buf[obs][block_idx][ibs, i, idx_in_block]
        else:
            return b_buf[obs][block_idx][ibs, idx_in_block, i]

    # TODO this is hella slow. parallelize, vectorize, and tile
    for obs in range(outer_batch_size):
        for ibs in range(inner_batch_size):
            for m in range(M):
                for n in range(N):
                    var accum = Scalar[c_type](0.0)
                    for k in range(K):
                        var a_val = a_buf[obs, ibs, m, k].cast[c_type]()
                        var b_val = _b_input(obs, ibs, n, k).cast[c_type]()
                        accum += a_val * b_val
                    c_buf[(obs, ibs, m, n)] = accum


@always_inline
fn is_ndbuffer_close[
    rank: Int, type: DType
](
    a: NDBuffer[type, rank],
    b: NDBuffer[type, rank],
    abs_tol: Scalar[type] = 1e-5,
    rel_tol: Scalar[type] = 1e-4,
    print_wrong_value: Bool = True,
    max_num_print: Int = 1,
) -> Bool:
    """Compare if two NDBuffers are close within input tolerance.

    It prints out up to `max_num_print` difference values if `print_wrong_value`
    is set to True.

    Returns:
        Returns True if they are within tolerance.
    """
    debug_assert(
        a.dynamic_shape == b.dynamic_shape
        and a.dynamic_stride == b.dynamic_stride,
        "Input buffers must have the same shape and stride.",
    )

    var num_errs = 0
    var is_close = True

    for i in range(a.num_elements()):
        var nd_idx = _compute_nd_index(a, i)
        var expect = a.load[width=1](nd_idx)
        var actual = b.load[width=1](nd_idx)
        if not isclose(expect, actual, atol=abs_tol, rtol=rel_tol):
            is_close = False
            if print_wrong_value and num_errs < max_num_print:
                print("At ", nd_idx, "expect", expect, "but get", actual)
                num_errs += 1
            else:
                return False

    return is_close


def test_mha_block_list[type: DType, seq_len: Int, is_k_transposed: Bool]():
    print("seq_len:", seq_len, "k_trans?:", is_k_transposed)
    # Query, key, value dimensions.
    alias batch_size = 1
    alias num_heads = 12
    alias depth = 64
    alias mask_val = Float32(-1e10)
    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    # this needs to align with tile sizes for target arch, otherwise we will
    # try to load beyond the block size for a given tile.
    # TODO figure out a way to retrieve max tile size from single source
    alias block_size = 32 if has_neon() else 64

    # Q, K, V shapes.
    alias BHSD = DimList(batch_size, num_heads, seq_len, depth)

    alias qkv_size = batch_size * num_heads * seq_len * depth
    alias kv_block_size = num_heads * block_size * depth

    alias v_block_shape = DimList(num_heads, block_size, depth)
    alias k_block_shape = DimList(
        num_heads, depth, block_size
    ) if is_k_transposed else v_block_shape

    # Allocate memory for all variables.
    var q_ptr = DTypePointer[type].alloc(qkv_size)
    var k_ptr = DTypePointer[type].alloc(qkv_size)
    var v_ptr = DTypePointer[type].alloc(qkv_size)
    var mask_ptr = DTypePointer[type].alloc(seq_len * seq_len)
    var output_ptr = DTypePointer[type].alloc(qkv_size)
    var block_list_output_ptr = DTypePointer[type].alloc(qkv_size)

    # Q, K, V are randomly initialized.
    rand(q_ptr, qkv_size)
    rand(k_ptr, qkv_size)
    rand(v_ptr, qkv_size)
    var q = NDBuffer[type, 4, BHSD](q_ptr)
    var k = NDBuffer[type, 4, BHSD](k_ptr)
    var v = NDBuffer[type, 4, BHSD](v_ptr)

    # intialize block list cache  with values from the contiguous tensor
    var num_full_blocks = seq_len // block_size
    var tokens_in_last_block = seq_len % block_size
    var k_block_list = List[List[NDBuffer[type, 3]]]()
    var v_block_list = List[List[NDBuffer[type, 3]]]()
    for bs in range(batch_size):
        k_block_list.append(List[NDBuffer[type, 3]]())
        v_block_list.append(List[NDBuffer[type, 3]]())
        for block_idx in range(ceildiv(seq_len, block_size)):
            var k_block_ptr = DTypePointer[type].alloc(kv_block_size)
            var v_block_ptr = DTypePointer[type].alloc(kv_block_size)

            if block_idx < num_full_blocks:
                for h in range(num_heads):
                    memcpy(
                        v_block_ptr + h * block_size * depth,
                        v._offset((bs, h, block_idx * block_size, 0)),
                        depth * block_size,
                    )

                    @parameter
                    if is_k_transposed:
                        for d in range(depth):
                            for b in range(block_size):
                                k_block_ptr[
                                    h * depth * block_size + d * block_size + b
                                ] = k[
                                    Index(bs, h, block_idx * block_size + b, d)
                                ]
                    else:
                        memcpy(
                            k_block_ptr + h * block_size * depth,
                            k._offset((bs, h, block_idx * block_size, 0)),
                            depth * block_size,
                        )
            else:
                for s in range(tokens_in_last_block):
                    for h in range(num_heads):
                        memcpy(
                            v_block_ptr + h * block_size * depth + s * depth,
                            v._offset((bs, h, block_idx * block_size + s, 0)),
                            depth,
                        )

                        @parameter
                        if is_k_transposed:
                            for d in range(depth):
                                k_block_ptr[
                                    h * depth * block_size + d * block_size + s
                                ] = k[
                                    Index(bs, h, block_idx * block_size + s, d)
                                ]
                        else:
                            memcpy(
                                k_block_ptr
                                + h * block_size * depth
                                + s * depth,
                                k._offset(
                                    (bs, h, block_idx * block_size + s, 0)
                                ),
                                depth,
                            )

            k_block_list[-1].append(
                NDBuffer[type, 3](
                    k_block_ptr, k_block_shape
                ).make_dims_unknown()
            )
            v_block_list[-1].append(
                NDBuffer[type, 3](
                    v_block_ptr, v_block_shape
                ).make_dims_unknown()
            )

    # Set triangular mask
    for b in range(seq_len):
        for i in range(b + 1):
            mask_ptr[b * seq_len + i] = 0
        for i in range(b + 1, seq_len):
            mask_ptr[b * seq_len + i] = mask_val.cast[type]()

    # Contruct buffers.

    var mask = NDBuffer[type, 2](mask_ptr, Index(seq_len, seq_len))
    var mask_4d = NDBuffer[type, 4](
        mask_ptr,
        Index(batch_size, num_heads, seq_len, seq_len),
        Index(0, 0, seq_len, 1),
    )
    var output = NDBuffer[type, 4, BHSD](output_ptr)
    var mha_output = NDBuffer[type, 4, BHSD](block_list_output_ptr)

    @parameter
    fn do_naive_attn():
        _naive_attention[type, True](
            output.make_dims_unknown(),
            q.make_dims_unknown(),
            k.make_dims_unknown(),
            v.make_dims_unknown(),
            mask,
            scale,
        )

    var naive_results = run[do_naive_attn](
        num_warmup=1,
        max_iters=10,
    )
    var naive_mean = naive_results.mean()
    print("Naive Attn:", naive_mean, "s")

    @parameter
    fn do_naive_block_list_attn():
        _naive_block_list_attention[type, is_k_transposed](
            mha_output.make_dims_unknown(),
            q.make_dims_unknown(),
            k_block_list,
            v_block_list,
            mask,
            seq_len,
            scale,
        )

    var naive_bl_results = run[do_naive_block_list_attn](
        num_warmup=1,
        max_iters=10,
    )
    var naive_bl_mean = naive_bl_results.mean()
    print("Naive BlockList Attn:", naive_bl_mean, "s")
    print("Naive/BlockList:", naive_mean / naive_bl_mean, "X")

    assert_true(
        is_ndbuffer_close(
            output.make_dims_unknown(),
            mha_output.make_dims_unknown(),
        )
    )

    @parameter
    if is_k_transposed:
        var fa_output_ptr = DTypePointer[type].alloc(qkv_size)
        var fa_output = NDBuffer[type, 4, BHSD](fa_output_ptr)

        @parameter
        fn input_k_fn[
            width: Int, rank: Int
        ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
            constrained[
                width <= block_size, "Expected width to be less than block size"
            ]()
            # idx == bs, num_heads, head_dim, kv_len
            var bs = idx[0]
            var head_idx = idx[1]
            var head_d_idx = idx[2]
            var seq = idx[3]
            var block_idx = seq // block_size
            var idx_in_block = seq % block_size

            return k_block_list[bs][block_idx].load[width=width](
                (head_idx, head_d_idx, idx_in_block)
            )

        @parameter
        fn input_v_fn[
            width: Int, rank: Int
        ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
            # idx == bs, num_heads, kv_len, head_dim
            constrained[
                width <= block_size, "Expected width to be less than block size"
            ]()
            var bs = idx[0]
            var head_idx = idx[1]
            var head_d_idx = idx[3]
            var seq = idx[2]
            var block_idx = seq // block_size
            var idx_in_block = seq % block_size
            return v_block_list[bs][block_idx].load[width=width](
                head_idx, idx_in_block, head_d_idx
            )

        @parameter
        fn input_mask_fn[
            width: Int, rank: Int
        ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
            return mask_4d.load[width=width](rebind[StaticIntTuple[4]](idx))

        var fa_k_shape = StaticIntTuple[4](
            batch_size, num_heads, depth, seq_len
        )
        var fa_v_shape = StaticIntTuple[4](
            batch_size, num_heads, seq_len, depth
        )

        @parameter
        fn do_flash_attn():
            flash_attention[type, 4, input_k_fn, input_v_fn, input_mask_fn,](
                q.make_dims_unknown(),
                fa_k_shape,
                fa_v_shape,
                fa_output.make_dims_unknown(),
                scale,
            )

        var flash_attn_results = run[do_flash_attn](
            num_warmup=1,
            max_iters=10,
        )
        var flash_attn_mean = flash_attn_results.mean()
        print("Flash Attn:", flash_attn_mean, "s")
        print("Naive/FlashAttn:", naive_mean / flash_attn_mean, "X")
        assert_true(
            is_ndbuffer_close(
                output.make_dims_unknown(),
                fa_output.make_dims_unknown(),
            )
        )
        fa_output_ptr.free()

    print()

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    mask_ptr.free()
    output_ptr.free()
    block_list_output_ptr.free()
    for i in range(len(v_block_list)):
        for j in range(len(v_block_list[i])):
            v_block_list[i][j].data.free()
            k_block_list[i][j].data.free()


def main():
    test_mha_block_list[DType.float32, 128, False]()
    test_mha_block_list[DType.float32, 2, False]()
    test_mha_block_list[DType.float32, 135, False]()
    test_mha_block_list[DType.float32, 128, True]()
    test_mha_block_list[DType.float32, 2, True]()
    test_mha_block_list[DType.float32, 135, True]()

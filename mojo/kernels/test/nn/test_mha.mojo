# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from math import div_ceil, min, abs, rsqrt, isclose
from memory.buffer import NDBuffer, _compute_nd_index
from memory.unsafe import DTypePointer
from random import rand
from runtime.llcl import Runtime
from utils.index import Index
from utils.list import DimList
from testing import assert_true

from BatchedMatmul import batched_matmul
from NN.Softmax import softmax
from NN.MultiHeadAttention import fused_attention, _naive_attention


# TODO: Move this function to a common space.
@always_inline
fn is_ndbuffer_close[
    rank: Int, type: DType
](
    a: NDBuffer[type, rank],
    b: NDBuffer[type, rank],
    abs_tol: SIMD[type, 1] = 1e-5,
    rel_tol: SIMD[type, 1] = 1e-4,
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
        let nd_idx = _compute_nd_index(a, i)
        let expect = a.simd_load[1](nd_idx)
        let actual = b.simd_load[1](nd_idx)
        if not isclose(expect, actual, abs_tol, rel_tol):
            is_close = False
            if print_wrong_value and num_errs < max_num_print:
                print("At ", nd_idx, "expect", expect, "but get", actual)
                num_errs += 1
            else:
                return False

    return is_close


alias type = DType.float32


def test_mha():
    # Query, key, value dimensions.
    alias batch_size = 1
    alias num_heads = 12
    alias seq_len = 128
    alias depth = 64
    alias mask_val = Float32(-1e10)
    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))

    # Q, K, V shapes.
    alias BHSD = DimList(batch_size, num_heads, seq_len, depth)

    alias qkv_size = batch_size * num_heads * seq_len * depth

    # Allocate memory for all variables.
    let q_ptr = DTypePointer[type].alloc(qkv_size)
    let k_ptr = DTypePointer[type].alloc(qkv_size)
    let v_ptr = DTypePointer[type].alloc(qkv_size)
    let mask_ptr = DTypePointer[type].alloc(seq_len * seq_len)
    let output_ptr = DTypePointer[type].alloc(qkv_size)
    let mha_output_ptr = DTypePointer[type].alloc(qkv_size)

    # Q, K, V are randomly initialized.
    rand(q_ptr, qkv_size)
    rand(k_ptr, qkv_size)
    rand(v_ptr, qkv_size)

    # Mask is set for half of the sequence.
    for b in range(seq_len):
        for i in range(seq_len // 2):
            mask_ptr[b * seq_len + i] = 0
        for i in range(seq_len // 2, seq_len):
            mask_ptr[b * seq_len + i] = mask_val

    # Contruct buffers.
    let q = NDBuffer[type, 4, BHSD](q_ptr)
    let v = NDBuffer[type, 4, BHSD](v_ptr)
    let mask = NDBuffer[type, 2](mask_ptr, Index(seq_len, seq_len))
    let output = NDBuffer[type, 4, BHSD](output_ptr)
    let mha_output = NDBuffer[type, 4, BHSD](mha_output_ptr)

    @parameter
    @always_inline
    fn test_body[transpose_k: Bool]() raises:
        let k_shape = Index(
            batch_size, num_heads, seq_len, depth
        ) if transpose_k else Index(batch_size, num_heads, depth, seq_len)
        let k = NDBuffer[type, 4](k_ptr, k_shape)

        _naive_attention[type, transpose_k](
            output.make_dims_unknown(),
            q.make_dims_unknown(),
            k.make_dims_unknown(),
            v.make_dims_unknown(),
            mask,
            scale,
        )

        fused_attention[
            4,
            2,
            BHSD,
            DimList.create_unknown[4](),
            BHSD,
            DimList.create_unknown[2](),
            BHSD,
            type,
            type,
            type,
            type,
            type,
            transpose_k=transpose_k,
            add_attn_mask=True,
        ](mha_output, q, k, v, mask, scale, Float32())

        assert_true(
            is_ndbuffer_close(
                output.make_dims_unknown(), mha_output.make_dims_unknown()
            )
        )

    test_body[transpose_k=False]()
    test_body[transpose_k=True]()

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    mask_ptr.free()
    output_ptr.free()
    mha_output_ptr.free()


def main():
    test_mha()

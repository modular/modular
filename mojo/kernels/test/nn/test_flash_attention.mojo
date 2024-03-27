# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from buffer import NDBuffer
from math import exp, isclose, max
from nn.flash_attention import flash_attention
from random import rand, seed
from testing import assert_equal
from utils.index import Index, product


def reference_attention[
    type: DType, rank: Int
](
    q_nd: NDBuffer[type, rank],
    k_nd: NDBuffer[type, rank],
    v_nd: NDBuffer[type, rank],
    mask_nd: NDBuffer[type, rank],
    output_nd: NDBuffer[type, rank],
    scale: Float32,
):
    fn flatten_to_3d(buf: NDBuffer[type, rank]) -> NDBuffer[type, 3]:
        var shape = buf.get_shape()
        var shape_3d = Index(
            product(shape, 0, rank - 2), shape[rank - 2], shape[rank - 1]
        )
        return NDBuffer[type, 3](buf.data, shape_3d)

    var q_3d = flatten_to_3d(q_nd)
    var k_3d = flatten_to_3d(k_nd)
    var v_3d = flatten_to_3d(v_nd)
    var mask_3d = flatten_to_3d(mask_nd)
    var output_3d = flatten_to_3d(output_nd)

    var batch_count = q_3d.dim(0)
    var seq_len = q_3d.dim(1)
    var depth_dim = q_3d.dim(2)
    var kv_seq_len = k_3d.dim(2)

    assert_equal(batch_count, k_3d.dim(0))
    assert_equal(depth_dim, k_3d.dim(1))

    assert_equal(batch_count, v_3d.dim(0))
    assert_equal(kv_seq_len, v_3d.dim(1))
    assert_equal(depth_dim, v_3d.dim(2))

    assert_equal(batch_count, mask_3d.dim(0))
    assert_equal(seq_len, mask_3d.dim(1))
    assert_equal(kv_seq_len, mask_3d.dim(2))

    assert_equal(q_3d.get_shape(), output_3d.get_shape())

    var score_ptr = DTypePointer[type].alloc(seq_len * kv_seq_len)
    var score_2d = NDBuffer[type, 2](score_ptr, Index(seq_len, kv_seq_len))

    for b in range(batch_count):
        # Compute: `score = Q @ K`
        for m in range(seq_len):
            for n in range(kv_seq_len):
                var accum = Scalar[type](0)
                for k in range(depth_dim):
                    accum = q_3d[Index(b, m, k)].fma(
                        k_3d[Index(b, k, n)], accum
                    )
                score_2d[Index(m, n)] = accum

        # Apply scaling and masking to the score buffer
        for m in range(seq_len):
            for n in range(kv_seq_len):
                score_2d[Index(m, n)] = (
                    score_2d[Index(m, n)] * scale.cast[type]()
                    + mask_3d[Index(b, m, n)]
                )

        # Compute: `score = softmax(score)`
        for m in range(seq_len):
            var max_val = Scalar[type].MIN
            for n in range(kv_seq_len):
                max_val = max(max_val, score_2d[Index(m, n)])

            var sum_val = Scalar[type](0)
            for n in range(kv_seq_len):
                var exp_val = exp(score_2d[Index(m, n)] - max_val)
                score_2d[Index(m, n)] = exp_val
                sum_val += exp_val

            for n in range(kv_seq_len):
                score_2d[Index(m, n)] = score_2d[Index(m, n)] / sum_val

        # Compute: `output = score @ V`
        for m in range(seq_len):
            for n in range(depth_dim):
                var accum = Scalar[type](0)
                for k in range(kv_seq_len):
                    accum = score_2d[Index(m, k)].fma(
                        v_3d[Index(b, k, n)], accum
                    )
                output_3d[Index(b, m, n)] = accum

    score_ptr.free()


def test_case[
    type: DType, batch_rank: Int
](
    batch_dims: StaticIntTuple[batch_rank],
    seq_len: Int,
    kv_seq_len: Int,
    depth_dim: Int,
    scale: Float32,
):
    alias rank = batch_rank + 2

    fn build_shape(
        batch_dims: StaticIntTuple[batch_rank], x: Int, y: Int
    ) -> StaticIntTuple[rank]:
        var shape = StaticIntTuple[rank]()

        @unroll
        for i in range(batch_rank):
            shape[i] = batch_dims[i]

        shape[rank - 2] = x
        shape[rank - 1] = y

        return shape

    var q_shape = build_shape(batch_dims, seq_len, depth_dim)
    var k_shape = build_shape(batch_dims, depth_dim, kv_seq_len)
    var v_shape = build_shape(batch_dims, kv_seq_len, depth_dim)
    var mask_shape = build_shape(batch_dims, seq_len, kv_seq_len)
    var output_shape = build_shape(batch_dims, seq_len, depth_dim)

    var q_ptr = DTypePointer[type].alloc(q_shape.flattened_length())
    var k_ptr = DTypePointer[type].alloc(k_shape.flattened_length())
    var v_ptr = DTypePointer[type].alloc(v_shape.flattened_length())
    var mask_ptr = DTypePointer[type].alloc(mask_shape.flattened_length())
    var output_ptr = DTypePointer[type].alloc(output_shape.flattened_length())
    var ref_output_ptr = DTypePointer[type].alloc(
        output_shape.flattened_length()
    )

    seed(0)
    rand(q_ptr, q_shape.flattened_length())
    rand(k_ptr, k_shape.flattened_length())
    rand(v_ptr, v_shape.flattened_length())
    rand(mask_ptr, mask_shape.flattened_length())

    var q = NDBuffer[type, rank](q_ptr, q_shape)
    var k = NDBuffer[type, rank](k_ptr, k_shape)
    var v = NDBuffer[type, rank](v_ptr, v_shape)
    var mask = NDBuffer[type, rank](mask_ptr, mask_shape)
    var output = NDBuffer[type, rank](output_ptr, output_shape)
    var ref_output = NDBuffer[type, rank](ref_output_ptr, output_shape)

    reference_attention[type, rank](
        q.make_dims_unknown(),
        k.make_dims_unknown(),
        v.make_dims_unknown(),
        mask.make_dims_unknown(),
        ref_output.make_dims_unknown(),
        scale,
    )

    @parameter
    @always_inline
    fn input_k_fn[
        simd_width: Int, _rank: Int
    ](idx: StaticIntTuple[_rank]) -> SIMD[type, simd_width]:
        return k.load[width=simd_width](rebind[StaticIntTuple[rank]](idx))

    @parameter
    @always_inline
    fn input_v_fn[
        simd_width: Int, _rank: Int
    ](idx: StaticIntTuple[_rank]) -> SIMD[type, simd_width]:
        return v.load[width=simd_width](rebind[StaticIntTuple[rank]](idx))

    @parameter
    @always_inline
    fn mask_fn[
        simd_width: Int, _rank: Int
    ](idx: StaticIntTuple[_rank]) -> SIMD[type, simd_width]:
        return mask.load[width=simd_width](rebind[StaticIntTuple[rank]](idx))

    flash_attention[type, rank, input_k_fn, input_v_fn, mask_fn](
        q.make_dims_unknown(),
        k.get_shape(),
        v.get_shape(),
        output.make_dims_unknown(),
        scale,
    )

    var mismatches = 0
    for i in range(output.num_elements()):
        if not isclose(
            output.data[i], ref_output.data[i], atol=1e-5, rtol=1e-4
        ):
            if mismatches == 0:
                print(
                    "Found mismatches for",
                    batch_dims,
                    seq_len,
                    kv_seq_len,
                    depth_dim,
                )

            print(
                "Mismatch at",
                output.get_nd_index(i),
                output.data[i],
                ref_output.data[i],
            )

            mismatches = mismatches + 1
            if mismatches > 5:
                break

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    mask_ptr.free()
    output_ptr.free()
    ref_output_ptr.free()


def test_flash_attention[type: DType]():
    test_case[type](
        batch_dims=Index(1, 8),
        seq_len=1,
        kv_seq_len=503,
        depth_dim=128,
        scale=0.125,
    )
    test_case[type](
        batch_dims=Index(2, 3),
        seq_len=128,
        kv_seq_len=128,
        depth_dim=63,
        scale=0.25,
    )
    test_case[type](
        batch_dims=Index(8),
        seq_len=64,
        kv_seq_len=64,
        depth_dim=384,
        scale=0.25,
    )


def main():
    test_flash_attention[DType.float32]()

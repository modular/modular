# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from buffer import NDBuffer
from buffer.list import Dim, DimList
from math import exp, isclose
from nn.flash_attention import flash_attention, flash_attention_split_kv
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


@value
@register_passable("trivial")
struct TestCaseConfig[batch_rank: Int]:
    """Test case workload configuration hyperparameters."""

    var batch_dims: StaticIntTuple[batch_rank]
    var seq_len: Int
    var kv_seq_len: Int
    """Total KV sequence length including previous and current forwards."""
    var depth_dim: Int
    var scale: Float32

    @always_inline
    def prev_seq_len(self) -> Int:
        """Returns the KV cache length from previous iterations."""
        return self.kv_seq_len - self.seq_len


def verify_output[
    type: DType, batch_rank: Int, rank: Int, output_static_shape: DimList
](
    output: NDBuffer[type, rank, output_static_shape],
    ref_output: NDBuffer[type, rank],
    cfg: TestCaseConfig[batch_rank],
) -> None:
    """Compares `output` and `ref_output` elementwise, printing up to 5 mismatches.
    """
    var mismatches = 0
    for i in range(output.num_elements()):
        if not isclose(
            output.data[i], ref_output.data[i], atol=1e-5, rtol=1e-4
        ):
            if mismatches == 0:
                print(
                    "Found mismatches for",
                    cfg.batch_dims,
                    cfg.seq_len,
                    cfg.kv_seq_len,
                    cfg.depth_dim,
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


def build_ndbuffer[
    type: DType,
    rank: Int,
    *,
    static_shape: DimList = DimList.create_unknown[rank](),
](shape: StaticIntTuple[rank]) -> NDBuffer[type, rank, static_shape]:
    var ptr = DTypePointer[type].alloc(shape.flattened_length())
    rand(ptr, shape.flattened_length())
    return NDBuffer[type, rank, static_shape](ptr, shape)


def test_case[
    type: DType,
    batch_rank: Int,
    output_static_shape: DimList = DimList.create_unknown[batch_rank + 2](),
](cfg: TestCaseConfig[batch_rank]):
    alias rank = batch_rank + 2

    @always_inline
    @parameter
    fn build_shape(x: Int, y: Int) -> StaticIntTuple[rank]:
        var shape = StaticIntTuple[rank]()

        @unroll
        for i in range(batch_rank):
            shape[i] = cfg.batch_dims[i]

        shape[rank - 2] = x
        shape[rank - 1] = y

        return shape

    seed(42)

    # Allocate the QKV tensors from the current sequence.
    var q = build_ndbuffer[type](build_shape(cfg.seq_len, cfg.depth_dim))
    var k = build_ndbuffer[type](build_shape(cfg.depth_dim, cfg.kv_seq_len))
    var v = build_ndbuffer[type](build_shape(cfg.kv_seq_len, cfg.depth_dim))

    # Allocate the attention mask.
    var mask = build_ndbuffer[type](build_shape(cfg.seq_len, cfg.kv_seq_len))

    # Allocate output and reference output buffers.
    var output = build_ndbuffer[type, static_shape=output_static_shape](
        build_shape(cfg.seq_len, cfg.depth_dim)
    )
    var ref_output = build_ndbuffer[type](
        build_shape(cfg.seq_len, cfg.depth_dim)
    )

    reference_attention(q, k, v, mask, ref_output, cfg.scale)

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

    flash_attention[
        type, rank, input_k_fn, input_v_fn, mask_fn, output_static_shape
    ](q, k.get_shape(), v.get_shape(), output, cfg.scale)

    verify_output(output, ref_output, cfg)

    q.data.free()
    k.data.free()
    v.data.free()
    mask.data.free()
    output.data.free()
    ref_output.data.free()


def test_flash_attention[type: DType]():
    test_case[type](
        TestCaseConfig(
            batch_dims=Index(1, 8),
            seq_len=1,
            kv_seq_len=503,
            depth_dim=128,
            scale=0.125,
        )
    )
    test_case[type](
        TestCaseConfig(
            batch_dims=Index(2, 3),
            seq_len=128,
            kv_seq_len=128,
            depth_dim=63,
            scale=0.25,
        )
    )
    test_case[type](
        TestCaseConfig(
            batch_dims=Index(8),
            seq_len=64,
            kv_seq_len=64,
            depth_dim=384,
            scale=0.25,
        )
    )
    test_case[type, 1, DimList(Dim(), Dim(), 160)](
        TestCaseConfig(
            batch_dims=Index(1),
            seq_len=100,
            kv_seq_len=300,
            depth_dim=160,
            scale=0.1,
        )
    )
    test_case[type, 1, DimList(Dim(), Dim(), 300)](
        TestCaseConfig(
            batch_dims=Index(1),
            seq_len=100,
            kv_seq_len=64,
            depth_dim=300,
            scale=0.1,
        )
    )


def test_case_split_kv[
    type: DType,
    batch_rank: Int,
    output_static_shape: DimList = DimList.create_unknown[batch_rank + 2](),
](cfg: TestCaseConfig[batch_rank]):
    # For now only allow Q.shape = [B, H, S, D].
    constrained[batch_rank == 2]()

    # Rank of Q.
    alias rank = batch_rank + 2
    # Rank of the KV cache.
    alias kv_rank = rank + 1

    # Reshaped k_cache, v_cache to simulate split KV setup.
    @always_inline
    @parameter
    fn build_shape[
        shape_rank: Int = rank
    ](x: Int, y: Int) -> StaticIntTuple[shape_rank]:
        var shape = StaticIntTuple[shape_rank]()

        @parameter
        if shape_rank == kv_rank:
            # Unsqueeze the output shape with a 1-dim.
            shape[0] = 1

            @unroll
            for i in range(batch_rank):
                shape[i + 1] = cfg.batch_dims[i]
        else:
            # Copy the batch dims without unsqueezing.
            @unroll
            for i in range(batch_rank):
                shape[i] = cfg.batch_dims[i]

        # In either case set the last two dimensions to [x, y].
        shape[shape_rank - 2] = x
        shape[shape_rank - 1] = y

        return shape

    seed(42)

    # Allocate the KV cache for the previous sequence.
    var k_cache = build_ndbuffer[type](
        build_shape[kv_rank](cfg.depth_dim, cfg.prev_seq_len())
    )
    var v_cache = build_ndbuffer[type](
        build_shape[kv_rank](cfg.prev_seq_len(), cfg.depth_dim)
    )

    # Allocate the QKV tensors from the current sequence.
    var q = build_ndbuffer[type](build_shape(cfg.seq_len, cfg.depth_dim))
    var k = build_ndbuffer[type](build_shape(cfg.depth_dim, cfg.seq_len))
    var v = build_ndbuffer[type](build_shape(cfg.seq_len, cfg.depth_dim))

    # Allocate the attention mask.
    var mask = build_ndbuffer[type](build_shape(cfg.seq_len, cfg.kv_seq_len))

    # Allocate output and reference output buffers.
    var output = build_ndbuffer[type, static_shape=output_static_shape](
        build_shape(cfg.seq_len, cfg.depth_dim)
    )
    var ref_output = build_ndbuffer[type](
        build_shape(cfg.seq_len, cfg.depth_dim)
    )

    # Allocate reference KV cache.
    var ref_k = build_ndbuffer[type](build_shape(cfg.depth_dim, cfg.kv_seq_len))
    var ref_v = build_ndbuffer[type](build_shape(cfg.kv_seq_len, cfg.depth_dim))

    # Copy previous KV cache and current KV tensors into a single buffer for
    # computing reference attention.
    for b in range(cfg.batch_dims[0]):
        for h in range(cfg.batch_dims[1]):
            for s in range(cfg.prev_seq_len()):
                for d in range(cfg.depth_dim):
                    ref_k[StaticIntTuple[rank](b, h, d, s)] = k_cache[
                        StaticIntTuple[kv_rank](0, b, h, d, s)
                    ]
                    ref_v[StaticIntTuple[rank](b, h, s, d)] = v_cache[
                        StaticIntTuple[kv_rank](0, b, h, s, d)
                    ]

            for s in range(cfg.prev_seq_len(), cfg.kv_seq_len):
                for d in range(cfg.depth_dim):
                    ref_k[StaticIntTuple[rank](b, h, d, s)] = k[
                        StaticIntTuple[rank](b, h, d, s - cfg.prev_seq_len())
                    ]
                    ref_v[StaticIntTuple[rank](b, h, s, d)] = v[
                        StaticIntTuple[rank](b, h, s - cfg.prev_seq_len(), d)
                    ]

    # Compute reference outputs for comparison.
    reference_attention(q, ref_k, ref_v, mask, ref_output, cfg.scale)

    # Define lambda to unsqueeze indices with a leading 1-dim.
    # In other words [B, H, S, D] becomes [1, B, H, S, D].
    @always_inline
    fn idx_to_kv_idx(
        idx: StaticIntTuple[rank],
    ) raises -> StaticIntTuple[kv_rank]:
        var kv_idx = StaticIntTuple[kv_rank]()
        kv_idx[0] = 0

        @unroll
        for i in range(rank):
            kv_idx[i + 1] = idx[i]

        return kv_idx

    # Define input lambdas for split KV cache attn `flash_attention_split_kv`.
    @parameter
    @always_inline
    fn input_k_cache_fn[
        simd_width: Int, _rank: Int
    ](idx: StaticIntTuple[_rank]) -> SIMD[type, simd_width]:
        return k_cache.load[width=simd_width](
            rebind[StaticIntTuple[kv_rank]](idx)
        )

    @parameter
    @always_inline
    fn input_v_cache_fn[
        simd_width: Int, _rank: Int
    ](idx: StaticIntTuple[_rank]) -> SIMD[type, simd_width]:
        return v_cache.load[width=simd_width](
            rebind[StaticIntTuple[kv_rank]](idx)
        )

    @parameter
    @always_inline
    fn mask_fn[
        simd_width: Int, _rank: Int
    ](idx: StaticIntTuple[_rank]) -> SIMD[type, simd_width]:
        return mask.load[width=simd_width](rebind[StaticIntTuple[rank]](idx))

    flash_attention_split_kv[
        type,
        rank,
        input_k_cache_fn,
        input_v_cache_fn,
        mask_fn,
        output_static_shape,
    ](
        q,
        k,
        v,
        rebind[StaticIntTuple[rank + 1]](k_cache.get_shape()),
        rebind[StaticIntTuple[rank + 1]](v_cache.get_shape()),
        output,
        cfg.scale,
    )

    verify_output(output, ref_output, cfg)

    k_cache.data.free()
    v_cache.data.free()
    q.data.free()
    k.data.free()
    v.data.free()
    mask.data.free()
    output.data.free()
    ref_output.data.free()
    ref_k.data.free()
    ref_v.data.free()


def test_flash_attention_split_kv[type: DType]():
    for kv_seq_len in range(1, 128):
        test_case_split_kv[type](
            TestCaseConfig(
                batch_dims=Index(1, 1),
                seq_len=1,
                kv_seq_len=kv_seq_len,
                depth_dim=1,
                scale=0.125,
            )
        )
    test_case_split_kv[type](
        TestCaseConfig(
            batch_dims=Index(1, 8),
            seq_len=1,
            kv_seq_len=503,
            depth_dim=128,
            scale=0.125,
        )
    )
    test_case_split_kv[type](
        TestCaseConfig(
            batch_dims=Index(2, 3),
            seq_len=128,
            kv_seq_len=128,
            depth_dim=63,
            scale=0.25,
        )
    )


def main():
    test_flash_attention[DType.float32]()
    test_flash_attention_split_kv[DType.float32]()

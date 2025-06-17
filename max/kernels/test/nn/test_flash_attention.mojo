# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from math import exp, isclose
from random import rand, seed

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from nn.flash_attention import flash_attention, flash_attention_split_kv
from testing import assert_equal

from utils import IndexList
from utils.index import Index


def reference_attention_bshd[
    type: DType, rank: Int
](
    q_nd: NDBuffer[type, rank],
    k_nd: NDBuffer[type, rank],
    v_nd: NDBuffer[type, rank],
    mask_nd: NDBuffer[type, rank],
    output_nd: NDBuffer[mut=True, type, rank],
    scale: Float32,
):
    fn reshape_4d(buf: NDBuffer[type, rank]) -> NDBuffer[type, 4, buf.origin]:
        var shape = buf.get_shape()
        var num_heads = shape[rank - 2] if rank == 4 else 1
        var shape_4d = Index(shape[0], shape[1], num_heads, shape[rank - 1])
        return NDBuffer[type, 4](buf.data, shape_4d)

    fn reshape_mask_4d(
        buf: NDBuffer[type, rank]
    ) -> NDBuffer[type, 4, buf.origin]:
        var shape = buf.get_shape()
        var num_heads = shape[1] if rank == 4 else 1
        var shape_4d = Index(
            shape[0], num_heads, shape[rank - 2], shape[rank - 1]
        )
        return NDBuffer[type, 4](buf.data, shape_4d)

    var q_4d = reshape_4d(q_nd)
    var k_4d = reshape_4d(k_nd)
    var v_4d = reshape_4d(v_nd)
    var mask_4d = reshape_mask_4d(mask_nd)
    var output_4d = reshape_4d(output_nd)

    var batch_count = q_4d.dim(0)
    var seq_len = q_4d.dim(1)
    var num_heads = q_4d.dim(2)
    var depth_dim = q_4d.dim(3)
    var kv_seq_len = v_4d.dim(1)
    var kv_num_heads = v_4d.dim(2)

    assert_equal(num_heads % kv_num_heads, 0)

    var kv_group_count = num_heads // kv_num_heads

    assert_equal(batch_count, k_4d.dim(0))
    assert_equal(kv_seq_len, k_4d.dim(1))
    assert_equal(kv_num_heads, k_4d.dim(2))
    assert_equal(depth_dim, k_4d.dim(3))

    assert_equal(batch_count, v_4d.dim(0))
    assert_equal(depth_dim, v_4d.dim(3))

    assert_equal(batch_count, mask_4d.dim(0))
    assert_equal(num_heads, mask_4d.dim(1))
    assert_equal(seq_len, mask_4d.dim(2))
    assert_equal(kv_seq_len, mask_4d.dim(3))

    assert_equal(q_4d.get_shape(), output_4d.get_shape())

    var score_ptr = UnsafePointer[Scalar[type]].alloc(seq_len * kv_seq_len)
    var score_2d = NDBuffer[type, 2](score_ptr, Index(seq_len, kv_seq_len))

    for b in range(batch_count):
        for h in range(num_heads):
            var kv_h = h // kv_group_count

            # Compute: `score = Q @ K`
            for m in range(seq_len):
                for n in range(kv_seq_len):
                    var accum = Scalar[type](0)
                    for k in range(depth_dim):
                        accum = q_4d[Index(b, m, h, k)].fma(
                            k_4d[Index(b, n, kv_h, k)], accum
                        )
                    score_2d[Index(m, n)] = accum

            # Apply scaling and masking to the score buffer
            for m in range(seq_len):
                for n in range(kv_seq_len):
                    score_2d[Index(m, n)] = (
                        score_2d[Index(m, n)] * scale.cast[type]()
                        + mask_4d[Index(b, h, m, n)]
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
                            v_4d[Index(b, k, kv_h, n)], accum
                        )
                    output_4d[Index(b, m, h, n)] = accum

    score_ptr.free()


@fieldwise_init
@register_passable("trivial")
struct TestCaseConfig[batch_rank: Int](Copyable, Movable):
    """Test case workload configuration hyperparameters."""

    alias rank = batch_rank + 2
    alias kv_cache_rank = Self.rank + 1

    var batch_dims: IndexList[batch_rank]
    var seq_len: Int
    var kv_num_heads: Int
    var kv_seq_len: Int
    """Total KV sequence length including previous and current forwards."""
    var depth_dim: Int
    var scale: Float32

    @always_inline
    fn prev_seq_len(self) -> Int:
        """Returns the KV cache length from previous iterations."""
        return self.kv_seq_len - self.seq_len

    @always_inline
    fn build_shape[
        *, shape_rank: Int = Self.rank, is_kv: Bool = False
    ](self, x: Int, y: Int) -> IndexList[shape_rank]:
        var shape = IndexList[shape_rank]()

        @parameter
        if shape_rank == self.kv_cache_rank:
            # Unsqueeze the output shape with a 1-dim.
            shape[0] = 1

            @parameter
            for i in range(batch_rank):
                shape[i + 1] = self.batch_dims[i]
        else:
            # Copy the batch dims without unsqueezing.
            @parameter
            for i in range(batch_rank):
                shape[i] = self.batch_dims[i]

        # Replace the number of query heads with the number of KV heads.
        @parameter
        if is_kv and batch_rank == 2:
            shape[shape_rank - 3] = self.kv_num_heads

        shape[shape_rank - 2] = x
        shape[shape_rank - 1] = y

        return shape

    @always_inline
    fn build_shape_bshd[
        *, shape_rank: Int = Self.rank, is_kv: Bool = False
    ](self, x: Int, y: Int) -> IndexList[shape_rank]:
        var shape = IndexList[shape_rank]()

        @parameter
        if shape_rank == self.kv_cache_rank:
            # Unsqueeze the output shape with a 1-dim.
            shape[0] = 1
            shape[1] = self.batch_dims[0]

            @parameter
            for i in range(1, batch_rank):
                shape[i + 2] = self.batch_dims[i]
        else:
            shape[0] = self.batch_dims[0]

            # Copy the batch dims without unsqueezing.
            @parameter
            for i in range(1, batch_rank):
                shape[i + 1] = self.batch_dims[i]

        # Replace the number of query heads with the number of KV heads.
        @parameter
        if is_kv and batch_rank == 2:
            shape[shape_rank - 2] = self.kv_num_heads

        shape[1] = x
        shape[shape_rank - 1] = y

        return shape


def verify_output[
    type: DType, batch_rank: Int, rank: Int
](
    output: NDBuffer[type, rank],
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
                    cfg.kv_num_heads,
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
](shape: IndexList[rank]) -> NDBuffer[
    type, rank, MutableAnyOrigin, static_shape
]:
    var ptr = UnsafePointer[Scalar[type]].alloc(shape.flattened_length())
    rand(ptr, shape.flattened_length())
    return NDBuffer[type, rank, _, static_shape](ptr, shape)


def test_case[
    type: DType,
    batch_rank: Int,
    *,
    output_static_shape: DimList = DimList.create_unknown[batch_rank + 2](),
](cfg: TestCaseConfig[batch_rank]):
    seed(42)

    # Allocate the QKV tensors.
    var q_shape = cfg.build_shape_bshd(cfg.seq_len, cfg.depth_dim)
    var kv_shape = cfg.build_shape_bshd[is_kv=True](
        cfg.kv_seq_len, cfg.depth_dim
    )
    var q = build_ndbuffer[type](q_shape)
    var k = build_ndbuffer[type](kv_shape)
    var v = build_ndbuffer[type](kv_shape)

    # Allocate the attention mask.
    var mask = build_ndbuffer[type](
        cfg.build_shape(cfg.seq_len, cfg.kv_seq_len)
    )

    # Allocate output and reference output buffers.
    var output = build_ndbuffer[type, static_shape=output_static_shape](q_shape)
    var ref_output = build_ndbuffer[type](q_shape)

    reference_attention_bshd(q, k, v, mask, ref_output, cfg.scale)

    @parameter
    @always_inline
    fn input_k_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return k.load[width=simd_width](rebind[IndexList[k.rank]](idx))

    @parameter
    @always_inline
    fn input_v_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return v.load[width=simd_width](rebind[IndexList[v.rank]](idx))

    @parameter
    @always_inline
    fn mask_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return mask.load[width=simd_width](rebind[IndexList[mask.rank]](idx))

    flash_attention[input_k_fn, input_v_fn, mask_fn](
        q, k.get_shape(), v.get_shape(), mask.get_shape(), output, cfg.scale
    )

    verify_output(output.make_dims_unknown(), ref_output, cfg)

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
            kv_num_heads=8,
            kv_seq_len=503,
            depth_dim=128,
            scale=0.125,
        )
    )
    test_case[type](
        TestCaseConfig(
            batch_dims=Index(4, 12),
            seq_len=1,
            kv_num_heads=4,
            kv_seq_len=503,
            depth_dim=64,
            scale=0.125,
        )
    )
    test_case[type](
        TestCaseConfig(
            batch_dims=Index(2, 3),
            seq_len=128,
            kv_num_heads=3,
            kv_seq_len=128,
            depth_dim=63,
            scale=0.25,
        )
    )
    test_case[type](
        TestCaseConfig(
            batch_dims=Index(8),
            seq_len=64,
            kv_num_heads=1,
            kv_seq_len=64,
            depth_dim=384,
            scale=0.25,
        )
    )
    test_case[type, output_static_shape = DimList(Dim(), Dim(), 128)](
        TestCaseConfig(
            batch_dims=Index(1),
            seq_len=55,
            kv_num_heads=1,
            kv_seq_len=127,
            depth_dim=128,
            scale=0.2,
        )
    )
    test_case[type, output_static_shape = DimList(Dim(), Dim(), 160)](
        TestCaseConfig(
            batch_dims=Index(1),
            seq_len=100,
            kv_num_heads=1,
            kv_seq_len=300,
            depth_dim=160,
            scale=0.1,
        )
    )
    test_case[type, output_static_shape = DimList(Dim(), Dim(), 300)](
        TestCaseConfig(
            batch_dims=Index(1),
            seq_len=100,
            kv_num_heads=1,
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
    # For now only allow Q.shape = [B, S, H, D].
    constrained[batch_rank == 2]()

    seed(42)

    # Allocate the QKV tensors.
    var q_shape = cfg.build_shape_bshd(cfg.seq_len, cfg.depth_dim)
    var kv_shape = cfg.build_shape_bshd[is_kv=True](
        cfg.kv_seq_len, cfg.depth_dim
    )
    var q = build_ndbuffer[type](q_shape)
    var k = build_ndbuffer[type](kv_shape)
    var v = build_ndbuffer[type](kv_shape)

    # Allocate the attention mask.
    var mask = build_ndbuffer[type](
        cfg.build_shape(cfg.seq_len, cfg.kv_seq_len)
    )

    # Allocate output and reference output buffers.
    var output = build_ndbuffer[type, static_shape=output_static_shape](q_shape)
    var ref_output = build_ndbuffer[type](q_shape)

    # Compute reference outputs for comparison.
    reference_attention_bshd(q, k, v, mask, ref_output, cfg.scale)

    # Define input lambdas for split KV cache attn `flash_attention_split_kv`.
    @parameter
    @always_inline
    fn input_k_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return k.load[width=simd_width](
            IndexList[k.rank](
                idx[0], idx[1] + cfg.prev_seq_len(), idx[2], idx[3]
            )
        )

    @parameter
    @always_inline
    fn input_v_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return v.load[width=simd_width](
            IndexList[v.rank](
                idx[0], idx[1] + cfg.prev_seq_len(), idx[2], idx[3]
            )
        )

    @parameter
    @always_inline
    fn input_k_cache_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return k.load[width=simd_width](
            IndexList[k.rank](idx[1], idx[3], idx[2], idx[4])
        )

    @parameter
    @always_inline
    fn input_v_cache_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return v.load[width=simd_width](
            IndexList[v.rank](idx[1], idx[3], idx[2], idx[4])
        )

    @parameter
    @always_inline
    fn mask_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return mask.load[width=simd_width](rebind[IndexList[mask.rank]](idx))

    var kv_present_shape = cfg.build_shape_bshd[is_kv=True](
        cfg.seq_len, cfg.depth_dim
    )
    var kv_past_shape = cfg.build_shape[
        shape_rank = cfg.kv_cache_rank, is_kv=True
    ](cfg.prev_seq_len(), cfg.depth_dim)

    flash_attention_split_kv[
        type=type,
        rank = batch_rank + 2,
        input_k_fn,
        input_v_fn,
        input_k_cache_fn,
        input_v_cache_fn,
        mask_fn,
    ](
        q,
        kv_present_shape,
        kv_present_shape,
        rebind[IndexList[cfg.rank + 1]](kv_past_shape),
        rebind[IndexList[cfg.rank + 1]](kv_past_shape),
        mask.get_shape(),
        output,
        cfg.scale,
    )

    verify_output(output.make_dims_unknown(), ref_output, cfg)

    q.data.free()
    k.data.free()
    v.data.free()
    mask.data.free()
    output.data.free()
    ref_output.data.free()


def test_flash_attention_split_kv[type: DType]():
    for kv_seq_len in range(1, 128):
        test_case_split_kv[type](
            TestCaseConfig(
                batch_dims=Index(1, 1),
                seq_len=1,
                kv_num_heads=1,
                kv_seq_len=kv_seq_len,
                depth_dim=1,
                scale=0.125,
            )
        )
    test_case_split_kv[type](
        TestCaseConfig(
            batch_dims=Index(1, 8),
            seq_len=1,
            kv_num_heads=8,
            kv_seq_len=503,
            depth_dim=128,
            scale=0.125,
        )
    )
    test_case_split_kv[type](
        TestCaseConfig(
            batch_dims=Index(1, 8),
            seq_len=1,
            kv_num_heads=2,
            kv_seq_len=503,
            depth_dim=128,
            scale=0.125,
        )
    )
    test_case_split_kv[type](
        TestCaseConfig(
            batch_dims=Index(5, 24),
            seq_len=23,
            kv_num_heads=8,
            kv_seq_len=57,
            depth_dim=128,
            scale=0.125,
        )
    )
    test_case_split_kv[type](
        TestCaseConfig(
            batch_dims=Index(2, 3),
            seq_len=128,
            kv_num_heads=3,
            kv_seq_len=128,
            depth_dim=63,
            scale=0.25,
        )
    )


def main():
    test_flash_attention[DType.float32]()
    test_flash_attention_split_kv[DType.float32]()

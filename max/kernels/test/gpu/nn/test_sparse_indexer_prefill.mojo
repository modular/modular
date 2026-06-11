# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Correctness tests for the prefill-path MSA indexer.

Inputs are random bf16, ragged across the batch. An independent host oracle
(f32) recomputes each block score as the max of `q . k * sm_scale` over that
query token's in-range *causal* keys (`prefix_len + local_index + 1`), applies
init/local forcing, and derives the top-k. The score buffer is compared to the
oracle (tolerant for computed blocks, exact for forced), and the top-k output is
checked with a value invariant plus forced-block presence, distinctness, and the
`-1` sentinel tail. A planted large key just past a token's causal boundary
makes a causality violation detectable; a K padding sentinel makes an over-read
past a row's last key detectable.
"""

from std.collections import Set
from std.gpu.host import DeviceContext
from std.math import ceildiv, max, min
from std.random import rand
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)
from std.utils.index import Index

from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)

from msa.sparse_indexer_prefill import (
    sparse_indexer_prefill,
    sparse_indexer_prefill_score,
    sparse_indexer_prefill_topk,
)
from nn.attention.mha_operand import RaggedMHAOperand

comptime SCORE_ATOL = 1e-2
comptime SCORE_RTOL = 1e-2
comptime INIT_SCORE = Float32(1.0e30)
comptime LOCAL_SCORE = Float32(1.0e29)


def _batch_of(t: Int, batch: Int, iro: List[Int]) -> Int:
    var b = 0
    for bi in range(batch):
        if t < iro[bi + 1]:
            b = bi
            break
    return b


def _host_block_score(
    q: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    k: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    t: Int,
    h: Int,
    blk: Int,
    key_off_b: Int,
    num_keys: Int,
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
    sm_scale: Float32,
) -> Float32:
    var key_start = blk * block_size
    var key_end = min(key_start + block_size, num_keys)
    var q_off = (t * num_index_heads + h) * idx_head_dim
    var blk_max = Float32(-3.0e38)
    for key in range(key_start, key_end):
        var k_off = (key_off_b + key) * idx_head_dim
        var dot = Float32(0)
        for d in range(idx_head_dim):
            dot += (
                q[q_off + d].cast[DType.float32]()
                * k[k_off + d].cast[DType.float32]()
            )
        var s = dot * sm_scale
        if s > blk_max:
            blk_max = s
    return blk_max


def _forced_score(
    raw: Float32, blk: Int, num_blocks: Int, init_blocks: Int, local_blocks: Int
) -> Float32:
    var local_start = max(0, num_blocks - local_blocks)
    var val = raw
    if blk < init_blocks:
        val = INIT_SCORE
    if blk >= local_start:
        val = LOCAL_SCORE
    return val


def _run_case[
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
](
    extend_lens: List[Int],
    prefix_lens_list: List[Int],
    topk: Int,
    init_blocks: Int,
    local_blocks: Int,
    plant_key: Int,  # global key index set to a large value (-1 = none)
    sm_scale: Float32,
    ctx: DeviceContext,
) raises:
    var batch = len(extend_lens)
    var iro = [0]  # input row offsets (query tokens)
    var cro = [0]  # cache row offsets (keys)
    for b in range(batch):
        iro.append(iro[b] + extend_lens[b])
        cro.append(cro[b] + prefix_lens_list[b] + extend_lens[b])
    var total_q = iro[batch]
    var total_keys = cro[batch]
    var max_num_blocks = 0
    for b in range(batch):
        max_num_blocks = max(
            max_num_blocks,
            ceildiv(prefix_lens_list[b] + extend_lens[b], block_size),
        )

    print(
        "  case: batch=",
        batch,
        " heads=",
        num_index_heads,
        " total_q=",
        total_q,
        " total_keys=",
        total_keys,
        " topk=",
        topk,
        " init=",
        init_blocks,
        " local=",
        local_blocks,
        " plant_key=",
        plant_key,
    )

    var q_n = total_q * num_index_heads * idx_head_dim
    var k_n = total_keys * idx_head_dim
    # Extra K rows filled with a large sentinel: an over-read past a row's last
    # key hits this and produces a detectably wrong block score.
    var k_pad = block_size * idx_head_dim
    var score_n = num_index_heads * total_q * max_num_blocks
    var out_n = num_index_heads * total_q * topk

    var q_host = ctx.enqueue_create_host_buffer[DType.bfloat16](q_n)
    var k_host = ctx.enqueue_create_host_buffer[DType.bfloat16](k_n + k_pad)
    rand(q_host.as_span())
    rand(k_host.as_span())
    for j in range(k_pad):
        k_host[k_n + j] = Scalar[DType.bfloat16](5.0)
    if plant_key >= 0:
        for d in range(idx_head_dim):
            k_host[plant_key * idx_head_dim + d] = Scalar[DType.bfloat16](5.0)

    var iro_host = ctx.enqueue_create_host_buffer[DType.uint32](batch + 1)
    var cro_host = ctx.enqueue_create_host_buffer[DType.uint32](batch + 1)
    var pl_host = ctx.enqueue_create_host_buffer[DType.uint32](batch)
    for b in range(batch + 1):
        iro_host[b] = UInt32(iro[b])
        cro_host[b] = UInt32(cro[b])
    for b in range(batch):
        pl_host[b] = UInt32(prefix_lens_list[b])
    ctx.synchronize()

    var q_dev = ctx.enqueue_create_buffer[DType.bfloat16](q_n)
    var k_dev = ctx.enqueue_create_buffer[DType.bfloat16](k_n + k_pad)
    var iro_dev = ctx.enqueue_create_buffer[DType.uint32](batch + 1)
    var cro_dev = ctx.enqueue_create_buffer[DType.uint32](batch + 1)
    var pl_dev = ctx.enqueue_create_buffer[DType.uint32](batch)
    var score_dev = ctx.enqueue_create_buffer[DType.float32](score_n)
    var out_dev = ctx.enqueue_create_buffer[DType.int32](out_n)
    ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
    ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)
    ctx.enqueue_copy(dst_buf=iro_dev, src_buf=iro_host)
    ctx.enqueue_copy(dst_buf=cro_dev, src_buf=cro_host)
    ctx.enqueue_copy(dst_buf=pl_dev, src_buf=pl_host)
    out_dev.enqueue_fill(Int32(-2))

    var q_t = TileTensor(
        q_dev, row_major((total_q, num_index_heads, idx_head_dim))
    )
    var iro_t = TileTensor(iro_dev, row_major(batch + 1))
    var pl_t = TileTensor(pl_dev, row_major(batch))
    var score_t = TileTensor(
        score_dev, row_major((num_index_heads, total_q, max_num_blocks))
    )
    var out_t = TileTensor(out_dev, row_major((num_index_heads, total_q, topk)))

    comptime k_layout = Layout.row_major(UNKNOWN_VALUE, 1, idx_head_dim)
    comptime cro_layout = Layout.row_major(UNKNOWN_VALUE)
    var k_lt = LayoutTensor[DType.bfloat16, k_layout, ImmutAnyOrigin](
        rebind[UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin]](
            k_dev.unsafe_ptr()
        ),
        RuntimeLayout[k_layout].row_major(Index(total_keys, 1, idx_head_dim)),
    )
    var cro_lt = LayoutTensor[DType.uint32, cro_layout, ImmutAnyOrigin](
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](cro_dev.unsafe_ptr()),
        RuntimeLayout[cro_layout].row_major(Index(batch + 1)),
    )
    var k_operand = RaggedMHAOperand(k_lt, cro_lt)

    # --- Score kernel, then inspect the score buffer. ---
    sparse_indexer_prefill_score[
        DType.bfloat16,
        type_of(k_operand),
        num_index_heads,
        idx_head_dim,
        block_size,
    ](
        q_t,
        k_operand,
        iro_t,
        pl_t,
        score_t,
        batch,
        total_q,
        max_num_blocks,
        init_blocks,
        local_blocks,
        sm_scale,
        ctx,
    )
    var score_host = ctx.enqueue_create_host_buffer[DType.float32](score_n)
    ctx.enqueue_copy(dst_buf=score_host, src_buf=score_dev)
    ctx.synchronize()

    var q_hp = q_host.unsafe_ptr()
    var k_hp = k_host.unsafe_ptr()

    for h in range(num_index_heads):
        for t in range(total_q):
            var b = _batch_of(t, batch, iro)
            var local_idx = t - iro[b]
            var num_keys = prefix_lens_list[b] + local_idx + 1
            var num_blocks = ceildiv(num_keys, block_size)
            for blk in range(num_blocks):
                var raw = _host_block_score(
                    q_hp,
                    k_hp,
                    t,
                    h,
                    blk,
                    cro[b],
                    num_keys,
                    num_index_heads,
                    idx_head_dim,
                    block_size,
                    sm_scale,
                )
                var expect = _forced_score(
                    raw, blk, num_blocks, init_blocks, local_blocks
                )
                var got = score_host[(h * total_q + t) * max_num_blocks + blk]
                if expect == INIT_SCORE or expect == LOCAL_SCORE:
                    assert_equal(got, expect, "forced score mismatch")
                else:
                    assert_almost_equal(
                        got, expect, atol=SCORE_ATOL, rtol=SCORE_RTOL
                    )

    # --- Top-k kernel, then check selected indices. ---
    sparse_indexer_prefill_topk[num_index_heads, block_size](
        iro_t,
        pl_t,
        score_t,
        out_t,
        batch,
        total_q,
        max_num_blocks,
        topk,
        ctx,
    )
    var out_host = ctx.enqueue_create_host_buffer[DType.int32](out_n)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()

    for h in range(num_index_heads):
        for t in range(total_q):
            var b = _batch_of(t, batch, iro)
            var local_idx = t - iro[b]
            var num_keys = prefix_lens_list[b] + local_idx + 1
            var num_blocks = ceildiv(num_keys, block_size)
            var k_batch = min(topk, num_blocks)
            var local_start = max(0, num_blocks - local_blocks)
            var base = (h * total_q + t) * topk

            var got = Set[Int]()
            for j in range(k_batch):
                var idx = Int(out_host[base + j])
                assert_true(
                    idx >= 0 and idx < num_blocks, "selected index out of range"
                )
                got.add(idx)
            assert_equal(len(got), k_batch, "duplicate/missing selected index")

            for j in range(k_batch, topk):
                assert_equal(Int(out_host[base + j]), -1, "tail must be -1")

            for blk in range(min(init_blocks, num_blocks)):
                assert_true(blk in got, "forced init block not selected")
            for blk in range(local_start, num_blocks):
                assert_true(blk in got, "forced local block not selected")

            # Top-k invariant: every selected score >= every non-selected score.
            var sel_min = Float32(3.0e38)
            var non_max = Float32(-3.0e38)
            for blk in range(num_blocks):
                var raw = _host_block_score(
                    q_hp,
                    k_hp,
                    t,
                    h,
                    blk,
                    cro[b],
                    num_keys,
                    num_index_heads,
                    idx_head_dim,
                    block_size,
                    sm_scale,
                )
                var sc = _forced_score(
                    raw, blk, num_blocks, init_blocks, local_blocks
                )
                if blk in got:
                    if sc < sel_min:
                        sel_min = sc
                else:
                    if sc > non_max:
                        non_max = sc
            if k_batch < num_blocks:
                assert_true(
                    sel_min >= non_max - SCORE_ATOL,
                    "selected block ranks below an excluded block",
                )

    # Exercise the public entry into fresh buffers and confirm it matches the
    # two-kernel path above.
    var score_dev2 = ctx.enqueue_create_buffer[DType.float32](score_n)
    var out_dev2 = ctx.enqueue_create_buffer[DType.int32](out_n)
    out_dev2.enqueue_fill(Int32(-2))
    var score_t2 = TileTensor(
        score_dev2, row_major((num_index_heads, total_q, max_num_blocks))
    )
    var out_t2 = TileTensor(
        out_dev2, row_major((num_index_heads, total_q, topk))
    )
    sparse_indexer_prefill[
        DType.bfloat16,
        type_of(k_operand),
        num_index_heads,
        idx_head_dim,
        block_size,
    ](
        q_t,
        k_operand,
        iro_t,
        pl_t,
        score_t2,
        out_t2,
        batch,
        total_q,
        max_num_blocks,
        topk,
        init_blocks,
        local_blocks,
        sm_scale,
        ctx,
    )
    var out_host2 = ctx.enqueue_create_host_buffer[DType.int32](out_n)
    ctx.enqueue_copy(dst_buf=out_host2, src_buf=out_dev2)
    ctx.synchronize()
    for i in range(out_n):
        assert_equal(
            out_host2[i],
            out_host[i],
            "public entry differs from two-kernel path",
        )

    _ = q_dev
    _ = k_dev
    _ = iro_dev
    _ = cro_dev
    _ = pl_dev
    _ = score_dev
    _ = out_dev
    _ = score_dev2
    _ = out_dev2


def _run_forcing_case(ctx: DeviceContext) raises:
    """Forcing check, independent of the local_start formula.

    A single query token (via a large prefix) sees many blocks. Forced blocks
    (first `init_blocks`, last `local_blocks`) are given a zero real score, so
    they are selected only if forcing works; unforced zero-score blocks are
    excluded.
    """
    comptime H = 4
    comptime D = 128
    comptime BS = 128
    var prefix = 2559  # single token sees prefix + 1 = 2560 keys = 20 blocks
    var seq_len = prefix + 1
    var num_blocks = seq_len // BS
    var topk = 4
    var init_blocks = 1  # forces block 0
    var local_blocks = 1  # forces the last block (num_blocks - 1)
    var sm = Float32(0.08838834764831845)
    # 0 and last are forced; 10 and last-1 are not -> must be excluded.
    var zeroed = [0, 10, num_blocks - 2, num_blocks - 1]
    print("  forcing case: blocks=", num_blocks, " topk=", topk)

    var k_n = seq_len * D
    var k_pad = BS * D
    var q_host = ctx.enqueue_create_host_buffer[DType.bfloat16](H * D)
    var k_host = ctx.enqueue_create_host_buffer[DType.bfloat16](k_n + k_pad)
    rand(q_host.as_span())
    rand(k_host.as_span())
    for j in range(k_pad):
        k_host[k_n + j] = Scalar[DType.bfloat16](5.0)
    for zi in range(len(zeroed)):
        var blk = zeroed[zi]
        for key in range(blk * BS, (blk + 1) * BS):
            for d in range(D):
                k_host[key * D + d] = Scalar[DType.bfloat16](0.0)
    ctx.synchronize()

    var iro_host = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var cro_host = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var pl_host = ctx.enqueue_create_host_buffer[DType.uint32](1)
    iro_host[0] = 0
    iro_host[1] = 1
    cro_host[0] = 0
    cro_host[1] = UInt32(seq_len)
    pl_host[0] = UInt32(prefix)

    var q_dev = ctx.enqueue_create_buffer[DType.bfloat16](H * D)
    var k_dev = ctx.enqueue_create_buffer[DType.bfloat16](k_n + k_pad)
    var iro_dev = ctx.enqueue_create_buffer[DType.uint32](2)
    var cro_dev = ctx.enqueue_create_buffer[DType.uint32](2)
    var pl_dev = ctx.enqueue_create_buffer[DType.uint32](1)
    var score_dev = ctx.enqueue_create_buffer[DType.float32](H * num_blocks)
    var out_dev = ctx.enqueue_create_buffer[DType.int32](H * topk)
    ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
    ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)
    ctx.enqueue_copy(dst_buf=iro_dev, src_buf=iro_host)
    ctx.enqueue_copy(dst_buf=cro_dev, src_buf=cro_host)
    ctx.enqueue_copy(dst_buf=pl_dev, src_buf=pl_host)
    out_dev.enqueue_fill(Int32(-2))

    var q_t = TileTensor(q_dev, row_major((1, H, D)))
    var iro_t = TileTensor(iro_dev, row_major(2))
    var pl_t = TileTensor(pl_dev, row_major(1))
    var score_t = TileTensor(score_dev, row_major((H, 1, num_blocks)))
    var out_t = TileTensor(out_dev, row_major((H, 1, topk)))

    comptime k_layout = Layout.row_major(UNKNOWN_VALUE, 1, D)
    comptime cro_layout = Layout.row_major(UNKNOWN_VALUE)
    var k_lt = LayoutTensor[DType.bfloat16, k_layout, ImmutAnyOrigin](
        rebind[UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin]](
            k_dev.unsafe_ptr()
        ),
        RuntimeLayout[k_layout].row_major(Index(seq_len, 1, D)),
    )
    var cro_lt = LayoutTensor[DType.uint32, cro_layout, ImmutAnyOrigin](
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](cro_dev.unsafe_ptr()),
        RuntimeLayout[cro_layout].row_major(Index(2)),
    )
    var k_operand = RaggedMHAOperand(k_lt, cro_lt)

    sparse_indexer_prefill[DType.bfloat16, type_of(k_operand), H, D, BS](
        q_t,
        k_operand,
        iro_t,
        pl_t,
        score_t,
        out_t,
        1,
        1,
        num_blocks,
        topk,
        init_blocks,
        local_blocks,
        sm,
        ctx,
    )
    var out_host = ctx.enqueue_create_host_buffer[DType.int32](H * topk)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()

    for h in range(H):
        var got = Set[Int]()
        for j in range(topk):
            got.add(Int(out_host[h * topk + j]))
        assert_true(0 in got, "init-forced block 0 not selected")
        assert_true(
            num_blocks - 1 in got, "local-forced last block not selected"
        )
        assert_false(10 in got, "unforced zero-score block 10 selected")
        assert_false(
            num_blocks - 2 in got, "unforced zero-score block (last-1) selected"
        )
    _ = q_dev
    _ = k_dev
    _ = iro_dev
    _ = cro_dev
    _ = pl_dev
    _ = score_dev
    _ = out_dev


def main() raises:
    comptime SM = Float32(0.08838834764831845)  # 128 ** -0.5
    with DeviceContext() as ctx:
        # Fresh prefill (no prefix), multi-batch, varied extend lengths, local=1.
        _run_case[4, 128, 128](
            [300, 100, 512], [0, 0, 0], 16, 0, 1, -1, SM, ctx
        )
        # init + local forcing.
        _run_case[4, 128, 128]([512, 200], [0, 0], 16, 2, 1, -1, SM, ctx)
        # Chunked prefill: nonzero prefix lengths.
        _run_case[4, 128, 128]([64, 96], [500, 1000], 16, 0, 1, -1, SM, ctx)
        # local=0: diagonal (final) block's computed max is USED, plus a planted
        # large key just past some tokens' causal boundary -> a causality
        # violation would read it and be caught by the score check.
        _run_case[4, 128, 128]([260], [0], 16, 0, 0, 100, SM, ctx)
        # Non-M3 head/dim config for generality.
        _run_case[2, 64, 64]([130, 70], [0, 0], 8, 1, 1, -1, SM, ctx)
        # A token with num_blocks > block_dim (exercises the block stride loop):
        # prefix 20000 + 2 extend -> ~157 blocks for the last token.
        _run_case[4, 128, 128]([2], [20000], 16, 0, 1, -1, SM, ctx)
        # Functional forcing check (independent of the local_start formula).
        _run_forcing_case(ctx)
        print("all prefill indexer cases passed")

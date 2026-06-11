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
"""Correctness tests for the decode-path MSA indexer.

Inputs are random bf16. An independent host oracle (f32) recomputes each block
score as the max of `q . k * sm_scale` over that block's in-range keys, applies
init/local forcing, and derives the top-k. The score buffer is read back BEFORE
`block_select_topk` mutates it and compared to the oracle (tolerant for computed
blocks, exact for forced blocks), which directly checks the scoring math
including the partial final block. The top-k output is checked with a
reference-independent invariant (every selected block's score >= every
non-selected block's), plus forced-block presence, distinctness, and the `-1`
sentinel tail. Cases cover partial final blocks, multiple heads/batches, init
and local forcing, topk > num_blocks, and a non-M3 head/dim config.
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

from msa.sparse_indexer_decode import (
    sparse_indexer_decode,
    sparse_indexer_decode_score,
    sparse_indexer_decode_topk,
)
from nn.attention.mha_operand import RaggedMHAOperand

comptime SCORE_ATOL = 1e-2
comptime SCORE_RTOL = 1e-2
comptime INIT_SCORE = Float32(1.0e30)
comptime LOCAL_SCORE = Float32(1.0e29)


def _host_block_score(
    q: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    k: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: Int,
    h: Int,
    blk: Int,
    off_b: Int,
    num_keys: Int,
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
    sm_scale: Float32,
) -> Float32:
    """Oracle block score (no forcing): max over the block's in-range keys."""
    var key_start = blk * block_size
    var key_end = min(key_start + block_size, num_keys)
    var q_off = (b * num_index_heads + h) * idx_head_dim
    var blk_max = Float32(-3.0e38)
    for key in range(key_start, key_end):
        var k_off = (off_b + key) * idx_head_dim
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
    # Local (1e29) takes priority over init (1e30); matches the kernel.
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
    seq_lens_list: List[Int],
    topk: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
    ctx: DeviceContext,
) raises:
    var batch = len(seq_lens_list)
    var total_keys = 0
    for i in range(batch):
        total_keys += seq_lens_list[i]
    var max_num_blocks = 0
    for i in range(batch):
        max_num_blocks = max(
            max_num_blocks, ceildiv(seq_lens_list[i], block_size)
        )

    print(
        "  case: batch=",
        batch,
        " heads=",
        num_index_heads,
        " dim=",
        idx_head_dim,
        " block=",
        block_size,
        " topk=",
        topk,
        " init=",
        init_blocks,
        " local=",
        local_blocks,
        " total_keys=",
        total_keys,
    )

    var q_n = batch * num_index_heads * idx_head_dim
    var k_n = total_keys * idx_head_dim
    # Extra K rows filled with a large sentinel: an over-read of a partial
    # final block hits this and produces a detectably wrong block score.
    var k_pad = block_size * idx_head_dim
    var score_n = num_index_heads * batch * max_num_blocks
    var out_n = num_index_heads * batch * topk

    # Host inputs.
    var q_host = ctx.enqueue_create_host_buffer[DType.bfloat16](q_n)
    var k_host = ctx.enqueue_create_host_buffer[DType.bfloat16](k_n + k_pad)
    var sl_host = ctx.enqueue_create_host_buffer[DType.uint32](batch)
    var cro_host = ctx.enqueue_create_host_buffer[DType.uint32](batch + 1)
    rand(q_host.as_span())
    rand(k_host.as_span())
    for j in range(k_pad):
        k_host[k_n + j] = Scalar[DType.bfloat16](5.0)
    var running: UInt32 = 0
    for b in range(batch):
        sl_host[b] = UInt32(seq_lens_list[b])
        cro_host[b] = running
        running += UInt32(seq_lens_list[b])
    cro_host[batch] = running
    ctx.synchronize()

    # Device buffers.
    var q_dev = ctx.enqueue_create_buffer[DType.bfloat16](q_n)
    var k_dev = ctx.enqueue_create_buffer[DType.bfloat16](k_n + k_pad)
    var sl_dev = ctx.enqueue_create_buffer[DType.uint32](batch)
    var cro_dev = ctx.enqueue_create_buffer[DType.uint32](batch + 1)
    var score_dev = ctx.enqueue_create_buffer[DType.float32](score_n)
    var out_dev = ctx.enqueue_create_buffer[DType.int32](out_n)
    ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
    ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)
    ctx.enqueue_copy(dst_buf=sl_dev, src_buf=sl_host)
    ctx.enqueue_copy(dst_buf=cro_dev, src_buf=cro_host)
    out_dev.enqueue_fill(Int32(-2))  # poison

    var q_t = TileTensor(
        q_dev, row_major((batch, num_index_heads, idx_head_dim))
    )
    var sl_t = TileTensor(sl_dev, row_major(batch))
    var score_t = TileTensor(
        score_dev, row_major((num_index_heads, batch, max_num_blocks))
    )
    var out_t = TileTensor(out_dev, row_major((num_index_heads, batch, topk)))

    # Build the ragged index-K operand: [total_keys, 1, idx_head_dim].
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

    # --- Launch score kernel only, then inspect the score buffer. ---
    sparse_indexer_decode_score[
        DType.bfloat16,
        type_of(k_operand),
        num_index_heads,
        idx_head_dim,
        block_size,
    ](
        q_t,
        k_operand,
        sl_t,
        score_t,
        batch,
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
        for b in range(batch):
            var off_b = Int(cro_host[b])
            var num_keys = seq_lens_list[b]
            var num_blocks = ceildiv(num_keys, block_size)
            for blk in range(num_blocks):
                var raw = _host_block_score(
                    q_hp,
                    k_hp,
                    b,
                    h,
                    blk,
                    off_b,
                    num_keys,
                    num_index_heads,
                    idx_head_dim,
                    block_size,
                    sm_scale,
                )
                var expect = _forced_score(
                    raw, blk, num_blocks, init_blocks, local_blocks
                )
                var got = score_host[(h * batch + b) * max_num_blocks + blk]
                if expect == INIT_SCORE or expect == LOCAL_SCORE:
                    assert_equal(got, expect, "forced score mismatch")
                else:
                    assert_almost_equal(
                        got, expect, atol=SCORE_ATOL, rtol=SCORE_RTOL
                    )

    # --- Launch top-k kernel, then check the selected indices. ---
    sparse_indexer_decode_topk[num_index_heads, block_size](
        sl_t,
        score_t,
        out_t,
        batch,
        max_num_blocks,
        topk,
        ctx,
    )
    var out_host = ctx.enqueue_create_host_buffer[DType.int32](out_n)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()

    for h in range(num_index_heads):
        for b in range(batch):
            var off_b = Int(cro_host[b])
            var num_keys = seq_lens_list[b]
            var num_blocks = ceildiv(num_keys, block_size)
            var k_batch = min(topk, num_blocks)
            var local_start = max(0, num_blocks - local_blocks)
            var base = (h * batch + b) * topk

            var got = Set[Int]()
            for j in range(k_batch):
                var idx = Int(out_host[base + j])
                assert_true(
                    idx >= 0 and idx < num_blocks, "selected index out of range"
                )
                got.add(idx)
            assert_equal(len(got), k_batch, "duplicate/missing selected index")

            # Sentinel tail.
            for j in range(k_batch, topk):
                assert_equal(Int(out_host[base + j]), -1, "tail must be -1")

            # Forced blocks must be selected.
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
                    b,
                    h,
                    blk,
                    off_b,
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

    # Exercise the public entry into fresh buffers and confirm it produces the
    # same indices as the (already-validated) two-kernel path above.
    var score_dev2 = ctx.enqueue_create_buffer[DType.float32](score_n)
    var out_dev2 = ctx.enqueue_create_buffer[DType.int32](out_n)
    out_dev2.enqueue_fill(Int32(-2))
    var score_t2 = TileTensor(
        score_dev2, row_major((num_index_heads, batch, max_num_blocks))
    )
    var out_t2 = TileTensor(out_dev2, row_major((num_index_heads, batch, topk)))
    sparse_indexer_decode[
        DType.bfloat16,
        type_of(k_operand),
        num_index_heads,
        idx_head_dim,
        block_size,
    ](
        q_t,
        k_operand,
        sl_t,
        score_t2,
        out_t2,
        batch,
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

    _ = score_dev2
    _ = out_dev2
    _ = q_dev
    _ = k_dev
    _ = sl_dev
    _ = cro_dev
    _ = score_dev
    _ = out_dev


def _run_forcing_case(ctx: DeviceContext) raises:
    """Forcing check: forced blocks are given a zero real score, so they are
    selected only if forcing works; unforced zero-score blocks are excluded.
    """
    comptime H = 4
    comptime D = 128
    comptime BS = 128
    var seq = 2560  # 20 full blocks
    var num_blocks = seq // BS
    var topk = 4
    var init_blocks = 1  # forces block 0
    var local_blocks = 1  # forces the last block (num_blocks - 1)
    var sm = Float32(0.08838834764831845)
    # Blocks whose keys are zeroed -> zero real score. 0 and last are forced;
    # 10 and last-1 are not, and must be excluded.
    var zeroed = [0, 10, num_blocks - 2, num_blocks - 1]
    print("  forcing case: blocks=", num_blocks, " topk=", topk)

    var k_n = seq * D
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

    var sl_host = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var cro_host = ctx.enqueue_create_host_buffer[DType.uint32](2)
    sl_host[0] = UInt32(seq)
    cro_host[0] = 0
    cro_host[1] = UInt32(seq)

    var q_dev = ctx.enqueue_create_buffer[DType.bfloat16](H * D)
    var k_dev = ctx.enqueue_create_buffer[DType.bfloat16](k_n + k_pad)
    var sl_dev = ctx.enqueue_create_buffer[DType.uint32](1)
    var cro_dev = ctx.enqueue_create_buffer[DType.uint32](2)
    var score_dev = ctx.enqueue_create_buffer[DType.float32](H * num_blocks)
    var out_dev = ctx.enqueue_create_buffer[DType.int32](H * topk)
    ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
    ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)
    ctx.enqueue_copy(dst_buf=sl_dev, src_buf=sl_host)
    ctx.enqueue_copy(dst_buf=cro_dev, src_buf=cro_host)
    out_dev.enqueue_fill(Int32(-2))

    var q_t = TileTensor(q_dev, row_major((1, H, D)))
    var sl_t = TileTensor(sl_dev, row_major(1))
    var score_t = TileTensor(score_dev, row_major((H, 1, num_blocks)))
    var out_t = TileTensor(out_dev, row_major((H, 1, topk)))

    comptime k_layout = Layout.row_major(UNKNOWN_VALUE, 1, D)
    comptime cro_layout = Layout.row_major(UNKNOWN_VALUE)
    var k_lt = LayoutTensor[DType.bfloat16, k_layout, ImmutAnyOrigin](
        rebind[UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin]](
            k_dev.unsafe_ptr()
        ),
        RuntimeLayout[k_layout].row_major(Index(seq, 1, D)),
    )
    var cro_lt = LayoutTensor[DType.uint32, cro_layout, ImmutAnyOrigin](
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](cro_dev.unsafe_ptr()),
        RuntimeLayout[cro_layout].row_major(Index(2)),
    )
    var k_operand = RaggedMHAOperand(k_lt, cro_lt)

    sparse_indexer_decode[DType.bfloat16, type_of(k_operand), H, D, BS](
        q_t,
        k_operand,
        sl_t,
        score_t,
        out_t,
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
        # Forced blocks (zero score) selected only if forcing works.
        assert_true(0 in got, "init-forced block 0 not selected")
        assert_true(
            num_blocks - 1 in got, "local-forced last block not selected"
        )
        # Unforced zero-score blocks must be excluded.
        assert_false(10 in got, "unforced zero-score block 10 selected")
        assert_false(
            num_blocks - 2 in got, "unforced zero-score block (last-1) selected"
        )
    _ = q_dev
    _ = k_dev
    _ = sl_dev
    _ = cro_dev
    _ = score_dev
    _ = out_dev


def main() raises:
    comptime SM = Float32(0.08838834764831845)  # 128 ** -0.5
    with DeviceContext() as ctx:
        # M3 config (heads=4, dim=128, block=128, topk=16).
        # Real selection (num_blocks > topk) + partial final blocks + local.
        _run_case[4, 128, 128]([2560, 4096, 3000], 16, 0, 1, SM, ctx)
        # init + local forcing.
        _run_case[4, 128, 128]([2560, 4096], 16, 2, 1, SM, ctx)
        # topk > num_blocks (short sequences) -> all selected, -1 tail.
        _run_case[4, 128, 128]([128, 300], 16, 0, 1, SM, ctx)
        # local=0: the partial final block's computed max is USED (not forced),
        # so this exercises and checks the partial-block scoring directly.
        _run_case[4, 128, 128]([256, 3000], 16, 0, 0, SM, ctx)
        # Non-M3 head/dim config for generality.
        _run_case[2, 64, 64]([200, 512, 64], 8, 1, 1, SM, ctx)
        # num_blocks > block_dim: each thread strides over multiple blocks;
        # partial final block (local=0) checked at scale.
        _run_case[4, 128, 128]([17000], 16, 0, 0, SM, ctx)
        # Functional forcing check (independent of the local_start formula).
        _run_forcing_case(ctx)
        print("all decode indexer cases passed")

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
"""Host reverse-CSR builder for KV-block-major sparse MHA, SM100.

Inverts the query-major selection `q2k_indices [head_kv, total_q, topK]` (per
query token, the batch-local KV-BLOCK ids it attends, `< 0` = unused) into the
KV-block-major CSR the block-major forward/combine kernels consume: for each
(batch, kv-block) pair the list of queries that selected it.  Sequential CPU
build; `k2q_csr_device` is the GPU port and the oracle for it.

A CSR "row" is one (batch, kv_block) pair.  Rows are numbered LEVEL-MAJOR round-
robin: all batches' block-0 first (batch order, skipping batches with no
block-0), then all block-1, etc., so `scheduler_metadata`'s
`(row_linear, batch, kv_block)` stays consistent with the device builder.

Contract tensors emitted (all i32, batch-local q indices):
- `k2q_row_ptr [head_kv, total_rows + 1]`   exclusive prefix of per-row counts.
- `scheduler_metadata [work_capacity, 6]`   (head_kv, row_linear, q_begin,
  q_count, batch, kv_block) + `work_count [1]`.  Each non-empty row is split
  into `ceil(row_count / q_per_cta)` work items (q-chunking) so a row selected
  by more than `q_per_cta` queries is served by multiple CTAs.
- `split_counts [B, max_seqlen_q, head_kv]` per-query valid-block count.
- `qsplit_indices [head_kv, total_q * topK]` `q | (split_slot << 24)`,
  topK <= 255.
"""

from std.math import ceildiv


comptime _UNUSED: Int32 = -1
"""Padding / unused-slot sentinel in `qsplit_indices`."""


struct K2qCsr(Movable):
    """Reverse-CSR + schedule for one sparse-MHA forward pass (host-built).

    All buffers are owned `List[Int32]`; index them with the strides documented
    on each field.  `total_rows`, `max_kv_blocks` and `work_capacity` are the
    sizing the future kernel allocates against.
    """

    var head_kv: Int
    var total_q: Int
    var topk: Int
    var batch: Int
    var blk_kv: Int
    var max_seqlen_q: Int

    # Sizing the kernels allocate against.
    var total_rows: Int
    var max_kv_blocks: Int
    var work_capacity: Int

    # Contract tensors.
    var k2q_row_ptr: List[Int32]  # [head_kv, total_rows + 1]
    var qsplit_indices: List[Int32]  # [head_kv, total_q * topk]
    var scheduler_metadata: List[Int32]  # [work_capacity, 6]
    var work_count: Int  # number of valid scheduler rows
    var split_counts: List[Int32]  # [batch, max_seqlen_q, head_kv]

    # (batch, kv_block) for each row_linear.
    var row_coords: List[Int32]  # [total_rows, 2]

    def __init__(
        out self,
        head_kv: Int,
        total_q: Int,
        topk: Int,
        batch: Int,
        blk_kv: Int,
        max_seqlen_q: Int,
        total_rows: Int,
        max_kv_blocks: Int,
        work_capacity: Int,
    ):
        self.head_kv = head_kv
        self.total_q = total_q
        self.topk = topk
        self.batch = batch
        self.blk_kv = blk_kv
        self.max_seqlen_q = max_seqlen_q
        self.total_rows = total_rows
        self.max_kv_blocks = max_kv_blocks
        self.work_capacity = work_capacity
        self.k2q_row_ptr = List[Int32]()
        self.qsplit_indices = List[Int32]()
        self.scheduler_metadata = List[Int32]()
        self.work_count = 0
        self.split_counts = List[Int32]()
        self.row_coords = List[Int32]()


def _rows_in_batch(seqlen_k: Int, blk_kv: Int) -> Int:
    """Number of kv-blocks (CSR rows) batch `b` owns."""
    return ceildiv(seqlen_k, blk_kv)


def balanced_target_q_per_cta(
    total_q: Int,
    topk: Int,
    blk_kv: Int,
    head_kv: Int,
    num_sms: Int,
    bm: Int = 128,
) -> Int:
    """Load-balanced queries-per-CTA cap for the scheduler q-chunking.

    Targets ~`num_sms*2` work items so each CTA processes `ceil(q_count / bm)`
    Q-groups against ONE resident KV block (instead of one CTA per `bm` queries).
    Rounded up to a multiple of `bm` (the fwd loops in `bm`-query groups), floored
    at `bm`, and capped at `topk * blk_kv * 2` so a few huge rows can't starve the
    SMs.
    """
    var groups_upper = ceildiv(max(total_q * topk * head_kv, 1), bm)
    var desired = max(num_sms * 2, 1)
    var groups_per_cta = min(512, max(1, ceildiv(groups_upper, desired)))
    var occupancy_target = groups_per_cta * bm
    var cap = max(bm, topk * blk_kv * 2)
    var target = min(max(occupancy_target, bm), cap)
    return ceildiv(target, bm) * bm


def build_k2q_csr(
    q2k_indices: List[Int32],  # [head_kv, total_q, topk], batch-local block ids
    cu_seqlens_q: List[Int32],  # [batch + 1]
    cu_seqlens_k: List[Int32],  # [batch + 1]
    head_kv: Int,
    total_q: Int,
    topk: Int,
    blk_kv: Int,
    max_seqlen_q: Int,
    max_seqlen_k: Int,
    q_per_cta: Int = 128,
) raises -> K2qCsr:
    """Builds the reverse-CSR + schedule from the query-major selection.

    `q2k_indices[(h * total_q + g) * topk + t]` is the batch-local KV-block id
    that global query token `g`'s slot `t` selected for kv-head `h`, or `< 0` if
    unused.  Queries are packed by batch via `cu_seqlens_q`; batch-local q is
    `g - cu_seqlens_q[b]`.

    Inverts sequentially.  Row numbering is level-major round-robin (block-0 of
    every active batch, then block-1, ...).

    A non-empty row is q-chunked into `ceil(row_count / q_per_cta)` work items
    (`q_per_cta <= BM`, the fwd CTA's query cap) so rows selected by more than
    `q_per_cta` queries are served by multiple CTAs rather than truncated.

    Note that `topk <= 255` (qsplit packs split_slot in the high byte) and
    `max_seqlen_q < 2^24` (qsplit packs q in the low 24 bits).
    """
    var batch = len(cu_seqlens_q) - 1

    # ---- Sizing -------------------------------------------------------------
    # total_rows = sum over batches of ceil(seqlen_k_b / blk_kv) -- exact.
    var total_rows = 0
    for b in range(batch):
        var seqlen_k = Int(cu_seqlens_k[b + 1] - cu_seqlens_k[b])
        total_rows += _rows_in_batch(seqlen_k, blk_kv)
    var max_kv_blocks = ceildiv(max(max_seqlen_k, blk_kv), blk_kv)
    # Q-chunked upper bound: every non-empty row is >=1 work item
    # (<= total_rows*head_kv rows), and the extra chunks across all rows total
    # <= ceil(total edges / q_per_cta).
    var work_capacity = total_rows * head_kv + ceildiv(
        total_q * topk * head_kv, q_per_cta
    )

    var csr = K2qCsr(
        head_kv,
        total_q,
        topk,
        batch,
        blk_kv,
        max_seqlen_q,
        total_rows,
        max_kv_blocks,
        work_capacity,
    )

    # ---- M: round-robin row map ---------------------------------------------
    # row_map[b, level] = row_linear if batch b is active at this kv-block level,
    # else -1.  row_coords[row_linear] = (b, level).
    var row_map = List[Int32](length=batch * max_kv_blocks, fill=_UNUSED)
    csr.row_coords = List[Int32](length=total_rows * 2, fill=Int32(0))
    for level in range(max_kv_blocks):
        var rows_before = 0
        for b in range(batch):
            var seqlen_k = Int(cu_seqlens_k[b + 1] - cu_seqlens_k[b])
            var rb = _rows_in_batch(seqlen_k, blk_kv)
            rows_before += min(rb, level)
        var active_before = 0
        for b in range(batch):
            var seqlen_k = Int(cu_seqlens_k[b + 1] - cu_seqlens_k[b])
            var rb = _rows_in_batch(seqlen_k, blk_kv)
            if rb > level:
                var row_linear = rows_before + active_before
                row_map[b * max_kv_blocks + level] = Int32(row_linear)
                csr.row_coords[row_linear * 2] = Int32(b)
                csr.row_coords[row_linear * 2 + 1] = Int32(level)
                active_before += 1

    # ---- H + PR: per-row counts -> exclusive prefix (k2q_row_ptr) ------------
    # row_counts[h, r] = number of (q, block) edges mapping to CSR row r.
    csr.k2q_row_ptr = List[Int32](
        length=head_kv * (total_rows + 1), fill=Int32(0)
    )
    var row_counts = List[Int32](length=head_kv * total_rows, fill=Int32(0))
    for h in range(head_kv):
        for g in range(total_q):
            var b = _batch_of(cu_seqlens_q, g, batch)
            var rmap_base = b * max_kv_blocks
            var q2k_base = (h * total_q + g) * topk
            for t in range(topk):
                var kvb = Int(q2k_indices[q2k_base + t])
                if kvb >= 0 and kvb < max_kv_blocks:
                    var row = Int(row_map[rmap_base + kvb])
                    if row >= 0 and row < total_rows:
                        row_counts[h * total_rows + row] += 1

    for h in range(head_kv):
        var rp_base = h * (total_rows + 1)
        var running = 0
        csr.k2q_row_ptr[rp_base] = Int32(0)
        for r in range(total_rows):
            running += Int(row_counts[h * total_rows + r])
            csr.k2q_row_ptr[rp_base + r + 1] = Int32(running)

    # ---- S: scatter, q-ascending within row (qsplit) -------------------------
    # split_counts[b, qloc, h] = valid-block count for that query.
    # qsplit packs qloc in the low 24 bits and the per-q split slot (rank among
    # that q's valid edges) in the top 8.
    var nnz = total_q * topk
    csr.qsplit_indices = List[Int32](length=head_kv * nnz, fill=_UNUSED)
    csr.split_counts = List[Int32](
        length=batch * max_seqlen_q * head_kv, fill=Int32(0)
    )

    # Per-row write cursor, reset per head.  Iterating g ascending and writing at
    # row_ptr[row] + cursor[row] yields q-ascending order within every row.
    var cursor = List[Int32](length=total_rows, fill=Int32(0))
    for h in range(head_kv):
        for r in range(total_rows):
            cursor[r] = 0
        var rp_base = h * (total_rows + 1)
        var qidx_base = h * nnz
        for g in range(total_q):
            var b = _batch_of(cu_seqlens_q, g, batch)
            var qloc = g - Int(cu_seqlens_q[b])
            var rmap_base = b * max_kv_blocks
            var q2k_base = (h * total_q + g) * topk
            var split_slot = 0  # rank among this q's valid edges
            for t in range(topk):
                var kvb = Int(q2k_indices[q2k_base + t])
                if kvb < 0 or kvb >= max_kv_blocks:
                    continue
                var row = Int(row_map[rmap_base + kvb])
                if row < 0 or row >= total_rows:
                    continue
                var slot = Int(cursor[row])
                cursor[row] += 1
                var out_pos = Int(csr.k2q_row_ptr[rp_base + row]) + slot
                csr.qsplit_indices[qidx_base + out_pos] = Int32(
                    qloc | ((split_slot & 0xFF) << 24)
                )
                split_slot += 1
            # split_count == number of valid edges this q produced.
            csr.split_counts[(b * max_seqlen_q + qloc) * head_kv + h] = Int32(
                split_slot
            )

    # ---- Schedule (q-chunking) ----------------------------------------------
    # Split each non-empty row into ceil(row_count / q_per_cta) chunks; chunk c
    # is (h, row, c*q_per_cta, min(q_per_cta, row_count - q_begin), b, lvl).
    csr.scheduler_metadata = List[Int32](
        length=work_capacity * 6, fill=Int32(0)
    )
    var work = 0
    for h in range(head_kv):
        var rp_base = h * (total_rows + 1)
        for r in range(total_rows):
            var row_count = Int(
                csr.k2q_row_ptr[rp_base + r + 1] - csr.k2q_row_ptr[rp_base + r]
            )
            if row_count == 0:
                continue
            var num_chunks = ceildiv(row_count, q_per_cta)
            for c in range(num_chunks):
                var q_begin = c * q_per_cta
                var q_count = min(q_per_cta, row_count - q_begin)
                var meta = work * 6
                csr.scheduler_metadata[meta + 0] = Int32(h)
                csr.scheduler_metadata[meta + 1] = Int32(r)
                csr.scheduler_metadata[meta + 2] = Int32(q_begin)
                csr.scheduler_metadata[meta + 3] = Int32(q_count)
                csr.scheduler_metadata[meta + 4] = csr.row_coords[r * 2]
                csr.scheduler_metadata[meta + 5] = csr.row_coords[r * 2 + 1]
                work += 1
    csr.work_count = work

    return csr^


def _batch_of(cu_seqlens_q: List[Int32], g: Int, batch: Int) raises -> Int:
    """Batch owning global query token `g` (cu_seqlens_q[b] <= g < [b+1])."""
    for b in range(batch):
        if g < Int(cu_seqlens_q[b + 1]):
            return b
    return batch - 1

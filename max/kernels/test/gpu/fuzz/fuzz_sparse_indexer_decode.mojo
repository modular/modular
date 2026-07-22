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
#
# Fuzz target: the MiniMax-M3 sparse-attention (MSA) indexer DECODE path
# (`sparse_indexer_decode_score` + `sparse_indexer_decode_topk`, in
# Kernels/lib/msa/sparse_indexer_decode.mojo). This is the per-step indexer that
# scores every KV block (block-max of `q . k * sm_scale`), applies init/local
# forcing, and selects the top-k blocks to attend to.
#
# Why this is a richer bug surface than block_select_topk: the score kernel does
# split-K over the KV-block dimension (num_chunks, chosen internally from the
# batch), a lane-split f32 dot reduction, partial-final-block handling, and
# init/local forcing -- all classic edge-bug territory. (block_select_topk, the
# selection core it feeds, is fuzzed separately by fuzz_sparse_indexer.)
#
# Runtime fuzz axes (CaseSpec, all Int): batch, the per-batch sequence lengths
# (derived from min/max/pattern -- num_keys % block_size is the partial-final-
# block modulus, and the KEYS_PER_ITER remainder is the historical deadlock
# class), topk, init_blocks, local_blocks, value distribution, seed. The kernel
# config (num_index_heads / idx_head_dim / block_size) is compile-time, baked via
# `-D` (default the M3 config 4 / 128 / 128; build with -D for the nh=1
# register-MMA geometry or a non-M3 head/dim).
#
# Oracles:
#  - `ref` (--check 1, default): an independent f32 host oracle recomputes each
#    block score (max over the block's in-range keys) and the init/local forcing,
#    and compares the score buffer (tolerant for computed blocks, exact for
#    forced). The top-k output is checked with the tie-robust invariant (every
#    selected block's forced-score >= every non-selected block's). Emits
#    FUZZ_NUMERIC_FAIL. The K buffer carries a large sentinel pad past the last
#    key, so a partial-final-block over-read produces a detectably wrong score.
#  - VALIDITY CONTRACT (always on): out_idxs are in-range / distinct, with a -1
#    sentinel tail, on a fresh -2-poisoned output. Rides under diff / memcheck /
#    redzone too. Emits FUZZ_CONTRACT_FAIL.
#  - `schedule` (--schedule N): re-run the score kernel N times on the same input
#    and flag any non-bit-exact score buffer -- an inter-chunk (split-K) race /
#    nondeterminism the single-shot ref pass could miss.
#
# `--inject N` (default 0; never set by the orchestrator) is the oracle canary:
#   1 = corrupt one selected index to a low-scoring block -> trips `ref`;
#   2 = write an out-of-range index -> trips the validity contract under diff.

from std.collections import Set
from std.gpu.host import DeviceContext
from std.math import ceildiv, max, min, sqrt
from std.random import randn, random_ui64, seed as set_seed
from std.sys.defines import get_defined_int

from layout import Idx, TileTensor, row_major

from msa.sparse_indexer_decode import (
    sparse_indexer_decode_score,
    sparse_indexer_decode_topk,
)
from nn.attention.mha_operand import RaggedMHAOperand

from _fuzz import (
    VD_ALL_EQUAL,
    VD_NORMAL,
    VD_SPARSE,
    VD_UNIFORM,
    boundary_int,
    collect_args,
    fill_all_equal,
    fill_sparse,
    fill_uniform,
    flag,
    flag_int,
)

comptime kv_type = DType.bfloat16
comptime out_idx_type = DType.int32

# Compile-time kernel config (one instantiation per build, like fuzz_matmul's
# N/K). Default = the MiniMax-M3 indexer (heads=4, dim=128, block=128); override
# with -D for the nh=1 register-MMA geometry or a non-M3 head/dim.
comptime num_index_heads = get_defined_int["sidx_num_heads", 4]()
comptime idx_head_dim = get_defined_int["sidx_head_dim", 128]()
comptime block_size = get_defined_int["sidx_block_size", 128]()

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()

comptime INIT_SCORE = Float32(1.0e30)
comptime LOCAL_SCORE = Float32(1.0e29)
comptime SCORE_ATOL = Float32(1.0e-2)
comptime SCORE_RTOL = Float32(1.0e-2)

# Sequence-length disparity patterns (deterministic from the scalar spec, so a
# `single` case reproduces a ragged batch exactly).
comptime PAT_EQUAL = 0
comptime PAT_RAMP = 1
comptime PAT_ALT = 2
comptime PAT_ONE_BIG = 3
comptime NUM_PATTERNS = 4


def _dist_name(d: Int) -> String:
    if d == VD_NORMAL:
        return "normal"
    if d == VD_SPARSE:
        return "sparse"
    if d == VD_ALL_EQUAL:
        return "all_equal"
    return "uniform"


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var batch: Int
    var max_seq: Int  # longest per-batch sequence (the dominant axis)
    var min_seq: Int  # shortest per-batch sequence
    var pattern: Int  # seq-length disparity pattern (PAT_*)
    var topk: Int  # blocks to select per (head, batch)
    var init_blocks: Int  # forced leading blocks (score 1e30)
    var local_blocks: Int  # forced trailing blocks (score 1e29)
    var dist: Int  # value-distribution id (finite-only; see _dist_name)
    var seed: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "batch=",
            self.batch,
            " max_seq=",
            self.max_seq,
            " min_seq=",
            self.min_seq,
            " pattern=",
            self.pattern,
            " topk=",
            self.topk,
            " init_blocks=",
            self.init_blocks,
            " local_blocks=",
            self.local_blocks,
            " dist=",
            _dist_name(self.dist),
            " seed=",
            self.seed,
        )


def derive_seq_lens(spec: CaseSpec) -> List[Int]:
    """Per-batch sequence lengths from (batch, min, max, pattern)."""
    var out = List[Int]()
    var bs = spec.batch
    var hi = max(1, spec.max_seq)
    var lo = max(1, min(spec.min_seq, hi))
    for i in range(bs):
        if spec.pattern == PAT_EQUAL:
            out.append(hi)
        elif spec.pattern == PAT_RAMP:
            if bs <= 1:
                out.append(hi)
            else:
                out.append(lo + (hi - lo) * i // (bs - 1))
        elif spec.pattern == PAT_ALT:
            out.append(hi if (i % 2 == 0) else lo)
        else:  # PAT_ONE_BIG
            out.append(hi if i == 0 else lo)
    return out^


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var batch = boundary_int(1, 8, 1)
        # max_seq biased around block_size and its multiples: the partial-final-
        # block and (hi-1) remainder edges (e.g. 8191 % 128 = 127) live there.
        var max_seq = boundary_int(1, 16384, block_size)
        var min_seq = boundary_int(1, max_seq, block_size)
        var pattern = Int(random_ui64(0, UInt64(NUM_PATTERNS - 1)))
        var topk = boundary_int(1, 256, 16)
        var init_blocks = boundary_int(0, 4, 1)
        var local_blocks = boundary_int(0, 4, 1)
        var droll = Int(random_ui64(0, 3))
        var dist: Int
        if droll == 0:
            dist = VD_NORMAL
        elif droll == 1:
            dist = VD_SPARSE
        elif droll == 2:
            dist = VD_ALL_EQUAL
        else:
            dist = VD_UNIFORM
        var the_seed = Int(random_ui64(1, 1_000_000))
        specs.append(
            CaseSpec(
                batch,
                max_seq,
                min_seq,
                pattern,
                topk,
                init_blocks,
                local_blocks,
                dist,
                the_seed,
            )
        )
    return specs^


def _fill_stable[
    dtype: DType
](span: Span[mut=True, Scalar[dtype], _], dist: Int):
    """Finite, bounded fill (|x| ~ 0.5) -- keeps bf16 dot products in a range
    where the f32 reference tolerance holds. NaN/Inf/huge are excluded (they have
    no meaningful block-max reference)."""
    if dist == VD_SPARSE:
        fill_sparse(span, density=0.1, lo=-0.5, hi=0.5)
    elif dist == VD_ALL_EQUAL:
        fill_all_equal(span, value=0.5)
    elif dist == VD_NORMAL:
        randn(span, mean=0.0, standard_deviation=0.5)
    else:  # VD_UNIFORM
        fill_uniform(span, lo=-0.5, hi=0.5)


def _host_block_score(
    q: UnsafePointer[mut=False, Scalar[kv_type], _],
    k: UnsafePointer[mut=False, Scalar[kv_type], _],
    b: Int,
    h: Int,
    blk: Int,
    off_b: Int,
    num_keys: Int,
    sm_scale: Float32,
) -> Float32:
    """Oracle block score (no forcing): max over the block's in-range keys of
    `q . k * sm_scale`, using the SAME bf16 input values the kernel sees."""
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
    """Init/local forcing (local takes priority over init; matches the kernel).
    """
    var local_start = max(0, num_blocks - local_blocks)
    var val = raw
    if blk < init_blocks:
        val = INIT_SCORE
    if blk >= local_start:
        val = LOCAL_SCORE
    return val


def run_one_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    schedule_repeats: Int = 0,
    inject: Int = 0,
) raises:
    var seq_lens = derive_seq_lens(spec)
    var batch = spec.batch
    var topk = spec.topk
    var init_blocks = spec.init_blocks
    var local_blocks = spec.local_blocks
    var sm_scale = Float32(1.0) / sqrt(Float32(idx_head_dim))

    var total_keys = 0
    var max_num_blocks = 0
    for i in range(batch):
        total_keys += seq_lens[i]
        max_num_blocks = max(max_num_blocks, ceildiv(seq_lens[i], block_size))

    var q_n = batch * num_index_heads * idx_head_dim
    var k_n = total_keys * idx_head_dim
    # Sentinel pad past the last key: a partial-final-block over-read lands here
    # and yields a detectably wrong block score.
    var k_pad = block_size * idx_head_dim
    var score_n = num_index_heads * batch * max_num_blocks
    var out_n = num_index_heads * batch * topk

    # --- host inputs ---------------------------------------------------------
    var q_host = ctx.enqueue_create_host_buffer[kv_type](q_n)
    var k_host = ctx.enqueue_create_host_buffer[kv_type](k_n + k_pad)
    var sl_host = ctx.enqueue_create_host_buffer[DType.uint32](batch)
    var cro_host = ctx.enqueue_create_host_buffer[DType.uint32](batch + 1)
    _fill_stable(q_host.as_span(), spec.dist)
    _fill_stable(Span(ptr=k_host.unsafe_ptr(), length=k_n), spec.dist)
    for j in range(k_pad):
        k_host[k_n + j] = Scalar[kv_type](5.0)
    var running: UInt32 = 0
    for b in range(batch):
        sl_host[b] = UInt32(seq_lens[b])
        cro_host[b] = running
        running += UInt32(seq_lens[b])
    cro_host[batch] = running
    ctx.synchronize()

    # --- device buffers ------------------------------------------------------
    var q_dev = ctx.enqueue_create_buffer[kv_type](q_n)
    var k_dev = ctx.enqueue_create_buffer[kv_type](k_n + k_pad)
    var sl_dev = ctx.enqueue_create_buffer[DType.uint32](batch)
    var cro_dev = ctx.enqueue_create_buffer[DType.uint32](batch + 1)
    var score_dev = ctx.enqueue_create_buffer[DType.float32](score_n)
    var out_dev = ctx.enqueue_create_buffer[out_idx_type](out_n)
    ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
    ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)
    ctx.enqueue_copy(dst_buf=sl_dev, src_buf=sl_host)
    ctx.enqueue_copy(dst_buf=cro_dev, src_buf=cro_host)
    out_dev.enqueue_fill(Int32(-2))  # poison: must be overwritten

    var q_t = TileTensor(
        q_dev, row_major((batch, num_index_heads, idx_head_dim))
    )
    var sl_t = TileTensor(sl_dev, row_major(batch))
    var score_t = TileTensor(
        score_dev, row_major((num_index_heads, batch, max_num_blocks))
    )
    var out_t = TileTensor(out_dev, row_major((num_index_heads, batch, topk)))

    # `seq_lens` here is the full inclusive key count, so the decode kernels'
    # in-step add must be 0: pass an all-zeros `input_row_offsets`.
    var iro_dev = ctx.enqueue_create_buffer[DType.uint32](batch + 1)
    iro_dev.enqueue_fill(UInt32(0))
    var iro_t = TileTensor(iro_dev, row_major(batch + 1))

    # Ragged index-K operand: [total_keys, 1, idx_head_dim].
    var k_buf = TileTensor(
        rebind[UnsafePointer[Scalar[kv_type], ImmutAnyOrigin]](
            k_dev.unsafe_ptr()
        ),
        row_major((total_keys, Idx[1], Idx[idx_head_dim])),
    )
    var cro_buf = TileTensor(
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](cro_dev.unsafe_ptr()),
        row_major((batch + 1,)),
    )
    var k_operand = RaggedMHAOperand(k_buf, cro_buf)

    # --- schedule oracle: re-run the score kernel N times on the same input and
    # flag any non-bit-exact score buffer (a split-K / inter-chunk race) --------
    if schedule_repeats > 0:
        var ref_host = ctx.enqueue_create_host_buffer[DType.float32](score_n)
        var n_runs = max(2, schedule_repeats)
        for r in range(n_runs):
            score_dev.enqueue_fill(Float32(0))
            sparse_indexer_decode_score[
                kv_type,
                type_of(k_operand),
                num_index_heads,
                idx_head_dim,
                block_size,
            ](
                q_t,
                k_operand,
                sl_t,
                iro_t,
                score_t,
                batch,
                max_num_blocks,
                init_blocks,
                local_blocks,
                sm_scale,
                ctx,
            )
            var sh = ctx.enqueue_create_host_buffer[DType.float32](score_n)
            ctx.enqueue_copy(dst_buf=sh, src_buf=score_dev)
            ctx.synchronize()
            if r == 0:
                for i in range(score_n):
                    ref_host[i] = sh[i]
            else:
                for i in range(score_n):
                    if sh[i] != ref_host[i]:
                        print(
                            "FUZZ_NUMERIC_FAIL kind=score_nondeterminism run=",
                            r,
                            "idx=",
                            i,
                            "got=",
                            sh[i],
                            "ref=",
                            ref_host[i],
                        )
                        raise Error(
                            "decode score kernel nondeterministic (split-K"
                            " race)"
                        )
        _ = q_dev
        _ = k_dev
        _ = sl_dev
        _ = cro_dev
        _ = iro_dev
        _ = score_dev
        _ = out_dev
        return

    # --- score kernel; snapshot the score buffer BEFORE top-k mutates it -----
    sparse_indexer_decode_score[
        kv_type,
        type_of(k_operand),
        num_index_heads,
        idx_head_dim,
        block_size,
    ](
        q_t,
        k_operand,
        sl_t,
        iro_t,
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

    # --- top-k kernel (block_select_topk; mutates score_t in place) ----------
    sparse_indexer_decode_topk[num_index_heads, block_size](
        sl_t,
        iro_t,
        score_t,
        out_t,
        batch,
        max_num_blocks,
        topk,
        ctx,
    )
    var out_host = ctx.enqueue_create_host_buffer[out_idx_type](out_n)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()

    if inject != 0:
        _apply_inject(inject, out_host.unsafe_ptr(), max_num_blocks)

    var q_hp = q_host.unsafe_ptr()
    var k_hp = k_host.unsafe_ptr()

    # --- score-buffer correctness (ref) --------------------------------------
    if check:
        for h in range(num_index_heads):
            for b in range(batch):
                var off_b = Int(cro_host[b])
                var num_keys = seq_lens[b]
                var num_blocks = ceildiv(num_keys, block_size)
                for blk in range(num_blocks):
                    var raw = _host_block_score(
                        q_hp, k_hp, b, h, blk, off_b, num_keys, sm_scale
                    )
                    var expect = _forced_score(
                        raw, blk, num_blocks, init_blocks, local_blocks
                    )
                    var got = score_host[(h * batch + b) * max_num_blocks + blk]
                    if expect == INIT_SCORE or expect == LOCAL_SCORE:
                        if got != expect:
                            print(
                                "FUZZ_NUMERIC_FAIL kind=forced_score h=",
                                h,
                                "b=",
                                b,
                                "blk=",
                                blk,
                                "got=",
                                got,
                                "expect=",
                                expect,
                            )
                            raise Error("forced block score mismatch")
                    elif not _close(got, expect):
                        print(
                            "FUZZ_NUMERIC_FAIL kind=block_score h=",
                            h,
                            "b=",
                            b,
                            "blk=",
                            blk,
                            "num_blocks=",
                            num_blocks,
                            "got=",
                            got,
                            "expect=",
                            expect,
                            "dist=",
                            _dist_name(spec.dist),
                        )
                        raise Error("computed block score mismatch")

    # --- top-k validity contract (always) + invariant (ref) ------------------
    for h in range(num_index_heads):
        for b in range(batch):
            var off_b = Int(cro_host[b])
            var num_keys = seq_lens[b]
            var num_blocks = ceildiv(num_keys, block_size)
            var k_batch = min(topk, num_blocks)
            var base = (h * batch + b) * topk

            var got = Set[Int]()
            for j in range(k_batch):
                var idx = Int(out_host[base + j])
                if idx < 0 or idx >= num_blocks:
                    print(
                        "FUZZ_CONTRACT_FAIL kind=index_out_of_range h=",
                        h,
                        "b=",
                        b,
                        "pos=",
                        j,
                        "idx=",
                        idx,
                        "num_blocks=",
                        num_blocks,
                    )
                    raise Error("selected block index out of range")
                if idx in got:
                    print(
                        "FUZZ_CONTRACT_FAIL kind=duplicate_index h=",
                        h,
                        "b=",
                        b,
                        "idx=",
                        idx,
                    )
                    raise Error("duplicate selected block index")
                got.add(idx)

            for j in range(k_batch, topk):
                if Int(out_host[base + j]) != -1:
                    print(
                        "FUZZ_CONTRACT_FAIL kind=tail_not_sentinel h=",
                        h,
                        "b=",
                        b,
                        "pos=",
                        j,
                        "got=",
                        Int(out_host[base + j]),
                    )
                    raise Error("output tail must be -1")

            if check and k_batch < num_blocks:
                _check_topk_invariant(
                    q_hp,
                    k_hp,
                    b,
                    h,
                    off_b,
                    num_keys,
                    num_blocks,
                    init_blocks,
                    local_blocks,
                    sm_scale,
                    got,
                )

    _ = q_dev
    _ = k_dev
    _ = sl_dev
    _ = cro_dev
    _ = iro_dev
    _ = score_dev
    _ = out_dev


def _close(got: Float32, expect: Float32) -> Bool:
    var diff = abs(got - expect)
    return diff <= SCORE_ATOL + SCORE_RTOL * abs(expect)


def _check_topk_invariant(
    q_hp: UnsafePointer[mut=False, Scalar[kv_type], _],
    k_hp: UnsafePointer[mut=False, Scalar[kv_type], _],
    b: Int,
    h: Int,
    off_b: Int,
    num_keys: Int,
    num_blocks: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
    got: Set[Int],
) raises:
    """Tie-robust top-k: every selected block's forced-score >= every
    non-selected block's. Subsumes the forced-block guarantee."""
    var sel_min = Float32(3.0e38)
    var non_max = Float32(-3.0e38)
    for blk in range(num_blocks):
        var raw = _host_block_score(
            q_hp, k_hp, b, h, blk, off_b, num_keys, sm_scale
        )
        var sc = _forced_score(raw, blk, num_blocks, init_blocks, local_blocks)
        if blk in got:
            if sc < sel_min:
                sel_min = sc
        else:
            if sc > non_max:
                non_max = sc
    if sel_min < non_max - SCORE_ATOL:
        print(
            "FUZZ_NUMERIC_FAIL kind=topk_invariant h=",
            h,
            "b=",
            b,
            "sel_min=",
            sel_min,
            "non_max=",
            non_max,
        )
        raise Error("selected block ranks below an excluded block")


def _apply_inject(
    inject: Int,
    out_host: UnsafePointer[mut=True, Scalar[out_idx_type], _],
    max_num_blocks: Int,
):
    """Corrupt out_idxs[0] to prove an oracle can FAIL (positive control)."""
    if inject == 2:
        # Out-of-range index -> trips the validity contract (even under diff).
        out_host[0] = Scalar[out_idx_type](max_num_blocks + 1)
        return
    # inject == 1: report block 0 as selected at a different output slot,
    # creating a duplicate-free but wrong selection only `ref` can judge. We
    # simply force out_idxs[0] to a likely-low block (the last one), which the
    # invariant flags when it was not actually top-k.
    out_host[0] = Scalar[out_idx_type](max_num_blocks - 1)


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var schedule_repeats = flag_int(args, "--schedule", 0)
    var inject = flag_int(args, "--inject", 0)
    set_seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "batch=",
                specs[i].batch,
                "max_seq=",
                specs[i].max_seq,
                "min_seq=",
                specs[i].min_seq,
                "pattern=",
                specs[i].pattern,
                "topk=",
                specs[i].topk,
                "init_blocks=",
                specs[i].init_blocks,
                "local_blocks=",
                specs[i].local_blocks,
                "dist=",
                specs[i].dist,
                "seed=",
                specs[i].seed,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--batch", 2),
            flag_int(args, "--max_seq", 2560),
            flag_int(args, "--min_seq", 2560),
            flag_int(args, "--pattern", PAT_EQUAL),
            flag_int(args, "--topk", 16),
            flag_int(args, "--init_blocks", 0),
            flag_int(args, "--local_blocks", 1),
            flag_int(args, "--dist", VD_UNIFORM),
            flag_int(args, "--seed", the_seed),
        )
        print("FUZZ_SINGLE ", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check, schedule_repeats, inject)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_sparse_indexer_decode heads=",
        num_index_heads,
        "dim=",
        idx_head_dim,
        "block=",
        block_size,
        "seed=",
        the_seed,
        "budget=",
        the_budget,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check, schedule_repeats, inject)
    print("=== done:", len(specs), "cases ===")

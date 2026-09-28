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
# Per-kernel fuzz target: AMD (gfx950) MLA prefill attention
# (`Attention.mla_prefill` in `nn/attention/gpu/amd_structured/mla_prefill.mojo`,
# reached via the public `flare_mla_prefill` entry point in
# `nn/attention/gpu/mla.mojo`).
#
# Targets a deferred-scale online-softmax defect (KERN-2826-class): the score
# tile stays unscaled while the running row max is carried in scaled-log2
# units, so a query row whose last tile is fully causally-masked reseeds its
# max from the stale scaled value, scales it a second time, and overflows
# `exp2` in `calculate_correction`. Both overflow and underflow tails end in
# NaN.
#
# Reachability needs two things at once, so `--oracle diff`/small budgets
# rarely trip it:
#   1. An EXPOSED row: `BM=128 > BN=64` means a query block spans two key
#      tiles, so rows `(row % 128) < 64` in the block's final tile are past
#      their own diagonal (see the CaseSpec docstring).
#   2. A LARGE scaled logit at that row: `mu = q.k / sqrt(192) >~ 94`
#      (`rowsum` overflow) or `>= 99.03` (strict fp32 `exp2` overflow); the
#      underflow tail is `mu <= -115.28`. Ordinary logits sit in the 10-40
#      band, so `spike_mode=1` engineers `mu` directly.
#
# Oracle: `contract` (finite Q/K/V/K_rope -> finite output). `--report_rows 1`
# prints which rows go non-finite, to check against the exposure predicate.
#
# num_heads is compile-time (`-D mla_num_heads`, default 16; K3 runs 12 heads
# per device at TP8 -- rebuild with `-D mla_num_heads=12` to match). dtype is
# fixed at bfloat16: the kernel runs the same deferred-scale softmax for both
# bf16 and fp8, so bf16 alone already exercises the defect.
#
# Three argv modes so the Python orchestrator can drive it with per-case
# timeout + process isolation (a hanging case only kills its own subprocess):
#
#   --mode list-specs --seed S --budget B
#       Print generated specs (`FUZZ_SPEC ...` lines), no GPU work.
#   --mode single --batch_size .. --num_keys .. --spike_mode .. \
#       --mu_target_x10 .. [--contract 1] [--report_rows 1]
#       Run exactly one case. Prints `FUZZ_RESULT verdict=PASS` on success; a
#       hang times out; a crash exits non-zero.
#   --mode fuzz --seed S --budget B   (default)
#       Generate + run a batch in-process (standalone convenience).

from std.math import isfinite, min, sqrt
from std.random import random_ui64, seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from nn.attention.gpu.mla import flare_mla_prefill
from nn.attention.mha_mask import CausalMask

from _fuzz import boundary_int, collect_args, fill_normal, flag, flag_int

# ===----------------------------------------------------------------------=== #
# Fixed MLA shape (DeepSeek-V2/V3 / Kimi K3). num_heads is -D-overridable.
# ===----------------------------------------------------------------------=== #

comptime qkv_type = DType.bfloat16
comptime output_type = DType.bfloat16

comptime Q_DEPTH = 192  # qk_nope_head_dim(128) + qk_rope_head_dim(64)
comptime NOPE_DEPTH = 128  # K/V/output head dim (v_head_dim == qk_nope_head_dim)
comptime CACHE_DEPTH = 576  # kv_lora_rank(512) + qk_rope_head_dim(64)
comptime CACHE_NUM_HEADS = 1  # MLA: a single shared latent KV head

# 1/sqrt(Q_DEPTH), K3's actual runtime scale. The
# defect's magnitude bar is derived against this exact value.
comptime SCALE = Float32(0.07216878)

comptime NUM_HEADS = get_defined_int["mla_num_heads", 16]()

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()
comptime TILE = 64  # BN: the modulus that decides final-block exposure


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    """One fuzz case: a single-shot context-encoding MLA prefill request.

    `batch_size` sequences, each of length `num_keys` (context encoding:
    start_pos=0, so num_keys == seq_len). For a full 128-row query block,
    rows `(row % 128) < 64` are exposed (their last 64-column tile lies past
    their own diagonal); `boundary_int`'s `tile=64` biases toward this class.

    `spike_mode`/`mu_target_x10` engineer the scaled attention logit
    `mu = q.k / sqrt(192)` directly, by setting one shared channel (index 0
    of the Q/K nope segment) to a computed magnitude on every token -- see
    `_apply_spike`. i.i.d. random fills rarely reach `mu ~= 94-99` (or
    <= -115), so this is the axis that distinguishes this target from a
    generic memory-safety fuzzer.
    """

    var batch_size: Int
    var num_keys: Int  # == seq_len; context encoding, start_pos=0
    var spike_mode: Int  # 0: background noise only (must always be finite)
    var mu_target_x10: Int  # target mu*10 (signed); read only if spike_mode=1

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "batch_size=",
            self.batch_size,
            " num_keys=",
            self.num_keys,
            " spike_mode=",
            self.spike_mode,
            " mu_target_x10=",
            self.mu_target_x10,
        )


def _gen_mu_target_x10() -> Int:
    """Boundary-aware target for `mu = q.k / sqrt(192)`, encoded as `mu*10`.

    Biased around the thresholds from the header docstring: the strict
    fp32-overflow bar (`~99.03`), the practical `rowsum`-overflow bar
    (`~94.4`), the underflow tail (`~-115.28`), an ordinary-magnitude
    control, and a wide uniform sweep.
    """
    var roll = Int(random_ui64(0, 9))
    if roll == 0:
        return 0
    if roll == 1:
        return 300  # ordinary transformer-scale logit; must stay finite
    if roll == 2:
        return 700
    if roll == 3:
        return 940  # practical overflow bar
    if roll == 4:
        return 990  # strict fp32-overflow bar (99.03)
    if roll == 5:
        return 1100
    if roll == 6:
        return -940
    if roll == 7:
        return -1153  # underflow tail bar (-115.28)
    if roll == 8:
        return -1200
    return Int(random_ui64(0, 2400)) - 1200


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var batch_size = boundary_int(1, 8, 4)
        var num_keys = boundary_int(1, 4096, TILE)
        # Bias toward spiking: a defect-hunting fuzzer should spend most of
        # its budget where the defect can actually fire.
        var spike_mode = 1 if Int(random_ui64(0, 3)) != 0 else 0
        var mu_target_x10 = _gen_mu_target_x10()
        specs.append(CaseSpec(batch_size, num_keys, spike_mode, mu_target_x10))
    return specs^


# ===----------------------------------------------------------------------=== #
# Value fill: background noise + an engineered single-channel magnitude spike
# ===----------------------------------------------------------------------=== #


def _apply_spike[
    dtype: DType
](
    span: Span[mut=True, Scalar[dtype], _],
    num_rows: Int,
    num_heads: Int,
    head_dim: Int,
    value: Float64,
):
    """Overwrites channel 0 of every (row, head) vector with `value`.

    Applied to every token uniformly: the online softmax is invariant to an
    offset applied to every key column, so only exposed rows -- whose last
    tile's mask discards this column instead of maxing against it -- hit the
    defect. This probes every exposed row in the batch in one launch.
    """
    for r in range(num_rows):
        for h in range(num_heads):
            span[(r * num_heads + h) * head_dim] = value.cast[dtype]()


def _nonfinite_rows(
    vals: Span[Scalar[output_type], _], q_rows: Int, num_heads: Int, depth: Int
) -> List[Int]:
    """Returns the global row indices (0..q_rows) with any non-finite output."""
    var bad = List[Int]()
    var per_row = num_heads * depth
    for r in range(q_rows):
        var base = r * per_row
        var any_bad = False
        for i in range(per_row):
            if not isfinite(vals[base + i]):
                any_bad = True
                break
        if any_bad:
            bad.append(r)
    return bad^


def _print_nonfinite_rows(bad: List[Int], seq_len: Int):
    """Prints which rows went non-finite, decomposed as (batch, local, local%128).

    `local % 128 < 64` is the exposure predicate for a full query block (see
    the CaseSpec docstring).
    """
    print("FUZZ_NONFINITE_ROWS n=", len(bad), "seq_len=", seq_len)
    comptime max_print = 512
    var shown = min(len(bad), max_print)
    for i in range(shown):
        var r = bad[i]
        var local = r % seq_len
        print(
            "  row=",
            r,
            "batch=",
            r // seq_len,
            "local=",
            local,
            "local_mod128=",
            local % 128,
        )
    if len(bad) > max_print:
        print("  ... (", len(bad) - max_print, " more)")


# ===----------------------------------------------------------------------=== #
# One case: build a single-shot ragged MLA-prefill batch and launch.
# ===----------------------------------------------------------------------=== #


def run_one_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    contract: Bool = False,
    report_rows: Bool = False,
) raises:
    var batch_size = spec.batch_size
    var num_keys = spec.num_keys
    var seq_len = num_keys  # context encoding: the whole prompt in one shot

    var q_rows = batch_size * seq_len
    var kv_rows = batch_size * num_keys

    var q_size = q_rows * NUM_HEADS * Q_DEPTH
    var k_size = kv_rows * NUM_HEADS * NOPE_DEPTH
    var v_size = k_size
    var o_size = q_rows * NUM_HEADS * NOPE_DEPTH
    var cache_size = batch_size * num_keys * CACHE_NUM_HEADS * CACHE_DEPTH

    var q_host = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_host = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var v_host = ctx.enqueue_create_host_buffer[qkv_type](v_size)
    var cache_host = ctx.enqueue_create_host_buffer[qkv_type](cache_size)

    # Background: ordinary, softmax-stable magnitude everywhere.
    fill_normal(q_host.as_span(), mean=0.0, std=0.5)
    fill_normal(k_host.as_span(), mean=0.0, std=0.5)
    fill_normal(v_host.as_span(), mean=0.0, std=0.5)
    fill_normal(cache_host.as_span(), mean=0.0, std=0.5)

    if spec.spike_mode == 1:
        var mu = Float64(spec.mu_target_x10) / 10.0
        # q.k (one aligned channel) = m^2 * sign(mu); mu = (q.k)/sqrt(Q_DEPTH).
        var m = sqrt(abs(mu) * sqrt(Float64(Q_DEPTH)))
        var k_sign = 1.0 if mu >= 0.0 else -1.0
        _apply_spike(q_host.as_span(), q_rows, NUM_HEADS, Q_DEPTH, m)
        _apply_spike(
            k_host.as_span(), kv_rows, NUM_HEADS, NOPE_DEPTH, m * k_sign
        )

    var input_row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    var cache_row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        input_row_offsets_host[i] = UInt32(i * seq_len)
        cache_row_offsets_host[i] = UInt32(i * num_keys)

    var q_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_dev = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_dev = ctx.enqueue_create_buffer[qkv_type](v_size)
    var cache_dev = ctx.enqueue_create_buffer[qkv_type](cache_size)
    var o_dev = ctx.enqueue_create_buffer[output_type](o_size)
    var input_row_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var cache_row_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )

    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(v_dev, v_host)
    ctx.enqueue_copy(cache_dev, cache_host)
    ctx.enqueue_copy(input_row_offsets_dev, input_row_offsets_host)
    ctx.enqueue_copy(cache_row_offsets_dev, cache_row_offsets_host)
    ctx.synchronize()

    # Ragged Q/K/V (batch*seq_len rows, no padding); K_rope (`cache`) is the
    # contiguous BSHD buffer flare_mla_prefill wraps separately.
    var q_tt = TileTensor(
        q_dev, row_major(Coord(q_rows, Idx[NUM_HEADS], Idx[Q_DEPTH]))
    )
    var k_tt = TileTensor(
        k_dev, row_major(Coord(kv_rows, Idx[NUM_HEADS], Idx[NOPE_DEPTH]))
    )
    var v_tt = TileTensor(
        v_dev, row_major(Coord(kv_rows, Idx[NUM_HEADS], Idx[NOPE_DEPTH]))
    )
    var cache_tt = TileTensor(
        cache_dev,
        row_major(
            Coord(batch_size, num_keys, Idx[CACHE_NUM_HEADS], Idx[CACHE_DEPTH])
        ),
    )
    var o_tt = TileTensor(
        o_dev, row_major(Coord(q_rows, Idx[NUM_HEADS], Idx[NOPE_DEPTH]))
    )
    var input_row_offsets_tt = TileTensor(
        input_row_offsets_dev, row_major(Coord(batch_size + 1))
    )
    var cache_row_offsets_tt = TileTensor(
        cache_row_offsets_dev, row_major(Coord(batch_size + 1))
    )

    # Kernel under test: on AMD this routes to
    # `Attention[...].mla_prefill(k_rope)` in amd_structured/mla_prefill.mojo.
    flare_mla_prefill[rank=q_tt.rank](
        o_tt,
        q_tt,
        k_tt,
        v_tt,
        cache_tt,
        CausalMask(),
        input_row_offsets_tt,
        cache_row_offsets_tt,
        SCALE,
        ctx,
        q_max_seq_len=seq_len,
    )
    ctx.synchronize()

    if contract or report_rows:
        var o_host = ctx.enqueue_create_host_buffer[output_type](o_size)
        ctx.enqueue_copy(o_host, o_dev)
        ctx.synchronize()
        var bad_rows = _nonfinite_rows(
            o_host.as_span(), q_rows, NUM_HEADS, NOPE_DEPTH
        )
        if report_rows:
            _print_nonfinite_rows(bad_rows, seq_len)
        if contract and len(bad_rows) > 0:
            print("FUZZ_CONTRACT_FAIL n_bad_rows=", len(bad_rows))
            raise Error(
                "MLA prefill produced non-finite output from finite input"
            )

    _ = q_dev
    _ = k_dev
    _ = v_dev
    _ = cache_dev
    _ = o_dev
    _ = input_row_offsets_dev
    _ = cache_row_offsets_dev


# ===----------------------------------------------------------------------=== #
# Mode dispatch (argv handling shared from _fuzz)
# ===----------------------------------------------------------------------=== #


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var contract = flag_int(args, "--contract", 0) == 1
    var report_rows = flag_int(args, "--report_rows", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "batch_size=",
                specs[i].batch_size,
                "num_keys=",
                specs[i].num_keys,
                "spike_mode=",
                specs[i].spike_mode,
                "mu_target_x10=",
                specs[i].mu_target_x10,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--batch_size", 1),
            flag_int(args, "--num_keys", 543),
            flag_int(args, "--spike_mode", 1),
            flag_int(args, "--mu_target_x10", 990),
        )
        print("FUZZ_SINGLE ", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, contract, report_rows)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_mla_prefill num_heads=",
        NUM_HEADS,
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
            run_one_case(ctx, specs[i], contract, report_rows)
    print("=== done:", len(specs), "cases ===")

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
# Per-kernel fuzz target: MHA CausalPaddingMask (see gpu-kernels-fuzzing-design.md).
#
# A seeded, boundary-aware shape generator drives flash_attention +
# mha_gpu_naive with a CausalPaddingMask. The oracle is memory safety
# (run under run_sanitizer.sh memcheck / the redzone or poison allocator), so
# input *values* are irrelevant -- only the *shape* and the mask matter.
#
# This target supports three argv modes so the Python orchestrator can drive it
# with per-case timeout + process isolation (a hanging case only kills its own
# subprocess), and so a single build serves generation, single-case execution,
# and standalone fuzzing:
#
#   --mode list-specs --seed S --budget B
#       Print the generated specs (machine-readable `FUZZ_SPEC ...` lines), no
#       GPU work. The orchestrator enumerates these, then runs each via `single`.
#
#   --mode single --seq-len S --num-keys N --valid-length V
#       Run exactly one case (for orchestration, shrinking, and corpus replay).
#       Prints `FUZZ_RESULT verdict=PASS` on success; a hang times out; a crash
#       exits non-zero.
#
#   --mode fuzz --seed S --budget B   (default)
#       Generate + run a batch in-process (standalone convenience). Uses the
#       SAFE spec space (excludes the known num_keys==1 & valid_length==0 hang)
#       so a single in-process run does not wedge; use the orchestrator for the
#       full space.
#
# Compile-time defaults (overridable by argv): `-D fuzz_seed=<n>`,
# `-D budget=<n>`, `-D depth=<n>`.

from std.math import max, min
from std.random import rand, random_ui64, seed
from std.sys.defines import get_defined_int

from std.gpu.host import DeviceContext
from layout import Idx, Layout, LayoutTensor, TileTensor, row_major
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import CausalPaddingMask

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check


# Fixed configuration. These match a historical OOB-trigger config (bf16, depth=128,
# num_heads=32, group=1) so decode cases land on the SM100 1q path. `depth` is a
# compile-time define because attention kernels specialize on it.
comptime qkv_type = DType.bfloat16
comptime depth = get_defined_int["depth", 128]()
comptime num_heads = 32
comptime group = 1
comptime kv_num_heads = num_heads // group
comptime scale = Float32(0.125)

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()
comptime TILE = 128  # attention BN boundary -- the interesting modulus.


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    """One fuzz case: the runtime-varied attention shape."""

    var seq_len: Int
    var num_keys: Int
    var valid_length: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "seq_len=",
            self.seq_len,
            " num_keys=",
            self.num_keys,
            " valid_length=",
            self.valid_length,
        )


def gen_specs(n: Int, safe: Bool) -> List[CaseSpec]:
    """Generate `n` boundary-aware cases (decode-biased).

    When `safe` is True, excludes the degenerate num_keys==1 / valid_length==0
    region (which hangs the decode kernel) so an in-process run does not wedge.
    When False, generates the full space so an isolated orchestrator can explore
    (and surface) those degenerate configs.
    """
    var specs = List[CaseSpec]()
    var nk_lo = 16 if safe else 1
    var vl_lo = 1 if safe else 0
    for _ in range(n):
        var num_keys = boundary_int(nk_lo, 1024, TILE)
        # Bias 3/4 of cases to decode (seq_len == 1) -- the SM100 1q path. Keep
        # prefill small and within [1, num_keys] (valid causal range): attention
        # is O(seq_len*num_keys), so big prefill is slow under memcheck without
        # adding boundary coverage small shapes don't already give.
        var seq_len: Int
        if Int(random_ui64(0, 3)) != 0:
            seq_len = 1
        else:
            seq_len = boundary_int(1, min(num_keys, 256), TILE)
        # valid_length in [vl_lo, num_keys], boundary-biased (full / full-1 /
        # half / edge) where the padding boundary meets the causal boundary.
        var vroll = Int(random_ui64(0, 5))
        var valid_length: Int
        if vroll == 0:
            valid_length = vl_lo
        elif vroll == 1:
            valid_length = num_keys
        elif vroll == 2:
            valid_length = max(vl_lo, num_keys - 1)
        elif vroll == 3:
            valid_length = max(vl_lo, num_keys // 2)
        else:
            valid_length = Int(random_ui64(UInt64(vl_lo), UInt64(num_keys)))
        specs.append(CaseSpec(seq_len, num_keys, valid_length))
    return specs^


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    """Allocate, fill with random data, and launch attention once."""
    comptime batch_size = 1
    var seq_len = spec.seq_len
    var num_keys = spec.num_keys

    var q_size = batch_size * num_heads * seq_len * depth
    var kv_size = batch_size * kv_num_heads * num_keys * depth

    var q_host = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    var v_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    rand(q_host.as_span())
    rand(k_host.as_span())
    rand(v_host.as_span())

    var q_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var v_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var o_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(v_dev, v_host)

    var q = TileTensor(
        q_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    var k = TileTensor(
        k_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var v = TileTensor(
        v_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var o = TileTensor(
        o_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )

    # `valid_lengths` is a 1-element [num_seqs=1] uint32 tensor, kept alive on
    # `vl_dev` across the launches (so a finding is the real OOB, not a UAF).
    var vl_dev = ctx.enqueue_create_buffer[DType.uint32](1)
    ctx.enqueue_memset(vl_dev, UInt32(spec.valid_length))
    var vl = LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin](
        vl_dev.unsafe_ptr()
    )
    var mask = CausalPaddingMask(vl)

    flash_attention(o, q, k, v, mask, scale, ctx)
    ctx.synchronize()

    # Also exercise the naive reference path (mha.mojo bmm0).
    var o_ref_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var o_ref = TileTensor(
        o_ref_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    mha_gpu_naive(
        q,
        k,
        v,
        mask,
        o_ref,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
    )
    ctx.synchronize()

    if check:
        # Numerical oracle: flash_attention (o) vs the naive reference (o_ref),
        # bf16 tolerances (matching the differential test's split-K allowance).
        var o_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
        var o_ref_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
        ctx.enqueue_copy(o_h, o_dev)
        ctx.enqueue_copy(o_ref_h, o_ref_dev)
        ctx.synchronize()
        if not numeric_check(
            o_h.as_span(), o_ref_h.as_span(), atol=2e-2, rtol=2e-2
        ):
            raise Error("flash_attention vs naive mismatch")

    _ = q_dev
    _ = k_dev
    _ = v_dev
    _ = o_dev
    _ = o_ref_dev
    _ = vl_dev


def run_schedule_case(ctx: DeviceContext, spec: CaseSpec, repeats: Int) raises:
    """Schedule amplification: force split-K decode (num_partitions=2) and run
    `repeats` times on the same input, flagging any non-bit-exact output. A
    difference means the inter-block split-K reduction is order-dependent (a
    race / nondeterminism) -- which racecheck (intra-block only) cannot see.
    """
    comptime batch_size = 1
    comptime seq_len = 1  # decode -> the split-K path
    var num_keys = max(2, spec.num_keys)  # large enough to split; avoid hang

    var q_size = batch_size * num_heads * seq_len * depth
    var kv_size = batch_size * kv_num_heads * num_keys * depth

    var q_host = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    var v_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    rand(q_host.as_span())
    rand(k_host.as_span())
    rand(v_host.as_span())

    var q_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var v_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var o_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(v_dev, v_host)

    var q = TileTensor(
        q_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    var k = TileTensor(
        k_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var v = TileTensor(
        v_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var o = TileTensor(
        o_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )

    var vl_dev = ctx.enqueue_create_buffer[DType.uint32](1)
    ctx.enqueue_memset(vl_dev, UInt32(num_keys))  # full (no padding)
    var vl = LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin](
        vl_dev.unsafe_ptr()
    )
    var mask = CausalPaddingMask(vl)
    var np = Optional[Int](2)  # force split-K

    flash_attention(o, q, k, v, mask, scale, ctx, num_partitions=np)
    ctx.synchronize()
    var first_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    ctx.enqueue_copy(first_h, o_dev)
    ctx.synchronize()

    for _ in range(repeats - 1):
        flash_attention(o, q, k, v, mask, scale, ctx, num_partitions=np)
        ctx.synchronize()
        var rep_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
        ctx.enqueue_copy(rep_h, o_dev)
        ctx.synchronize()
        # Bit-exact (atol=rtol=0): any difference is nondeterminism.
        if not numeric_check(
            rep_h.as_span(), first_h.as_span(), atol=0.0, rtol=0.0
        ):
            raise Error("flash_attention split-K nondeterminism (schedule)")

    _ = q_dev
    _ = k_dev
    _ = v_dev
    _ = o_dev
    _ = vl_dev


# ===----------------------------------------------------------------------=== #
# Mode dispatch (argv handling shared from _fuzz)
# ===----------------------------------------------------------------------=== #


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var schedule_repeats = flag_int(args, "--schedule", 0)
    seed(the_seed)

    if mode == "list-specs":
        # Generation only -- the orchestrator enumerates these, no GPU work.
        var specs = gen_specs(the_budget, safe=False)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "seq_len=",
                specs[i].seq_len,
                "num_keys=",
                specs[i].num_keys,
                "valid_length=",
                specs[i].valid_length,
            )
        return

    if mode == "single":
        # Flag names match the FUZZ_SPEC keys (underscored) so the orchestrator
        # can drive any target generically with `--<spec_key> <value>`.
        var sl = flag_int(args, "--seq_len", 1)
        var nk = flag_int(args, "--num_keys", 16)
        var vl = flag_int(args, "--valid_length", 8)
        print("FUZZ_SINGLE seq_len=", sl, "num_keys=", nk, "valid_length=", vl)
        with DeviceContext() as ctx:
            if schedule_repeats > 0:
                run_schedule_case(ctx, CaseSpec(sl, nk, vl), schedule_repeats)
            else:
                run_one_case(ctx, CaseSpec(sl, nk, vl), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    # Default: in-process standalone fuzz over the SAFE space.
    print(
        "=== fuzz_mha_causal seed=",
        the_seed,
        "budget=",
        the_budget,
        "depth=",
        depth,
        "(in-process, safe space) ===",
    )
    var specs = gen_specs(the_budget, safe=True)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            if schedule_repeats > 0:
                run_schedule_case(ctx, specs[i], schedule_repeats)
            else:
                run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")

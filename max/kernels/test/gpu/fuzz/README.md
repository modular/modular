# Kernel-level fuzzing for Mojo GPU kernels

A boundary-aware fuzzer that searches the kernel input space (shapes, value
distributions, launch configs) and feeds each generated case into a correctness
oracle, with shrinking and a replayable regression corpus. It is the
input-generation layer on top of the in-tree oracles — memory safety (the
redzone/poison device allocators and NVIDIA Compute Sanitizer), numerical
correctness (a higher-precision reference), special-value contracts, and
inter-block race checks.

## How it works

Three layers; only the top is new:

1. **Generate** (Python orchestrator + a Mojo harness) — boundary-aware specs
   from a seed, plus value-distribution fills (uniform/normal/sparse/large/
   all-equal and NaN/Inf/denormal/±0 injection).
2. **Execute** (per-kernel Mojo target) — one `run_one_case` per kernel that
   allocates, fills, launches, and (optionally) compares; reused from the
   existing per-kernel test lifecycle.
3. **Oracle** — memory safety (redzone/poison allocators, Compute Sanitizer),
   numerical correctness (`ref`, vs a higher-precision reference), special-value
   contracts (`contract`), inter-block races (`schedule`), and hangs/crashes
   (timeouts / exit codes).

The orchestrator runs each case in its **own subprocess with a per-case
timeout**, so a hanging case only kills its own process (it does not wedge the
run), and on a failure it **shrinks** the spec to a minimal repro and writes a
corpus entry.

## Files

| File                           | Role                                                                                                                                                                                                                                                               |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `fuzz.py`                      | Python orchestrator: build, enumerate, run-per-case, classify, shrink, corpus, replay.                                                                                                                                                                             |
| `_fuzz.mojo`                   | Reusable harness: `boundary_int`, value-distribution fills, argv helpers.                                                                                                                                                                                          |
| `fuzz_mha_causal.mojo`         | Target: MHA `CausalPaddingMask` boundary fuzz (memory-safety oracle).                                                                                                                                                                                              |
| `fuzz_mha_nullmask.mojo`       | Target: MHA `NullMask` (non-causal full attention; memory-safety + ref).                                                                                                                                                                                           |
| `fuzz_softmax.mojo`            | Target: `_softmax_gpu` boundary fuzz (+ value-distribution axis; ref / contract / racecheck).                                                                                                                                                                      |
| `fuzz_rms_norm.mojo`           | Target: `rms_norm_gpu` boundary fuzz.                                                                                                                                                                                                                              |
| `fuzz_layer_norm.mojo`         | Target: `layer_norm_gpu` boundary fuzz (memory-safety + numerical ref).                                                                                                                                                                                            |
| `fuzz_matmul.mojo`             | Target: `_matmul_gpu` SM100 tuned bf16 (M runtime; N/K via `-D`); ref = fp32-accum naive.                                                                                                                                                                          |
| `fuzz_block_scaled_fp4.mojo`   | Target: block-scaled FP4 SM100 matmul (small_bn); reproduces a live OOB read.                                                                                                                                                                                      |
| `fuzz_mla_decode.mojo`         | Target: `generic_flare_mla_decode_kv_cache_ragged` ragged paged-KV MLA decode (ref/diff/schedule/memcheck; `-D mla_num_heads=128|64|16`).                                                                                                                          |
| `fuzz_fused_rope_rmsnorm.mojo` | Target: `mla_fused_rope_rmsnorm_quantization` -> `fused_rope_rmsnorm_quantization_kernel`: fused MLA RoPE + paged KV-cache RMSNorm + quantize-write (memcheck/ref; ragged `post_seq_idx` drives the freqs row + paged-store index; `-D frrq_num_heads=128|64|16`). |
| `fuzz_topk_sampling.mojo`      | Target: `fused_token_sampling_gpu` token sampler, gumbel/served + top-k routes (validity contract; diff/poison/initcheck/memcheck).                                                                                                                                |
| `fuzz_moe_indices.mojo`        | Target: `moe_create_indices` (uninitialized-read oracle).                                                                                                                                                                                                          |
| `fuzz_ep_combine.mojo`         | Target: EP MoE combine `send_tokens_back`: fuzzed `src_info` offsets -> memcheck catches the unvalidated wild P2P write (SERVOPT-1458).                                                                                                                            |
| `fuzz_oob_canary.mojo`         | Positive control: a deliberate shape-dependent OOB write that proves the oracle pipeline.                                                                                                                                                                          |
| `fuzz_numeric_canary.mojo`     | Positive control for the `ref` oracle: a deliberate shape-dependent wrong answer.                                                                                                                                                                                  |
| `test_fuzz_fills.mojo`         | CPU unit test for `_fuzz.mojo` (no GPU).                                                                                                                                                                                                                           |
| `repro_decode_hang.mojo`       | Standalone minimal repro probe for the decode-path hang.                                                                                                                                                                                                           |
| `corpus/<target>/*.json`       | Regression corpus — specs + their expected verdict (replayed as a gate).                                                                                                                                                                                           |

## Running

```bash
# Memory-safety fuzz of the MHA causal-padding kernel under memcheck:
python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal \
    --oracle memcheck --budget 32 --seed 12345

# Quick diff-oracle smoke (catches hangs + crashes; no sanitizer, fast):
python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal --oracle diff --budget 24

# Reproduce one explicit case (no generation):
python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal --oracle diff \
    --spec seq_len=1,num_keys=1,valid_length=0

# EP combine write-offset containment under memcheck (SERVOPT-1458). dist=2
# is the one-row-past OOB; dist=0 is the in-bounds control:
python3 max/kernels/test/gpu/fuzz/fuzz.py --target ep_combine \
    --oracle memcheck --spec dist=2

# Fused MLA RoPE + KV-cache RMSNorm: memory-safety sweep, then the dual-output
# numerical oracle (dense roped query + paged K cache read back via its page
# table). The ragged shape drives post_seq_idx = cache_length(b) + token_idx,
# which indexes both freqs_cis and the paged store:
python3 max/kernels/test/gpu/fuzz/fuzz.py --target fused_rope_rmsnorm \
    --oracle memcheck --budget 32 --seed 12345
python3 max/kernels/test/gpu/fuzz/fuzz.py --target fused_rope_rmsnorm \
    --oracle ref --budget 16

# Replay the regression corpus (the deterministic gate):
python3 max/kernels/test/gpu/fuzz/fuzz.py --replay-corpus --timeout 30
```

Oracles (`--oracle`):

- `diff` — hangs (timeout) and crashes (exit code); no sanitizer.
- `ref` — numerical correctness vs a higher-precision reference: FP64 CPU for
  softmax/rms_norm/layer_norm, the naive MHA for attention, an fp32-accum naive
  matmul for `_matmul_gpu`. Emits `FUZZ_NUMERIC_FAIL`.
- `contract` — inject NaN/Inf/large and check a finiteness/propagation contract
  (softmax: every output is NaN or in [0, 1], never Inf/out-of-range). Robust
  where a `ref` tolerance diff would false-positive on NaN/Inf.
- `schedule` — inter-block race check: force a split-K decomposition and re-run
  the same input N times, flagging any non-bit-exact output.
- `redzone` — OOB writes, ~native, AMD-capable (validated on MI355).
- `poison` — uninitialized reads surface as NaN.
- `memcheck` / `initcheck` — Compute Sanitizer with the device pool disabled
  (exact kernel line) for OOB / uninitialized reads.
- `racecheck` / `synccheck` — intra-block shared-memory races / barrier bugs.

Which targets support which extra oracle: `ref` (softmax, rms_norm, layer_norm,
matmul, mha_causal, mha_nullmask, numeric_canary, mla_decode,
fused_rope_rmsnorm), `contract`
(softmax), `schedule` (mha_causal, mla_decode). softmax also fuzzes the input
value distribution (a
`dist` spec field: uniform/normal/sparse/large/all-equal); NaN/Inf specials are
reachable via `--dist 5` but kept out of the auto-mix — they drive the
`contract` oracle, not `ref`.

> Oracle reality: the redzone/poison allocators catch **writes / uninitialized
> reads**; OOB **reads** need `memcheck` with the device pool disabled (the
> orchestrator sets `MEMORY_MANAGER_SIZE=0` in the subprocess env, which works
> because it runs the built binary directly rather than via `bazel test`).

## What it has found

A B200 sweep across the targets (with the redzone allocator also validated on
MI355) found the main kernels clean across the memory-safety, numerical, and
race oracles, plus two live bugs and two reclassifications:

- a robustness **hang** in `flash_attention` + `CausalPaddingMask` decode at the
  degenerate corner `num_keys == 1` and `valid_length == 0` (each alone passes);
- a live **out-of-bounds read** in the block-scaled FP4 `small_bn` matmul,
  reproduced from generated shapes and shrunk to `m == 1` (M-independent);
- a previously-suspected MHA causal-mask OOB read is **fixed** (no longer
  reproduces under memcheck, including its original trigger shape);
- a previously-suspected MoE uninitialized read is a **test-harness artifact**
  (a never-written buffer copied back by the test), not a kernel bug.

The two live findings are recorded under `corpus/` and replayed by the gate.

A later sweep of the ragged paged-KV MLA decode entry point
(`generic_flare_mla_decode_kv_cache_ragged`, the DeepSeek/Kimi production decode
op) found it **clean**: 1145 cases across diff/ref/schedule/memcheck and three
head configs (128/64/16) with zero findings, including the targeted
ragged-last-tile boundaries of the F-1/KERN-2339 OOB class.
Two PASS regression anchors are recorded under `corpus/mla_decode/`.

A sweep of the production token sampler (`fused_token_sampling_gpu`) found
**four live bugs** on B200 — the gumbel/served argmax emits token `-1` for
all-NaN logit rows (any temperature) and for all-`<=0` rows at the unclamped
`temperature == 0` divide; on multi-block shapes (`vocab > 256`) the `p == -1`
sentinel escalates to a memcheck-confirmed OOB read in `_topk_stage2`
(`topk.mojo:1524`) emitting arbitrary garbage token ids; and a per-row
`top_k == 0` leaves its output row unwritten (silent under `diff`; caught by
poison/initcheck).
Four FAIL anchors + one PASS control are recorded under `corpus/topk_sampling/`.

A sweep of the fused MLA RoPE + KV-cache RMSNorm prefill op
(`mla_fused_rope_rmsnorm_quantization` ->
`fused_rope_rmsnorm_quantization_kernel`, the DeepSeek/Kimi op with no isolated
unit test) found it **clean**: across the boundary-rich ragged distribution
(batch size, per-batch cache lengths / start positions, per-batch new-token
counts) it had no hangs/crashes (`diff`), no out-of-bounds memory (`memcheck` +
device-pool-off, including the `post_seq_idx % PAGE_SIZE` page-crossing edge on
both the `freqs_cis` row read and the paged K-cache store), and matched a
higher-precision fp64 CPU reference on **both** outputs (`ref`): the dense
interleaved-complex roped query (`q_rope_output`) and the paged K cache read
back through its page table (the RMSNorm'd latent columns + the roped rope
columns), at `num_q_heads` 16 and 128. Two PASS regression anchors (one
`memcheck` page-cross, one `ref` mid-page cache offset) are recorded under
`corpus/fused_rope_rmsnorm/`.

## Adding a kernel target

The `add-kernel-fuzz-target` Claude Code skill walks this end-to-end — picking
the fuzz axes, oracle, and reference; writing the target; wiring the build; and
validating locally (run `/add-kernel-fuzz-target`, or just ask to fuzz a
kernel). The summary:

1. Write `fuzz_<kernel>.mojo` with a `CaseSpec` (its fuzzable fields) and a
   `run_one_case(ctx, spec)`. Support the three argv modes — `list-specs`
   (print `FUZZ_SPEC <key>=<val> ...`), `single` (read `--<key>` per field, run
   one case, print `FUZZ_RESULT verdict=PASS`), and `fuzz` (in-process batch).
   Reuse `boundary_int` and the argv helpers from `_fuzz.mojo`.
2. Add a `mojo_test` target in `BUILD.bazel` with `srcs = ["_fuzz.mojo",
   "fuzz_<kernel>.mojo"]`, `main = "fuzz_<kernel>.mojo"`, `tags = ["gpu",
   "manual"]`.
3. Register it in `_TARGETS` in `fuzz.py` (name, bazel target, binary path,
   default oracle).

Spec field names == `FUZZ_SPEC` keys == the target's `--<key>` flags, so the
orchestrator drives any target generically.

## CI integration

The tooling is local-first (proven before any CI spend). Two lanes follow the
design's non-gating → gating rollout:

- **Presubmit (gating, fast, deterministic):** build the fuzz targets, then run
  the corpus-replay gate. Same seed/spec → same verdict, so it never flakes; it
  fails only when a verdict drifts (a regression, a fixed bug whose corpus entry
  needs updating, or a broken oracle).

  ```bash
  ./bazelw build //max/kernels/test/gpu/fuzz:all
  python3 max/kernels/test/gpu/fuzz/fuzz.py --replay-corpus --no-build --timeout 30
  ```

- **Nightly (non-gating / notify-only, slow):** a time-boxed live search per
  oracle. New findings file an issue + notify; the lane does not redden `main`.
  Proposed `ci/default/postsubmit.json` lane (apply when ready):

  ```json
  {
    "name": "kernel-fuzz-b200",
    "command": "ci/default/test.sh --config=ci-remote-b200 --config=ci-postsubmit -- python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal --oracle memcheck --budget 200 --seed $BUILDKITE_BUILD_NUMBER",
    "queue": "persistent-b200",
    "soft_fail": true
  }
  ```

  Validate the redzone/poison allocators on MI355 before adding an AMD lane.

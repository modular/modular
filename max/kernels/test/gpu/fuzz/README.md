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
3. **Oracle** — classifies each case as PASS or a specific failure (see
   [Oracles](#oracles)).

The orchestrator runs each case in its **own subprocess with a per-case
timeout**, so a hanging case only kills its own process (it does not wedge the
run). On a failure it **shrinks** the spec to a minimal repro and writes a
corpus entry that the replay gate then locks in.

## Running

The orchestrator drives every target generically. The only required choice is
`--target` (which kernel) and `--oracle` (which bug class); everything else has
a default. Run `--help` to list the currently registered targets — each ships
with a sensible default oracle that `--oracle` overrides.

```bash
# List targets and flags:
python3 max/kernels/test/gpu/fuzz/fuzz.py --help

# Memory-safety fuzz of a kernel under Compute Sanitizer:
python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal \
    --oracle memcheck --budget 32 --seed 12345

# Quick diff-oracle smoke (catches hangs + crashes; no sanitizer, fast):
python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal \
    --oracle diff --budget 24

# Reproduce one explicit case (no generation):
python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal --oracle diff \
    --spec seq_len=1,num_keys=1,valid_length=0

# Replay the regression corpus (the deterministic gate):
python3 max/kernels/test/gpu/fuzz/fuzz.py --replay-corpus --timeout 30
```

Confirmed failures and their shrunk specs are recorded under `corpus/<target>/`
with their expected verdict, and the replay gate re-runs them deterministically.

## Oracles

Select with `--oracle`:

- `diff` — hangs (timeout) and crashes (exit code); no sanitizer.
- `ref` — numerical correctness vs a higher-precision reference (e.g. an FP64
  CPU recompute or an fp32-accum naive kernel). Emits `FUZZ_NUMERIC_FAIL`.
- `contract` — inject NaN/Inf/large and check a finiteness/propagation contract
  (for example: every softmax output is NaN or in `[0, 1]`, never Inf or
  out-of-range). Robust where a `ref` tolerance diff would false-positive on
  NaN/Inf.
- `schedule` — inter-block race check: force a split-K decomposition and re-run
  the same input N times, flagging any non-bit-exact output.
- `redzone` — OOB writes, ~native speed, AMD-capable (validated on MI355).
- `poison` — NaN-fills every device allocation (`MODULAR_DEBUG_DEVICE_ALLOCATOR=
  poison-all`), so an uninitialized read propagates NaN into the output and
  trips the diff/ref check. ~native speed, no kernel instrumentation. (The
  allocator's other tier, `uninitialized-poison`, covers only graph-driver
  tensors and needs an instrumented build, so it does not apply to the fuzz
  harness's directly-allocated device buffers.)
- `memcheck` / `initcheck` — Compute Sanitizer with the device pool disabled
  (exact kernel line) for OOB / uninitialized reads.
- `racecheck` / `synccheck` — intra-block shared-memory races / barrier bugs.

Not every target supports every oracle. Each target declares a default oracle
(its primary bug class) and a `ref`/`contract`/`schedule` mode only where a
reference, special-value contract, or split-K decomposition exists. Targets that
fuzz the input value distribution expose a `dist` spec field
(uniform/normal/sparse/large/all-equal); NaN/Inf specials are reachable but kept
out of the auto-mix — they drive the `contract` oracle, not `ref`.

> Oracle reality: the redzone/poison allocators catch **writes / uninitialized
> reads**; OOB **reads** need `memcheck` with the device pool disabled (the
> orchestrator sets `MEMORY_MANAGER_SIZE=0` in the subprocess env, which works
> because it runs the built binary directly rather than via `bazel test`).

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

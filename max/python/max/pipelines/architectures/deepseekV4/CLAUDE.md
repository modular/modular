# deepseekV4 — agent notes

DeepSeek-V4-Flash-0731. Prefill + decode numerically verified against a vLLM
golden on the minimized 8-layer model (per-cell ≤ 2× the torch reference's own
floor). Not yet: native fp8/fp4 execution, serving/KV-cache integration.
Full records: `max/docs/internal/dsv4-bringup/` (start at `OVERVIEW.md`;
issue/decision numbers below refer to `ISSUES.md` / `DECISIONS.md` there).

## Invariants — do not "clean these up"

- **Positive slice bounds on the last axis, everywhere.** A negative bound on
  a concat result can return a *rotated* tensor (Issue 34, silent wrong
  answers, cost a day). Every `[..., : width - rd]` is written that way on
  purpose and commented.
- **The QAT fake-quant ops are model semantics, not optimization**
  (`fp8_qat_quantize` / `fp4_qat_quantize` on the KV latent, indexer q,
  compressed entries). The checkpoint was trained with them; removing one
  changes the numbers.
- **Rank-3 folds are workarounds, not style.** `sparse_attention`, the
  indexer score matmul, and `_output_projection` fold `(b, s)` or use the
  group axis as batch because (a) the fused-epilogue rank-4 bmm path fails to
  instantiate at N%128==0, K≥128 (Issue 32), and (b) broadcasting `wo_a` gets
  materialized as a 24.75 GB per-layer constant at seq=198. Reverting to the
  "natural" rank-4 forms reintroduces both.
- **`decode_token`, not `decode`**, on `DeepseekV4Attention`:
  `DSparkAttention.decode` already exists with a different signature and the
  mypy LSP aspect rejects the override.
- **Decode skips the indexer** — exact only while
  `max_position // 4 ≤ index_topk` (top-k then keeps every closed entry and
  attention is order-invariant). The 10-step driver asserts the bound; longer
  contexts must bring the indexer (and its compressor state) into decode.
- **Dense MoE dispatch is minimized-model-only.** Every expert sees every
  token; at 256 experts that is 42× the routed work. First thing to replace
  for the real checkpoint (grouped W4A8 kernel exists — PROBE-B G3).
- RoPE: no mscale, and two schedules — compressed layers use
  `compress_rope_theta` + YaRN, window-only layers base theta, YaRN off.
- `self.exported` on the attention is a graph-build-time stash so
  `prefill_with_state` can emit cache-init tensors without changing `__call__`
  signatures; dead code unless the caller outputs it.

## Working on this code

- Build + typecheck:
  `./bazelw build //max/python/max/pipelines/architectures/deepseekV4:deepseekV4`
  (mypy runs as an aspect). Adding a *new* `.py` file requires rebuilding the
  pipelines venv; editing existing files does not.
- Gates were throwaway scripts (not committed); how to reproduce them, the
  golden's location and its gotchas (cold-start run doesn't count; DSpark
  golden is not run-to-run reproducible; logprobs are stored at 1/16
  granularity) are in `PROGRESS.md` and `REFERENCE-NOTES.md`.
- Decode state is caller-held (D17): ring / zone / compressor states are graph
  inputs and outputs; all position-derived indices are host-computed inputs,
  so one compiled decode graph serves every prompt and position. Serving
  integration replaces this with a `compressed` KV group (Issue 30,
  PROBE-A-REPORT).

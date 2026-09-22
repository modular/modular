# deepseekV4 — agent notes

DeepSeek-V4-Flash-0731. Prefill + decode numerically verified against a vLLM
golden on the minimized 8-layer model (per-cell ≤ 2× the torch reference's own
floor), running on MAX's paged KV cache (seven leaves, `layers/cache.py`).
Not yet: native fp8/fp4 execution, the ragged serving graph, a fused
attention kernel over the leaves. Full records:
`max/docs/internal/dsv4-bringup/` (start at `OVERVIEW.md`; issue/decision
numbers below refer to `ISSUES.md` / `DECISIONS.md` there).

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
- **One forward path.** `DeepseekV4Attention.__call__(x, s, cache)` is
  prefill (`cache_lengths == 0`), chunked prefill / prefix-cache resume
  (`> 0`, `s > 1`) and decode (`s == 1`). Every position-derived index is
  computed in-graph from `cache_lengths`; nothing is host-fed. Do not add a
  second "decode" entry point.
- **A compressed zone is a `slots_per_page = page_size // ratio` leaf written
  by the stock ragged store with `cache_lengths // ratio`** (the kernel takes
  the page size from the buffer's static slot dim). Each chunk stores
  `ceil(s / ratio)` candidate entries; the trailing one may be an unclosed
  window and lands on the open slot, which is allocated and unread
  (`layers/csa.py` docstring has the proof). Do not add a conditional store.
- **The compressor is stateless.** Its open state is the last `coff · ratio`
  raw `wkv` / `wgate` rows in a sliding-window leaf (`*_state`, K/V of an MHA
  leaf); `ape` is added at read by slot. Chunk boundaries anywhere are
  therefore exact, which prefix-cache resumes rely on.
- **The indexer runs in decode too**, over `[closed entries from its zone
  leaf, this chunk's candidates]`; `top_k` keeps `min(index_topk, n)`, so at
  bringup lengths it selects everything and is exact by construction.
- **Attention reads the leaves through
  `mo.latent_sparse_attention.ragged.paged`**
  (`max.nn.kernels.latent_sparse_attention_ragged`): window by position from
  `cache_lengths`, zone by the entry numbers `CompressedStream.entries`
  resolves, both leaves already holding the chunk's rows. Window-only layers
  pass the window leaf as the compressed operand with a `[rows, 0]` entry list.
  The compressor state and indexer candidate reads are still `buffer_load` +
  `gather_nd` per layer -- a copy of the leaf per read; do not size a real
  deployment on them. The cache-less path (`cache=None`) keeps the
  gathered-table attention as the gates' reference.
- **Dense MoE dispatch is minimized-model-only.** Every expert sees every
  token; at 256 experts that is 42× the routed work. First thing to replace
  for the real checkpoint (grouped W4A8 kernel exists — PROBE-B G3).
- RoPE: no mscale, and two schedules — compressed layers use
  `compress_rope_theta` + YaRN, window-only layers base theta, YaRN off.
  `apply_rope_tail` takes a `[seq, ...]` or `[batch, seq, ...]` table.

## Working on this code

- Build + typecheck:
  `./bazelw build //max/python/max/pipelines/architectures/deepseekV4:deepseekV4`
  (mypy runs as an aspect). Adding a *new* `.py` file requires rebuilding the
  pipelines venv; editing existing files does not.
- Leaf geometry lives in `model_config.kv_leaf_specs` / `build_kv_params`;
  `DeepseekV4Cache.from_groups` maps the unflattened params tree back to
  leaves in that order. Ratios the model does not use have no leaf.
- Gates were throwaway scripts (not committed); how to reproduce them, the
  golden's location and its gotchas (cold-start run doesn't count; DSpark
  golden is not run-to-run reproducible; logprobs are stored at 1/16
  granularity) are in `PROGRESS.md` and `REFERENCE-NOTES.md`. The cache gates
  drive a `JengaKVCacheManager` exactly as a pipeline does (claim / alloc /
  runtime_inputs / execute / update / step).

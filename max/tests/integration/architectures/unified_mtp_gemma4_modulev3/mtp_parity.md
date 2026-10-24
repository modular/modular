# MTP parity: ModuleV3 (eager) vs graph API (V2) — Gemma 4 31B + assistant

Greedy-decoding parity instructions for `google/gemma-4-31B-it` (BF16) and
`nvidia/Gemma-4-31B-IT-NVFP4` (modelopt NVFP4, `float4_e2m1fnx2`) with the
`google/gemma-4-31B-it-assistant` MTP draft, between the graph-API (V2)
`UnifiedMTPGemma4ForCausalLM` architecture and the ModuleV3
`UnifiedMTPGemma4ForCausalLM_ModuleV3` architecture. Run each command on
the same host/GPUs and diff the generated text; the two stacks must
produce identical token sequences (and identical accepted-token counts)
under greedy sampling.

Both commands use the same fixed prompt set, temperature 0 (greedy), a
fixed seed, 3 speculative tokens, and 64 new tokens. `--prefer-module-v3`
selects the ModuleV3 architecture; its absence selects the graph (V2)
architecture. The MTP flags are identical for both stacks.

## BF16 target, TP1 (single GPU)

```bash
# V2 (graph API)
./bazelw run //max/python/max/_entrypoints:pipelines -- generate \
  --model-path google/gemma-4-31B-it \
  --draft-model google/gemma-4-31B-it-assistant \
  --speculative-method mtp \
  --num-speculative-tokens 3 \
  --devices gpu:0 \
  --temperature 0.0 \
  --seed 0 \
  --max-new-tokens 64 \
  --prompt "The capital of France is"

# V3 (ModuleV3)
./bazelw run //max/python/max/_entrypoints:pipelines -- generate \
  --model-path google/gemma-4-31B-it \
  --draft-model google/gemma-4-31B-it-assistant \
  --speculative-method mtp \
  --num-speculative-tokens 3 \
  --devices gpu:0 \
  --prefer-module-v3 \
  --temperature 0.0 \
  --seed 0 \
  --max-new-tokens 64 \
  --prompt "The capital of France is"
```

## BF16 target, TP2 (two GPUs)

```bash
# V2 (graph API)
./bazelw run //max/python/max/_entrypoints:pipelines -- generate \
  --model-path google/gemma-4-31B-it \
  --draft-model google/gemma-4-31B-it-assistant \
  --speculative-method mtp \
  --num-speculative-tokens 3 \
  --devices gpu:0,1 \
  --temperature 0.0 \
  --seed 0 \
  --max-new-tokens 64 \
  --prompt "The capital of France is"

# V3 (ModuleV3)
./bazelw run //max/python/max/_entrypoints:pipelines -- generate \
  --model-path google/gemma-4-31B-it \
  --draft-model google/gemma-4-31B-it-assistant \
  --speculative-method mtp \
  --num-speculative-tokens 3 \
  --devices gpu:0,1 \
  --prefer-module-v3 \
  --temperature 0.0 \
  --seed 0 \
  --max-new-tokens 64 \
  --prompt "The capital of France is"
```

## NVFP4 target

Repeat each pair above with `--model-path
nvidia/Gemma-4-31B-IT-NVFP4` in place of `google/gemma-4-31B-it` (the
draft checkpoint is unchanged; it stays BF16). See
`gemma4_modulev3/nvfp4_parity.md` for the NVFP4 quantization notes that
apply to the target here as well.

## Additional prompts

Repeat each pair (V2, then V3) for the remaining fixed prompts to
harden the check against single-prompt luck:

```bash
PROMPTS=(
  "The capital of France is"
  "Write a Python function that reverses a string:"
  "Summarize the plot of Romeo and Juliet in one sentence:"
  "1 + 1 ="
  "Explain photosynthesis to a five year old:"
)
```

For each prompt, assert the two outputs (V2 without
`--prefer-module-v3`, V3 with it) are byte-identical at both
`--devices gpu:0` and `--devices gpu:0,1`, and that the reported
accepted-token statistics match. TP1 vs TP2 outputs are expected to be
numerically close but need not match bit-exactly (different reduction
orders).

## Notes

- Both stacks run the same `SequentialDriver` verify/accept/propose
  loop with the same draft: a Q-only cross-attention assistant reading
  the TARGET's KV caches (`DraftCache.TARGET`; the draft allocates no
  cache of its own). Only the execution stack differs (graph API vs
  ModuleV3 `Module.compile`).
- The ModuleV3 model shares the target's `embed_tokens` with the draft
  by module aliasing (one weight, two registry paths), matching the
  graph API's shared weight.
- `--use-greedy-acceptance` (if passed) must be passed to BOTH
  commands; parity is asserted for whatever acceptance mode is chosen,
  as long as the two stacks agree.

# NVFP4 parity: ModuleV3 (eager) vs graph API (V2) — Gemma 4 31B

Greedy-decoding parity instructions for `nvidia/Gemma-4-31B-IT-NVFP4`
(modelopt NVFP4, `float4_e2m1fnx2`) between the graph-API (V2)
`Gemma4ForConditionalGeneration` architecture and the ModuleV3
`Gemma4ForConditionalGeneration_ModuleV3` architecture. Run each command
on the same host/GPUs and diff the generated text; the two stacks must
produce identical token sequences under greedy sampling.

Both commands use the same fixed prompt set, temperature 0 (greedy), a
fixed seed, and 64 new tokens. `--prefer-module-v3` selects the ModuleV3
architecture; its absence selects the graph (V2) architecture.

## TP1 (single GPU)

```bash
# V2 (graph API)
./bazelw run //max/python/max/_entrypoints:pipelines -- generate \
  --model-path nvidia/Gemma-4-31B-IT-NVFP4 \
  --devices gpu:0 \
  --temperature 0.0 \
  --seed 0 \
  --max-new-tokens 64 \
  --prompt "The capital of France is"

# V3 (ModuleV3)
./bazelw run //max/python/max/_entrypoints:pipelines -- generate \
  --model-path nvidia/Gemma-4-31B-IT-NVFP4 \
  --devices gpu:0 \
  --prefer-module-v3 \
  --temperature 0.0 \
  --seed 0 \
  --max-new-tokens 64 \
  --prompt "The capital of France is"
```

## TP2 (two GPUs)

```bash
# V2 (graph API)
./bazelw run //max/python/max/_entrypoints:pipelines -- generate \
  --model-path nvidia/Gemma-4-31B-IT-NVFP4 \
  --devices gpu:0,1 \
  --temperature 0.0 \
  --seed 0 \
  --max-new-tokens 64 \
  --prompt "The capital of France is"

# V3 (ModuleV3)
./bazelw run //max/python/max/_entrypoints:pipelines -- generate \
  --model-path nvidia/Gemma-4-31B-IT-NVFP4 \
  --devices gpu:0,1 \
  --prefer-module-v3 \
  --temperature 0.0 \
  --seed 0 \
  --max-new-tokens 64 \
  --prompt "The capital of France is"
```

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

For each prompt, assert the two outputs (V2 without `--prefer-module-v3`,
V3 with it) are byte-identical at both `--devices gpu:0` and
`--devices gpu:0,1`. TP1 vs TP2 outputs are expected to be numerically
close but need not match bit-exactly (different reduction orders).

## Notes

- The first-party checkpoint keeps every `self_attn` and `lm_head` in
  BF16 (modelopt `ignore` globs); only the dense MLPs are NVFP4.
- ModuleV3 reuses the shared ModuleV3 NVFP4 layers in `common_layers`
  (`QuantizedMLP` / `QuantizedLinear` / `NVFP4Tensor` /
  `quant_ops.matmul`). The fused gate/up matmul is exact for this
  checkpoint: modelopt emits bit-identical `input_scale`/`weight_scale_2`
  for gate/up in each fused group, matching `concat_nvfp4`'s
  first-tensor scale semantics (and making V2's `max()` trivially
  agree).
- KV-cache static scales (`.k_scale`/`.v_scale`) are dropped by the
  shared language converter for both stacks.

# Benchmarking cascade

Measure cascade framework overhead with `benchmark_serving` against a running
cascade server. Echo mode swaps the GPU model for a token-replay worker, so the
numbers isolate the framework path (HF tokenization, cross-pool worker
streaming, incremental detokenization) with no model forward pass. Echo mode
needs no GPU and no `HF_TOKEN` when you use the public `modularai` tokenizer.

## Run the model-benchmarking baseline

Use two terminals. The concurrency (600) and prompt count (10000) match what we
use for model benchmarking.

1. Start the echo-mode server (leave it running):

```bash
./bazelw run --config=production \
  //max/python/max/experimental/cascade/serve:main -- \
  --models.main.model-path echo:modularai/Llama-3.1-8B-Instruct-GGUF \
  --host localhost --port 8000
```

   Wait for `Uvicorn running on http://localhost:8000`. Sanity check:
   `curl -s http://localhost:8000/health` returns `OK`.

2. Run the sharegpt benchmark client:

```bash
./bazelw run --config=production //max/python/max/_entrypoints:benchmark -- \
  benchmark \
  --model modularai/Llama-3.1-8B-Instruct-GGUF \
  --dataset-name sharegpt \
  --num-prompts 10000 \
  --max-concurrency 600 \
  --host localhost --port 8000 \
  --seed 0 \
  --result-filename /tmp/cascade_sharegpt_c600_n10000.json \
  --always-save-result
```

The summary table prints to stdout; full metrics land in `--result-filename`.

## Notes

- Defaults already match the server: `--backend modular`, `--endpoint
  /v1/chat/completions`. Cascade only serves `/v1/chat/completions`.
- The tokenizer (from `--model`) is downloaded from the public
  `modularai/Llama-3.1-8B-Instruct-GGUF` repo; the gated `meta-llama` repo would
  need an `HF_TOKEN`.
- Cascade returns no `usage` field, so the client computes token counts from its
  own tokenizer (it prints a warning to that effect) and the
  `localhost:8001/metrics` scrape warnings are harmless (cascade exposes no
  Prometheus endpoint).
- Vary `--max-concurrency` (accepts a comma-separated sweep, e.g. `1,32,64,600`)
  and `--num-prompts` for other operating points.

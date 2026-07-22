# Cascade optimization log

Running record of cascade framework-overhead optimizations. Each block below is
one change, measured against the same echo-mode sharegpt workload (concurrency
600, 10000 prompts). See [benchmarking.md](benchmarking.md) for the full
procedure.

The client command is invariant across all runs:

```bash
./bazelw run --config=production //max/python/max/_entrypoints:benchmark -- \
  benchmark \
  --model modularai/Llama-3.1-8B-Instruct-GGUF \
  --dataset-name sharegpt --num-prompts 10000 --max-concurrency 600 \
  --host localhost --port 8000 --seed 0 \
  --result-filename /tmp/cascade_sharegpt_c600_n10000.json --always-save-result
```

Append a new block (copy the Baseline block) per change as we go.

## Baseline

(no changes)

Server command:

```bash
./bazelw run --config=production \
  //max/python/max/experimental/cascade/serve:main -- \
  --models.main.model-path echo:modularai/Llama-3.1-8B-Instruct-GGUF \
  --host localhost --port 8000
```

Results:

- Req/s: 108.5
- Output tok/s (mean): 15847
- TTFT mean / p99 (ms): 3337.9 / 5336.4
- E2E mean / p99 (ms): 3551.1 / 5589.8
- TPOT mean (ms): 0.04
- Requests: 9867/10000 succeeded (133 filtered client-side); duration 90.9 s

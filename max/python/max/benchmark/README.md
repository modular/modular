# Benchmark MAX

This directory contains tools to benchmark the performance of an LLM model
server—measuring throughput, latency, and resource utilization. You can use
these scripts to compare other serving backends, namely
[vLLM](https://github.com/vllm-project/vllm), against MAX.

The `benchmark_serving.py` script is adapted from
[vLLM](https://github.com/vllm-project/vllm/blob/main/benchmarks),
licensed under Apache 2.0. We forked this script to ensure consistency with
vLLM's measurement methodology and extended it with features we found helpful,
such as client-side GPU metric collection via `max.diagnostics`.

`benchmark_serving.py` supports:
- text generation
- text-to-image generation
- image-to-image generation via `/v1/responses`

For image-to-image benchmarks, use:
- `--dataset-name local-image --dataset-path /path/to/file.jsonl` for a generic
  local JSONL dataset with `prompt` and `image_path` rows
- `--dataset-name random-image` for a synthetic image-edit workload backed by a
  generated local placeholder image

Example `random-image` run:

```bash
python max/python/max/benchmark/benchmark_serving.py \
  --model black-forest-labs/FLUX.2-dev \
  --endpoint /v1/responses \
  --benchmark-task image-to-image \
  --dataset-name random-image \
  --num-prompts 20 \
  --image-width 1024 \
  --image-height 1024
```

`random-image` is analogous to the placeholder-image path used in diffusion
serving benchmarks such as vLLM-Omni: MAX generates a white PNG in the system
temp directory and reuses it for each request in the run.

For `benchmark_serving.py` usage instructions, see [Benchmarking a MAX
endpoint](/max/docs/max-benchmarking.md).

> [!NOTE]
> This benchmarking script is also available with the `max benchmark` command,
> which you can get by installing `modular` with pip, uv, conda, or pixi
> package managers. Try it now by following the detailed guide to [benchmark
> MAX on GPUs](https://docs.modular.com/max/deploy/benchmark).

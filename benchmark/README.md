# Benchmark MAX

This directory contains tools to benchmark the performance of the
[MAX inference engine](https://docs.modular.com/max/serve/). You can also use
these scripts to compare different LLM serving backends such as
[vLLM](https://github.com/vllm-project/vllm) and
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) against MAX. The
benchmarking tools measure throughput, latency, and resource utilization
metrics.

Key features:

- Tests any OpenAI-compatible HTTP endpoint
- Supports both chat and completion APIs
- Measures detailed latency metrics
- Works with hosted services

> The `benchmark_serving.py` script is adapted from
> [vLLM](https://github.com/vllm-project/vllm/blob/main/benchmarks),
> licensed under Apache 2.0. We forked this script to ensure consistency with
> vLLM's measurement methodology and extended it with features we found helpful,
> such as client-side NVIDIA GPU metric collection via `nvitop`.

## Table of contents

- [Get started](#get-started)
- [Basic usage](#basic-usage)
- [Reference](#reference)
- [Troubleshooting](#troubleshooting)

## Get started

If this is your first time benchmarking a MAX endpoint, we recommend that you
follow our [tutorial to benchmark MAX on a
GPU](https://docs.modular.com/max/tutorials/benchmark-max-serve/).

## Basic usage

You can benchmark any HTTP endpoint that implements
OpenAI-compatible APIs as follows.

First enter the local virtual environment:

```bash
# Either clone main (nightly):
git clone https://github.com/modular/modular.git
# Or clone stable:
# git clone -b stable https://github.com/modular/modular.git

cd modular/benchmark

pixi shell
```

Now install `modular` based on whether you cloned the main or stable branch:

```bash
pixi add pip

# If you cloned main:
pip install --pre modular \
  --index-url https://dl.modular.com/public/nightly/python/simple/

# If you cloned stable:
# pip install modular
```

Then run the benchmark script while specifying your active
MAX endpoint, model, and corresponding dataset to
use for benchmarking (for more detail, see our [benchmarking
tutorial](https://docs.modular.com/max/tutorials/benchmark-max-serve)):

```bash
python benchmark_serving.py \
    --base-url https://company_url.xyz \
    --endpoint /v1/completions \
    --backend modular \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --num-prompts 500
```

To exit the virtual environment shell simply run `exit`.

### Output

Results are in JSON format. If you use the `--save-results` flag, the file is
saved locally with the following naming convention:

```bash
{backend}-{request_rate}qps-{model_name}-{timestamp}.json
```

The output should look similar to the following:

```bash
============ Serving Benchmark Result ============
Successful requests:                     500
Failed requests:                         0
Benchmark duration (s):                  46.27
Total input tokens:                      100895
Total generated tokens:                  106511
Request throughput (req/s):              10.81
Input token throughput (tok/s):          2180.51
Output token throughput (tok/s):         2301.89
---------------Time to First Token----------------
Mean TTFT (ms):                          15539.31
Median TTFT (ms):                        15068.37
P99 TTFT (ms):                           33034.17
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          34.23
Median TPOT (ms):                        28.47
P99 TPOT (ms):                           138.55
---------------Inter-token Latency----------------
Mean ITL (ms):                           26.76
Median ITL (ms):                         5.42
P99 ITL (ms):                            228.45
-------------------Token Stats--------------------
Max input tokens:                        933
Max output tokens:                       806
Max total tokens:                        1570
--------------------GPU Stats---------------------
GPU Utilization (%):                     94.74
Peak GPU Memory Used (MiB):              37228.12
GPU Memory Available (MiB):              3216.25
==================================================
```

### Key metrics explained

- **Request throughput**: Number of complete requests processed per second
- **Input token throughput**: Number of input tokens processed per second
- **Output token throughput**: Number of tokens generated per second
- **TTFT**: Time to first token (TTFT), the time from request start to first
token generation
- **TPOT**: Time per output token (TPOT), the average time taken to generate
each output token
- **ITL**: Inter-token latency (ITL), the average time between consecutive token
or token-chunk generations
- **GPU utilization**: Percentage of time during which at least one GPU kernel
is being executed
- **Peak GPU memory used**: Peak memory usage during benchmark run

## Reference

### Command line arguments for `benchmark_serving.py`

- Backend configuration:
  - `--backend`: Choose from `modular` (MAX `v1/completions` endpoint),
  `modular-chat` (MAX `v1/chat/completions` endpoint), `vllm` (vLLM), or`trt-llm`
  (TensorRT-LLM)
  - `--model`: Hugging Face model ID or local path
- Load generation:
  - `--num-prompts`: Number of prompts to process (`int`, default: `500`)
  - `--request-rate`: Request rate in requests/second (`int`, default: `inf`)
  - `--seed`: The random seed used to sample the dataset (`int`, default: `0`)
- Serving options
  - `--base-url`: Base URL of the API service
  - `--endpoint`: Specific API endpoint (`/v1/completions` or
  `/v1/chat/completions`)
  - `--tokenizer`: Hugging Face tokenizer to use (can be different from model)
  - `--dataset-name`: (default:`sharegpt`) Real-world conversation data in the
  form of variable length prompts and responses. ShareGPT is automatically
  downloaded if not already present.
- Additional options
  - `--collect-gpu-stats`: Report GPU utilization and memory consumption.
  Only works when running `benchmark_serving.py` on the same instance as
  the server, and only on NVIDIA GPUs.
  - `--save-results`: Saves results to a local JSON file.

## Troubleshooting

### Memory issues

- Reduce batch size
- Check GPU memory availability: `nvidia-smi` or `rocm-smi`

### Permission issues

- Verify `HF_TOKEN` is set correctly
- Ensure model access on Hugging Face

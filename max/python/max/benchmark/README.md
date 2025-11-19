# Benchmark MAX

This directory contains tools to benchmark the performance of an LLM model
server—measuring throughput, latency, and resource utilization. You can use
these scripts to compare other serving backends, namely
[vLLM](https://github.com/vllm-project/vllm), against MAX.

> [!NOTE]
> This benchmarking script is also available with the `max benchmark` command,
> which you can get by installing `modular` with pip, uv, conda, or pixi
> package managers. Try it now by following the [MAX
> quickstart guide](https://docs.modular.com/max/get-started).

Key features:

- Tests any OpenAI-compatible HTTP endpoint
- Supports both chat and completion APIs
- Measures detailed latency metrics
- Works with hosted services

> The `benchmark_serving.py` script is adapted from
> [vLLM](https://github.com/vllm-project/vllm/blob/main/benchmarks),
> licensed under Apache 2.0. We forked this script to ensure consistency with
> vLLM's measurement methodology and extended it with features we found helpful,
> such as client-side GPU metric collection via `max.diagnostics`.

## Get started

If this is your first time benchmarking MAX, we suggest you read the following
docs that walk you through the process to create and benchmark an endpoint:

- To benchmark an endpoint created with the `max serve` CLI, read the [MAX
quickstart guide](https://docs.modular.com/max/get-started).

- To benchmark an endpoint created with our Docker container, read the tutorial
to [Benchmark MAX on NVIDIA or AMD
GPUs](https://docs.modular.com/max/tutorials/benchmark-max-serve).

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

Now install `modular` based on whether you cloned the main or stable branch
(because the benchmarking script depends on `max` modules):

```bash
pixi add pip

# If you cloned main:
pip install --pre modular \
  --index-url https://dl.modular.com/public/nightly/python/simple/

# If you cloned stable:
# pip install modular
```

Then run the benchmark script, specifying the model and a dataset:

```bash
python benchmark_serving.py \
  --model google/gemma-3-27b-it \
  --backend modular \
  --endpoint /v1/chat/completions \
  --dataset-name sonnet \
  --num-prompts 500 \
  --sonnet-input-len 512 \
  --output-lengths 256 \
  --sonnet-prefix-len 200
```

To see all the available options, run:

```sh
python benchmark_serving.py --help
```

For more information, see the [`max benchmark`
documentation](/max/cli/benchmark).

## Config files

When you want to save your benchmark configurations, you can define them in a
YAML file and pass it with the `--config-file` option.

The YAML file supports all the same properties as the `benchmark_serving.py`
arguments, except in the YAML file the properties **must use `snake_case`
names** instead of hyphenated names—for example, `--num-prompts` becomes
`num_prompts`. And all properties must be nested under a top level
`benchmark_config` key. For example:

```sh
benchmark_config:
  model: google/gemma-3-27b-it
  backend: modular
  endpoint: /v1/chat/completions
  dataset_name: sonnet
  ...
```

To help you reproduce our own benchmarks, we've made some of our config files
available in the [`configs`](configs/) directory. For example, copy our
[`gemma-3-27b-sonnet-decode-heavy-prefix200.yaml`](https://github.com/modular/modular/tree/main/benchmark/configs)
file from GitHub, and you can benchmark Gemma3-27B with this command:

```sh
max benchmark --config-file gemma-3-27b-sonnet-decode-heavy-prefix200.yaml
```

## Output

Results are printed to the terminal but you can also save a JSON-formatted
file by adding the `--save-result` flag. It's saved to the local directory
with this naming convention:

```bash
{backend}-{request_rate}qps-{model_name}-{timestamp}.json
```

You can change the file name with `--result-filename` and change the directory
with `--result-dir`.

The output in the terminal should look similar to the following:

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

## Troubleshooting

### Memory issues

- Reduce batch size
- Check GPU memory availability: `nvidia-smi` or `rocm-smi`

### Permission issues

- Verify `HF_TOKEN` is set correctly
- Ensure model access on Hugging Face

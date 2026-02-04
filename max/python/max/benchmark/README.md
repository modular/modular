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

For `benchmark_serving.py` usage instructions, see [Benchmarking a MAX
endpoint](/max/docs/max-benchmarking.md).

> [!NOTE]
> This benchmarking script is also available with the `max benchmark` command,
> which you can get by installing `modular` with pip, uv, conda, or pixi
> package managers. Try it now by following the detailed guide to [benchmark
> MAX on GPUs](https://docs.modular.com/max/deploy/benchmark).

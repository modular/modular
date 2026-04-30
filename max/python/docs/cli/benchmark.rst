:title: max benchmark

.. raw:: markdown

    Runs comprehensive benchmark tests on an active model server to measure
    performance metrics including throughput, latency, and resource utilization.
    For a complete walkthrough, see the tutorial to [benchmark MAX on a
    GPU](/max/deploy/benchmark).

    Before running this command, make sure the model server is running, via [`max
    serve`](/max/cli/serve).

    For example, here's how to benchmark the `google/gemma-3-27b-it` model
    already running on localhost:

    ```sh
    max benchmark \
      --model google/gemma-3-27b-it \
      --backend modular \
      --endpoint /v1/chat/completions \
      --num-prompts 50 \
      --dataset-name arxiv-summarization \
      --arxiv-summarization-input-len 12000 \
      --max-output-len 1200
    ```

    When it's done, you'll see the results printed to the terminal.

    By default, it sends inference requests to `localhost:8000`, but you can change
    that with the `--host` and `--port` arguments.

    To save the results to a JSON file, set `--result-filename` to the path you
    want (the value can include a directory, which is created if needed):

    ```sh
    max benchmark ... --result-filename results/gemma-run.json
    ```

    Instead of passing all these benchmark options, you can pass a configuration
    file. See [Configuration file](#benchmark-configuration-file) below.

    :::note

    The `max benchmark` command is a convenient packaging for our open-source
    [`benchmark_serving.py`](https://github.com/modular/modular/tree/main/max/python/max/benchmark#benchmark-max)
    script and accepts all the same options.

    :::

    ## Usage

    Run `max benchmark` with one or more options:

    ```sh
    max benchmark [OPTIONS]
    ```

    ## Options

    The full option list is long. The most useful options group as follows. For
    everything else, run `max benchmark --help` or see the [benchmarking script
    source code](https://github.com/modular/modular/tree/main/max/python/max/benchmark).

    - Backend configuration:

      - `--backend`: Server type to benchmark. Choices: `modular`,
        `modular-chat`, `vllm`, `vllm-chat`, `sglang`, `sglang-chat`,
        `trtllm`, `trtllm-chat`. Default: `modular`.

      - `--model`: Hugging Face model ID or local path.

      - `--endpoint`: Specific API endpoint, such as `/v1/completions` or
        `/v1/chat/completions`. Default: `/v1/chat/completions`.

      - `--base-url`: Base URL of the API service. Overrides `--host` and
        `--port` when set.

      - `--host`: Server host. Default: `localhost`.

      - `--port`: Server port. Default: `8000`.

      - `--tokenizer`: Hugging Face tokenizer to use. Defaults to the model's
        tokenizer.

    - Load generation:

      - `--num-prompts`: Number of prompts to process. Default: unset (driven by
        the dataset and duration).

      - `--request-rate`: Requests per second. Accepts a single value or a
        comma-separated sweep (such as `1,2,4,8`). Default: `inf` (no rate
        limit).

      - `--max-concurrency`: Maximum concurrent requests. Accepts a single
        integer or a comma-separated sweep.

      - `--seed`: Random seed used to sample the dataset. Default: `0`.

    - Dataset selection:

      - `--dataset-name`: Dataset to benchmark on. Determines the dataset class
        and processing logic. Default: `sharegpt`. See [Datasets](#datasets)
        below.

      - `--dataset-path`: Path to a local dataset file that overrides the
        default source for the chosen `--dataset-name`.

    - Output control:

      - `--max-output-len`: Maximum output length per request, in tokens.

      - `--temperature`, `--top-p`, `--top-k`: Sampling parameters forwarded to
        the server.

    - LoRA traffic:

      - `--lora`: Optional LoRA name to send with each request.

      - `--lora-paths`: Paths to existing LoRA adapters. Each entry is either
        `path` or `name=path`.

      - `--lora-uniform-traffic-ratio`: Probability (between `0.0` and `1.0`)
        that any given request targets a randomly selected LoRA instead of the
        base model. Default: `0.0`.

      - `--per-lora-traffic-ratio`: Per-adapter traffic ratios, in the same
        order as `--lora-paths`. Sum must not exceed `1.0`; the remainder goes
        to the base model. Overrides `--lora-uniform-traffic-ratio` when set.

      - `--max-concurrent-lora-ops`: Maximum concurrent LoRA load and unload
        operations. Default: `1`.

    - Result saving:

      - `--result-filename`: Path to a JSON file for benchmark results. When
        unset, no file is written. The path may include directories that the
        command creates if they don't exist.

      - `--metadata`: Key-value pairs (such as `--metadata version=0.3.3 tp=1`)
        recorded alongside the run in the result JSON.

      - `--log-dir`: Directory for log output. Default:
        `<backend>-latency-Y.m.d-H.M.S`.

    - Stats collection:

      - `--collect-gpu-stats` / `--no-collect-gpu-stats`: Report GPU utilization
        and memory consumption (NVIDIA only). Enabled by default. Only works
        when `max benchmark` runs on the same instance as the server.

      - `--collect-cpu-stats` / `--no-collect-cpu-stats`: Report CPU stats.
        Enabled by default.

      - `--collect-server-stats` / `--no-collect-server-stats`: Report server
        stats. Enabled by default.

    - Configuration file:

      - `--config-file`: Path to a YAML file containing all benchmark options.
        Replaces individual command line flags. See [Configuration
        file](#benchmark-configuration-file) below.

    ### Datasets

    The `--dataset-name` option supports the following datasets. For any
    dataset that has configurable flags, those flags are listed inline.

    You can override the default data source for most datasets using
    `--dataset-path`. You must always set `--dataset-name` so the tool knows
    how to process the file.

    #### Text

    - `sharegpt` (default): Conversational dataset with human-AI exchanges,
      from Hugging Face Datasets.

    - `axolotl`: Local dataset in Axolotl format with human/assistant
      conversation segments. Pair with `--dataset-path`.

    - `obfuscated-conversations`: Local obfuscated conversation dataset. Pair
      with `--dataset-path` to point at a local JSONL file.

      - `--obfuscated-conversations-average-output-len`: Average output length
        when per-request output lengths are not provided. Default: `175`.
      - `--obfuscated-conversations-coefficient-of-variation`: Coefficient of
        variation for output length. Default: `0.1`.
      - `--obfuscated-conversations-shuffle` /
        `--no-obfuscated-conversations-shuffle`: Shuffle the dataset.
        Disabled by default.

    - `arxiv-summarization`: Research paper summarization dataset, from Hugging
      Face Datasets.

      - `--arxiv-summarization-input-len`: Input tokens per request.
        Default: `15000`.

    - `sonnet`: Poetry dataset using local text files of poem lines.

      - `--sonnet-input-len`: Input tokens per request. Default: `550`.
      - `--sonnet-prefix-len`: Shared prefix tokens per request. Default: `200`.

    - `random`: Synthetically generated dataset with configurable token
      distributions.

      - `--random-input-len`: Input tokens per request. Accepts a constant or a
        distribution string: `N(mean,std)`, `U(lower,upper)`, `DU(lower,upper)`,
        `NB(n,p)`, `G(shape,scale)`, or `LN(mean,std)`. Use `;` to set
        separate distributions for the first and subsequent turns (for example,
        `N(2048,200);N(512,50)`). Default: `1024`.
      - `--random-output-len`: Output tokens per request. Same format as
        `--random-input-len`. Default: `128`.
      - `--random-num-turns`: Turns per session. Same format as
        `--random-input-len`. Default: `1`.
      - `--random-sys-prompt-ratio`: Fraction of the input length to use as a
        system prompt. Range: `0.0`–`1.0`. Default: `0.0`.
      - `--random-max-num-unique-sys-prompt`: Maximum number of distinct system
        prompts to generate. Default: `1`.
      - `--warm-shared-prefix` / `--no-warm-shared-prefix`: Send each unique
        shared prefix as a single-token request before the run to prime
        prefix-cache KV entries. Requires `--random-sys-prompt-ratio > 0`.
        Disabled by default.
      - `--random-image-count`: Images to attach per request (enables vision
        mode on this dataset). Default: `0`.
      - `--random-image-size`: Pixel dimensions of generated images (for
        example, `512x512`). Used with `--random-image-count`.

    - `synthetic`: Synthetic text generation workload with multiturn support.
      Also supports `--warm-shared-prefix` (see `random` above).

    #### Code

    - `instruct-coder`: Instruction-following coding dataset with multiturn
      support.

    - `agentic-code`: Multiturn agentic coding workload with tool-call turns.

    - `code_debug`: Long-context code debugging dataset with multiple-choice
      questions, from Hugging Face Datasets.

    #### Vision

    - `batch-job`: Batch image workload.

      - `--batch-job-image-dir`: Directory where the server can access images
        (file reference mode). When unset, images are embedded as base64.

    - `local-image`: Local images for vision benchmarks. Pair with
      `--dataset-path`.

    - `vision-arena`: Vision-language benchmark dataset with images and
      associated questions for multimodal model evaluation, from Hugging Face
      Datasets.

    - `synthetic-pixel`: Synthetic pixel-generation workload for image-output
      backends.

    ### Configuration file {#benchmark-configuration-file}

    The `--config-file` option points at a YAML file containing all benchmark
    options as a replacement for individual command line flags. Define every
    option (corresponding to a `max benchmark` flag) under a top-level
    `benchmark_config` key.

    :::caution

    In the YAML file, the properties **must use `snake_case` names** instead of
    the hyphenated names from the command line. For example, `--num-prompts`
    becomes `num_prompts`.

    :::

    For example, instead of specifying configurations on the command line like
    this:

    ```sh
    max benchmark \
      --model google/gemma-3-27b-it \
      --backend modular \
      --endpoint /v1/chat/completions \
      --host localhost \
      --port 8000 \
      --num-prompts 50 \
      --dataset-name arxiv-summarization \
      --arxiv-summarization-input-len 12000 \
      --max-output-len 1200
    ```

    Create this configuration file:

    ```yaml title="gemma-benchmark.yaml"
    benchmark_config:
      model: google/gemma-3-27b-it
      backend: modular
      endpoint: /v1/chat/completions
      host: localhost
      port: 8000
      num_prompts: 50
      dataset_name: arxiv-summarization
      arxiv_summarization_input_len: 12000
      max_output_len: 1200
    ```

    Then run the benchmark by passing that file:

    ```sh
    max benchmark --config-file gemma-benchmark.yaml
    ```

    For more config file examples, see our [benchmark configs on
    GitHub](https://github.com/modular/modular/tree/main/max/python/max/benchmark/configs).

    For a walkthrough of setting up an endpoint and running a benchmark, see
    the [quickstart guide](/max/get-started).

    ## Output

    Each run prints the following metrics on completion:

    - **Request throughput**: number of complete requests processed per second.
    - **Input token throughput**: number of input tokens processed per second.
    - **Output token throughput**: number of tokens generated per second.
    - **TTFT** (time to first token): time from request start to first token
      generation.
    - **TPOT** (time per output token): average time taken to generate each
      output token.
    - **ITL** (inter-token latency): average time between consecutive token or
      token-chunk generations.

    When `--collect-gpu-stats` is enabled, the run also reports:

    - **GPU utilization**: percentage of time during which at least one GPU
      kernel is executing.
    - **Peak GPU memory used**: peak memory usage during the benchmark run.

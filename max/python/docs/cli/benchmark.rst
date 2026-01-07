:title: max benchmark

.. raw:: markdown

    Runs comprehensive benchmark tests on an active model server to measure
    performance metrics including throughput, latency, and resource utilization.

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

    If you want to save the results, add the `--save-result` option, which creates
    a JSON file in the local path with the following naming convention:

    ```bash
    {backend}-{request_rate}qps-{model_name}-{timestamp}.json
    ```

    But you can specify the file name with `--result-filename` and change the
    directory with `--result-dir`.

    Instead of passing all these benchmark options, you can instead pass a
    configuration file. See [Configuration file](#benchmark-configuration-file)
    below.

    :::note

    The `max benchmark` command is a convenient packaging for our open-source
    [`benchmark_serving.py`](https://github.com/modular/modular/tree/main/benchmark#benchmark-max)
    script and accepts all the same options.

    :::

    ## Usage

    ```sh
    max benchmark [OPTIONS]
    ```

    ## Options

    This list of options is not exhaustive. For more information, run `max
    benchmark --help` or see the [benchmarking script source
    code](https://github.com/modular/modular/tree/main/benchmark).

    - Backend configuration:

      - `--backend`: Choose from `modular` (MAX `v1/completions` endpoint),
      `modular-chat` (MAX `v1/chat/completions` endpoint), or `vllm` (vLLM)

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

      - `--dataset-name`: (Required; default:`sharegpt`) Specifies which type of
      benchmark dataset to use. This determines the dataset class and processing
      logic. See [Datasets](#datasets) below.

      - `--dataset-path`: Path to a local dataset file that overrides the default
      dataset source for the specified `dataset-name`. The file format must match
      the expected format for the specified `dataset-name` (such as JSON for
      `axolotl`, JSONL for `obfuscated-conversations`, plain text for `sonnet`).

    - Additional options

      - `--collect-gpu-stats`: Report GPU utilization and memory consumption
        for both NVIDIA and AMD GPUs. Only works when running `max benchmark`
        on the same instance as the server.

      - `--save-results`: Saves results to a local JSON file.

    - LoRA benchmarking options

      The benchmark script supports testing LoRA adapter performance for
      supported models and target modules:

      - `--num-loras`: Number of LoRA adapters to test. If > 0, test LoRA
        adapters will be generated.
      - `--lora-rank`: LoRA rank (r parameter) for generated adapters. Controls
        the dimension of the low-rank decomposition.
      - `--lora-output-dir`: Directory to save generated LoRA adapters.
        Defaults to `/tmp/loras`.
      - `--lora-paths`: Paths to existing LoRA adapters to use instead of
        generating new ones.
      - `--lora-request-ratio`: Ratio of requests to send with LoRA adapters
        (0.0-1.0). For example, 0.5 means 50% of requests use LoRA.
      - `--max-num-loras`: Maximum number of LoRA adapters cached on GPU.
        This should match the server configuration.
      - `--lora-target-modules`: List of module names to apply LoRA to when
        generating random test adapters (e.g., `q_proj`, `k_proj`, `v_proj`,
        `o_proj`). Only used when `--num-loras` > 0 and generating adapters
        (not when using existing `--lora-paths`).

      - `--config-file`: Path to a YAML file containing benchmark configuration.
        The configuration file is a YAML file that contains key-value pairs for all
        your benchmark configurations (as a replacement for individual command line
        options). See [Configuration file](#benchmark-configuration-file) below.

    ### Datasets

    The `--dataset-name` option supports several dataset names/formats you can
    use for benchmarking:

    - `arxiv-summarization` - Research paper summarization dataset containing
    academic papers with abstracts for training summarization models, from Hugging
    Face Datasets.

    - `axolotl` - Local dataset in Axolotl format with conversation segments
    labeled as human/assistant text, from Hugging Face Datasets.

    - `code_debug` - Long-context code debugging dataset containing code with
    multiple choice debugging questions for testing long-context understanding,
    from Hugging Face Datasets.

    - `obfuscated-conversations` - Local dataset with obfuscated conversation data.
    You must pair this with the `--dataset-path` option to specify the local JSONL
    file.

    - `random` - Synthetically generated random dataset that creates random
    token sequences with configurable input/output lengths and distributions.

    - `sharegpt` - Conversational dataset containing human-AI conversations for
    chat model evaluation, from Hugging Face Datasets.

    - `sonnet` - Poetry dataset using local text files containing poem lines,
    from Hugging Face Datasets.

    - `vision-arena` - Vision-language benchmark dataset containing images with
    associated questions for multimodal model evaluation, from Hugging Face
    Datasets.

    You can override the default dataset source for any of these using the
    `--dataset-path` option (except for generated datasets like `random`), but you
    must always specify a `--dataset-name` so the tool knows how to process the
    dataset format.

    ### Configuration file {#benchmark-configuration-file}

    The `--config-file` option allows you to specify a YAML file containing all
    your benchmark configurations, as a replacement for individual command line
    options. Simply define all the configuration options (corresponding to the `max
    benchmark` command line options) in a YAML file, all nested under the
    `benchmark_config` key.

    :::caution

    In the YAML file, the properties **must use `snake_case` names** instead of
    using the hyphenated names from the command line options. For example,
    `--num-prompts` becomes `num_prompts`.

    :::

    For instance, instead of specifying all configurations in the command line like
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

    And then run the benchmark by passing that file:

    ```sh
    max benchmark --config-file gemma-benchmark.yaml
    ```

    For more config file examples, see our [benchmark configs on
    GitHub](https://github.com/modular/modular/tree/main/benchmark/configs).

    For a walkthrough of setting up an endpoint and running a benchmark, see the
    [quickstart guide](/max/get-started).

    ## Output

    Here's an explanation of the most important metrics printed upon completion:

    - **Request throughput**: Number of complete requests processed per second
    - **Input token throughput**: Number of input tokens processed per second
    - **Output token throughput**: Number of tokens generated per second
    - **TTFT**: Time to first token—the time from request start to first
    token generation
    - **TPOT**: Time per output token—the average time taken to generate
    each output token
    - **ITL**: Inter-token latency—the average time between consecutive token
    or token-chunk generations

    If `--collect-gpu-stats` is set, you'll also see these:

    - **GPU utilization**: Percentage of time during which at least one GPU kernel
    is being executed
    - **Peak GPU memory used**: Peak memory usage during benchmark run

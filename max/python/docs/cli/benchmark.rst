:title: max benchmark

.. raw:: markdown

    Runs comprehensive benchmark tests on an active model server to measure
    performance metrics including throughput, latency, and resource utilization.
    For a complete walkthrough, see the tutorial to [benchmark MAX on a
    GPU](/serve/benchmark).

    Before running this command, make sure the model server is running, via [`max
    serve`](/cli/serve).

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

    Each run also writes a log directory (`sweep-serving-<timestamp>` by
    default, or the path you set with `--log-dir`) holding a `results.csv`
    file and one result JSON file per `--max-concurrency` value.

    To also save the results to a JSON file at a path you choose, set
    `--result-filename` (the value can include a directory, which is created
    if needed):

    ```sh
    max benchmark ... --result-filename results/gemma-run.json
    ```

    Instead of passing all these benchmark options, you can pass a configuration
    file. See [Configuration file](#benchmark-configuration-file) below.

    :::note

    The `max benchmark` command is a convenient packaging for our open-source
    [`benchmark_serving.py`](https://github.com/modular/modular/tree/main/max/python/max/benchmark#benchmark-max)
    script and accepts all the same options, plus the `--profile` options
    described below.

    :::

    ## Usage

    Run `max benchmark` with one or more options:

    ```sh
    max benchmark [OPTIONS]
    ```

    You can also set any option with an environment variable named `MODULAR_`
    plus the option name in upper snake case. For example,
    `MODULAR_NUM_PROMPTS=50` is the same as `--num-prompts 50`.

    ## Options

    The options are grouped into the following sections. To load options from
    a YAML file with `--config-file`, see [Configuration
    file](#benchmark-configuration-file). To see the options from the command
    line, run `max benchmark --help`, or see the [benchmarking script source
    code](https://github.com/modular/modular/tree/main/max/python/max/benchmark).

    ### Backend configuration

    - `--backend`: Server type to benchmark. Choices: `modular`,
      `modular-chat`, `vllm`, `vllm-chat`, `sglang`, `sglang-chat`,
      `trtllm`, `trtllm-chat`, `atom`, `atom-chat`, `mach`, `mcloud`.
      Default: `modular`.

    - `--model`: Hugging Face model ID or local path.

    - `--endpoint`: API endpoint to send requests to. Choices:
      `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
      `/v1/images/generations`, `/v1/videos`, `/v1/videos/sync`,
      `/v2/models/ensemble/generate_stream`. Default:
      `/v1/chat/completions`.

    - `--benchmark-task`: Type of workload to send. Choices:
      `text-generation`, `text-to-image`, `image-to-image`,
      `text-to-video`, `image-to-video`. Default: `text-generation`. For
      the image and video tasks, the endpoint is chosen from the backend
      when you don't set `--endpoint`. See [Pixel
      generation](#pixel-generation).

    - `--num-gpus`: Number of GPUs the server uses. Set this to also report
      throughput per GPU.

    - `--base-url`: Base URL of the API service. Overrides `--host` and
      `--port` when set. Requests carry the `OPENAI_API_KEY` environment
      variable as a bearer token.

    - `--host`: Server host. Default: `localhost`.

    - `--port`: Server port. Default: `8000`.

    - `--tokenizer`: Hugging Face tokenizer to use. Defaults to the model's
      tokenizer.

    - `--tokenizer-revision`: Commit revision to load the tokenizer at.

    - `--tokenizer-local-files-only`: Load the tokenizer from the local
      Hugging Face cache only.

    - `--model-max-length`: Override the tokenizer's maximum length. Use this
      when the server's context limit is lower than the tokenizer's.

    - `--trust-remote-code`: Trust remote code when loading the tokenizer
      from Hugging Face.

    ### Load generation

    - `--num-prompts`: Number of single-turn prompts to process. With
      `--num-chat-sessions`, it instead caps the total number of requests
      (turns) sent across all sessions. Default: `1000`, unless you set
      `--num-prompts-multiplier`.

    - `--num-chat-sessions`: Number of multiturn chat sessions to drive.
      Required for `chat-judge`. Use with multiturn-capable datasets
      (`random`, `synthetic`, `chat-judge`, `code_debug`, `instruct-coder`,
      `agentic-code`, and `nemotron-opencode`) instead of `--num-prompts`.
      Turns per session are dataset-specific; see
      [Dataset selection](#dataset-selection).

    - `--num-prompts-multiplier`: Set the number of prompts for each
      `--max-concurrency` value to this multiplier times that concurrency.
      Can't be combined with `--num-prompts`, and every `--max-concurrency`
      value must be bounded.

    - `--max-benchmark-duration-s`: Maximum duration of each run, in
      seconds. A run stops when it has sent all its prompts or reaches this
      limit, whichever comes first. Default: `300` when you set
      `--num-prompts-multiplier` or don't set `--num-prompts`; otherwise no
      limit.

    - `--request-rate`: Requests per second. Accepts a single value or a
      comma-separated sweep (such as `1,2,4,8`). Default: `inf` (no rate
      limit).

    - `--burstiness`: Burstiness of request arrivals when `--request-rate`
      is finite. `1.0` is a Poisson process. Values below `1.0` make
      arrivals burstier and values above it make them more uniform.
      Default: `1.0`.

    - `--max-concurrency`: Maximum concurrent requests. Accepts a single
      integer or a comma-separated sweep. Use `none` for no limit (for
      example, `1,8,none`). Default: no limit.

    - `--max-concurrent-conversations`: Maximum number of multiturn chat
      sessions open at once, for KV cache stress tests. Each session runs to
      completion before another starts, so all open sessions keep their KV
      cache on the server. Set `--max-concurrency` no higher than this value
      to cap the turns in flight.

    - `--seed`: Random seed for the workload generator (input/output lengths,
      session structure, and content). Default: `24301` (fixed for
      reproducibility). Pass `--seed none` (or `seed: null` in a workload YAML)
      to draw a fresh random seed; the drawn value is logged and recorded with
      the results.

    - `--kv-block-size`: KV cache block size in tokens for the per-turn cache
      retention metric. Default: `128`. Should match the server's
      `--kv-cache-page-size` so the retention metric is accurate; a mismatch
      does not affect the benchmark run itself.

    - `--fit-distributions`: Reshape multiturn workloads using the `random_*`
      flags and `--delay-between-chat-turns`. Requires `--num-chat-sessions`
      with `instruct-coder`, `agentic-code`, or `nemotron-opencode`. Turn
      count comes from `--random-num-turns` (see the `random` dataset below).

    - `--agentic-tool-profiles`: Append an agent loop after each human
      turn, synthesized from per-tool distributions rather than replayed.
      A profile is the load one tool call produces, not a tool the model may
      invoke -- nothing here is sent as the OpenAI `tools` field, which is
      `--tool-calls` / `--no-tool-calls` on `nemotron-opencode`. Requires
      `--fit-distributions`, `--num-chat-sessions`, and
      `--agentic-rounds-per-turn`. Takes either a path to a YAML file or an
      inline YAML/JSON mapping -- the inline form is the file's own content,
      so there is one shape to learn:

      ```sh
      --agentic-tool-profiles '{"tools":[{"input-len":"N(200,80)","output-len":"N(30,10)"}]}'
      ```

      A file is easier once there is more than one tool:

      ```yaml
      tools:
        - weight: 5              # a shell-like tool: small result, fast
          input-len: N(30,20)
          output-len: N(25,8)
          delay: N(50,20)
        - weight: 2              # a file read: large result, slower
          input-len: LN(7.5,1.2)
          output-len: N(60,20)
          delay: N(300,120)
      ```

      A session then runs as:

      ```text
      S U0 (A1* U1*  A2* U2*  ...  Ax* Ux*) A0   U1 A1   ...
      ```

      Starred messages are the loop: `Aj*` is the call to tool `j`, `Uj*` is
      that tool's result. Each `Uj*` is one request. Delete the parentheses
      and `S U0 A0 U1 A1` is exactly what a turn with no rounds produces.
      Each round picks a tool by `weight`, which is relative rather than a
      probability and defaults to 1, so omitting it everywhere gives a fair
      share and adding a tool never requires rebalancing the others.

      Per tool, `input-len` is the result it returns (`Uj*`) and `delay` is
      how long it took. `output-len` is the assistant message that *invokes*
      it (`Aj*`) -- a tool call, so usually small -- because that is where
      the decode actually sits. `--random-output-len` describes `A0`, the
      prose answer that closes the block, since that is the reply the human
      sees.

      Tool results go on the wire as `role: user`, which keeps strict
      user/assistant alternation and renders on every chat template. Human
      turns keep using the `random_*` flags, so
      `--random-input-len "first;rest"` still shapes first-versus-remaining
      human turns; `;` is rejected inside a per-tool value, where it would
      rebind on reorder.

      Supported on `instruct-coder`, `agentic-code`, and `nemotron-opencode`
      with `--fit-distributions` (not on `random` or `synthetic`). On
      `agentic-code`, rounds draw their bodies from recorded `tool`
      messages; the other two reuse the user-text pool.

    - `--agentic-rounds-per-turn`: How many agent-loop rounds follow each human
      turn, sampled once per turn. Accepts a constant or a
      distribution string. Requires `--agentic-tool-profiles`.

    - `--delay-between-chat-turns`: Delay between chat turns in
      milliseconds. Accepts a constant or a distribution string (same format as
      `--random-input-len`). With `--agentic-tool-profiles`, this is the
      human think-time: rounds use their own per-tool `delay`, and only the
      reply that closes a block waits this long.

    - `--chat-warmup-delay-ms`: Delay between starting chat sessions, in
      milliseconds. Default: `0`.

    - `--force-unique-runs`: Prepend one random prefix, shared by every
      request in the run, to each prompt. Requests in the run can still
      share prefix cache entries, but a later run can't reuse this run's
      cache. Not supported for `image-to-image`.

    - `--use-session-id-as-cache-salt`: Send each multiturn chat session's
      ID as the `X-Cache-Salt` header on all of that session's requests, to
      measure per-session KV cache isolation in MAX.

    - `--strip-harmony-wire-tags`: For multiturn workloads, remove Harmony
      wire tags (such as `<|channel|>`) from each response before it is sent
      back as chat history. A Harmony chat template rejects these tags in
      message text.

    - `--workload-config`: YAML file specifying benchmark workload options
      (hyphenated keys such as `num-prompts` and `seed`). CLI flags override
      values from this file. Relative file paths in the YAML resolve against
      the YAML file's directory. Using a workload file also turns on
      `--skip-test-prompt`.

    ### Warmup and measurement window

    - `--skip-test-prompt`: Skip the single test request sent before the
      benchmark. Useful for external profiling.

    - `--skip-first-n-requests`, `--skip-last-n-requests`: Number of
      requests to leave out of the metrics at the start and end of each run.
      When unset and the run has no rate limit and a `--max-concurrency`
      above 1, each defaults to the `--max-concurrency` value, which
      excludes the ramp up and ramp down. Otherwise, each defaults to `0`.

    - `--steady-state-filtering` / `--no-steady-state-filtering`: Report
      metrics over an automatically detected steady-state window, and
      reject per-request latency outliers. With
      `--no-steady-state-filtering`, metrics cover every successful request
      left after the skip options. Enabled by default.

    - `--ignore-first-turn-stats`: Leave the first turn of each multiturn
      chat session out of the metrics.

    - `--warmup-multiturn` / `--no-warmup-multiturn`: For multiturn
      workloads, start each chat session at a random later turn and replay
      its earlier turns as warmup, so the measured run begins with the turn
      mix of a long-running server. Enabled by default. Formerly
      `--warmup-to-steady-state`, which still works.

    - `--warmup-oversample-factor`: Number of candidate sessions per warmup
      slot. `0` disables length-biased warmup, `1` picks warmup turns
      uniformly at random, and `2` or more biases picks toward longer
      sessions. Default: `8`.

    - `--warmup-delay-biased`: Bias warmup selection by the total delay
      between turns instead of turn count. Needs `--delay-between-chat-turns`.

    - `--warmup-delay-estimated-ttft-ms`,
      `--warmup-delay-estimated-tpot-ms`: Estimated TTFT and TPOT in
      milliseconds. With these, `--warmup-delay-biased` weights each turn by
      its estimated generation time plus delay. Require
      `--warmup-delay-biased`. Default: `0` (off).

    - `--warmup-concurrency`: Maximum requests in flight during
      `--warm-shared-prefix` priming and multiturn warmup. Default: `128`.

    ### Dataset selection

    - `--dataset-name`: Dataset to benchmark on. Determines the dataset class
      and processing logic. Default: `sharegpt`. The supported datasets
      are listed below.

    - `--dataset-path`: Path to a local dataset file. Supported only for
      datasets that accept a local override, as listed below.

    The `--dataset-name` option supports the following datasets. For any
    dataset that has configurable flags, those flags are listed inline.

    Some datasets download from Hugging Face Hub or Hugging Face Datasets
    automatically. Others require `--dataset-path` or generate prompts in
    memory. Only `axolotl`, `sonnet`, `chat-judge`, `obfuscated-conversations`,
    `instruct-coder`, `batch-job`, and `local-image` read `--dataset-path`.
    `nemotron-opencode` rejects it, and the other datasets ignore it.

    #### Text

    - `sharegpt` (default): Conversational dataset with human-AI exchanges,
      from Hugging Face Hub (`anon8231489123/ShareGPT_Vicuna_unfiltered`).

    - `artificial-analysis`: Prompts modeled on the Artificial Analysis
      performance benchmark. Each request pairs arXiv article text, from
      Hugging Face Datasets (`ccdv/arxiv-summarization`), with one of several
      tasks such as summarization or Q&A generation, trimmed to a target
      input length. Set lengths with `--random-input-len` and
      `--random-output-len` (see the `random` dataset below).

    - `axolotl`: Dataset in Axolotl format with human/assistant conversation
      segments. Uses a packaged default file; override with `--dataset-path`.

    - `chat-judge`: LLM-as-judge multiturn workload backed by a local JSONL
      session file. Each turn inlines prior context in the user message, so the
      driver sends `[system?, user]` per turn without accumulating assistant
      responses. Requires `--dataset-path` and `--num-chat-sessions` (single-turn
      mode isn't supported).

      Example session (each session goes on one line in the file; this one
      is expanded for readability):

      ```json
      {
        "session_id": "s1",
        "turns": [
          {"text": "You are a safety judge.", "role": "system"},
          {"text": "Rate this content: ..."}
        ]
      }
      ```

    - `obfuscated-conversations`: Local obfuscated conversation dataset.
      Requires `--dataset-path` pointing at a local JSONL file.

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

    - `sonnet`: Poetry dataset using poem lines from a packaged text file.
      Override with `--dataset-path`.

      - `--sonnet-input-len`: Input tokens per request. Default: `550`.
      - `--sonnet-prefix-len`: Shared prefix tokens per request. Default: `200`.

    - `random`: Synthetically generated dataset with configurable token
      distributions.

      - `--random-input-len`: Input tokens per request. Accepts a constant or a
        distribution string: `N(mean,std)`, `U(lower,upper)`, `DU(lower,upper)`,
        `NB(n,p)`, `G(shape,scale)`, `LN(mean,std)`, `Burr12(c,d,scale)`, or
        `Cat(v1:w1,v2:w2,...)` for an explicit weighted set of values (for
        example, `Cat(512:0.7,1024:0.2,2048:0.1)`; omit `:weight` for a
        uniform choice among the listed values). Use `;` to set
        separate distributions for the first and subsequent turns (for example,
        `N(2048,200);N(512,50)`). Default: `1024`.
      - `--random-output-len`: Output tokens per request. Same format as
        `--random-input-len`. Default: `128`.
      - `--random-num-turns`: Turns per session. Same format as
        `--random-input-len`. Default: `1`. Used by `random` and `synthetic`
        multiturn workloads, and by `--fit-distributions` on `instruct-coder`,
        `agentic-code`, and `nemotron-opencode`.
      - `--random-sys-prompt-ratio`: Fraction of the input length to use as a
        system prompt. Range: `0.0`–`1.0`. Default: `0.0`.
      - `--random-max-num-unique-sys-prompt`: Maximum number of distinct system
        prompts to generate. Default: `1`.
      - `--warm-shared-prefix` / `--no-warm-shared-prefix`: Send each unique
        shared prefix as a single-token request before the run to prime
        prefix-cache KV entries. Supported on `random` and `synthetic`, and on
        `instruct-coder` and `agentic-code` with `--fit-distributions`.
        Requires `--random-sys-prompt-ratio > 0`. Disabled by default.
      - `--random-image-count`: Images to attach per request (enables vision
        mode on this dataset). Single-turn (`--num-prompts`) only. Default:
        `0`.
      - `--random-image-size`: Pixel dimensions of generated images (for
        example, `512,512`). Used with `--random-image-count`.
      - `--image-fraction`, `--image-count`, `--image-long-side`,
        `--image-aspect-ratio`, `--image-turn`: Dataset-agnostic image
        mixing, usable with any text-generation dataset (including
        `instruct-coder`). See [Multimodal input](#multimodal-input). Mutually
        exclusive with `--random-image-count` / `--random-image-size`.
        `--image-long-side` and `--image-aspect-ratio` both accept any
        distribution string, including `Cat(...)`, so you can match an
        empirical shape distribution measured from real traffic instead of
        approximating it with a parametric one — for example,
        `--image-long-side "Cat(1024:0.7,512:0.2,2048:0.1)"`.

    - `synthetic`: Synthetic text generation workload that uses the same
      distribution flags as `random`, but generates synthetic token IDs instead
      of vocabulary text. Supports multiturn via `--num-chat-sessions` and the
      `random_*` flags listed above. Also supports `--warm-shared-prefix`.

    #### Code

    - `instruct-coder`: Instruction-following coding dataset from Hugging Face
      Hub (`likaixin/InstructCoder`). Supports single-turn (`--num-prompts`) and
      multiturn (`--num-chat-sessions`) modes.

      If using with multiturn, it groups editing tasks at their
      natural token lengths, by default (up to 5 turns per session). With
      `--fit-distributions`, turn count comes from `--random-num-turns` instead,
      and per-turn input/output lengths and inter-turn delays follow the same
      distributions as the `random` dataset, via `random_*` flags and
      `--delay-between-chat-turns`; prompts are padded or truncated to match
      those targets.

    - `agentic-code`: Multiturn agentic coding workload with tool-call turns,
      from Hugging Face Hub (`novita/agentic_code_dataset_22`). Supports
      single-turn (`--num-prompts`) and multiturn (`--num-chat-sessions`) modes.
      By default, each session replays a full recorded conversation (variable
      turn count). `--fit-distributions` behaves the same as for
      `instruct-coder`.

      - `--tool-calls` / `--no-tool-calls`: Include or strip tool-call turns and
        forward tool definitions. Default: enabled.

    - `nemotron-opencode`: Large-scale agentic coding traces from Hugging Face
      (`nvidia/Nemotron-SFT-OpenCode-v1`), streamed on demand. Includes tool
      schemas translated to OpenAI function-tool format. Doesn't support
      `--dataset-path`. Supports single-turn (`--num-prompts`) and multiturn
      (`--num-chat-sessions`) modes. By default, each session replays a full
      recorded conversation (variable turn count). `--fit-distributions`
      behaves the same as for `instruct-coder`.

      - `--tool-calls` / `--no-tool-calls`: Include or strip tool-call turns and
        forward tool definitions. Default: enabled.

    - `code_debug`: Long-context code debugging dataset with multiple-choice
      questions, from Hugging Face Hub (`xinrongzhang2022/InfiniteBench`).
      Single-turn via `--num-prompts`. Also supports a fixed two-turn long-context
      template via `--num-chat-sessions`.

    #### Vision

    - `batch-job`: Batch image workload in OpenAI Batch API format. Requires
      `--dataset-path` (tar archive or extracted directory with `jobs.jsonl`).

      - `--batch-job-image-dir`: Directory where the server can access images
        (file reference mode). When unset, images are embedded as base64.

    - `local-image`: Local input images for the `image-to-image` and
      `image-to-video` tasks (see `--benchmark-task`). Requires
      `--dataset-path` (JSONL with `prompt` and `image_path` per line).

    - `vision-arena`: Vision-language benchmark dataset with images and
      associated questions for multimodal model evaluation, from Hugging Face
      Datasets (`lmarena-ai/vision-arena-bench-v0.1`).

    - `synthetic-pixel`: Synthetic prompts generated in memory for the
      pixel-generation tasks. The only dataset for `text-to-image` and
      `text-to-video`, and also supports `image-to-image` and
      `image-to-video`.

    ### Output control

    - `--max-output-len`: Maximum output length per request, in tokens.

    - `--output-lengths`: Output length for every request, as an integer or
      a path to a YAML file with an `output_lengths` list (such as one
      written by `--record-output-lengths`). Not supported by
      `arxiv-summarization`.

    - `--disable-ignore-eos`: Let requests stop at an EOS token. By default,
      requests ignore EOS and generate their full target length, so every
      backend generates the same number of tokens. Speculative decoding
      needs this flag, because a draft model can't predict tokens past a
      forced EOS.

    - `--temperature`, `--top-p`, `--top-k`: Sampling parameters forwarded to
      the server.

    - `--thinking-temperature`: Temperature for tokens inside a
      `<think>...</think>` block. MAX only; other backends ignore or reject
      it.

    - `--extra-body`: Extra top-level fields to add to every text-generation
      request body, as an inline JSON object or a path to a JSON or YAML
      file. For example, `'{"stop": ["}"]}'`. These fields override any
      field with the same name set by another option.

    - `--response-format`: OpenAI `response_format` for structured output,
      as an inline JSON object or a path to a JSON or YAML file. Not applied
      to `chat-judge`.

    - `--response-format-fraction`: Fraction (`0.0`–`1.0`) of requests, or
      of eligible turns in multiturn sessions, that use `--response-format`.
      Default: `1.0`.

    - `--response-format-turn`: Which turns of a multiturn session
      `--response-format-fraction` can select. Choices: `every`, `first`,
      `last`. Default: `every`.

    - `--tools`: OpenAI chat-completions tool definitions to send, as an
      inline JSON list or a path to a JSON or YAML file. Not applied to
      `chat-judge` or to requests that already include tools from the
      dataset. Unrelated to `--agentic-tool-profiles`.

    - `--tools-fraction`: Fraction (`0.0`–`1.0`) of requests, or of
      multiturn sessions, that include `--tools`. A selected session sends
      the tools on every turn. Default: `1.0`.

    ### Multimodal input

    These options mix generated images into the prompts of any
    text-generation dataset. They require the `/v1/chat/completions`
    endpoint and can't be combined with `--random-image-count` or
    `--random-image-size`.

    - `--image-fraction`: Fraction (`0.0`–`1.0`) of requests, or of
      multiturn sessions, that include images. Default: `0.0`.

    - `--image-count`: Number of images per selected request or turn, as a
      constant or a distribution string (such as `DU(1,4)`). Default: `1`.

    - `--image-long-side`: Length of each image's longer side, in pixels, as
      a constant or a distribution string. Default: `512`.

    - `--image-aspect-ratio`: Width-to-height ratio of each image, as a
      constant or a distribution string. Default: `1.0` (square).

    - `--image-turn`: Which turns of a multiturn session get images.
      Choices: `first`, `last`, `every`. Default: `first`.

    - `--image-num-turns`: Number of turns to keep in a session selected
      for images, as a constant or a distribution string. Longer sessions
      are cut to this length. Default: unset (sessions keep their length).

    The distribution strings use the same format as `--random-input-len`.

    ### Pixel generation

    These options apply when `--benchmark-task` is an image or video task.

    - `--image-width`, `--image-height`: Output image size in pixels.

    - `--image-steps`: Number of denoising steps.

    - `--image-guidance-scale`: Guidance scale.

    - `--image-negative-prompt`: Negative prompt.

    - `--image-seed`: Seed for deterministic generation.

    - `--num-frames`: Number of frames to generate. Required for
      `text-to-video` and `image-to-video`.

    ### LoRA traffic

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

    ### Result saving

    - `--result-filename`: Path to a JSON file for benchmark results, in
      addition to the result files in `--log-dir`. When unset, no file is
      written at a custom path. The path may include directories that the
      command creates if they don't exist.

    - `--always-save-result`: Write result files even when the run fails
      validation (for example, when some requests fail). The command still
      exits with code 1.

    - `--metadata`: Key-value pairs (such as `--metadata version=0.3.3 tp=1`)
      recorded alongside the run in the result JSON.

    - `--log-dir`: Directory for `results.csv` and the per-run result JSON
      files. Default: `sweep-serving-Y.m.d-H.M.S`. With `--dry-run`, no
      directory is created unless you set this option.

    - `--latency-percentiles`: Comma-separated latency percentiles to
      include in the CSV results. Only `50`, `90`, `95`, and `99` are
      supported. Default: `50,90,95,99`.

    - `--goodput`: Per-request latency targets in milliseconds, as
      space-separated `key:value` pairs with keys `ttft`, `tpot`, and
      `e2el` (for example, `--goodput ttft:500 tpot:50`). The results then
      report request goodput (successful requests per second that met every
      target) and goodput attainment (the share that met every target).
      Text generation only.

    - `--record-output-lengths`: Path to a YAML file to save each request's
      output length. Pass the file to `--output-lengths` to replay the same
      lengths later.

    - `--record-request-text`: Include each request's generated text in the
      per-request records of the result JSON.

    - `--workload-stats-json`: Path to write statistics about the requests
      the workload sends (lengths by turn and the shares of requests with
      tools, images, or structured output) as JSON. Combine with `--dry-run`
      to get them without a server.

    - `--engine-state-json`: Path to write the engine state the run
      measured (prefix cache hit rate and mean decode batch size) as JSON.
      Requires a single `--max-concurrency` and `--request-rate` value.

    ### Stats collection

    - `--collect-gpu-stats` / `--no-collect-gpu-stats`: Report GPU utilization
      and memory consumption (NVIDIA only). Enabled by default. Reads local
      GPUs unless you set `--gpu-metrics-host` or
      `--gpu-metrics-prometheus-url`, so without those it only measures the
      server when `max benchmark` runs on the same machine.

    - `--collect-cpu-stats` / `--no-collect-cpu-stats`: Report CPU stats.
      Enabled by default.

    - `--collect-server-stats` / `--no-collect-server-stats`: Report server
      stats. Enabled by default.

    - `--server-pids`: PIDs to monitor for CPU stats, including their
      children, instead of finding the server process from its port. Useful
      when the server runs in a container.

    - `--metrics-urls.<label>`: Prometheus metrics endpoint to scrape for
      server stats, keyed by a label of your choice (for example,
      `--metrics-urls.engine-0=http://host:8001/metrics`). Repeat for
      multiple endpoints. When unset, the endpoint is derived from `--host`
      and `--port`.

    - `--gpu-metrics-host`: DCGM exporter to read GPU stats from when
      `max benchmark` runs on a different machine than the server. Accepts
      a host, `host:port`, or full URL; a bare host expands to
      `http://<host>:9400/metrics`. Accepts a comma-separated list for
      multi-node servers.

    - `--gpu-metrics-prometheus-url`: Base URL of a Prometheus server that
      scrapes the server's DCGM exporter. Takes precedence over
      `--gpu-metrics-host`. Requires `--gpu-metrics-selector`.

    - `--gpu-metrics-selector`: PromQL label matchers, without braces, that
      select the server's GPUs in `--gpu-metrics-prometheus-url` (for
      example, `'namespace="bench",pod=~"engine-.*"'`).

    ### Repeated runs

    - `--num-iters`: Number of times to run each `--max-concurrency` and
      `--request-rate` combination. With more than one, the results come
      from the run with the median request throughput. Default: `1`.

    - `--flush-prefix-cache` / `--no-flush-prefix-cache`: Reset the server's
      prefix cache before each run. Enabled by default. The `trtllm`,
      `trtllm-chat`, and `mcloud` backends have no reset endpoint, so pass
      `--no-flush-prefix-cache` with them.

    - `--server-ready-timeout-s`: Seconds to wait for the server's health
      endpoint to respond before the benchmark starts. Default: `0` (check
      once).

    ### Debugging

    - `--dry-run`: Build the dataset, print workload stats and a warmup
      preview, then exit without contacting the server.

    - `--print-workload-stats`: Print the distribution of input lengths,
      output lengths, turns, and delays in the workload.

    - `--print-inputs-and-outputs`: Print every input prompt and, for text
      generation, every generated output.

    - `--disable-tqdm`: Hide the progress bar.

    - `--verbose`: Enable debug logging.

    ### Profiling

    - `--profile`: Capture an Nsight Systems GPU trace and print a ranked
      top-N kernel summary when the run finishes. (Translates to `--trace`
      internally.) The server must already be running under `nsys launch`
      (unlike `max generate --profile`, which re-execs the client under
      `nsys profile`).

    - `--profile-output`: Path for the `.nsys-rep` file when `--profile` is
      set. Default: `$BUILD_WORKSPACE_DIRECTORY/max-profile.nsys-rep`, or
      `max-profile.nsys-rep` in the current directory.

    - `--profile-top-n`: Number of kernels to show in the summary table.
      Default: `15`.

    - `--trace`: Enable nsys tracing (lower-level alternative to
      `--profile` without the post-run kernel summary). Requires the server
      to run under `nsys launch`. NVIDIA GPUs only.

    - `--trace-file`: Path to save the `nsys` trace when using `--trace`
      directly. Default: `$BUILD_WORKSPACE_DIRECTORY/profile.nsys-rep`, or
      `./profile.nsys-rep`.

    - `--trace-session`: Optional `nsys` session name to trace.

    ### Configuration file {#benchmark-configuration-file}

    The `--config-file` option loads benchmark settings from YAML instead of
    spelling out every flag on the command line. Define options under a top-level
    `benchmark_config` key, or put them at the top level of the file. CLI flags
    and `MODULAR_*` environment variables override values from the file when
    both are supplied.

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
    the [quickstart guide](/get-started).

    ## Output

    Each run prints the following metrics on completion. Token throughput and
    latency metrics are reported as a mean and percentiles.

    - **Request throughput**: number of complete requests processed per second.
    - **Input throughput**: number of input tokens processed per second.
    - **Output throughput**: number of tokens generated per second.
    - **TTFT** (time to first token): time from request start to first token
      generation.
    - **TPOT** (time per output token): time taken to generate each output
      token after the first, computed per request as
      `(latency - TTFT) / (output_tokens - 1)`.
    - **ITL** (inter-token latency): time between consecutive token or
      token-chunk generations.
    - **Request latency**: time from request start to the end of the
      response.

    For multiturn workloads, the run also reports:

    - **Per-turn cached token rate**: percentage of each turn's prompt tokens
      served from prefix cache (when the server reports token statistics).
    - **Per-turn KV cache retention**: for each turn after the first, the
      percentage of the previous turn's block-aligned prefix that remains cached.
      Surfaces when the server drops cached tokens across turns (distinct from
      cached token rate, whose denominator includes new and uncacheable tokens).
      Configure block alignment with `--kv-block-size`.

    When `--collect-gpu-stats` is enabled, the run also reports:

    - **GPU utilization**: percentage of time during which at least one GPU
      kernel is executing.
    - **Peak GPU memory**: peak memory usage during the benchmark run.
    - **Available GPU memory**: lowest free GPU memory during the run.

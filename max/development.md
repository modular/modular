# MAX framework and model development

This document covers the essentials of developing within the MAX open source
project.

If this is your first time contributing, first read everything in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Set up your environment

To get started, you need to do the following:

1. [Fork the repo and create a branch](../CONTRIBUTING.md#how-to-create-a-pull-request).

2. If you're using VS Code, [Install the Mojo VS Code
  extension](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo)

3. Many of our examples use [`pixi`](https://pixi.sh/latest/), which you can
    install with this command:

    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

    This is not necessary for much of the development process, because the
    Bazel build system will take care of installing the necessary components
    for you.

Now you're ready to start developing.

## Building the MAX framework

The Modular repository uses [Bazel](https://bazel.build/), a fast, scalable
build and test tool to ensure reproducible builds through dependency tracking
and caching.

To build and test your changes to the MAX framework, run the following
`./bazelw` commands from the top-level directory of the repository (where the
`bazel` folder is located).

Be sure that you meet the
[system requirements](https://docs.modular.com/max/packages/install#system-requirements).

If you're developing on macOS, you need Xcode 16.0 or later and macOS 15.0 or
later. You may need to run `xcodebuild -downloadComponent MetalToolchain`, which
downloads the Metal utilities required for GPU programming in later versions of
Xcode.

## Testing the MAX framework

To run all MAX tests, you can run:

```bash
./bazelw test //max/...
```

## Testing only a subset of the MAX framework

You can run all the tests within a specific subdirectory by simply
specifying the subdirectory and using `/...`. For example:

```bash
./bazelw test //max/tests/integration/API/python/graph/...
./bazelw test //max/tests/tests/torch/...
```

To find all the test targets, you can run:

```bash
./bazelw query 'tests(//max/tests/...)'
```

### Running MAX pipelines

When developing a new model architecture, or testing MAX API changes against
existing models, you can use Bazel commands like the following:

```bash
# Generate text with a model
./bazelw run //max/python/max/entrypoints:pipelines -- generate \
  --model modularai/Llama-3.1-8B-Instruct-GGUF \
  --prompt "Hello, world!"

# Serve a model locally
./bazelw run //max/python/max/entrypoints:pipelines -- serve \
  --model modularai/Llama-3.1-8B-Instruct-GGUF

# Run with custom configuration
./bazelw run //max/python/max/entrypoints:pipelines -- generate \
  --model model.gguf \
  --max-new-tokens 256 \
  --temperature 0.7
```

> [!NOTE]
> If you need to make changes to the core kernels library or Mojo standard
> library to support your model, currently the graph compiler does not use
> local changes to Mojo libraries. To see these reflected in your model, place
> the new Mojo code for these kernels in
> [custom operations](https://docs.modular.com/max/develop/build-custom-ops)
> and use these within your model.

## Contributing a new model architecture

To contribute a new model architecture to MAX, you can follow these steps:

1. Create new directory in `architectures/`
2. Implement model components:
   - `model.py`: Core model implementation
   - `model_config.py`: Configuration class
   - `arch.py`: Architecture registration
   - `weight_adapters.py`: Weight loading logic

3. Register the architecture:

    ```python
    @register_pipelines_model("your-model", provider="your-org")
    class YourModelConfig(HFModelConfig):
        ...
    ```

4. Test that the model builds and runs by performing text generation or serving
   up an endpoint using one of the commands in the previous section.

5. Validate the accuracy of the model using
    [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness). To do so
    for text LLMs, first start your model server in one terminal:

    ```bash
    max serve --model-path your-org/your-model-name
    ```

    and then run the GSM8K evaluation in another terminal:

    ```bash
    uvx --from 'lm-eval[api]' lm_eval \
    --model local-chat-completions \
    --tasks gsm8k_cot_llama \
    --model_args model=your-org/your-model-name,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=64,max_retries=1 \
    --apply_chat_template \
    --limit 320 \
    --seed 42 \
    --gen_kwargs seed=42,temperature=0 \
    --fewshot_as_multiturn
    ```

    This will give you an accuracy score that you can compare against a
    reference implementation in vLLM or SGLang. For benchmarking other kinds
    of models, or more details on this process, see
    [the MAX custom model architecture documentation](./python/max/pipelines/architectures/README.md)

## Formatting changes

Please make sure your changes are formatted before submitting a pull request.
Otherwise, CI will fail in its lint and formatting checks.  `bazel` setup
provides a `format` command.  So, you can format your changes like so:

```bash
./bazelw run format
```

It is advised, to avoid forgetting, to set-up `pre-commit`, which will format
your changes automatically at each commit, and will also ensure that you
always have the latest linting tools applied.

To do so, install pre-commit:

```bash
pixi x pre-commit install
```

and that's it!

If you need to manually apply the `pre-commit`, for example, if you
made a commit with the github UI, you can do `pixi x pre-commit run --all-files`,
and it will apply the formatting to all Mojo and Python files.

You can also consider setting up your editor to automatically format
Mojo and Python files upon saving.

### Raising a PR

Follow the steps to
[create a pull request](../CONTRIBUTING.md#create-a-pull-request).

Congratulations! You've now got an idea on how to contribute to the MAX
framework, test your changes, and raise a PR.

If you're still having issues, reach out on
[Discord](https://modul.ar/discord).

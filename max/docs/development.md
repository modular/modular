# MAX framework developer guide

This is an introduction to developing within the MAX open source project. If
you plan to contribute changes back to the repo, first read everything in
[CONTRIBUTING.md](/max/CONTRIBUTING.md).

If you just want to build with MAX and aren't interested in developing in the
source code, instead see the [MAX quickstart
guide](https://docs.modular.com/max/get-started).

## Set up your environment

First, make sure your system meets the [MAX system
requirements](https://docs.modular.com/max/packages/install#system-requirements).
The same requirements that apply to the `modular` package apply to developing
in this repo.

In particular, if you're on macOS, make sure you have Metal utilities (for GPU
programming in recent versions of Xcode)—try `xcodebuild -downloadComponent
MetalToolchain`.

Then you can get started:

1. Fork the repo, clone it, and create a branch.

2. Optionally, install [`pixi`](https://pixi.sh/latest/). We use it in our
code examples for package management and virtual environments.

    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

3. Optionally, [install the Mojo
  extension](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo)
  in VS Code or Cursor.

That's it.

The build system uses [Bazel](https://bazel.build/), and if you don't have it,
the `bazelw` script in the next step installs it.

## Test the MAX framework

From the repo root, run this `bazelw` command to run all the MAX tests:

```bash
./bazelw test //max/...
```

If it's your first time, it starts by installing the Bazel version manager,
[Bazelisk](https://github.com/bazelbuild/bazelisk), which then installs Bazel.

### Know the local test prerequisites

Not every MAX test has the same local requirements. Before running a broad
target, check which of these constraints apply:

- **`HF_TOKEN`**: Some tests exercise gated Hugging Face repos or fetch remote
  config files. Export a token before running them:

  ```bash
  export HF_TOKEN="hf_..."
  ```

- **Model downloads**: Some integration tests resolve model snapshots through
  the local Hugging Face cache and expect the snapshot to be present already.
  On a fresh machine, warm the cache first with:

  ```bash
  bazel run //max/tests/integration/tools:download_models -- \
    meta-llama/Llama-3.2-1B-Instruct
  ```

- **GPU requirements**: Many integration targets are tagged `gpu` and will not
  run successfully on a CPU-only machine. Prefer CPU or pure unit targets if
  your change does not touch GPU execution.
- **Networked tests**: Targets tagged `requires-network` may contact Hugging
  Face or other remote endpoints and are more likely to fail in restricted or
  offline environments.

### Test a subset of the MAX framework

You can run all the tests within a specific subdirectory by simply
specifying the subdirectory and using `/...`. For example:

```bash
./bazelw test //max/tests/integration/graph/...
./bazelw test //max/tests/tests/torch/...
```

To find all the test targets, you can run:

```bash
./bazelw query 'tests(//max/tests/...)'
```

### Use a minimal test matrix

For local iteration, start with the smallest target set that matches your
change. The following commands are good default subsets:

| Change type | Suggested command | Typical prerequisites |
| --- | --- | --- |
| Core Python logic and lightweight local regression | `./bazelw test //max/tests/tests:cpu_local_tests` | No GPU; usually no `HF_TOKEN` |
| Serve process-control unit tests | `./bazelw test //max/tests/tests/serve/unit:tests` | CPU-only, but slower than the default local suite |
| Pipeline library or architecture logic that should stay CPU-safe | `./bazelw test //max/tests/tests/pipelines/... //max/tests/integration/pipelines:tests` | Network may be needed for some pipeline tests |
| Tokenization or HF-backed pipeline integration | `./bazelw test //max/tests/integration/pipelines/tokenization:tests //max/tests/integration/architectures/internvl_network_tests:tests` | `HF_TOKEN`, network, and a GPU-capable machine |
| GPU runtime, graph, or kernel-facing changes | `./bazelw test //max/tests/tests:test_interpreter_ops_gpu //max/tests/integration/pipelines:tests_gpu` | GPU required; network often required |

If you are unsure whether a target needs network or GPUs, inspect its Bazel
rule for tags such as `gpu` or `requires-network`, or for `env_inherit =
["HF_TOKEN"]`.

## Run a MAX pipeline

When developing a new model architecture, or testing MAX API changes against
existing models, you can use the following Bazel commands to run inference.

> [!NOTE]
> Some models require Hugging Face authentication to load model weights, so
> you should set your [HF access token](https://huggingface.co/settings/tokens)
> as an environment variable:
>
> ```sh
> export HF_TOKEN="hf_..."
> ```

For example, this `entrypoints:pipelines generate` command is equivalent to
running inference with [`max
generate`](https://docs.modular.com/max/cli/generate):

```bash
./bazelw run //max/python/max/entrypoints:pipelines -- generate \
  --model OpenGVLab/InternVL3-8B-Instruct \
  --prompt "Hello, world!"
```

And this is equivalent to creating an endpoint with [`max
serve`](https://docs.modular.com/max/cli/serve):

```bash
./bazelw run //max/python/max/entrypoints:pipelines -- serve \
  --model OpenGVLab/InternVL3-8B-Instruct \
  --trust-remote-code
```

## Start developing

Here are some docs to help start developing in the MAX framework:

- [Contributing new model architectures](/max/docs/contributing-models.md)
- [Benchmarking a MAX endpoint](/max/docs/max-benchmarking.md)
- [Benchmarking Mojo kernels with `kbench`](/max/docs/kernel-benchmarking.md)
- [Kernel profiling with Nsight Compute](/max/docs/kernel-profiling.md)
- [Contributing changes to the repo](../../CONTRIBUTING.md#contributing-changes)

For more documentation, see [docs.modular.com](https://docs.modular.com).

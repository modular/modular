# MAX unreleased changelog

This is a list of UNRELEASED MAX changes.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

## UNRELEASED

### Highlights {#26-2-highlights}

### Documentation {#26-2-docs}

### MAX models {#26-2-models}

- Add support for Olmo3ForCausalLM architecture.
- Add support for Qwen/Qwen3-30B-A3B-Instruct-2507 which is a MOE model.
- Add multi-GPU tensor parallelism support for Qwen3 and Qwen3-MoE models.
- Remove legacy Gemma 3 multimodal implementation and the
  `MODULAR_MAX_DISABLE_GEMMA3_VISION` environment variable.
- Implement multi-GPU support (tensor parallelism) for GPT-OSS.
- Common MAX models like Qwen 2.5 can now run on AMD RDNA consumer GPUs.

### MAX framework {#26-2-max}

#### Inference server {#26-2-max-serve}

- Enabled overlap scheduling for select model architectures like
  `LlamaForCausalLM_Legacy` by default. This optimization reduces CPU overhead
  by overlapping python host code with GPU kernel execution. This optimization
  is currently incompatible with some features such as structured outputs or cpu
  models. This feature is very experimental! You can forcibly disable it via
  `--no-enable-overlap-scheduler --force`.

#### `max` CLI {#26-2-max-cli}

#### Python API {#26-2-max-python}

- Keep a global MLIR context active and drop per-graph context plumbing so
  algebraic dims and graph/custom op construction work without an explicit
  context manager. Threadpool-backed MAX paths now scope worker-thread MLIR
  usage to the default context automatically.

- Added the `prod` op for computing the product of elements along an axis,
  available as `max.graph.ops.prod`, `max.functional.prod`, and
  `Tensor.prod()`.

### Breaking changes {#26-2-breaking}

- **`PipelineConfig.max_length` has been removed**. The `max_length` parameter
  now resides at the model configuration level as **`MAXModelConfig.max_length`**
  (accessible as `config.model.max_length`). This change correctly places the
  parameter at the model level since it describes model capacity (maximum sequence
  length the model can process), not pipeline runtime behavior. Update all
  configurations and code to use `model.max_length` instead of the removed
  `max_length` field at the pipeline level.

- **`PipelineModel` no longer accepts the `encoding` parameter**. The
  `encoding` parameter has been removed from `PipelineModel.__init__` and all
  subclasses. The encoding is now automatically inferred from
  `pipeline_config.model.quantization_encoding`. This change eliminates
  redundant parameter passing and ensures a single source of truth for
  quantization encoding configuration. Users who directly instantiate
  `PipelineModel` subclasses (advanced use case) should remove the `encoding`
  argument from constructor calls.

#### Mojo API {#26-2-max-mojo}

#### Custom ops {#26-2-custom-ops}

### MAX kernels {#26-2-max-kernels}

<!-- Please place Layout/LayoutTensor changes under "Library changes" in the
     **Mojo changelog**, since the layout package is packaged with and
     documented alongside Mojo. -->

### Mojo language {#26-2-mojo}

For all the updates to the Mojo language, standard library, and tools,
including all GPU programming and `Layout`/`LayoutTensor` changes, see the [Mojo
changelog](/mojo/changelog)

# MAX Integration Tools

This directory contains utilities for testing, debugging, and validating MAX pipelines.

## Tools

### compare_buffers.py

Compares MAX buffers with PyTorch tensors to validate numerical accuracy.
Supports comparing individual files or auto-matching tensors with same names
across directories.

**File formats:**

- `.max` files for MAX buffers (generated via `debug_model --output-dir` or
  `ops.print()`)
- `.pt` files for PyTorch tensors

To generate `.max` files with `ops.print()`, configure the
`InferenceSession` with:

```python
session.set_debug_print_options(
    "BINARY_MAX_CHECKPOINT",
    output_directory="/tmp/max"
)
```

**Example:**

```bash
bazel run //max/tests/integration/tools:compare_buffers -- \
    --torch-tensor ref.pt --max-buffer out.max --rtol 1e-5 --atol 1e-8
```

Use `--help` for more options including directory comparison and tolerance settings.

### debug_model.py

Runs models with debugging capabilities to inspect intermediate outputs and
diagnose issues. Supports both MAX and PyTorch frameworks with layer-level
inspection.

**Example:**

```bash
bazel run //max/tests/integration/tools:debug_model -- \
    --framework max --pipeline meta-llama/Llama-3.2-1B-Instruct
```

Use `--help` for more options including number of layers, prompts, and output configuration.

### generate_llm_logits.py

Generates reference logits from LLM models using MAX, PyTorch, or vLLM
frameworks. Useful for creating golden files to validate model accuracy across
different implementations.

**Example:**

```bash
bazel run //max/tests/integration/tools:generate_llm_logits -- \
    --framework max --pipeline meta-llama/Llama-3.2-1B-Instruct --output-path logits.json
```

Use `--help` for more options including batch size, encoding, and reference comparison.

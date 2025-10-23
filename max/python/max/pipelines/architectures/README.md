# Contributing custom model architectures

This directory contains the model architectures supported by MAX pipelines. Each
architecture defines how MAX reconstructs models from Hugging Face as
performance-optimized MAX graphs.

## Overview

MAX comes with built-in support for many popular model architectures like
`Gemma3ForCausalLM`, `Qwen2ForCausalLM`, and `LlamaForCausalLM`. However, you can
contribute your own custom model architecture that can be served natively by MAX
using the `max serve` command.

For more information see our [custom model architectures
tutorial](https://docs.modular.com/max/tutorials/serve-custom-model-architectures/).

## Get started

### 1. Set up your development environment

Clone the repository and navigate to this directory:

```bash
git clone https://github.com/modular/modular.git
cd modular/SDK/lib/API/python/max/pipelines/architectures
```

### 2. Create your architecture directory

Create a new directory for your architecture (e.g., `my_model/`) with the
required file structure:

```text
my_model/
├── __init__.py          # Exports your architecture for discovery
├── arch.py              # Defines the SupportedArchitecture configuration
├── model.py             # Implements the main model logic and graph building
├── model_config.py      # Handles model configuration and parameter parsing
└── weight_adapters.py   # Converts weights between different formats
```

> Copy an existing architecture folder and rename it to your custom architecture
> name, then customize it as needed.

### 3. Register your architecture

Add your architecture to the main [`__init__.py`](__init__.py) file:

```python
# Add import
from .my_model import my_model_arch

# Add to architectures list
architectures = [
    # ... existing architectures ...
    my_model_arch,
    # ... rest of architectures ...
]
```

## Requirements

Your custom architecture must:

1. **Follow the naming convention**: The architecture name in `arch.py` must
exactly match the model class name in your Hugging Face model's configuration.

2. **Implement required methods**: Your model class must inherit from
`PipelineModel` and implement the required methods.

3. **Handle weight conversion**: Provide weight adapters for supported formats
(at minimum SafeTensors).

4. **Include proper configuration**: Handle parameter mapping from Hugging Face
config to your internal format.

## Testing your architecture

Test your custom architecture locally using the `--custom-architectures` flag:

```bash
max serve \
  --model your-org/your-model-name \
  --custom-architectures path/to/your/architecture
```

## Detailed tutorial

For a comprehensive step-by-step guide with complete code examples, see our
[custom model architectures
tutorial](https://docs.modular.com/max/tutorials/serve-custom-model-architectures/).

## Contribution guidelines

Before submitting your custom architecture:

1. **Test thoroughly**: Ensure your architecture works with the `max serve` command.
2. **Follow existing patterns**: Study similar architectures in this directory.
3. **Document your code**: Include clear docstrings and comments.
4. **Handle edge cases**: Ensure robust error handling and validation.
5. **Performance considerations**: Optimize for inference performance.

## Support

- For detailed examples, explore the existing architecture implementations in
this directory.

- For questions or issues, please open a GitHub issue.

- For comprehensive documentation, visit the [MAX
documentation](https://docs.modular.com).

## Examples

This directory contains various architecture families you can use as reference:

- **LLaMA family**: `llama/` - Popular open-source language models.
- **Gemma family**: `gemma/` - Google's Gemma models.
- **Qwen family**: `qwen/` - Alibaba's Qwen models.

Each subdirectory represents a different model family with its own implementation
that you can study and adapt for your custom architecture.

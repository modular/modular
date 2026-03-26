# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Repository Overview

This is the `max` directory of Modular AI's codebase containing the MAX platform
SDK, APIs, and tools. The SDK provides:

- **Graph API**: High-level API for building and executing computational graphs
- **Driver API**: Low-level interface for device and tensor management
- **Engine API**: Inference session management and execution
- **Max Python APIs**: Python bindings for MAX platform functionality
- **Pipelines**: Pre-built architectures for LLMs and other models
- **Serving**: Production-ready model serving infrastructure

## Build Commands

### Building SDK Components

```bash
# Build all SDK targets
./bazelw build //max/...

# Build specific components
./bazelw build //max/python/max
./bazelw build //max/python/max/entrypoints:pipelines
./bazelw build //max/python/max/serve

# Build and install MAX CLI
./bazelw run //max/python/max/entrypoints:pipelines
```

### Running Tests

```bash
# Run all max tests
./bazelw test //max/...

# Run specific test suites
./bazelw test //max/tests/integration/graph
./bazelw test //max/tests/integration/architectures/llama3:test_cross_attention

# Run tests with specific arguments
./bazelw test --test_arg=-k --test_arg=test_attention \
  //max/tests/integration/architectures/llama3:test_cross_attention

# Run GPU tests remotely
bt-h100 //max/tests/integration/architectures/llama3:tests_gpu
bt-b200 //max/tests/integration/architectures/llama3:tests_gpu
```

## Key SDK Components

### Graph API (`//max/python/max/graph`)

- Core graph construction and manipulation
- Operations library (`ops/`)
- Weight loading and management
- Quantization support

### Pipelines (`//max/python/max/pipelines`)

- Pre-built model architectures (`architectures/`)
- Configuration management
- Tokenization and sampling
- Memory estimation and optimization

### Serving Infrastructure (`//max/python/max/serve`)

- API server implementation
- Request routing (OpenAI, KServe, SageMaker compatible)
- Scheduler for efficient batch processing
- KV cache management for LLMs
- Performance monitoring and telemetry

### Engine API (`//max/python/max/engine`)

- Inference session management
- Input/output processing
- Model loading and execution

## Development Workflow

### Running Pipelines

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

### Working with Graph API

```python
# Example graph construction
from max.graph import Graph, TensorType
from max.graph.ops import matmul, relu

g = Graph()
with g:
    x = g.arg(TensorType(shape=(None, 768), dtype=DType.float32))
    w = g.const(weight_data)
    y = matmul(x, w)
    z = relu(y)
    g.output(z)

# Compile and execute
session = InferenceSession(g)
result = session.run(input_data)
```

### Adding New Pipeline Architectures

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

## Testing Guidelines

### Unit Tests

- Graph API tests: `//max/tests/integration/graph`
- Pipeline tests: `//max/tests/integration`
- Serving tests: `//max/tests/integration/serve`

### Integration Tests

```bash
# Test full pipeline execution
./bazelw test //max/tests/integration:test_llama3

# Test serving infrastructure
./bazelw test //max/tests/integration/serve:test_tinyllama_serving_cpu
```

### Performance Testing

```bash
# Benchmark model performance
./bazelw run //max/python/max/entrypoints:pipelines -- benchmark \
  --model model

# Profile model execution
./bazelw run //max/python/max/entrypoints:pipelines -- profile \
  --model model
```

### Logit Verification Testing

To verify that pipeline outputs match PyTorch reference implementations:

```bash
# 1. Generate logits with MAX pipeline
./bazelw run //max/tests/integration/tools:generate_llm_logits -- \
  --device gpu \
  --framework max \
  --pipeline gemma3-1b \
  --encoding bfloat16 \
  --output /tmp/max-logits.json

# 2. Generate logits with PyTorch reference
./bazelw run //max/tests/integration/tools:generate_llm_logits -- \
  --device gpu \
  --framework torch \
  --pipeline gemma3-1b \
  --encoding bfloat16 \
  --output /tmp/torch-logits.json

# 3. Compare the logits
./bazelw run //max/tests/integration/accuracy:verify -- \
  --eval-metric cos,kl,tol \
  --relative-tolerance 1e-2 \
  --absolute-tolerance 1e-5 \
  --cos-dist-threshold 0.001 \
  --kl-div-threshold 0.01 \
  /tmp/max-logits.json /tmp/torch-logits.json

# Run verification pipeline directly (combines all steps)
./bazelw run //max/tests/integration/accuracy:verify_pipelines -- \
  --pipeline Gemma-3-1B-bfloat16 \
  --devices='gpu'

# Trigger Pipeline Logit Verification GitHub workflow (runs logit verification on CI)
gh workflow run "Pipeline Logit Verification" --ref <branch-name>
```

## Architecture Patterns

### Layer Implementation Pattern

Most neural network layers follow this structure:

1. Define configuration in `config.py`
2. Implement graph construction in layer class
3. Add weight loading in `weight_adapters.py`
4. Register with appropriate base class

### Memory Management

- Use `max.graph.weight.Weights` for efficient weight management
- Implement lazy loading for large models
- Use memory estimation before allocation

### Distributed Execution

- Models can be distributed across devices using `transfer_to` ops
- Implement custom sharding strategies in model classes
- Use collective operations for cross-device communication

## Common Development Tasks

### Accessing HuggingFace Model Configurations

When investigating model issues or comparing configurations between models:

1. **Model configurations are available on HuggingFace**:
   - Format: `https://huggingface.co/{org}/{model}/blob/main/config.json`
   - Example: `https://huggingface.co/google/gemma-3-12b-it/blob/main/config.json`

2. **For gated models requiring authentication**:

   ```bash
   # Check if HF_TOKEN is set
   echo "HF_TOKEN is: ${HF_TOKEN:-(not set)}"

   # If not set, warn the user that they need to set it
   # export HF_TOKEN=your_huggingface_token_here

   # Use HF_TOKEN with curl to access configs
   curl -s -H "Authorization: Bearer $HF_TOKEN" \
     https://huggingface.co/{org}/{model}/resolve/main/config.json | python3 -m json.tool

   # Example for Gemma models:
   curl -s -H "Authorization: Bearer $HF_TOKEN" \
     https://huggingface.co/google/gemma-3-12b-it/resolve/main/config.json | python3 -m json.tool
   ```

3. **Model versions are locked in**:
   - Check `max/tests/integration/hf-repo-lock.tsv` for
     exact revision hashes
   - This ensures reproducible builds and tests

### Adding New Operations

1. Implement operation in `//max/python/max/graph/ops/`
2. Add C++ binding if needed in `//max/python/max/_core/internal/`
3. Write comprehensive tests
4. Update documentation

### Working with Weights

```python
# Load weights from different formats
from max.graph.weights import load_pytorch, load_gguf, load_safetensors

weights = load_gguf("model.gguf")
weights = load_pytorch("model.pt")
weights = load_safetensors("model.safetensors")
```

## SDK-Specific Build Configurations

```bash
# Debug SDK components
c debug-sdk

# Optimize for serving
c serving-opt

# Enable profiling
c profile
```

## Important Notes

- Always run formatting before committing: `./bazelw run //:format`
- Use type hints throughout Python code
- Follow the Graph API style guide in `docs/GraphAPIStyleGuide.md`
- Write comprehensive tests for new features
- Document new architectures in `architectures/README.md`
- Performance improvements should include benchmarks
- Refer to docs/internal/PythonDocstringStyleGuide.md for Python docstring style.

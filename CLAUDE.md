# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Repository Overview

The Modular Platform is a unified platform for AI development and deployment
that includes:

- **MAX**: High-performance inference server with OpenAI-compatible endpoints
for LLMs and AI models
- **Mojo**: A new programming language that bridges Python and systems
programming, optimized for AI workloads

## Essential Build Commands

### Global Build System (Bazel)

All builds use the `./bazelw` wrapper from the repository root:

```bash
# Build everything
./bazelw build //...

# Build specific targets
./bazelw build //max/kernels/...
./bazelw build //mojo/stdlib/...

# Run tests
./bazelw test //...
./bazelw test //max/kernels/test/linalg:test_matmul

# Run tests with specific test arguments
./bazelw test --test_arg=-k --test_arg=test_attention //max/tests/...

# Find targets
./bazelw query '//max/...'
./bazelw query 'tests(//...)'

# Format code before committing
./bazelw run format
```

### Remote GPU Testing

```bash
# Run tests on NVIDIA H100
bt-h100 //max/tests/integration/pipelines/python/llama3:tests_gpu

# Run tests on NVIDIA B200
bt-b200 //max/tests/integration/pipelines/python/llama3:tests_gpu
```

### Pixi Environment Management

Many directories include `pixi.toml` files for environment management:

```bash
# Install Pixi environment (run once per directory)
pixi install

# Run Mojo files through Pixi
pixi run mojo [file.mojo]

# Format Mojo code
pixi run mojo format ./

# Use predefined tasks from pixi.toml
pixi run main              # Run main example
pixi run test              # Run tests

# List available tasks
pixi task list

# Install pre-commit hooks (recommended)
pixi x pre-commit install
```

### MAX Server Commands

```bash
# Install MAX nightly via pip
pip install modular --extra-index-url https://whl.modular.com/nightly/simple/

# Start OpenAI-compatible server
max serve --model modularai/Llama-3.1-8B-Instruct-GGUF

# Run pipelines via Bazel during development
./bazelw run //max/python/max/entrypoints:pipelines -- generate \
  --model modularai/Llama-3.1-8B-Instruct-GGUF \
  --prompt "Hello, world!"

./bazelw run //max/python/max/entrypoints:pipelines -- serve \
  --model modularai/Llama-3.1-8B-Instruct-GGUF
```

## High-Level Architecture

### Repository Structure

```text
modular/
├── mojo/                    # Mojo programming language
│   ├── stdlib/              # Standard library implementation
│   ├── docs/                # User documentation
│   ├── proposals/           # Language proposals (RFCs)
│   └── integration-test/    # Integration tests
├── max/                     # MAX framework
│   ├── kernels/             # High-performance Mojo kernels (GPU/CPU)
│   ├── python/max/serve/    # Python inference server (OpenAI-compatible)
│   ├── python/max/pipelines/# Model architectures (Python)
│   └── python/max/nn/       # Neural network operators (Python)
├── examples/                # Usage examples
└── bazel/                   # Build system configuration
```

### Key Architectural Patterns

1. **Language Separation**:
   - Low-level performance kernels in Mojo (`max/kernels/`)
   - High-level orchestration in Python (`max/python/max/`)

2. **Hardware Abstraction**:
   - Platform-specific optimizations via dispatch tables
   - Support for NVIDIA/AMD GPUs, Intel/Apple CPUs
   - Device-agnostic APIs with hardware-specific implementations

3. **Memory Management**:
   - Device contexts for GPU memory management (`DeviceContext` APIs)
   - Host/Device buffer abstractions
   - Careful lifetime management in Mojo code

4. **Testing Philosophy**:
   - Tests mirror source structure
   - Use `lit` tool with FileCheck validation
   - Migrating to `testing` module assertions

## Development Workflow

### Branch Strategy

- Work from `main` branch (synced with nightly builds)
- `stable` branch for released versions
- Create feature branches for significant changes

### Testing Requirements

```bash
# Run tests before committing
./bazelw test //path/to/your:target

# Run with sanitizers
./bazelw test --config=asan //...

# Multiple test runs
./bazelw test --runs_per_test=10 //...
```

### Code Style

- Use `mojo format` for Mojo code
- Run `./bazelw run format` before committing
- Sign commits with `git commit -s`

### Performance Development

```bash
# Run benchmarks with environment variables
./bazelw run //max/kernels/benchmarks/gpu:bench_matmul -- \
    env_get_int[M]=1024 env_get_int[N]=1024 env_get_int[K]=1024

# Use autotune tools
python max/kernels/benchmarks/autotune/kbench.py benchmarks/gpu/bench_matmul.yaml
```

## Critical Development Notes

### Mojo Development

- Use nightly Mojo builds for development
- Install nightly VS Code extension
- Avoid deprecated types like `Tensor` (use modern alternatives)
- Follow value semantics and ownership conventions
- Use `Reference` types with explicit lifetimes in APIs
- Always check function return values for errors

### MAX Kernel Development

- Fine-grained control over memory layout and parallelism
- Hardware-specific optimizations (tensor cores, SIMD)
- Ensure coalesced memory access patterns on GPU
- Minimize CPU-GPU synchronization points
- Avoid global state in kernels
- Performance improvements must include benchmarks

### macOS Development

- Requires Xcode 16.0+ and macOS 15.0+
- May need to run: `xcodebuild -downloadComponent MetalToolchain`

### Contributing New Model Architectures

1. Create new directory in `max/python/max/pipelines/architectures/`
2. Implement: `model.py`, `model_config.py`, `arch.py`, `weight_adapters.py`
3. Register with `@register_pipelines_model("your-model", provider="your-org")`
4. Validate accuracy using lm-eval:
   ```bash
   # Start model server
   max serve --model-path your-org/your-model-name

   # Run evaluation
   uvx --from 'lm-eval[api]' lm_eval \
     --model local-chat-completions \
     --tasks gsm8k_cot_llama \
     --model_args model=your-org/your-model-name,base_url=http://127.0.0.1:8000/v1/chat/completions
   ```

## Contributing Areas

Currently accepting contributions for:

- Mojo standard library (`/mojo/stdlib/`)
- MAX AI kernels (`/max/kernels/`)

Other areas are not open for external contributions.

## Platform Support

- Linux: x86_64, aarch64
- macOS: ARM64 (Apple Silicon) - Xcode 16.0+, macOS 15.0+
- Windows: Not currently supported

## LLM-friendly Documentation

- Docs index: <https://docs.modular.com/llms.txt>
- Mojo API docs: <https://docs.modular.com/llms-mojo.txt>
- Python API docs: <https://docs.modular.com/llms-python.txt>
- Comprehensive docs: <https://docs.modular.com/llms-full.txt>

## Git Commit Style

- **Atomic Commits:** Keep commits small and focused on a single logical change.
- **Descriptive Messages:** Explain the *why*, not just the *what*. Use imperative mood.
- **Commit titles:** Use `[Stdlib]`, `[Kernel]`, or `[GPU]` tags as appropriate.
- Wrap messages with `BEGIN_PUBLIC` and `END_PUBLIC`.
- Sign commits with `git commit -s`.

Example:

```git
[Kernels] Add fused attention kernel

BEGIN_PUBLIC
[Kernels] Add fused attention kernel

This adds a fused multi-head attention kernel to improve inference
performance by reducing memory bandwidth requirements.
END_PUBLIC
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Repository Overview

This is the MAX Kernels directory containing high-performance compute kernels
written in Mojo. These kernels serve as building blocks for numerical, machine
learning, and other performance-critical workloads. The repository is part of
Modular AI's larger codebase and uses Bazel as its build system.

## Build System

This project uses Bazel for building. Commands should be run through the
`./bazelw` wrapper script from the main Modular repository root.

### Essential Build Commands

```bash
# Build all kernels
./bazelw build //max/kernels/...

# Build a specific module
./bazelw build //max/kernels/src/linalg:linalg

# Build a specific benchmark
./bazelw build //max/kernels/benchmarks/gpu:bench_matmul

# Build and run a benchmark
./bazelw run //max/kernels/benchmarks/gpu:bench_matmul

# Run a specific test
./bazelw test //max/kernels/test/linalg:test_matmul

# Run all tests in a directory
./bazelw test //max/kernels/test/linalg/...

# Run GPU tests with specific hardware
./bazelw test --config=remote-a10 //max/kernels/test/gpu/...  # For A10 GPU
./bazelw test --config=remote-h100 //max/kernels/test/gpu/... # For H100 GPU
./bazelw test --config=remote-mi300 //max/kernels/test/gpu/... # For MI300 GPU
```

### Running Mojo Files Directly

```bash
# Run a Mojo file
./bazelw run //KGEN/tools/mojo -- /path/to/file.mojo

# Or use the bmojo alias (after sourcing start-modular.sh)
bmojo /path/to/file.mojo

# Debug a Mojo file
bd //KGEN/tools/mojo -- /path/to/file.mojo
```

## Code Architecture

### Directory Structure

- `src/`: Core kernel implementations
  - `linalg/`: Linear algebra operations (GEMM, GEMV, etc.)
  - `nn/`: Neural network operations (convolution, attention, pooling)
  - `quantization/`: Quantized operations
  - `layout/`: Memory layout utilities and tensor operations
  - `internal_utils/`: Internal utilities and helpers
  - `kv_cache/`: Key-value cache implementations
  - `Mogg/`: MOGG (Modular Graph Generator) related code
  - `register/`: Register-level operations
- `test/`: Unit tests mirroring source structure
  - Tests are organized by functionality (linalg, nn, gpu, etc.)
- `benchmarks/`: Performance benchmarks
  - `gpu/`: GPU-specific benchmarks with YAML configurations
  - `linalg/`: Linear algebra benchmarks
  - `nn/`: Neural network operation benchmarks
  - `autotune/`: Auto-tuning utilities and benchmarking tools

### Key Patterns

#### Kernel Implementation

- Kernels are written using Mojo's systems programming capabilities
- Fine-grained control over memory layout and parallelism
- Hardware-specific optimizations (CPU SIMD, GPU tensor cores)
- Vendor library integration (cuBLAS, Apple Accelerate)

#### Import Structure

```mojo
from linalg.matmul import matmul
from internal_utils import DeviceNDBuffer, HostNDBuffer
from gpu.host import DeviceContext
```

#### Test Files

- Tests files have a corresponding `.mojo` file in the test directory
- GPU tests are in the `test/gpu/` subdirectory
- Tests use assertions from the `testing` module

## Development Workflow

### Testing

```bash
# Run a specific test
./bazelw test //max/kernels/test/linalg:test_matmul

# Run tests with specific configurations
./bazelw test --config=asan //max/kernels/test/...  # With AddressSanitizer
./bazelw test --config=debug-modular //max/kernels/test/...  # Debug build
./bazelw test --runs_per_test=10 //max/kernels/test/...  # Multiple runs
```

### Benchmarking

```bash
# Run benchmarks using the benchmarking framework
./bazelw run //max/kernels/benchmarks/gpu:bench_matmul

# Run benchmarks with environment variables
./bazelw run //max/kernels/benchmarks/gpu:bench_matmul -- \
    env_get_int[M]=1024 env_get_int[N]=1024 env_get_int[K]=1024

# Use autotune tools for performance analysis
python benchmarks/autotune/kbench.py benchmarks/gpu/bench_matmul.yaml
```

### Format and Lint

```bash
# Format Mojo code
mojo format ./

# Run formatting through Bazel
./bazelw run //:format
```

## Platform-Specific Development

### GPU Development

- NVIDIA GPU support through CUDA/PTX
- AMD GPU support through ROCm
- Tests can be run on specific hardware using remote configs
- GPU kernels use device contexts and memory management

### CPU Optimizations

- Intel AMX support
- Apple AMX and Accelerate framework
- ARM NEON intrinsics
- x86 AVX/VNNI instructions

## Environment Variables

Many benchmarks and tests use environment variables for configuration:

- `env_get_int[]`: Get integer values
- `env_get_bool[]`: Get boolean flags  
- `env_get_dtype[]`: Get data type specifications

Example:

```bash
./bazelw run //max/kernels/benchmarks/gpu:bench_matmul -- \
    env_get_int[M]=512 env_get_bool[transpose_b]=true \
    env_get_dtype[type]=float16
```

## Debugging Tips

### Using LLDB

```bash
# Debug with bazel
bd //max/kernels/benchmarks/gpu:bench_matmul

# Debug in VSCode
bd --vscode //max/kernels/benchmarks/gpu:bench_matmul
```

### Common Debug Patterns

- Use `print()` for debugging values
- Enable assertions with `--enable_assertions`
- Use `--test_output=streamed` for immediate test output

## Performance Optimization

### Auto-tuning

The `benchmarks/autotune/` directory contains tools for:

- Running parameterized benchmarks (`kbench.py`)
- Comparing performance (`kdiff.py`)
- Plotting results (`kplot.py`)
- Profiling kernels (`kprofile.py`)

### Dispatch Tables

Platform-specific optimizations are selected through dispatch tables:

- `dispatch_table_a100_gpu.mojo`: NVIDIA A100 optimizations
- `dispatch_table_amd.mojo`: AMD GPU optimizations

## Contributing

Currently, external contributions are not being accepted, but you can:

- Report bugs through GitHub Issues
- Test kernels and provide feedback
- Stay updated through the [Modular forum](https://forum.modular.com/)

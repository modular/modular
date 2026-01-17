# B200 Kernel Comparison Benchmarks

Benchmarks comparing MAX kernels against external baselines on NVIDIA B200 GPUs.

## Benchmarks

| Target | Description | Baselines |
|--------|-------------|-----------|
| `bench_prefill` | MHA prefill (variable-length) | FlashInfer, flash-attention |
| `bench_decode` | MHA decode (single token) | FlashInfer (TRT-LLM backend) |
| `bench_mla_decode` | Multi-head Latent Attention decode | FlashInfer (TRT-LLM MLA) |
| `bench_grouped_gemm` | Grouped GEMM | DeepGEMM |
| `bench_ep_baseline` | Expert Parallelism dispatch/combine (adhoc) | DeepEP (optional) |

## Running Benchmarks

```bash
# Via kbench (recommended)
kbench bench_prefill.yaml
kbench bench_decode.yaml
kbench bench_mla_decode.yaml
kbench bench_grouped_gemm.yaml

# Via Bazel directly
./bazelw run //max/kernels/benchmarks/misc/comparison:bench_prefill
./bazelw run //max/kernels/benchmarks/misc/comparison:bench_decode
./bazelw run //max/kernels/benchmarks/misc/comparison:bench_ep_baseline
```

## Architecture

### Wheel Infrastructure

External baselines require SM100-specific builds not available on PyPI. The infrastructure:

```text
MODULE.bazel                             # http_file: fetch wheels from S3
    ↓
bazel/pip/blackwell_bench/               # pycross_wheel_library targets
    ↓
max/kernels/benchmarks/misc/comparison/  # modular_py_binary executables
```

**Key files:**

- `MODULE.bazel` - `http_file` rules fetch pre-built wheels from S3
- `bazel/pip/blackwell_bench/BUILD.bazel` - `pycross_wheel_library` targets
- `bazel/pip/blackwell_bench_wheels.bzl` - Helper macro `blackwell_bench_wheel()`

### Adding a Dependency

1. **Fetch wheel** in `MODULE.bazel`:

   ```starlark
   http_file(
       name = "mylib_sm100_wheel",
       downloaded_file_path = "mylib-1.0.0-cp312-cp312-linux_x86_64.whl",
       sha256 = "...",
       urls = ["https://modular-bazel-artifacts-public.s3.amazonaws.com/artifacts/..."],
   )
   ```

2. **Create library target** in `bazel/pip/blackwell_bench/BUILD.bazel`:

   ```starlark
   pycross_wheel_library(
       name = "mylib",
       wheel = "@mylib_sm100_wheel//file",
       deps = ["@modular_pip_lock_file_repo//deps:torch"],
   )
   ```

3. **Use in benchmark**:

   ```starlark
   deps = [blackwell_bench_wheel("mylib")]
   ```

## Rebuilding Wheels

Use `setup_bench_env.py` to build reproducible wheels from source. Repositories
are **automatically cloned** to `~/.cache/blackwell_bench/` if not present.

```bash
# Build all wheels (auto-clones repos if needed)
python setup_bench_env.py --build-wheels --wheel-dir ./sm100_wheels

# Build specific wheel
python setup_bench_env.py --build-wheels --wheel-dir ./sm100_wheels \
    --no-flashinfer --no-flashattn  # Only DeepGEMM

# Use custom source path (skips auto-clone)
python setup_bench_env.py --build-wheels --deepgemm-src ~/my/DeepGEMM

# Upload to S3
./utils/upload-public-bazel-artifact.sh deep_gemm sm100 ./sm100_wheels/*.whl
```

Then update `MODULE.bazel` with the new URL and sha256.

### Source Repositories

Repos are cloned from these URLs (shallow clone, `--depth 1`):

| Package | Repository |
|---------|------------|
| DeepGEMM | <https://github.com/deepseek-ai/DeepGEMM> |
| FlashInfer | <https://github.com/flashinfer-ai/flashinfer> |
| flash-attention | <https://github.com/Dao-AILab/flash-attention> |

Default cache location: `~/.cache/blackwell_bench/`

## Workarounds

### nvidia-cutlass-dsl .pth files

Bazel doesn't process `.pth` files. The `nvidia-cutlass-dsl` package uses
one to add `python_packages/` (containing `cutlass`) to sys.path. We use
the `imports` attribute:

```starlark
imports = ["../../../../rules_pycross++.../python_packages"]
```

### flash-attention pure Python wheel

The flash-attention wheel is built as pure Python
(`FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`) because we only use the cute
interface (JIT-compiled via nvidia-cutlass-dsl). The benchmark code creates
a stub `flash_attn` module to bypass `__init__.py` which imports the
missing CUDA extension.

### S3 URL encoding for wheel filenames

PEP 440 local versions use `+` (e.g., `2.2.0+38f8ef7`), which appears in
wheel filenames like `deep_gemm-2.2.0+38f8ef7-cp312-...whl`. S3 objects
with `+` in filenames require URL encoding (`%2B`) for HTTP access, but
Bazel's `http_file` doesn't encode special characters, causing 403 errors.

**Solution:** `setup_bench_env.py --build-wheels` automatically renames
wheels to replace `+` with `_` (e.g., `deep_gemm-2.2.0_38f8ef7-...whl`).
The wheel contents are unchanged—only the filename is sanitized.

## EP Baseline Benchmark (Adhoc)

The `bench_ep_baseline` benchmark compares MAX EP dispatch/combine operations
against DeepSeek-AI's DeepEP library for expert parallelism.

### Running

```bash
br //max/kernels/benchmarks/misc/comparison:bench_ep_baseline -- \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288
```

### DeepEP Installation (Optional)

DeepEP is an optional baseline. If not installed, only MAX results are shown.

```bash
# Install DeepEP with NVSHMEM support
NVSHMEM_DIR=/path/to/installed/nvshmem python -m pip install \
    git+https://github.com/deepseek-ai/DeepEP.git
```

**Requirements:**

- NVSHMEM library installed
- CUDA 12.3+ for SM90 or CUDA 11.0+ for SM80
- NVLink for intranode communication
- RDMA network for internode communication

See [DeepEP repository](https://github.com/deepseek-ai/DeepEP) for detailed
setup instructions.

### Parameters

```text
--num-tokens         Number of tokens per device (default: 128)
--hidden             Hidden size (default: 7168)
--num-topk           Number of experts per token (default: 8)
--num-experts        Total number of experts (default: 288)
--dispatch-dtype     Dispatch dtype: bf16/fp16/fp32 (default: bf16)
--combine-dtype      Combine dtype: bf16/fp16/fp32 (default: bf16)
--iters              Number of test iterations (default: 30)
--gpus-per-node      GPUs to use (0 = all visible, default: 0)
--profile            Print detailed Kineto profiling tables
```

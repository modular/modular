# B200 Kernel Comparison Benchmarks

Benchmarks comparing MAX kernels against external baselines on NVIDIA B200 GPUs.

## Benchmarks

| Target | Description | Baselines |
|--------|-------------|-----------|
| `bench_prefill` | MHA prefill (variable-length) | FlashInfer, flash-attention |
| `bench_decode` | MHA decode (single token) | FlashInfer (TRT-LLM backend) |
| `bench_mla_decode` | Multi-head Latent Attention decode | FlashInfer (TRT-LLM MLA) |
| `bench_grouped_gemm` | Grouped GEMM | DeepGEMM |

## Running Benchmarks

```bash
# Via Bazel (recommended)
./bazelw run //Kernels/benchmarks/comparison:bench_prefill
./bazelw run //Kernels/benchmarks/comparison:bench_decode

# Or use aliases after sourcing start-modular.sh
br //Kernels/benchmarks/comparison:bench_prefill
```

## Architecture

### Wheel Infrastructure

External baselines require SM100-specific builds not available on PyPI. The infrastructure:

```text
MODULE.bazel                    # http_file: fetch wheels from S3
    ↓
bazel/pip/blackwell_bench/      # pycross_wheel_library targets
    ↓
Kernels/benchmarks/comparison/  # modular_py_binary executables
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

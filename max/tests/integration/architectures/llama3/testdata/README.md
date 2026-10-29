# Pipelines test data

## Test data

The tiny llama checkpoints are committed fixtures in this directory, and the
llama3 tests consume them through the `testdata` filegroup in this
directory's `BUILD.bazel`:

- `tiny_llama.gguf` — float32 checkpoint
- `tiny_llama_bf16.gguf` — bfloat16 checkpoint

The generator that produced them (`gen_tiny_llama`) was removed together with
the ModularFramework component in commit `34c1a0a1f07`, so there is currently
no supported way to regenerate them. If you need to, recover the script from
git history:

```bash
git show 34c1a0a1f07^:ModularFramework/utils/gen_tiny_llama.py
```

Note: hidden dim must be a multiple of 8 (required for `fused_qkv_matmul`
kernel alignment).

The golden-values workflow that used to accompany these checkpoints
(`evaluate_llama`, the `test_llama_golden` and `torch_llama_golden` archives)
was removed with the old golden tests. See git history if you need those
instructions.

## Running the tests

The llama3 test targets are:

```bash
# GPU kernel tests (fp4 attention/linear, rms norm)
./bazelw test //max/tests/integration/architectures/llama3:tests_gpu

# Data-parallel graph test (runs on CPU)
./bazelw test //max/tests/integration/architectures/llama3:test_data_parallel_graph
```

`tests_gpu` requires a GPU. On a machine without one, use the remote execution
configs instead, for example:

```bash
./bazelw test --config=remote-b200 //max/tests/integration/architectures/llama3:tests_gpu
```

## Tokenizer data

`special_tokens_map.json`, `tokenizer_config.json` and `tokenizer.json` are
copied from the
[meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
HuggingFace model.

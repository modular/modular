
# Generate Test Data

Use these scripts to randomly generate a tiny llama checkpoint and compute
golden values.

## Tinyllama

To facilitate fast cycle times during local development, a tiny llama test
is included alongside the full weights. To re-generate the tiny llama
checkpoint, use the `gen_tiny_llama` target:

```bash
TESTDATA_DIR="$MODULAR_PATH/SDK/integration-test/pipelines/python/testdata"

./bazelw run //ModularFramework/utils:gen_tiny_llama --\
    --output=$TESTDATA_DIR/tiny_llama.gguf \
    --quantization-encoding=float32 \
    --n-layers=1 \
    --n-heads=1 \
    --n-kv-heads=1 \
    --hidden-dim=10
```

Then, you can use `evaluate_llama` to generate the golden values. The
CLI supports encoding (q4_k, float32, bfloat16) and model (tinyllama, llama3_1) parameters.
If either are not set they default to "all", so the typical command simply
points to the modular root so that the CLI can write the golden files for
each encoding/model pair to the test data folder.

```bash
./bazelw run //SDK/integration-test/pipelines/python:evaluate_llama --\
    --modular-path /path/to/modular \
    --encoding q4_k \ # float32, q4_k, bfloat16, or all (default)
    --model tinyllama # llama3_1, tinyllama, or all (default)
```

## Tokenizer data

`special_tokens_map.json`, `tokenizer_config.json` and `tokenizer.json` are
copied from the [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
HuggingFace model.

# Running the tests

The test target for CPU tests is:

```bash
./bazelw test //SDK/integration-test/pipelines/python:tests
```

For local development, it may be convenient to just run the tiny llama
tests, which you can select out using a pytest filter:

```bash
./bazelw test //SDK/integration-test/pipelines/python:tests --test_arg="-k test_llama[tiny-float32-llama3_1]"
```

Note that GPU tests have a different target:

```bash
./bazelw test //SDK/integration-test/pipelines/python:tests_gpu
```

# Registering new golden test data

We use `http_archive` to bundle and download the golden values from S3. To
add a file to this archive, you need to:

1. Download the existing archive by `cat WORKSPACE.bazel | grep test_llama_golden`
, finding the s3 URL (at time of writing this
was `https://modular-bazel-artifacts-public.s3.amazonaws.com/artifacts/test_llama_golden/1/bc9c5e599b005b20d8b176384c869808c6cf242f397b5fb5e694570dfe87dd0c/test_llama_golden.tar.gz`)
and downloading to your local machine (e.g., with wget).
1. Untar the existing archive `tar -xvf test_llama_golden.tar.gz`.
1. Add any additional files you want to register alongside.
1. Run `./utils/upload-public-bazel-artifact.sh test_llama_golden 1 *golden.json`
to package and upload the latest version.
1. The result of ^ will be a snippet like:

```bash
http_archive(
    name = "test_llama_golden",
    build_file_content = """
filegroup(
    name = "test_llama_golden",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)""",
    sha256 = "bc9c5e599b005b20d8b176384c869808c6cf242f397b5fb5e694570dfe87dd0c",
    url = "https://modular-bazel-artifacts-public.s3.amazonaws.com/artifacts/test_llama_golden/1/bc9c5e599b005b20d8b176384c869808c6cf242f397b5fb5e694570dfe87dd0c/test_llama_golden.tar.gz",
)
```

1. Find the associated section in `WORKSPACE.bazel`, delete it, and replace
with this newly generated value.

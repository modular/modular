
# Generate Test Data

Use these scripts to randomly generate a tiny llama checkpoint and compute
golden values.

```bash
TESTDATA_DIR="$MODULAR_PATH/SDK/integration-test/pipelines/python/testdata"

bazel run //ModularFramework/utils:gen_tiny_llama --\
    --output=$TESTDATA_DIR/tiny_llama.gguf \
    --quantization-encoding=float32 \
    --n-layers=1 \
    --n-heads=1 \
    --n-kv-heads=1 \
    --hidden-dim=10

bazel run //SDK/integration-test/pipelines/python:evaluate_llama --\
    --output=$TESTDATA_DIR/tiny_llama_golden.json \
    --weight-path=$TESTDATA_DIR/tiny_llama.gguf
```

## Tokenizer data

`special_tokens_map.json`, `tokenizer_config.json` and `tokenizer.json` are
copied from the [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
HuggingFace model.

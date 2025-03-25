# Llama 3.1

**Language:** Python

**API**: MAX Graph

This pipeline provides optimized support for the `LlamaForCausalLM` family
of large language models, as exemplified by the Llama 3.1 text completion
model. The model itself has been constructed in Python
using the [MAX Graph API](https://docs.modular.com/max/graph/).

The MAX Graph API provides an accessible interface to the construction of
flexible accelerated compute graphs, which are then optimized by the MAX
Engine's advanced graph compiler. This pipeline showcases how a large language
model can be fully defined using Python and MAX Graphs and then compiled for
optimal inference performance via the MAX Engine.

## Model

[Llama 3.1](https://llama.meta.com/llama3/) is an open source large language
model released by Meta. The structure of this implementation was inspired by
Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) and its [Mojo
port by Aydyn Tairov](https://github.com/tairov/llama2.mojo).

The text completion demo is compatible with the the official Llama 3
[text completion demo](https://github.com/meta-llama/llama3/blob/14aab0428d3ec3a9596f1dea06d9c564f9c0e35f/example_text_completion.py).

The default settings for this pipeline use the 8B set of pretrained weights in
`q4_k` quantized encodings.

## Usage

The easiest way to try out this pipeline is with our Magic command-line tool.

1. Install Magic on macOS and Ubuntu with this command:

   ```shell
   curl -ssL https://magic.modular.com | bash
   ```

   Then run the source command that's printed in your terminal.

   To see the available commands, you can run `magic --help`.
   [Learn more about Magic here](https://docs.modular.com/magic).

2. Clone the MAX examples repository:

   If you don't already have a local clone of this repository, create one via:

   ```shell
   git clone https://github.com/modular/max.git
   ```

   The following instructions assume that you're present within this pipeline's
   directory, and you can change to it after cloning:

   ```shell
   cd max/src/max/pipelines
   ```

3. Now run the Llama 3.1 text completion demo with the following command:

   ```shell
   magic run generate --model-path="modularai/Llama-3.1-8B-Instruct-GGUF" --prompt "I believe the meaning of life is"
   ```

4. Host a chat completion endpoint via MAX Serve.

   MAX Serve provides functionality to host performant OpenAI compatible
   endpoints using the FastAPI framework.

   You can configure the pipeline to be hosted by using the `--serve` argument.
   For example:

   ```shell
   magic run serve --model-path="modularai/Llama-3.1-8B-Instruct-GGUF"
   ```

   A request can be submitted via a cURL command.

   ```shell
   curl -N http://localhost:8000/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
       "model": "modularai/llama-3.1",
       "stream": true,
       "messages": [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "Who won the world series in 2020?"}
       ]
   }'
   ```

   Additionally, finetuned weights hosted on Hugging Face for any compatible
   `LlamaForCausalLM` model can be used with this optimized architecture
   via the `serve` command:

   ```shell
   magic run serve --model-path=meta-llama/Llama-3.2-1B
   ```

## Options

The following command-line options are available to customize operation of the
pipeline:

- `--max-length`: Controls the maximum length of the text sequence (includes the
  input tokens). Defaults to huggingface model config's max sequence length.
- `--max-new-tokens`: The maximum number of new tokens to generate. If a -1
  value is provided, the model will continue to generate tokens for the entire
  context length. (Default value: -1)
- `--prompt`: The text prompt to use for further generation.
- `--quantization-encoding`: The encoding to use for a datatype that can be
  quantized to a low bits per weight format. The options for quantized formats
  will download and cache default weights, but `float32` requires the use of
 `--weight-path` to specify locally downloaded full-precision weights for use
  in the model.
  Valid values: `q4_0`, `q4_k`, `q6_k`, `bfloat16`, `float32`.
  (Default value: `float32`).
- `--top-k`: Limits the sampling to the K most probable tokens. Default is 1.
- `--version`: Selects which version in the Llama 3 family to use.
  Valid values: `3`, `3.1`.
  (Default value: `3.1`)
- `--weight-path`: Overrides the default URL, and allows for an
  already-downloaded pretrained weight file to be used with the model.
- `--max-batch-size`: Specifies the maximum batch size to be used.
  Default is 1.
- `--devices`: Specifies the device(s) to use for the model. This can optionally
  be "cpu", "gpu", or a comma-separated list of GPU ID(s) like "gpu-0,gpu-1".
  Defaults to the first available GPU, or "cpu" if no GPUs are available.

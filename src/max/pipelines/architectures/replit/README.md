# Replit Code V1.5 3B

**Language:** Python

**API**: MAX Graph

This pipeline demonstrates code completion from an initial prompt using
Replit's Code V1.5 3B large language model. The model itself has been
constructed in Python using the
[MAX Graph API](https://docs.modular.com/max/graph/).

The MAX Graph API provides an accessible interface to the construction of
flexible accelerated compute graphs, which are then optimized by the MAX
Engine's advanced graph compiler. This pipeline showcases how a large language
model can be fully defined using Python and MAX Graphs and then compiled for
optimal inference performance via the MAX Engine.

## Model

[Replit Code](https://blog.replit.com/replit-code-v1_5) is an open source code
generation model trained on permissively licensed code and released by
[Replit](https://replit.com). The V1.5, 3B variant is the basis for this
implementation, and weights are
[obtained via Hugging Face](https://huggingface.co/replit/replit-code-v1-3b).

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
   cd max/src/max/
   ```

3. Now run the Replit code completion demo with the following command:

   ```shell
   magic run replit --prompt "def fibonacci(n):"
   ```

4. Host a code completion endpoint via MAX Serve.

   MAX Serve provides functionality to host performant OpenAI compatible
   endpoints using the FastAPI framework.

   You can configure the pipeline to be hosted by using the `--serve` argument.
   For example:

   ```shell
   magic run replit --serve
   ```

   A request can be submitted via a cURL command.

   ```shell
   curl -N http://localhost:8000/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
       "model": "replit/replit-code-v1_5-3b",
       "stream": true,
       "messages": [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "def fibonacci(n)"}
       ]
   }'
   ```

## Options

The following command-line options are available to customize operation of the
pipeline:

- `--max-length`: Controls the maximum length of the text sequence
  (includes the input tokens).
  (Default value: 512)
- `--max-new-tokens`: The maximum number of new tokens to generate. If a -1
  value is provided, the model will continue to generate tokens for the entire
  context length. (Default value: -1)
- `--prompt`: The text prompt to use for further generation.
- `--quantization-encoding`: The encoding to use for a datatype that can be
  quantized to a low bits per weight format. The options for quantized formats
  will download and cache default weights, but `float32` requires the use of
 `--weight-path` to specify locally downloaded full-precision weights for use
  in the model.
  Valid values: `bfloat16`, `float32`.
  (Default value: `float32`).
- `--save-to-serialized-model-path`: If specified, writes the serialized model
  to this path.
- `--serialized-model-path`: If specified, tries to load a serialized model
  from this path.
- `--top-k`: Limits the sampling to the K most probable tokens. Default is 1.
- `--weight-path`: Overrides the default URL, and allows for an
  already-downloaded pretrained weight file to be used with the model.
- `--max-batch-size`: Specifies the maximum batch size to be used.
  Default is 1.
- `--devices`: Specifies the device(s) to use for the model. This can optionally
  be "cpu" or a comma-separated list of GPU ID(s). Defaults to the first
  available GPU.

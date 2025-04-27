# MAX Pipelines

These are end-to-end pipelines that demonstrate the power of
[MAX](https://docs.modular.com/max/) for accelerating common AI workloads, and
more. Each of the supported pipelines can be served via an OpenAI-compatible
endpoint.

MAX can also serve most PyTorch-based large language models that are
present on Hugging Face, although not at the same performance as native MAX
Graph versions.

## Usage

The easiest way to try out any of the pipelines is with our Magic command-line
tool.

1. Install Magic on macOS and Linux with this command:

   ```shell
   curl -ssL https://magic.modular.com | bash
   ```

   Then run the source command that's printed in your terminal.

   To see the available commands, you can run `magic --help`.
   [Learn more about Magic here](https://docs.modular.com/magic).

2. Install max-pipelines command to run the pipelines.

   ```shell
   magic global install max-pipelines
   ```

3. Serve a model.

   ```shell
   max-pipelines serve --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B
   ```

See [https://builds.modular.com/](https://builds.modular.com/) to discover many
of the models supported by MAX.

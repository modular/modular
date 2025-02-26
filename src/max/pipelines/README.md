# MAX Pipelines

These are end-to-end pipelines that demonstrate the power of
[MAX](https://docs.modular.com/max/) for accelerating common AI workloads, and
more. Each of the supported pipelines can be served via an OpenAI-compatible
endpoint.

MAX can also serve most PyTorch-based large language models that are
present on Hugging Face, although not at the same performance as native MAX
Graph versions.

## Pipelines

Highly optimized MAX Graph implementations exist for several core model
architectures. These include:

- [Llama 3.1](llama3): A text completion pipeline using the Llama 3.1 model,
implemented using the MAX Graph API. This pipeline contains everything
needed to run a self-hosted large language model in the `LlamaForCausalLM`
family with state-of-the-art serving throughput.
- [Mistral](architectures/mistral): Support for the `MistralForCausalLM` family
of text completion models, by default using the Mistral NeMo 12B model. This pipeline
has been tuned for performance using the MAX Graph API.
- [Replit Code](architectures/replit): Code generation via the
Replit Code V1.5 3B model, implemented using the MAX Graph API.

Instructions for how to run each pipeline can be found in their respective
subdirectories, along with all configuration parameters. A shared driver is
used to execute the pipelines.

## Usage

The easiest way to try out any of the pipelines is with our Magic command-line
tool.

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

   The following instructions assume that you're present within this
   directory, and you can change to it after cloning:

   ```shell
   cd max/src/max/
   ```

3. Now run one of the text completion demos with any of following commands:

   ```shell
   magic run generate --model-path=modularai/llama-3.1 --prompt "Why is the sky blue?"
   magic run generate --model-path=modularai/replit-code-1.5 --prompt "def fibonacci(n):"
   magic run generate --model-path=mistralai/Mistral-Nemo-Instruct-2407 --prompt "Why is the sky blue?"
   ```

4. Host a chat completion endpoint via MAX Serve.

   MAX Serve provides functionality to host performant OpenAI compatible
   endpoints using the FastAPI framework.

   Finetuned weights hosted on Hugging Face can be used with one
   of these optimized pipeline architectures when serving via the `serve`
   command:

   ```shell
   magic run serve --model-path=modularai/llama-3.1
   ```

   A request can then be submitted via a cURL command.

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

## Verified Hugging Face model architectures

If you provide a repository ID for a Hugging Face large language model
that does not currently have an optimized MAX Graph implementation, MAX
falls back to serving a PyTorch eager version of the model.

The following table lists the model architectures tested to work with MAX.

| **Architecture** | **Example Model** |
| --- | --- |
| AquilaForCausalLM | BAAI/Aquila-7B |
| ChatGLMModel | THUDM/chatglm3-6b |
| GPT2LMHeadModel | openai-community/gpt2 |
| GPTJForCausalLM | EleutherAI/gpt-j-6b |
| LlamaForCausalLM | meta-llama/Llama-3.2-3B-Instruct |
| LlamaForCausalLM | Skywork/Skywork-o1-Open-Llama-3.1-8B |
| LlamaForCausalLM | deepseek-ai/deepseek-coder-1.3b-instruct |
| PhiForCausalLM | microsoft/phi-2 |
| Phi3ForCausalLM | microsoft/Phi-3-mini-4k-instruct |
| GraniteForCausalLM | ibm-granite/granite-3.1-8b-instruct |
| Qwen2ForCausalLM | Qwen/Qwen2.5-1.5B-Instruct |
| OlmoForCausalLM | allenai/OLMo-1B-0724-hf |

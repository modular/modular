# Pipeline Dataset Evaluation Configurations

These files are used by `.github/workflows/pipelineDatasetEvaluation.yaml`.
Paths in this directory are expected to be named `<HF repo ID>.sh`. Each file
is a shell snippet that will be sourced. It is expected to set variables
affecting the invocation.

Current variables used:

- **`batch_size`:** Optional. The server's `--max-batch-size` and lm-eval's
  `num_concurrent`. When unset, the server uses the architecture's default and
  lm-eval sends one request at a time.
- **`max_length`:** Optional. The server's `--max-length` and lm-eval's
  `max_length`. When unset, the server uses the architecture's default.
  Required with `local-completions`, which truncates prompts to it.
- **`lm_eval_model`:** Optional. The lm-eval model type: `local-completions`
  (default) applies the HF tokenizer's chat template on the client and calls
  `/v1/completions`; `local-chat-completions` calls `/v1/chat/completions`,
  so the server applies its own chat template, and adds `--apply_chat_template`.
  Use it for models whose HF tokenizer has no `chat_template`.
- **`extra_pipelines_args`:** Optional, array.
- **`extra_lm_eval_args`:** Optional, array. All instruct-tuned models will
  probably want to add the recommended instruct-evaluation arguments here
  (`--apply_chat_template --fewshot_as_multiturn`).
- **`tasks`:** Optional. Defaults to `leaderboard`.

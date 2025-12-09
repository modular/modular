# Pipeline Dataset Evaluation Configurations

These files are used by `.github/workflows/pipelineDatasetEvaluation.yaml`.
Paths in this directory are expected to be named `<HF repo ID>.sh`.  Each file
is a shell snippet that will be sourced.  It is expected to set variables
affecting the invocation.

Current variables used:

- **`batch_size`:** Required.
- **`max_length`:** Required.
- **`extra_pipelines_args`:** Optional, array.
- **`extra_lm_eval_args`:** Optional, array.  All instruct-tuned models will
  probably want to add the recommended instruct-evaluation arguments here
  (`--apply_chat_template --fewshot_as_multiturn`).
- **`tasks`:** Optional.  Defaults to `leaderboard`.

##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# shellcheck disable=SC2034  # Variables are used when sourced
batch_size=64
max_length=50000

extra_pipelines_args=(
  # --enable-echo  # Not needed for gsm8k_cot_llama
  --device-memory-utilization=0.6
)
extra_lm_eval_args=(
  --log_samples
  --apply_chat_template
  --fewshot_as_multiturn
  "--gen_kwargs=max_gen_toks=4096,seed=42,temperature=0"
)

# Increase generation timeout to 5000s
extra_lm_model_args=(
  timeout=5000
)

# These parameters (max_length=50000, max_gen_toks=4096, temp=0) are used with
# following thinking task:
tasks=gsm8k_cot_llama

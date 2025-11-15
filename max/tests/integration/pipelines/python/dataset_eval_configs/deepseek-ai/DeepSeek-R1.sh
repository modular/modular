##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

batch_size=32
max_length=4096
extra_pipelines_args=(
  --enable-echo
  --device-memory-utilization=0.6
  # TODO(MODELS-846): Currently required to avoid CUDA errors.
  --prefill-chunk-size=256
)
extra_lm_eval_args=(
  --log_samples
  --apply_chat_template
  --fewshot_as_multiturn
)

# Increase generation timeout to 5000s
extra_lm_model_args=(
  timeout=5000
)

##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

batch_size=50
max_length=4096
extra_pipelines_args=(--enable-echo)
extra_lm_eval_args=(
  --apply_chat_template
  --fewshot_as_multiturn
)

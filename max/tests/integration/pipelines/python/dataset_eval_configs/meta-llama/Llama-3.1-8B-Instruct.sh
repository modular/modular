##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

batch_size=512
max_length=16384  # Llama-3.1 supports longer contexts, increase from 4096
extra_pipelines_args=(--enable-echo)
extra_lm_eval_args=(
  --apply_chat_template
  --fewshot_as_multiturn
)

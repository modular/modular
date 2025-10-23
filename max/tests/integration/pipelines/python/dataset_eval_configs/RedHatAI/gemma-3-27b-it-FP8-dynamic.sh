##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# TODO(MODELS-815): Run all tasks after Gemma3 supports logprobs.
tasks=leaderboard_ifeval
batch_size=512
max_length=8192
extra_pipelines_args=(--enable-echo)
extra_lm_eval_args=(
  --apply_chat_template
  --fewshot_as_multiturn
)

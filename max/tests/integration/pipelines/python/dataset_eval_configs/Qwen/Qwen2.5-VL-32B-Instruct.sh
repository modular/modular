##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# shellcheck disable=SC2034  # Variables are used when sourced
batch_size=256
max_length=128000

evaluator=mistral-evals
tasks=mathvista,chartqa,docvqa,mmmu

extra_pipelines_args=(
  --trust-remote-code
  --no-enable-chunked-prefill
)

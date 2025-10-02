##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

batch_size=128
max_length=128000

evaluator=mistral-evals
tasks=mathvista,chartqa,docvqa

extra_pipelines_args=(
  --trust-remote-code
  --no-enable-prefix-caching
  --no-enable-chunked-prefill
)

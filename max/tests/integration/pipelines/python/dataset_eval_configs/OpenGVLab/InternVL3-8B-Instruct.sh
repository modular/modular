##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

batch_size=512
max_length=8192

evaluator=mistral-evals
tasks=mathvista,chartqa,docvqa

# NOTE: Change `max_dynamic_patch` here to sweep the number of patches.
# This affects the number of input vision tokens.
extra_pipelines_args=(
  --trust-remote-code
  --device-memory-utilization 0.8
  --no-enable-prefix-caching
  --no-enable-chunked-prefill
  --vision-config-overrides '{"max_dynamic_patch": 12}'
)


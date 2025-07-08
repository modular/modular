##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

batch_size=128
max_length=8192

evaluator=mistral-evals
tasks=mathvista,chartqa,docvqa

extra_pipelines_args=(--trust-remote-code --device-memory-utilization 0.8 --no-enable-prefix-caching --no-enable-chunked-prefill)
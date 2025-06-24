##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

batch_size=1
max_length=8192
evaluator=mistral-evals
tasks=mathvista,chartqa,docvqa

extra_pipelines_args=(--trust-remote-code)


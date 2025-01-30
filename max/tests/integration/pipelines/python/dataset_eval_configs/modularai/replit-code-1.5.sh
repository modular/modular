##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# TODO(AIPIPE-252): Replit broken with batch size > 1
batch_size=1
max_length=4096
extra_pipelines_args=(
  --max-new-tokens=512
  --trust-remote-code
)
tasks=human_eval

##===----------------------------------------------------------------------===##
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##

# shellcheck disable=SC2034  # Variables are used when sourced

# DeepSeek-V4-Flash-0731 at TP2 on 2x B200. The limits match the Mach export
# (mach/docs/models/deepseek-v4-flash.md): batch 32, 32K context, 4,096-token
# chunks. Mach serves it behind the orchestrator with
# `--tokenizer-config {"mode":"optimized"} --format-spec deepseek_v4`.
batch_size=32
max_length=32768

extra_pipelines_args=(
  --prefer-module-v3
  --max-batch-input-tokens 4096
  --enable-structured-output
)

model_profile=deepseek_v4
scenarios=
k2vv_mode=
circuit_breaker=0

# Thinking is on by default, so a request without max_tokens can reason to
# the context limit; the timeout covers that at 32 streams.
extra_fuzz_args="--max-concurrency $batch_size --timeout 900 --max-context-length $max_length"

# MAX Serve and Mach read the same list.
exclude_files+=("$(dirname "${BASH_SOURCE[0]}")/DeepSeek-V4-Flash-0731-known-failures.txt")

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
batch_size=64
max_length=32768

extra_pipelines_args=(
  --prefer-module-v3
  --device-memory-utilization=0.9
  --enable-prefix-caching
  --enable-structured-output
  # 400 on tool schemas the grammar cannot fully enforce, instead of
  # compiling a partial grammar the model can slip non-conforming
  # arguments through.
  --tool-call-policy default_strict_true_and_reject_unsupported
  # DSpark speculative decoding (speculative_method=dspark, K=7), per the
  # nemotron_h_modulev3 lightning_bf16_dspark recipe, on the NVFP4 target.
  --draft-model-path=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
  --speculative-method=dspark
  --num-speculative-tokens=7
  # Unscaled FP8 target KV cache, as in the lightning_nvfp4_dspark recipe.
  # The DSpark draft cache stays BF16.
  --kv-cache-format=float8_e4m3fn
)
model_profile=nemotron
scenarios=
k2vv_mode=
circuit_breaker=0

# Known failures were observed on the NVFP4 single-GPU lane (no speculative
# decoding, TP=1). Reusing the list until this lane has its own failure data.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/nemotron-known-failures.sh"

# TODO(SERVOPT-1726): fix the DSpark + structured-output truncation, then drop
# these two exclusions.
# Failures observed only on this lane: the response is cut off mid-string
# ("Unterminated string starting at: line 1 column 13") on structured output
# under DSpark speculative decoding. One test fails per run and which one
# varies (run 38523298418: reasoning-disabled-response-format-json-object;
# run 38525961906: structured_json_streaming), so this is likely a MAX bug
# rather than test-specific; excluded to get a baseline, not a fix. Not in the
# single-GPU list, so excluded here rather than in the shared .txt.
exclude="${exclude},openrouter_tests:reasoning-disabled-response-format-json-object"
exclude="${exclude},openrouter_tests:structured_json_streaming"

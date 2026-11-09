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

# Deliberately empty: no known-failure exemptions. A failing test is a bug to
# fix, not an entry to add here.
# TODO(SERVOPT-1726): MAX Serve under DSpark cuts structured output off
# mid-string, so the MAX lane can fail
# openrouter_tests:reasoning-disabled-response-format-json-object and
# openrouter_tests:structured_json_streaming. Both pass on Mach.
exclude=

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

# The ModuleV3 2xMI355 deployment, as in
# glm5_next_modulev3/recipes/glm_5_3_flash_fp8_tp_ep_2x_mi355.yaml, with prefix
# caching on so needle_prefix_cache tests it, as Mach serves it.
batch_size=8
max_length=16384

extra_pipelines_args=(
  --prefer-module-v3
  --ep-size 2
  --device-memory-utilization=0.78
  --enable-structured-output
)

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
# GLM-5.3 uses the GLM-4.5+ tool format the glm-5.1 scenarios target.
model_profile=glm-5.1
scenarios=
k2vv_mode=
circuit_breaker=0

# GLM-5.3's chat template has no thinking toggle: it always opens <think>, and
# "none" can only lower reasoning_effort to its floor rung. Both engines return
# reasoning on these thinking-off requests.
exclude="basic_reasoning_and_tool_usage:response-format-thinking-off"
exclude="${exclude},basic_reasoning_and_tool_usage:open-ended-thinking-off"
exclude="${exclude},basic_reasoning_and_tool_usage:special-token-content-thinking-off"
exclude="${exclude},basic_reasoning_and_tool_usage:content-parts-thinking-off"
exclude="${exclude},openrouter_tests:reasoning-disabled"

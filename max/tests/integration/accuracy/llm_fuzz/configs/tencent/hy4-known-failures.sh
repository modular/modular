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

# shellcheck disable=SC2034  # `exclude` is consumed when this file is sourced.

# Model behaviour that vLLM 0.30.0 serving the same checkpoint with its hy_v4
# parsers fails too, from the full validation run of 2026-10-14.
_hy4_vllm_failures=(
  # basic_reasoning_and_tool_usage: the model reasons until max_tokens and
  # writes no content (vLLM: FAIL, finish=length, 32777 reasoning chars).
  'basic_reasoning_and_tool_usage:special-token-content-thinking-on'
  # openrouter_tests: the chat template closes a trailing assistant message
  # with its hy_end token and adds no new generation prompt, so the model
  # usually ends the turn at once (vLLM: FAIL; 5 of 5 replays had empty
  # content).
  'openrouter_tests:assistant-prefill'
  # tc_advanced: on some iterations the model writes its call inside an
  # unclosed thinking block (vLLM: FAIL, 17/20).
  'tc_advanced:sequential_soak_20'
)

# Model behaviour seen only on MAX runs; vLLM 0.30.0 passed its single draw.
_hy4_model_failures=(
  # tc_basics: with tool_choice=auto the model sometimes spells the call as
  # JSON or a [tool_call] block instead of its own tags, so no call parses
  # (3 of 40 MAX replays; vLLM: INTERESTING, budget exhausted).
  'tc_basics:20_tools_auto'
)

# Intermittent: fail on some draws only.
_hy4_intermittent_failures=(
  # tc_schema_enforcement: the prompt asks only for values the schema
  # forbids; with the grammar blocking them the model sometimes returns a
  # shorter array than the check's minimum. That array is schema-valid: the
  # schema has no minItems.
  'tc_schema_enforcement:ref_defs_array_of_enums[tool_choice=required]'
  'tc_schema_enforcement:anyof_deeply_nested[tool_choice=required]'
  'tc_schema_enforcement:anyof_deeply_nested[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_nested_array_in_object[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_cross_referencing[tool_choice=required]'
  # tc_schema_enforcement: the prompt asks for an undeclared key; the schema
  # allows it (additionalProperties unset), the check does not.
  'tc_schema_enforcement:ref_defs_adversarial_all_constraints[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_adversarial_all_constraints[tool_choice=required]'
)

# Join into the comma-separated form the --exclude flag expects.
exclude=$(
  IFS=,
  printf "%s" "${_hy4_vllm_failures[*]},${_hy4_model_failures[*]},${_hy4_intermittent_failures[*]}"
)

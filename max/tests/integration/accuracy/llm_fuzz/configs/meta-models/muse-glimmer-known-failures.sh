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

# Model behaviour that vLLM 0.30.0 serving the same checkpoint with its
# muse_glimmer parsers fails too, and that failed at least once in three MAX
# validation runs on 2026-10-02. The chat template has no thinking-off
# switch, and without a grammar the model may decline a tool call, pick
# another tool, or write arguments that break the schema.
_muse_glimmer_vllm_failures=(
  # agentic_correctness
  'agentic_correctness:case_insensitive_tool_name'
  # basic_reasoning_and_tool_usage
  'basic_reasoning_and_tool_usage:content-parts-thinking-off'
  'basic_reasoning_and_tool_usage:open-ended-thinking-off'
  'basic_reasoning_and_tool_usage:special-token-content-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-auto-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-calculate-required-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-thinking-off'
  # openrouter_tests
  'openrouter_tests:reasoning-disabled'
  'openrouter_tests:reasoning-disabled-tool-choice-required'
  'openrouter_tests:streaming_reasoning_tool_choice_required'
  # production_resilience
  'production_resilience:streaming_vs_nonstreaming_match'
  # tc_basics
  'tc_basics:parallel_tc_correct_indices'
  'tc_basics:parallel_tc_two_tools'
  # tc_schema_enforcement
  'tc_schema_enforcement:anyof_string_or_integer[tool_choice=auto]'
  'tc_schema_enforcement:anyof_string_or_integer[tool_choice=required]'
  'tc_schema_enforcement:anyof_with_enum_branch[tool_choice=auto]'
  'tc_schema_enforcement:const_value_fail_open[tool_choice=auto]'
  'tc_schema_enforcement:const_value_fail_open[tool_choice=required]'
  'tc_schema_enforcement:many_properties_20[tool_choice=auto]'
  'tc_schema_enforcement:many_properties_20[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_array_of_enums[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_chain_a_b_c[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_cross_referencing[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_integer_enforcement[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_nested_items_type_enforcement[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_nested_items_type_enforcement[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_recursive_enum_leaf[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_recursive_enum_leaf[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_required_only_minimal[tool_choice=required]'
  'tc_schema_enforcement:type_list_object_with_properties[tool_choice=required]'
)

# Model behaviour seen only on MAX runs.
_muse_glimmer_model_failures=(
  # basic_reasoning_and_tool_usage: sampled at temperature 1.0; on some draws
  # the model closes its thinking block at once (one reasoning token) and
  # answers directly.
  'basic_reasoning_and_tool_usage:special-token-content-thinking-on'
  # openrouter_tests: sampled at temperature 1.0; the model sometimes
  # answers directly instead of calling the tool.
  'openrouter_tests:reasoning-enabled-tool-call-step-1'
  # tc_schema_enforcement: the model writes the bare boolean the prompt asks
  # for, which the schema's type forbids. At temperature 0 the output is a
  # near-tie path that depends on prefix-cache and prefill batch state, on MAX
  # and vLLM 0.30.0 alike.
  'tc_schema_enforcement:anyof_nested_in_array_items[tool_choice=auto]'
  # tc_schema_enforcement: the model declined a tool_choice=required call on
  # some sampled draws (temperature 1.0).
  'tc_schema_enforcement:anyof_nullable_boolean[tool_choice=required]'
  'tc_schema_enforcement:anyof_nullable_integer[tool_choice=required]'
  'tc_schema_enforcement:anyof_nullable_number[tool_choice=required]'
  'tc_schema_enforcement:anyof_nullable_object[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_boolean_type[tool_choice=required]'
  # basic_reasoning_and_tool_usage: the chat template has no thinking-off
  # switch, so the model reasons although the request turns thinking off.
  'basic_reasoning_and_tool_usage:response-format-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-and-response-format-thinking-off'
  # tc_schema_enforcement: the model writes an int where the schema asks for
  # a string; tool_choice=auto leaves argument values unconstrained (2 of 3
  # runs on 2026-10-02).
  'tc_schema_enforcement:ref_defs_chain_a_b_c[tool_choice=auto]'
  # tc_schema_enforcement: the model writes an argument value outside the
  # enum; the structural tag frames the call but leaves values free text.
  'tc_schema_enforcement:anyof_with_enum_branch[tool_choice=required]'
  # tool_arguments_json_stability: at temperature 0 the model reasons past
  # max_tokens=512 without closing its to=self message, so the forced-call
  # grammar, which starts after <|eom|>, never engages.
  'tool_arguments_json_stability:forced_search_args_json_32_runs'
)

# Join into the comma-separated form the --exclude flag expects.
exclude=$(
  IFS=,
  printf "%s" "${_muse_glimmer_vllm_failures[*]},${_muse_glimmer_model_failures[*]}"
)

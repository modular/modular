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
# parsers fails too, from the full validation run of 2026-10-14. The hy_v4
# structural tag frames a call, its name and its argument keys, but leaves
# argument values free text, so a prompt that asks for a value the schema
# forbids gets that value.
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
  # tc_schema_enforcement: the model writes the value the prompt asks for
  # rather than the type or shape the schema allows (vLLM: FAIL each).
  'tc_schema_enforcement:anyof_nested_in_array_items[tool_choice=auto]'
  'tc_schema_enforcement:anyof_nullable_integer[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_array_enforcement[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_array_enforcement[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_nested_array_in_object[tool_choice=required]'
  'tc_schema_enforcement:type_list_object_with_properties[tool_choice=required]'
  # tc_schema_enforcement: passed in the full run, failed on a re-run of the
  # same serve; the same value-shape draws as above (vLLM: FAIL, one entry
  # where two were asked).
  'tc_schema_enforcement:anyof_nested_in_array_items[tool_choice=required]'
)

# Model behaviour seen only on MAX runs; vLLM 0.30.0 passed its single draw.
_hy4_model_failures=(
  # tc_basics: with tool_choice=auto the model sometimes spells the call as
  # JSON or a [tool_call] block instead of its own tags, so no call parses
  # (3 of 40 MAX replays; vLLM: INTERESTING, budget exhausted).
  'tc_basics:20_tools_auto'
  # tc_schema_enforcement: the model writes the non-string values the prompt
  # asks for (vLLM: PASS, it wrote strings on its draw).
  'tc_schema_enforcement:type_list_array_with_items[tool_choice=required]'
  # tc_schema_enforcement: the model writes the string and boolean the prompt
  # asks for (vLLM: PASS, the model declined the call).
  'tc_schema_enforcement:anyof_multiple_required_fields[tool_choice=auto]'
  # tc_schema_enforcement: the model leaves prop2 out of the third object
  # (vLLM: PASS).
  'tc_schema_enforcement:ref_defs_nested_items_type_enforcement[tool_choice=auto]'
  # tc_schema_enforcement: passed in the full run and failed on a re-run of
  # the same serve, the same sampled value-shape draws as above (vLLM: PASS,
  # PASS, INTERESTING budget exhausted, PASS on its single draw).
  'tc_schema_enforcement:anyof_multiple_required_fields[tool_choice=required]'
  'tc_schema_enforcement:anyof_with_enum_branch[tool_choice=required]'
  'tc_schema_enforcement:anyof_deeply_nested[tool_choice=required]'
  'tc_schema_enforcement:ap_typed_list_dict_str_list_str[tool_choice=auto]'
)

# Join into the comma-separated form the --exclude flag expects.
exclude=$(
  IFS=,
  printf "%s" "${_hy4_vllm_failures[*]},${_hy4_model_failures[*]}"
)

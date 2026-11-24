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

# Measured on the full NVFP4 checkpoint through the orchestrator and Mach on
# eight B200s (expert parallel, DSpark drafter, graph capture). Each failing
# scenario was replayed on Mach without the drafter and on vLLM serving the
# same checkpoint with its muse parsers.

# Engine behaviour: vLLM fails the same request the same way.
_muse_spark_vllm_failures=(
  # The chat template has no thinking-off switch: reasoning is still written.
  'basic_reasoning_and_tool_usage:content-parts-thinking-off'
  'basic_reasoning_and_tool_usage:open-ended-thinking-off'
  'basic_reasoning_and_tool_usage:response-format-thinking-off'
  'basic_reasoning_and_tool_usage:special-token-content-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-and-response-format-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-auto-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-calculate-required-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-thinking-off'
  'openrouter_tests:reasoning-disabled'
  # The template drops reasoning from earlier assistant turns.
  'reasoning_carry_replay:carry_reasoning_content_key'
  'reasoning_carry_replay:carry_reasoning_key'
  # Hits the 16384-token cap before answering.
  'reasoning_runaway:engineering_q21'
  # Reasoning outlasts the 30 s client timeout under concurrent load. In
  # constrained_decoding most of its ~1600 requests time out, and which ones
  # changes from run to run, so the scenario is listed whole.
  'constrained_decoding'
  'pipeline_stall:mixed_prefill_decode_pressure'
  'thinking_tokens:thinking_many_concurrent'
  # zombie_horde: scenarios still running when the 300 s horde timeout
  # elapses, and the waves' own failures. The horde shuffles its waves, so
  # these names change from run to run.
  'zombie_horde:agentic_correctness__horde_timeout'
  'zombie_horde:constrained_decoding__horde_timeout'
  'zombie_horde:horde_final_verdict'
  'zombie_horde:openrouter_tests__reasoning-disabled'
  'zombie_horde:production_resilience__horde_timeout'
  'zombie_horde:structured_output_schema_limits__horde_timeout'
)

# Join into the comma-separated form the --exclude flag expects.
exclude=$(
  IFS=,
  printf "%s" "${_muse_spark_vllm_failures[*]}"
)

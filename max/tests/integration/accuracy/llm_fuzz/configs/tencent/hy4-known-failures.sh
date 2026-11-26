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

# Measured on the tencent/Hy4-FP8 checkpoint through the orchestrator and Mach
# on eight B200s, and in main's fuzz CI. Some box runs served Mach with
# --max-new-tokens 256, a default only for requests that set no max_tokens;
# main's CI serves Mach uncapped (131072). A vLLM result in a comment is from
# vLLM 7bce843 serving the same checkpoint with its hy_v4 parsers, unless the
# comment names another checkpoint; entries without one were not compared.

# Model behaviour: vLLM fails the same test.
_hy4_vllm_failures=(
  # Neither engine returns JSON content in any of the 100 runs.
  'json_schema_compliance:schema_compliance_100_runs'
  # The secret in an earlier turn's reasoning does not reach the answer.
  'reasoning_carry_replay:carry_reasoning_content_key'
  'reasoning_carry_replay:carry_reasoning_key'
  # Hits the 16384-token cap before answering (Mach 4/4 and 3/4 samples,
  # vLLM 4/4 and 2/4).
  'reasoning_runaway:engineering_q21'
  'reasoning_runaway:engineering_q46'
  # Concurrent reasoning outlasts the 30 s client timeout (Mach 10/10 and
  # 10/20 errors, vLLM 8/10 and 10/20).
  'thinking_tokens:thinking_many_concurrent'
  'thinking_tokens:thinking_rapid_alternation'
)

# The fuzz load shape against Hy4's reasoning length, not an engine defect.
_hy4_load_failures=(
  # 200 requests in flight on 32 seats with a 30 s client timeout, and Hy4
  # reasons for thousands of tokens by default: most requests time out on
  # both engines (on the same load, vLLM 1180/1575 vs Mach 580/1575), and
  # which ones changes from run to run, so the scenario is listed whole.
  'constrained_decoding'
  # Decode-heavy requests set max_tokens=4096 and the client times out at
  # 30 s. Mach honours the explicit max_tokens (~68 s at 10 concurrent; 0/20
  # errors with a 600 s timeout or with max_tokens 256); vLLM passes only on
  # the 256-cap box runs, where its generation-config override clamps an
  # explicit max_tokens to 256.
  'pipeline_stall:long_decode_block'
  'pipeline_stall:mixed_prefill_decode_pressure'
  # The 300 s horde deadline cuts the scenarios still running in the last
  # wave, on both engines, and reports each as an error. Hy4's long reasoning
  # makes that 1-4 scenarios per run, under different names each run, so the
  # scenario is listed whole. Each cut scenario passes on its own.
  'zombie_horde'
)

# Intermittent: fail on some draws only.
_hy4_intermittent_failures=(
  # Reasoning on with max_tokens 256: when reasoning runs past 256 tokens the
  # required call never starts (no call in 1/10 replays on Mach, 2-3/10 on
  # vLLM).
  'tool_calling:tool_schema_with_oneof_and_const'
  # Reasons until max_tokens (10000) and writes no content; vLLM does the
  # same on Hy4-preview.
  'basic_reasoning_and_tool_usage:special-token-content-thinking-on'
  # The fuzz client stops reading at the first 100 ms gap, and the two
  # 5-token answers take 113-171 ms on Mach; vLLM fails the same way.
  'connection_exhaustion:pipelined_requests'
  # Reasoning runs past max_tokens 1024 before the named call starts.
  'tool_schema_validation:tool_choice_specific_schema'
  # At temperature 0 a cold prefill takes a path that reasons past 512
  # tokens, so 1 of 32 runs has no call (cached prefills all call).
  'tool_arguments_json_stability:forced_search_args_json_32_runs'
  # The model sometimes writes a ChatML transcript as plain text in its
  # reasoning. '<|im_end|>' is not a Hy4 token, and both Mach's and vLLM's
  # hy_v4 parsers pass the text through; llm_fuzz's leak markers, written
  # for Kimi, flag it.
  'basic_reasoning_and_tool_usage:tool-call-calculate-required-thinking-on'
  # The model sometimes quotes its prompt's tool-call instructions in its
  # reasoning, '<tool_call>' included. The hy_v4 reasoning parser passes
  # reasoning text through, and llm_fuzz's leak markers, written for GLM,
  # flag it.
  'basic_reasoning_and_tool_usage:tool-call-thinking-on'
  # Asked for kinds the schema's enum leaves out, the model sometimes drops
  # those nodes and writes only the root, which the schema allows. The
  # reasoning before the call differs from run to run even at temperature 0.
  'tc_schema_enforcement:ref_defs_recursive_enum_leaf[tool_choice=auto]'
  # The model sometimes plans in Chinese in its reasoning (about 1 % of
  # responses on both engines; the content stays English).
  'output_correctness_under_burst:mixed_burst_short_story'
  'output_correctness_under_burst:streaming_long_story'
)

# Join into the comma-separated form the --exclude flag expects.
exclude=$(
  IFS=,
  printf "%s" "${_hy4_vllm_failures[*]},${_hy4_load_failures[*]},${_hy4_intermittent_failures[*]}"
)

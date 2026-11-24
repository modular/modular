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
# on eight B200s with the CI flags. Each failing test was compared with vLLM
# 7bce843 serving the same checkpoint with its hy_v4 parsers.

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
  # errors with a 600 s timeout or with max_tokens 256); vLLM passes in CI
  # only because its generation-config override clamps it to 256.
  'pipeline_stall:long_decode_block'
  'pipeline_stall:mixed_prefill_decode_pressure'
  # zombie_horde: scenarios still running when the 300 s horde timeout
  # elapses, on both engines. The horde shuffles its waves, so these names
  # change from run to run.
  'zombie_horde:output_correctness_under_burst__horde_timeout'
  'zombie_horde:parallel_tool_call_ids__horde_timeout'
  'zombie_horde:structured_output__horde_timeout'
  'zombie_horde:token_runaway__horde_timeout'
)

# Intermittent: fail on some draws only.
_hy4_intermittent_failures=(
  # Reasoning on with max_tokens 256: when reasoning runs past 256 tokens the
  # required call never starts (no call in 1/10 replays on Mach, 2-3/10 on
  # vLLM).
  'tool_calling:tool_schema_with_oneof_and_const'
)

# Join into the comma-separated form the --exclude flag expects.
exclude=$(
  IFS=,
  printf "%s" "${_hy4_vllm_failures[*]},${_hy4_load_failures[*]},${_hy4_intermittent_failures[*]}"
)

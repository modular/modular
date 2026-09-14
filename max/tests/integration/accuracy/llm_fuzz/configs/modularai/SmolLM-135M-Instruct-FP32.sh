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
# SmolLM-135M's max_position_embeddings is 2048 — far below the other configs
# here, and a hard ceiling rather than a tuning knob: anything larger makes
# Llama3Config refuse to infer a max_length and MAX Serve fail to start.
max_length=2048

extra_pipelines_args=(
  --device-memory-utilization=0.5
  --enable-prefix-caching
  # No --enable-structured-output: this entry serves on CPU, and MAX refuses
  # that flag there (pipelines/lib/config/config.py) because the feature is a
  # sampling-graph bitmask, not a server-side check -- a grammar backend masks
  # logits each decode step, which needs the GPU path. It gates only
  # user-supplied `response_format` schemas; tool-call grammars are separate
  # (enable_tool_call_constrained_decode, on by default) and irrelevant to a
  # model that never emits a tool call.
)

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
model_profile=smollm
scenarios=
# Two groups are dropped, for two different reasons, leaving a pass/fail smoke
# test of the serve path and the API surface rather than of model behaviour.
#
# Capability: SmolLM-135M never emits a tool call, so every test in the
# tool-calling and reasoning scenarios fails on the model rather than on
# anything this pipeline is meant to catch (runs 34537420496 and 34539449145:
# 166 failures between them, essentially all "no tool_calls returned").
#
# Unsupported on CPU: with --enable-structured-output gone (see above), a
# request carrying a `response_format` schema is rejected outright.
# json_schema_compliance and so_basics/so_advanced are excluded entirely --
# nothing in them would exercise anything, and json_schema_compliance scores a
# 0% schema rate as a hard FAIL. Only one *test* needs excluding beyond that:
# production_resilience:max_tokens_json_truncation calls so_chat with no branch
# for a rejected request, so it reports ERROR (run 34637196807). Everything
# else that touches response_format already maps the 400 to INTERESTING and is
# deliberately kept -- openai_spec_compliance's response_format_json_object and
# response_format_json_schema both self-report "server rejects ... (400)",
# which is accurate signal about this deployment rather than noise.
exclude=tc_basics,tc_advanced,tc_schema_enforcement,tc_streaming_protocol,tool_arguments_json_stability,agentic_correctness,basic_reasoning_and_tool_usage,openrouter_tests,concurrent_stress,json_schema_compliance,so_basics,so_advanced,production_resilience:max_tokens_json_truncation
k2vv_mode=
circuit_breaker=0

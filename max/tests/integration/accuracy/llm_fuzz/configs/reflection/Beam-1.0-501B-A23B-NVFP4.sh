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

# Fuzzed on Mach only (no MAX Serve lane), on four B200s with the recipe
# mach/docs/models/beam.md serves. The workflow compiles the MEFs at this
# length rather than mach_variants.py's 16384: at 16384 the schema prompt of
# structured_output:invalid_schema_huge_enum and the history of
# token_runaway:reasoning_schema_final_turn exceed the context, and the run
# reports the limit instead of the engine.
max_length=131072

# Mach's export admits 32 sequences per step (mach_variants.py).
batch_size=32

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
model_profile=beam
scenarios=
k2vv_mode=
circuit_breaker=0

# llm-fuzz defaults to 200 requests in flight. Against a batch of 32 the
# surplus queues, and the 30 s client timeout then measures the queue rather
# than the engine. A scenario that means to overload the server sets its own
# concurrency.
extra_fuzz_args="--max-concurrency $batch_size"

# Skip the known failures: tests vLLM fails the same way.
exclude_files+=("$(dirname "${BASH_SOURCE[0]}")/Beam-1.0-501B-A23B-NVFP4-known-failures.txt")

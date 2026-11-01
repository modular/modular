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

# Laguna S-2.1 NVFP4 on one B200. Mach serves this checkpoint from a MEF
# `gen_mef` exports at the limits below, and `--max-serve-url` points the
# fuzzer at that endpoint; without the flag the same limits launch the MAX
# Serve ModuleV3 arm. One file for both means a control run and a Mach run
# are capacity-matched by construction rather than by whoever typed the
# flags -- which is the only way their failure counts compare.

# Matches the Mach variant's `max_batch_size=64` (`mach_variants.py`) and the
# MAX smoke recipe's 64, so both engines admit the same concurrency and the
# lane's pressure scenarios exercise the same admission on each.
batch_size=64
max_length=262144

extra_pipelines_args=(
  # The graph-API architecture is still registered, so the ModuleV3 one
  # needs asking for by name. Mach can only serve the ModuleV3 export.
  --prefer-module-v3
  --enable-structured-output
  # Mach serves with its prefix cache on and no flag to turn it off, and the
  # reuse is not bit-exact, so a control run without this measures the cache
  # rather than the engine. `tc_basics:parallel_tc_two_tools` is the one that
  # showed it: both engines ask for both tools cold and one or none warm.
  --enable-prefix-caching
)

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
model_profile=laguna
scenarios=
k2vv_mode=
circuit_breaker=0

# Drive no more requests than the engine admits in a step, and allow a
# generation that runs to its cap.
#
# llm-fuzz defaults to 200 in flight, a property of the tool rather than of
# any deployment. Against a batch of 16 (this lane's earlier setting) the
# surplus queues, so a request waits far longer than it is served -- a median
# 52 s on the 2,950-test final tier -- and the 30 s default timeout fires on
# queueing before the correctness it was sent to measure is ever reached; 15
# of that tier's 29 failures were `transport error: TIMEOUT` even at 180 s.
# Matching the batch leaves the engine exactly as busy, since the batch is
# still full and the sweep's wall clock is unchanged, and leaves a timeout
# meaning what it says. A scenario that means to overload the server passes its own
# concurrency and is not affected.
#
# The timeout covers the longest legitimate generation, not the shortest. That
# is `reasoning_runaway`'s 16,384-token cap (`decode_heavy_max_tokens` is only
# 4,096 here, since the lane passes no `--max-num-tokens`). The MAX ModuleV3
# baseline at 64 concurrent streams has TPOT p50 25.7 ms and p99 about 29 ms,
# so a run to the cap takes about 475 s at p99, plus a TTFT of 0.08-0.15 s and
# an allowance for queueing: 600 s, which is also the per-request timeout that
# scenario sets for itself.
#
# `--max-context-length` pins the probes' context ceiling. In
# `--max-serve-url` mode the script already passes it; in managed mode the
# fuzzer would otherwise size probes from the HF config, and fall back to
# 4,096 tokens when that fetch fails or `--no-hf-fetch` is set, against
# the 262,144 the Mach arm is probed at.
extra_fuzz_args="--max-concurrency $batch_size --timeout 600 --max-context-length $max_length"

# Skip the known failures. `run-llm-fuzz.sh` reads every file named here into
# the --exclude list, so naming one is all this takes. MAX Serve and Mach read
# the same list. Each entry carries the control that makes it a known failure,
# and an id this checkpoint passes is in no list at all.
exclude_files+=("$(dirname "${BASH_SOURCE[0]}")/Laguna-S-2.1-NVFP4-known-failures.txt")

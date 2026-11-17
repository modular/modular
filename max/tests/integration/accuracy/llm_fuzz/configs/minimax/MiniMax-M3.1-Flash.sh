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

# MiniMax-M3.1-Flash on 8xB200, mirroring the production recipe
# max_private/minimax_m3/recipes/kadabra-m31-prod-7.max.yaml: DP=2 x EP=8,
# NVFP4 KV cache, DSpark draft (K=7, sampled proposals), VMM allocator with
# 32 MiB pages. Deviations: the tiered host+disk connector stands in for the
# dKV connector (dKV daemon, NIXL, EFA), and max_length caps the 1M
# production context at the value the M3 lanes use. The checkpoint is the
# one production serves: the privately placed copy in the runners' HF cache,
# not the Hub's `main`.

use_max_private=1
batch_size=64
max_length=131072

# private-hf-repo-lock.tsv pins the cached copy by an S3 fingerprint, not an
# HF commit, so only an offline lookup resolves it. Online, `main` resolves
# to the Hub's commit, which the cache does not keep.
export HF_HUB_OFFLINE=1

# The VMM allocator settings prod-7 runs with. This config is sourced by
# run-llm-fuzz.sh, so exporting them here puts them in the environment
# `max serve` inherits.
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY=1
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM_PAGE_MB=32

extra_pipelines_args=(
  --no-prefer-module-v3
  --ep-size 8
  --data-parallel-degree 2
  --trust-remote-code
  --kv-cache-format float4_e2m1fn
  --device-memory-utilization 0.8
  --no-enable-dp-cross-replica-prefix-copy
  --kv-connector-config '{"type":"tiered","host_offload_max_gb":512,"disk_offload_dir":"/tmp/max_kv_tiered_m31","disk_offload_max_gb":1024}'
  # The DSpark draft ships in the target repo's dspark/ folder.
  --draft-model-path MiniMaxAI/MiniMax-M3.1-Flash
  --draft-subfolder dspark
  --speculative-method dspark
  --num-speculative-tokens 7
  --draft-proposal sampled
  --enable-structured-output
  --tool-call-policy default_strict_false_and_best_effort
  --emit-reasoning-content
  --vision-cache-utilization 0.06
  --dp-ce-balance-threshold 0.95
  --dp-ce-balance-timeout-ms 1500
)

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
model_profile=minimax-m3
scenarios=
k2vv_mode=
circuit_breaker=0

# Start from the M3 list (derived from MiniMax's official M3 API) until an
# M3.1 list is derived the same way.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../minimax-m3-known-failures.sh"

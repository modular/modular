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

# Serve via //max_private:max_private: the tiered KV connector below needs the
# private kv_tier_connector pyo3 extension, which only the max_private serve
# binary bundles (same as the MiniMax-M3-MXFP8-ep-dp config).
use_max_private=1
batch_size=64
max_length=262144

extra_pipelines_args=(
  --device-memory-utilization=0.75
  --ep-size 8
  --data-parallel-degree 1
  --max-batch-input-tokens 4096
  --enable-prefix-caching
  --enable-structured-output
  --ep-use-allreduce
  --kv-cache-format float8_e4m3fn
  --kv-connector tiered
  --kv-connector-config '{"host_kvcache_swap_space_gb":512,"disk_offload_dir":"/tmp/max_kv_tiered","disk_offload_max_gb":1024}'
  --trust-remote-code
  # Eagle3 speculative decoding -- mirrors the Mammoth kimi-k27-nvfp4
  # benchmark component (TP attention + EP MoE, allreduce, tiered KV).
  # K2.7-Code has no dedicated Eagle3 draft, so the deployment (and this
  # config) reuses the K2.6 draft. As with K2.5/K2.6, eagle3 is bundled
  # into the ep-tp config rather than a separate variant.
  # disk_offload_dir is pointed at /tmp here (the deployment uses
  # /cache/max-cache, which is not mounted on the fuzz runner). The
  # draft.sliding_window override matches the value used with the K2.5 draft
  # (not declared in the K2.6 draft's HF config).
  --speculative-method eagle
  --num-speculative-tokens 3
  --draft-model-path nvidia/Kimi-K2.6-Eagle3
  --model-override draft.sliding_window=12288
  --draft-quantization-encoding bfloat16
)

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
model_profile=kimi-k2.5
scenarios=
k2vv_mode=full
circuit_breaker=0

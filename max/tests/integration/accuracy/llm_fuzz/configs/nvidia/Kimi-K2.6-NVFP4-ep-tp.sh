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
max_length=262144

extra_pipelines_args=(
  --device-memory-utilization=0.75
  --ep-size 8
  --data-parallel-degree 1
  --max-batch-input-tokens 4096
  --max-num-steps 1
  --enable-prefix-caching
  --enable-structured-output
  --ep-use-allreduce
  --kv-cache-format float8_e4m3fn
  --kv-connector tiered
  --kv-connector-config '{"host_kvcache_swap_space_gb":512,"disk_offload_dir":"/tmp/max_kv_tiered","disk_offload_max_gb":1024}'
  --trust-remote-code
  # Eagle3 speculative decoding -- mirrors the stage-1 max-kimi-k26-nvfp4-shadow
  # deployment and the nvfp4_kimi_k2_6_eagle_tiered_kvconnector_tpep_ar_8x_b200
  # recipe (TP attention + EP MoE, allreduce, tiered KV). As with K2.5, eagle3
  # is bundled into the ep-tp config rather than a separate variant.
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

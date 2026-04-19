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
pipeline=deepseek-ai/DeepSeek-R1

batch_size=64
max_length=128000

extra_pipelines_args=(
  --device-memory-utilization=0.95
  --ep-size 8
  --data-parallel-degree 8
  --max-batch-input-tokens 1024
)
extra_longbench_v2_args=(
  --max_new_tokens 8192
  --max_samples=100  # Limit samples for testing
  --max_context_length=100000 # https://github.com/THUDM/LongBench/issues/134
  --client_timeout=5000
  # --no_context  # Omit context (for measuring memorization)
)

evaluator=longbench-v2

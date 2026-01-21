##===----------------------------------------------------------------------===##
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
max_length=50000

extra_pipelines_args=(
  # --enable-echo  # Not needed for gsm8k_cot_llama
  --device-memory-utilization=0.6
  --prefill-chunk-size=128
  --ep-size=8
  --data-parallel-degree=8
)
extra_lm_eval_args=(
  --apply_chat_template
  --fewshot_as_multiturn
  "--gen_kwargs=max_gen_toks=4096,seed=42,temperature=0"
)

# Increase generation timeout to 5000s
extra_lm_model_args=(
  timeout=5000
)

# These parameters (max_length=50000, max_gen_toks=4096, temp=0) are used with
# following thinking task:
tasks=gsm8k_cot_llama

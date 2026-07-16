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
max_length=50000

extra_pipelines_args=(
  --ep-size 4
  --data-parallel-degree 4
  --max-batch-input-tokens 1024
  --trust-remote-code
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

tasks=gsm8k_cot_llama

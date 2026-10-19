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

# The HF tokenizer has no chat template; the server applies its own.
lm_eval_model=local-chat-completions

extra_lm_eval_args=(
  --fewshot_as_multiturn
  "--gen_kwargs=max_gen_toks=4096,seed=42,temperature=0"
)

tasks=gsm8k_cot_llama

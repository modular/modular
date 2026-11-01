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
max_length=32768

extra_pipelines_args=(
  --ep-size 8
  --speculative-method mtp
  --num-speculative-tokens 3
)
# no_think matches the full-test-set gsm8k comparison against vLLM.
extra_lm_eval_args=(
  --apply_chat_template
  --fewshot_as_multiturn
  '--gen_kwargs={"max_gen_toks": 4096, "seed": 42, "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "no_think"}}'
)

tasks=gsm8k_cot_llama

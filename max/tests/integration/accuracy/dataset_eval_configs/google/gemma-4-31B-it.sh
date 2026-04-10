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
batch_size=128
max_length=8192
extra_pipelines_args=(
    --enable-echo
    --max-num-steps 1
)
extra_lm_eval_args=(
    --fewshot_as_multiturn 
    --apply_chat_template
    "--gen_kwargs=seed=42,temperature=0,max_gen_toks=1024"
)
tasks=mmlu_pro

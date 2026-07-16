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
# LFM2 conv state cache only supports batch_size=1, so large task groups
# (leaderboard_bbh, leaderboard_mmlu_pro) exceed the CI time limit.
# Restrict to smaller tasks until batch_size>1 is supported.
batch_size=1
max_length=4096
tasks=leaderboard_gpqa,leaderboard_ifeval,leaderboard_musr
extra_pipelines_args=(--enable-echo)
extra_lm_eval_args=(
  --apply_chat_template
  --fewshot_as_multiturn
)

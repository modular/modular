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
batch_size=256
max_length=128000

evaluator=mistral-evals
tasks=mathvista,chartqa,docvqa,mmmu

extra_pipelines_args=(
  --trust-remote-code
  --no-enable-prefix-caching
  --no-enable-chunked-prefill
)

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
  --prefer-module-v3
  --device-memory-utilization=0.9
  --enable-prefix-caching
  --enable-structured-output
  # 400 on tool schemas the grammar cannot fully enforce, instead of
  # compiling a partial grammar the model can slip non-conforming
  # arguments through.
  --tool-call-policy default_strict_true_and_reject_unsupported
)
model_profile=nemotron
scenarios=
k2vv_mode=
circuit_breaker=0

# Known failures on this lane; see nemotron-known-failures.txt.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/nemotron-known-failures.sh"

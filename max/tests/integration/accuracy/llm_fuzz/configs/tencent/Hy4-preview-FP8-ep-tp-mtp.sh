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

# Mirrors max/pipelines/architectures/hy_v4/recipes/
# hy4_preview_fp8_tp_ep_8x_b200.yaml; the hy_v4 parsers are the arch default.
batch_size=64
max_length=131072

extra_pipelines_args=(
  --ep-size 8
  --speculative-method mtp
  --num-speculative-tokens 3
  --max-batch-input-tokens 4096
  --enable-in-flight-batching
  --enable-prefix-caching
  --enable-structured-output
)

model_profile=hy_v4
scenarios=
k2vv_mode=
circuit_breaker=0

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/hy4-known-failures.sh"

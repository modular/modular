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

# Mirrors max/pipelines/architectures/glm5_next/recipes/
# glm_5_3_flash_fp8_tp_ep_8x_b200.yaml; the glm45 parsers are the arch default.
batch_size=32
max_length=131072

extra_pipelines_args=(
  --ep-size 8
  --max-batch-input-tokens 8192
  --device-memory-utilization 0.78
  --enable-structured-output
)

model_profile=glm-5.1
scenarios=
k2vv_mode=
circuit_breaker=0

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/glm53-flash-known-failures.sh"

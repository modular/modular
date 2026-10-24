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

# shellcheck disable=SC2034  # `exclude` is consumed when this file is sourced.

# The list lives in nemotron-known-failures.txt so the Mammoth llm-fuzz image
# can read the same file.

mapfile -t _nemotron_known_failures < <(
  sed -e 's/#.*//' -e 's/[[:space:]]//g' \
    "$(dirname "${BASH_SOURCE[0]}")/nemotron-known-failures.txt" |
    grep -v '^$'
)

# Join into the comma-separated form the --exclude flag expects.
exclude=$(
  IFS=,
  printf "%s" "${_nemotron_known_failures[*]}"
)

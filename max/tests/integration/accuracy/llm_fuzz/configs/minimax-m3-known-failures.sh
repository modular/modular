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

# The list itself lives in minimax-m3-known-failures.txt so that the Mammoth
# llm-fuzz job can read the same file (see that file's header). A list kept
# in this script instead would have to be copied into the Mammoth overlay,
# and the two copies would drift.

mapfile -t _m3_known_failures < <(
  sed -e 's/#.*//' -e 's/[[:space:]]//g' \
    "$(dirname "${BASH_SOURCE[0]}")/minimax-m3-known-failures.txt" |
    grep -v '^$'
)

# Join into the comma-separated form the --exclude flag expects.
exclude=$(
  IFS=,
  printf "%s" "${_m3_known_failures[*]}"
)

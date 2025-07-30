#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT="${SCRIPT_DIR}"/../../..


FILTER="stdlib/test"
if [[ $# -gt 0 ]]; then
  # If an argument is provided, use it as the specific test directory
  FILTER="${1#./}" # remove leading relative file path if it has one
  if [[ -f ${FILTER} ]]; then
    FILTER=$(dirname $FILTER)
  fi
  FILTER="${FILTER%/}" # remove trailing / if it has one
fi

exec "$REPO_ROOT"/bazelw test //mojo/${FILTER}/...

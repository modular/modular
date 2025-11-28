#!/bin/bash
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

if [[ $# -gt 0 ]]; then
    echo "error: //:format does not accept arguments." >&2
    echo "Run './bazelw run //:format' without arguments to format all files." >&2
    exit 1
fi

# Set RUNFILES_DIR so the multirun script can find its dependencies
export RUNFILES_DIR="${0}.runfiles"
exec "${RUNFILES_DIR}/_main/bazel/lint/fix.bash"

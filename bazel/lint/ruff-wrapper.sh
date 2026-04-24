#!/usr/bin/env bash
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

set -euo pipefail

binary_root=$PWD/..

cd "$BUILD_WORKSPACE_DIRECTORY"

binary=$(find "$binary_root" -name ruff | head -n 1)

# In FAST mode, only process tracked files changed since the merge-base with
# origin/main so that untracked scratch files are left alone. When FAST is
# unset, fast_files stays empty and expands to no extra args below.
fast_files=""
if [[ -n "${FAST:-}" ]]; then
    merge_base=$(git merge-base origin/main HEAD 2>/dev/null || true)
    fast_files=$(git diff --diff-filter=d --name-only ${merge_base:+"$merge_base"} -- '*.py' '*.pyi')
    if [[ -z "$fast_files" ]]; then
        exit 0
    fi
fi

result=0
case "$1" in
    check)
        shift
        # shellcheck disable=SC2086
        "$binary" format --check --quiet --diff "$@" $fast_files || result=$?
        # shellcheck disable=SC2086
        "$binary" check --quiet "$@" $fast_files || result=$?
        ;;
    fix)
        shift
        # shellcheck disable=SC2086
        "$binary" format "$@" $fast_files || result=$?
        # shellcheck disable=SC2086
        "$binary" check --fix "$@" $fast_files || result=$?
        ;;
    *)
        echo "Unknown subcommand '$1' to Ruff wrapper" >&2
        result=1
        ;;
esac
exit $result

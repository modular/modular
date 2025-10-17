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

readonly binary=$(find $PWD -name markdownlint -path "*markdownlint_*")
readonly config=$(find $BUILD_WORKSPACE_DIRECTORY -name .markdownlint.yaml -path "*bazel/lint*")

if [[ -n "${FAST:-}" ]]; then
    cd "$BUILD_WORKSPACE_DIRECTORY"
    paths=$(git diff --name-only $(git merge-base origin/main HEAD) -- '*.md' '*.mdx' ':!:third-party/*')
    if [ ! -n "$paths" ]; then
        # markdownlint will just print help if no input paths, short circuit here
        exit 0
    fi
    # aspect_rules_js complains if we change directories and then run the binary, so go back
    cd -
else
    paths="."
fi

JS_BINARY__CHDIR="$BUILD_WORKSPACE_DIRECTORY" \
  "$binary" --config "$config" \
  --ignore-path "$BUILD_WORKSPACE_DIRECTORY/.gitignore" \
  --ignore "$BUILD_WORKSPACE_DIRECTORY/third-party" \
  "$@" "$paths" 2>&1 | sed 's/^/error: /'

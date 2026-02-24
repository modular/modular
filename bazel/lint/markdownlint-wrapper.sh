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

binary=$(find "$PWD" -name markdownlint -path "*markdownlint_*")
config="$BUILD_WORKSPACE_DIRECTORY/bazel/lint/.markdownlint.yaml"
if [[ ! -f "$config" ]]; then
    config="$BUILD_WORKSPACE_DIRECTORY/oss/modular/bazel/lint/.markdownlint.yaml"
fi
if [[ ! -f "$config" ]]; then
    echo "error: could not find .markdownlint.yaml config file" >&2
    exit 1
fi

if [[ -n "${FAST:-}" ]]; then
    cd "$BUILD_WORKSPACE_DIRECTORY"
    paths=$(git diff --diff-filter=d --name-only "$(git merge-base origin/main HEAD)" -- '*.md' '*.mdx' ':!:third-party/*')
    if [ -z "$paths" ]; then
        # markdownlint will just print help if no input paths, short circuit here
        exit 0
    fi
    # aspect_rules_js complains if we change directories and then run the binary, so go back
    cd - >/dev/null
else
    paths="."
fi

# Intentionally disable this, we want $paths to split into multiple args
# shellcheck disable=SC2086
JS_BINARY__CHDIR="$BUILD_WORKSPACE_DIRECTORY" \
  "$binary" --config "$config" \
  --ignore-path "$BUILD_WORKSPACE_DIRECTORY/.gitignore" \
  --ignore "$BUILD_WORKSPACE_DIRECTORY/third-party" \
  --ignore "$BUILD_WORKSPACE_DIRECTORY/oss/modular/oss.AGENTS.md" \
  "$@" $paths 2>&1 | sed 's/^/error: /'

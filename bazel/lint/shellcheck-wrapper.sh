#!/bin/bash
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

if [[ $OSTYPE == darwin* ]]; then
  platform=darwin.aarch64
else
  platform="linux.$(uname -m)"
fi

binary="$PWD/../+http_archive+shellcheck-$platform/shellcheck"
cd "$BUILD_WORKSPACE_DIRECTORY"

# TODO: Improve heuristic for finding new versions of this
# https://github.com/koalaman/shellcheck/issues/143
no_extension_files=(
  bazelw
  oss/modular/tools/bazel
  tools/bazel
  utils/benchmarking/tmp-tuning-utils/bbench
  utils/benchmarking/tmp-tuning-utils/nohup-kbench
  utils/packaging/conda/entrypoints/lldb-dap
)

exec git ls-files --cached --exclude-standard -- '*.sh' "${no_extension_files[@]}" | xargs "$binary" --format=gcc

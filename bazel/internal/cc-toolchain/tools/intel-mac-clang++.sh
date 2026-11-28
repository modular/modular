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

# The Intel Mac toolchain repo. Its path differs in some tests.
clang_root="$PWD/external/+intel_mac_toolchain+clang-macos-x86_64"
if [[ ! -d "$clang_root" ]]; then
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
  clang_root="$repo_root/../+intel_mac_toolchain+clang-macos-x86_64"
fi

"$clang_root/bin/clang++" "$@"

if [[ -n "${PARSE_HEADER:-}" ]]; then
  touch "${PARSE_HEADER}"
fi

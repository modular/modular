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

if [[ $OSTYPE == darwin* ]]; then
  platform=macos
else
  platform="linux-$(uname -m)"
fi

clang_root="$PWD/external/+http_archive+clang-$platform"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir/../../../.."
# File paths in some golang compiles
if [[ ! -d "$clang_root" ]]; then
  clang_root="$repo_root/external/+http_archive+clang-$platform"
fi
# File paths in tests differ
if [[ ! -d "$clang_root" ]]; then
  clang_root="$repo_root/../+http_archive+clang-$platform"
fi

exec "$clang_root/bin/clang" "$@"

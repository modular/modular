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

dsym_path=""
binary_path=""
linker_args=()
for arg in "$@"; do
  case "$arg" in
    # No interface libraries on Intel Macs: drop these arguments.
    --modular-ifs-input=* | --modular-ifs-output=*) ;;
    --modular-dsym-path=*) dsym_path="${arg#*=}" ;;
    --modular-binary-path=*) binary_path="${arg#*=}" ;;
    *) linker_args+=("$arg") ;;
  esac
done

if [[ "${BUILD_IFS:-}" == "yes" ]]; then
  echo "error: interface libraries aren't supported on Intel Macs yet" >&2
  exit 1
fi

"$clang_root/bin/clang++" "${linker_args[@]}"

if [[ -n "$dsym_path" ]]; then
  "$clang_root/bin/dsymutil" -o "$dsym_path" "$binary_path"
fi

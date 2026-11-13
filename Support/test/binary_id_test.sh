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

readonly binary=Support/test/binary_id_binary
if [[ "$OSTYPE" == darwin* ]]; then
  library=Support/test/libbinary_id_shared_library.dylib
else
  library=Support/test/libbinary_id_shared_library.so
fi

output=$($binary)

binary_id=$(echo "$output" | grep "binary id: " | cut -d" " -f3)
library_id=$(echo "$output" | grep "shared library id: " | cut -d" " -f4)

if [[ -z "$binary_id" || -z "$library_id" ]]; then
  echo "error: binary and library id should be non empty: $binary_id, $library_id" >&2
  exit 1
fi

if [[ "$binary_id" == "$library_id" ]]; then
  echo "error: binary and library id should differ: $binary_id, $library_id" >&2
  exit 1
fi

binary_build_id=$(echo "$output" | grep "binary build id: " | cut -d" " -f4)
if [[ "$binary_build_id" != "$binary_id" ]]; then
  echo "error: build id '$binary_build_id' should equal id '$binary_id'" >&2
  exit 1
fi

# The NVPTX toolchain path suffixes the id, but not the build id.
nvptx_output=$(MODULAR_NVPTX_COMPILER_PATH=/opt/cuda $binary)
nvptx_id=$(echo "$nvptx_output" | grep "^binary id: " | cut -d" " -f3)
nvptx_build_id=$(echo "$nvptx_output" | grep "binary build id: " | cut -d" " -f4)
if [[ "$nvptx_build_id" != "$binary_build_id" ||
      "$nvptx_id" != "$binary_build_id"-* ]]; then
  echo "error: with an NVPTX path, id '$nvptx_id' should extend build id" \
       "'$binary_build_id', and build id '$nvptx_build_id' should not change" >&2
  exit 1
fi

# Remote runfiles can hold separate copies of one file, such as a library under
# _solib. Match the loaded copy by name and contents rather than by path.
check_path() {
  local kind=$1 path=$2 expected=$3
  if [[ "$path" != /* || "$(basename "$path")" != "$(basename "$expected")" ]] ||
     ! cmp -s "$path" "$expected"; then
    echo "error: $kind path '$path' should be an absolute path to a copy of" \
         "'$expected'" >&2
    exit 1
  fi
}
check_path binary "$(echo "$output" | grep "^binary path: " | cut -d" " -f3-)" \
  "$binary"
check_path library \
  "$(echo "$output" | grep "shared library path: " | cut -d" " -f4-)" \
  "$library"

if [[ "$OSTYPE" == darwin* ]]; then
  real_binary_id=$(dwarfdump -u "$binary" | tr '[:upper:]' '[:lower:]' | tr -d '-')
  if ! echo "$real_binary_id" | grep -q "uuid: $binary_id"; then
    echo "error: binary id '$binary_id' does not match real id: '$real_binary_id'" >&2
    exit 1
  fi

  real_library_id=$(dwarfdump -u "$library" | tr '[:upper:]' '[:lower:]' | tr -d '-')
  if ! echo "$real_library_id" | grep -q "uuid: $library_id"; then
    echo "error: library id '$library_id' does not match real id: '$real_library_id'" >&2
    exit 1
  fi
else
  readobj=../+llvm_configure+llvm-project/llvm/llvm-readobj
  real_binary_id=$("$readobj" -n "$binary")
  if ! echo "$real_binary_id" | grep -q "Build ID: $binary_id"; then
    echo "error: binary id '$binary_id' does not match real id: '$real_binary_id'" >&2
    exit 1
  fi

  real_library_id=$("$readobj" -n "$library")
  if ! echo "$real_library_id" | grep -q "Build ID: $library_id"; then
    echo "error: library id '$library_id' does not match real id: '$real_library_id'" >&2
    exit 1
  fi
fi

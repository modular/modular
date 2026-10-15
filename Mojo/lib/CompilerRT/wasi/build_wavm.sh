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
# Build the WAVM runtime (trcrsired/WAVM, mt-2 branch) with wasm64 WASI
# support. This is the real runtime used to run `mojo build` wasm64 output;
# it implements the wasm64 preview1 ABI natively (args_get_wasm64, ...).
#
# Usage:
#   build_wavm.sh [output_dir] [llvm_dir]
#     output_dir  default: /tmp/WAVM-build
#     llvm_dir    default: autodetect (LLVM_DIR env, /usr/lib/llvm-*/lib/cmake/llvm)
set -euo pipefail

OUT="${1:-/tmp/WAVM-build}"
SRC_DIR="${OUT%-build}-src"
LLVM_DIR="${2:-}"

mkdir -p "$OUT"

# Clone the mt-2 branch once; subsequent runs reuse it.
if [[ ! -d "$SRC_DIR/.git" ]]; then
  git clone --branch mt-2 --depth=1 \
    https://github.com/trcrsired/WAVM.git "$SRC_DIR"
fi

# Keep the checkout on the pinned mt-2 head recorded in PINS.md.
WAVM_PIN="b17e158616333cb8c4e19a5b7d72fccde070b464"
if [[ "$(git -C "$SRC_DIR" rev-parse HEAD)" != "$WAVM_PIN" ]]; then
  git -C "$SRC_DIR" fetch --depth=1 origin "$WAVM_PIN"
  git -C "$SRC_DIR" checkout -f --detach "$WAVM_PIN"
fi

# Add the preview1 sock_accept stub so modules importing sock_accept_wasm64
# can instantiate (returns ENOSYS until a real socket implementation lands),
# and unpack AssertMetadata: the packed attribute gives the pointer members
# alignment 1, which the modern macOS linker rejects ("pointer not aligned").
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for patch in wavm-sock-accept.patch wavm-unpacked-assert-metadata.patch; do
  if [[ ! -f "$SCRIPT_DIR/$patch" ]]; then
    echo "error: missing patch file $SCRIPT_DIR/$patch" >&2
    exit 1
  fi
  # Already applied (idempotent re-runs) -> skip; any other failure is fatal:
  # silently skipping a drifted or unappliable patch would surface much
  # later as an unrelated linker or instantiation error.
  if git -C "$SRC_DIR" apply --reverse --check "$SCRIPT_DIR/$patch" >/dev/null 2>&1; then
    continue
  fi
  git -C "$SRC_DIR" apply "$SCRIPT_DIR/$patch"
done

# Autodetect an installed LLVM CMake package.
if [[ -z "$LLVM_DIR" ]]; then
  for cand in \
    /opt/homebrew/opt/llvm/lib/cmake/llvm \
    /opt/homebrew/Cellar/llvm/*/lib/cmake/llvm \
    /usr/lib/llvm-*/lib/cmake/llvm \
    /usr/local/llvm*/lib/cmake/llvm \
    /tmp/llvm*/*/llvm/lib/cmake/llvm; do
    if [[ -d "$cand" ]]; then
      LLVM_DIR="$cand"
      break
    fi
  done
fi
if [[ -z "$LLVM_DIR" || ! -f "$LLVM_DIR/LLVMConfig.cmake" ]]; then
  echo "error: could not locate LLVM CMake package (LLVMConfig.cmake)" >&2
  echo "       install llvm via brew (macOS) or llvm-dev (apt), or pass the LLVM_DIR as arg 2" >&2
  exit 1
fi

# WAVM defaults its C++ standard to 14; LLVM 17+ headers require C++17.
cmake_args=(
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DLLVM_DIR="$LLVM_DIR"
  -DWAVM_ENABLE_LIBFUZZER=OFF
  -DWAVM_CXX_STANDARD_VERSION=17
)
# On macOS compile with the system clang: Homebrew's clang drives lld, which
# duplicates the LLVM archives and rejects the fork's remaining quirks.
if [[ "$(uname)" == "Darwin" && -x /usr/bin/clang ]]; then
  cmake_args+=(-DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++)
fi

cmake -S "$SRC_DIR" -B "$OUT" "${cmake_args[@]}"

# CMake links every LLVM component as a static archive, which grows
# libWAVM.dylib past arm64's +/-128MB B/BL branch range. When the LLVM
# install ships a monolithic libLLVM shared library (brew does), swap
# the archives for it after configuring.
# -P resolves symlinked prefixes (brew's opt/ is a symlink into Cellar/),
# matching how LLVMConfig.cmake records the archive paths in link.txt.
LLVM_LIB_DIR="$(cd "$LLVM_DIR/../.." && pwd -P)"
if [[ -f "$LLVM_LIB_DIR/libLLVM.dylib" ]]; then
  find "$OUT" -name link.txt -print0 | while IFS= read -r -d '' linktxt; do
    if grep -q "$LLVM_LIB_DIR/libLLVM" "$linktxt"; then
      # First occurrence carries the -L search path; the rest dedupe to -lLLVM.
      # (sed -i.bak, not plain -i or BSD's -i '': portable across GNU/BSD sed.)
      sed -i.bak "s#$LLVM_LIB_DIR/libLLVM[A-Za-z0-9]*\.a#-L$LLVM_LIB_DIR -lLLVM#" "$linktxt"
      sed -i.bak -E "s#$LLVM_LIB_DIR/libLLVM[A-Za-z0-9]*\.a#-lLLVM#g" "$linktxt"
      rm -f "$linktxt.bak"
    fi
  done
fi

cmake --build "$OUT" --target wavm -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

if [[ ! -x "$OUT/bin/wavm" ]]; then
  echo "error: WAVM build did not produce $OUT/bin/wavm" >&2
  exit 1
fi

echo "built WAVM at $OUT/bin/wavm"

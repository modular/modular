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
# Build a wasm64 wasi-libc sysroot from WebAssembly/wasi-libc PR #444.
#
# The sysroot provides the real crt1-command.o, libc.a, and
# libclang_rt.builtins-wasm64.a used by `mojo build --wasi-sysroot`. It is
# built with the repo's pinned clang-22 (located via the Bazel external
# cache, the same way build_wasm64_shim.sh finds it).
#
# Usage:
#   build_wasi_sysroot.sh [output_dir]
#     output_dir  default: /tmp/wasi-libc
#   The sysroot lands at <output_dir>/sysroot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

OUT="${1:-/tmp/wasi-libc}"
WASI_LIBC_DIR="$OUT/src"
SYSROOT="$OUT/sysroot"
PR444_HEAD="33983baf671eaa50da2ec0f7c9e9eeceb1f089b1"

mkdir -p "$OUT"

# Locate the repo's pinned clang-22 toolchain. Prefer the Bazel external
# cache's host toolchain; fall back to PATH.
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_locate_clang.sh"
# shellcheck disable=SC2119
CLANG="$(kgen_locate_clang)"
if [ -z "$CLANG" ]; then
  echo "error: could not locate clang-22 (Bazel cache or PATH)" >&2
  exit 1
fi

# Clone PR #444 once; subsequent runs reuse and reset to the pinned head.
if [[ ! -d "$WASI_LIBC_DIR/.git" ]]; then
  git clone https://github.com/WebAssembly/wasi-libc.git "$WASI_LIBC_DIR"
fi
git -C "$WASI_LIBC_DIR" fetch origin "$PR444_HEAD" --depth=1
git -C "$WASI_LIBC_DIR" checkout -q "$PR444_HEAD"

# wasi-libc's Makefile needs the clang binary directory on PATH so it can
# find clang, llvm-ar, and the compiler-rt builtins for the target.
clang_bin="$(dirname "$CLANG")"
export PATH="$clang_bin:$PATH"

# Build the wasm64 sysroot into <output_dir>/sysroot. `no-check-symbols`
# skips the predefined-macro sanity check, which fails with newer clangs
# that define __wasip1__ (the PR #444 expected metadata predates it).
make -C "$WASI_LIBC_DIR" WASM64=yes SYSROOT="$SYSROOT" no-check-symbols \
  -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

if [[ ! -f "$SYSROOT/lib/wasm64-wasip1/libc.a" ]]; then
  echo "error: wasi-libc build did not produce $SYSROOT/lib/wasm64-wasip1/libc.a" >&2
  exit 1
fi

# wasi-libc does not install the wasm64 compiler-rt builtins archive; build it
# from the repo's LLVM compiler-rt sources and install it under the name
# mojo-build.cpp expects.
"$SCRIPT_DIR/build_wasm64_builtins.sh" "$OUT/builtins" "$ROOT"
cp "$OUT/builtins/libclang_rt.builtins-wasm64.a" \
  "$SYSROOT/lib/wasm64-wasip1/libclang_rt.builtins-wasm64.a"

echo "built wasm64 wasi-libc sysroot at $SYSROOT"

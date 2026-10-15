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
# Build the wasm64 KGEN_CompilerRT_* glue archive against the real wasi-libc
# wasm64 sysroot.
#
# Hand-driven fallback for non-Bazel workflows and the `bazel test` sandbox
# (the Bazel-built :wasi_sysroot already installs this archive into the
# sysroot; this script builds it standalone). Compiled with the repo's
# clang-22 and archived with llvm-ar. libc symbols and compiler-rt builtins
# are NOT compiled here — they come from the real wasi-libc sysroot and LLVM
# compiler-rt at link time.
#
# Usage:
#   build_wasm64_shim.sh [output_dir] [sysroot]
#     output_dir  default: /tmp/kgen_wasm_shim
#     sysroot     default: /tmp/wasi-libc/sysroot
#   WASI_CLANG env overrides the clang binary (needed under `bazel test`,
#   where the sandbox hides the Bazel cache toolchain).
set -euo pipefail

OUT="${1:-/tmp/kgen_wasm_shim}"
SYSROOT="${2:-/tmp/wasi-libc/sysroot}"
mkdir -p "$OUT"
# Remove stale objects: the archive is built from $OUT/*.o, so objects from
# earlier source lists (e.g. a removed wasi_libc.c) would otherwise be swept
# in and cause duplicate-symbol link errors.
rm -f "$OUT"/*.o "$OUT/libkgen_compilerrt_wasi.a"

if [[ ! -f "$SYSROOT/lib/wasm64-wasip1/libc.a" ]]; then
  echo "error: wasm64 wasi-libc sysroot not found at $SYSROOT" >&2
  echo "       build it first (wasi-libc PR #444: make WASM64=yes)" >&2
  exit 1
fi

# Locate the repo's pinned clang-22 toolchain. Prefer the Bazel external
# cache's host toolchain (note: Bazel's `external/` dir is a symlink, so
# `find -L`); fall back to PATH.
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/_locate_clang.sh"
CLANG="$(kgen_locate_clang "${WASI_CLANG:-}")"
if [ -z "$CLANG" ]; then
  echo "error: could not locate clang-22 (set WASI_CLANG, or use the Bazel cache or PATH)" >&2
  exit 1
fi
LLVM_AR="$(dirname "$CLANG")/llvm-ar"

SOURCES=(kgen_compilerrt_wasi wasi_alloc wasi_stubs wasi_libc_shims)

for src in "${SOURCES[@]}"; do
  "$CLANG" --target=wasm64-unknown-wasip1 --sysroot="$SYSROOT" -O2 \
    -c "$HERE/$src.c" -o "$OUT/$src.o"
done

"$LLVM_AR" rcs "$OUT/libkgen_compilerrt_wasi.a" "$OUT"/*.o
echo "built $OUT/libkgen_compilerrt_wasi.a"

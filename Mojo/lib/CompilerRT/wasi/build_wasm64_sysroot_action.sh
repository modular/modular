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
# Bazel action: build the wasm64 wasi-libc sysroot, the wasm64 compiler-rt
# builtins archive, and the KGEN_CompilerRT_* glue archive in one shot.
#
# This is the hermetic sibling of build_wasi_sysroot.sh +
# build_wasm64_builtins.sh + build_wasm64_shim.sh: same commands, but all
# inputs (wasi-libc sources, compiler-rt builtins sources, clang-22 tools)
# come from Bazel-declared action inputs instead of /tmp checkouts and
# Bazel-cache globbing.
#
# Args:
#   $1  clang bin dir (contains clang-22, llvm-ar, llvm-nm)
#   $2  wasi-libc source root (dir containing Makefile)
#   $3  compiler-rt builtins source dir (compiler-rt/lib/builtins)
#   $4  builtins source list (wasm64_builtins_sources.txt)
#   $5  shim source dir (this directory: kgen_compilerrt_wasi.c etc.)
#   $6  output sysroot directory (declared tree artifact)
set -euo pipefail

# Absolutize everything: `make -C` changes directory, so relative tool
# paths would not resolve inside the make recipes.
CLANG_BIN="$PWD/$1"
WASI_SRC="$PWD/$2"
BUILTINS_SRC="$PWD/$3"
BUILTINS_LIST="$PWD/$4"
SHIM_SRC="$PWD/$5"
OUT="$6"

CLANG="$CLANG_BIN/clang-22"
LLVM_AR="$CLANG_BIN/llvm-ar"
LLVM_NM="$CLANG_BIN/llvm-nm"
for tool in "$CLANG" "$LLVM_AR" "$LLVM_NM"; do
  if [[ ! -x "$tool" ]]; then
    echo "error: expected tool missing or not executable: $tool" >&2
    exit 1
  fi
done

# The action additionally needs POSIX host tools that Bazel does not
# declare: tar (source staging), make (wasi-libc's build system, which
# itself shells out to /bin/sh), and getconf (job count). The rule is
# tagged `no-remote-exec` so it always runs on such a host; fail loudly
# here instead of with an opaque make/tar error deep in the build.
for tool in tar make getconf; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required host tool not found on PATH: $tool" >&2
    exit 1
  fi
done

OUT_ABS="$PWD/$OUT"
# Sandbox inputs are read-only symlinks and wasi-libc's Makefile builds
# in-tree (OBJDIR=build/$(TARGET_TRIPLE)), so copy the sources to a
# writable scratch dir inside the action workspace first.
WORK="$PWD/wasm64-sysroot-work.$$"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/src"
tar -cf - -C "$WASI_SRC" . | tar -xf - -C "$WORK/src"

JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# `no-check-symbols` skips the predefined-macro sanity check, which fails
# with newer clangs that define __wasip1__ (the PR #444 expected metadata
# predates it). CC/NM/AR are passed explicitly: the Makefile derives NM/AR
# from CC by patsubst, which would produce `llvm-22-ar`-style names when
# CC is clang-22.
make -C "$WORK/src" WASM64=yes SYSROOT="$OUT_ABS" \
  CC="$CLANG" NM="$LLVM_NM" AR="$LLVM_AR" \
  no-check-symbols -j"$JOBS"

if [[ ! -f "$OUT_ABS/lib/wasm64-wasip1/libc.a" ]]; then
  echo "error: wasi-libc build did not produce libc.a" >&2
  exit 1
fi

# compiler-rt builtins: wasi-libc does not install the wasm64 builtins
# archive and the pinned clang-22 does not ship one, so compile the
# unconditional GENERIC_SOURCES subset and install it under the name
# mojo-build.cpp expects.
mkdir -p "$WORK/builtins"
while IFS= read -r src; do
  [[ -z "$src" || "$src" == \#* ]] && continue
  "$CLANG" --target=wasm64-unknown-wasip1 -O2 -fno-builtin \
    -I"$BUILTINS_SRC" \
    -c "$BUILTINS_SRC/$src" -o "$WORK/builtins/${src%.c}.o"
done < "$BUILTINS_LIST"
"$LLVM_AR" rcs "$OUT_ABS/lib/wasm64-wasip1/libclang_rt.builtins-wasm64.a" \
  "$WORK/builtins"/*.o

# KGEN_CompilerRT_* glue archive, installed inside the sysroot so the whole
# wasm64 link interface is one tree artifact.
SHIM_SOURCES=(kgen_compilerrt_wasi wasi_alloc wasi_stubs wasi_libc_shims)
objs=()
for src in "${SHIM_SOURCES[@]}"; do
  "$CLANG" --target=wasm64-unknown-wasip1 --sysroot="$OUT_ABS" -O2 \
    -c "$SHIM_SRC/$src.c" -o "$WORK/$src.o"
  objs+=("$WORK/$src.o")
done
"$LLVM_AR" rcs "$OUT_ABS/lib/wasm64-wasip1/libkgen_compilerrt_wasi.a" "${objs[@]}"

echo "built wasm64 sysroot at $OUT_ABS"

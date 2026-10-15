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
# Build the wasm64 compiler-rt builtins archive (libclang_rt.builtins-wasm64.a)
# from the repo's LLVM compiler-rt sources.
#
# The pinned clang-22 toolchain does not ship wasm64 WASI builtins in its
# resource dir, and wasi-libc does not install them, so they are compiled
# here from the LLVM source checked out as a Bazel external repo. The source
# list is the unconditional GENERIC_SOURCES + GENERIC_TF_SOURCES subset
# (no x87 80-bit float, no bfloat16), which is what wasm64 codegen needs.
#
# Usage:
#   build_wasm64_builtins.sh [output_dir] [repo_root]
#     output_dir  default: /tmp/wasm64_builtins
#     repo_root   default: the modular repo containing this script
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-/tmp/wasm64_builtins}"
ROOT="${2:-$(cd "$HERE/../../../.." && pwd)}"
mkdir -p "$OUT"

# Start clean: like build_wasm64_shim.sh, sweep stale .o files so objects
# from a previous run (a source removed from wasm64_builtins_sources.txt,
# or a partial failed build) are not swept into the archive.
rm -f "$OUT"/*.o "$OUT/libclang_rt.builtins-wasm64.a"

# Locate the repo's pinned clang-22 toolchain (same logic as the other
# build scripts).
# shellcheck source=/dev/null
source "$HERE/_locate_clang.sh"
# shellcheck disable=SC2119
CLANG="$(kgen_locate_clang)"
if [ -z "$CLANG" ]; then
  echo "error: could not locate clang-22 (Bazel cache or PATH)" >&2
  exit 1
fi
LLVM_AR="$(dirname "$CLANG")/llvm-ar"

# compiler-rt builtins sources: the Bazel external LLVM repo. Prefer the
# execroot layout (when invoked from one); otherwise search the standard
# Bazel caches. Run a Bazel build first so the external repo is checked out.
BUILTINS_SRC="$ROOT/external/+llvm_configure+llvm-project/compiler-rt/lib/builtins"
if [[ ! -d "$BUILTINS_SRC" ]]; then
  BUILTINS_SRC=""
  for cache in \
    "${HOME}/Library/Caches/bazel/_bazel_$(whoami)" \
    "${HOME}/.cache/bazel/_bazel_$(whoami)"; do
    [[ -d "$cache" ]] || continue
    BUILTINS_SRC="$(find -L "$cache" -maxdepth 8 -type d -path "*llvm-project*/compiler-rt/lib/builtins" 2>/dev/null | head -1 || true)"
    [[ -n "$BUILTINS_SRC" ]] && break
  done
fi
if [[ -z "$BUILTINS_SRC" || ! -d "$BUILTINS_SRC" ]]; then
  echo "error: compiler-rt builtins sources not found" >&2
  echo "       run a Bazel build first so the LLVM external repo is checked out" >&2
  exit 1
fi

while IFS= read -r src; do
  [[ -z "$src" || "$src" == \#* ]] && continue
  "$CLANG" --target=wasm64-unknown-wasip1 -O2 -fno-builtin \
    -I"$BUILTINS_SRC" \
    -c "$BUILTINS_SRC/$src" -o "$OUT/${src%.c}.o"
done < "$HERE/wasm64_builtins_sources.txt"

"$LLVM_AR" rcs "$OUT/libclang_rt.builtins-wasm64.a" "$OUT"/*.o
echo "built $OUT/libclang_rt.builtins-wasm64.a"

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
# End-to-end wasm64 cross-compile gate for the native `mojo` compiler.
#
# Cross-compiles the checked-in smoke programs to standard WASI commands
# and runs them on a real wasm64 preview1 runtime (wasmtime by default,
# WAVM via WASM64_RUNTIME=wavm), failing if any step or expected-output
# check fails.
#
# Usage: wasm64_smoke_test.sh [repo_root] [sysroot] [wavm]
#
# Environment overrides (positional args win where shown above):
#   WASM64_REPO_ROOT  repo checkout root (default: derived from script path)
#   WASM64_RUNTIME    wasmtime (default) | wavm
#   WASI_SYSROOT      wasm64 wasi-libc sysroot (default: Bazel-built
#                     :wasi_sysroot, else /tmp/wasi-libc/sysroot)
#   WASI_SHIM         KGEN_CompilerRT_* archive (default: inside the sysroot,
#                     else built by build_wasm64_shim.sh)
#   WAVM_BIN          WAVM runtime (default /tmp/WAVM-build/bin/wavm)
#   WASMTIME_RUNNER   wasmtime host runner (default tools/wasm64/wasmtime-runner.py)
#   MOJO_BIN          native mojo compiler (default bazel-bin/Mojo/tools/mojo/mojo)
#   MOJO_STDLIB       dir containing std.mojoc (default bazel-bin/Mojo/stdlib/std)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${WASM64_REPO_ROOT:-${1:-$(cd "$HERE/../../../.." && pwd)}}"
# wasmtime is the default runtime: the wasm targets now default
# +relaxed-simd, and the legacy WAVM predates that proposal (and memory64
# only via local patches), so it cannot run the default-feature output.
RUNTIME="${WASM64_RUNTIME:-wasmtime}"

# Shared clang-22 discovery (see _locate_clang.sh).
# shellcheck source=/dev/null
source "$HERE/_locate_clang.sh"

# Sysroot resolution: explicit arg/env, then the Bazel-built tree artifact
# (runfiles layout when run under `bazel test`, bazel-bin layout when run
# from a checkout), then the hand-built /tmp location.
SYSROOT="${2:-${WASI_SYSROOT:-}}"
if [[ -z "$SYSROOT" ]]; then
  for cand in \
    "$HERE/wasi_sysroot" \
    "$ROOT/bazel-bin/Mojo/lib/CompilerRT/wasi/wasi_sysroot" \
    /tmp/wasi-libc/sysroot; do
    if [[ -f "$cand/lib/wasm64-wasip1/libc.a" ]]; then
      SYSROOT="$cand"
      break
    fi
  done
fi
WAVM="${3:-${WAVM_BIN:-/tmp/WAVM-build/bin/wavm}}"
RUNNER="${WASMTIME_RUNNER:-$ROOT/tools/wasm64/wasmtime-runner.py}"
MOJO="${MOJO_BIN:-$ROOT/bazel-bin/Mojo/tools/mojo/mojo}"
STDLIB="${MOJO_STDLIB:-$ROOT/bazel-bin/Mojo/stdlib/std}"
TESTS_DIR="$HERE/tests"
WORK_DIR="$(mktemp -d /tmp/wasm64-smoke.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

if [[ ! -x "$MOJO" ]]; then
  echo "error: mojo binary not found at $MOJO (build //Mojo/tools/mojo:mojo first)" >&2
  exit 1
fi
if [[ ! -f "$STDLIB/std.mojoc" ]]; then
  echo "error: stdlib not found at $STDLIB/std.mojoc (build '@mojo//:std' with --config=build-mojo first)" >&2
  exit 1
fi
if [[ ! -f "$SYSROOT/lib/wasm64-wasip1/libc.a" ]]; then
  echo "error: wasm64 wasi-libc sysroot not found at $SYSROOT" >&2
  echo "       build //Mojo/lib/CompilerRT/wasi:wasi_sysroot (or see PINS.md)" >&2
  exit 1
fi

# The mojo compiler links with wasm-ld from PATH. If it is missing, prepend
# the Bazel cache's pinned clang-22 bin dir. Under `bazel test` $HOME is
# redirected into the sandbox, so first derive the real cache root from this
# script's own path (runfiles always live under
# <cache>/_bazel_<user>/<hash>/{sandbox,execroot}/); the $HOME glob is the
# fallback for direct invocations.
if ! command -v wasm-ld >/dev/null 2>&1; then
  # Under `bazel test` $HOME is redirected into the sandbox, so first derive
  # the real cache root from this script's own path (runfiles always live
  # under <cache>/_bazel_<user>/<hash>/{sandbox,execroot}/); otherwise the
  # standard Bazel caches (via the helper) are the fallback.
  CLANG=""
  if [[ "$HERE" == *"/_bazel_"* ]]; then
    CACHE_ROOT="${HERE%%/sandbox/*}"
    CACHE_ROOT="${CACHE_ROOT%%/execroot/*}"
    CLANG="$(kgen_locate_clang "$CACHE_ROOT/external")"
  else
    CLANG="$(kgen_locate_clang)"
  fi
  if [[ -n "$CLANG" && -x "$(dirname "$CLANG")/wasm-ld" ]]; then
    clang_bin="$(dirname "$CLANG")"
    export PATH="$clang_bin:$PATH"
  fi
fi

# The KGEN_CompilerRT_* glue archive ships inside the Bazel-built sysroot;
# otherwise build it against the given sysroot.
SHIM="${WASI_SHIM:-$SYSROOT/lib/wasm64-wasip1/libkgen_compilerrt_wasi.a}"
if [[ ! -f "$SHIM" ]]; then
  "$HERE/build_wasm64_shim.sh" "$WORK_DIR/shim" "$SYSROOT" >/dev/null
  SHIM="$WORK_DIR/shim/libkgen_compilerrt_wasi.a"
fi

# Run a wasm64 WASI command. wasi_features does real file I/O, so it gets a
# writable WASI root (WAVM --mount-root; the wasmtime runner preopens ".").
WASI_ROOT="$WORK_DIR/wasiroot"
mkdir -p "$WASI_ROOT"

run_wasm() {
  local wasm="$1"
  local needs_root="${2:-}"
  case "$RUNTIME" in
    wavm)
      if [[ ! -x "$WAVM" ]]; then
        if command -v wavm >/dev/null 2>&1; then
          WAVM="$(command -v wavm)"
        else
          echo "error: WAVM runtime not found at $WAVM (and not on PATH)" >&2
          echo "       build it from trcrsired/WAVM, or use WASM64_RUNTIME=wasmtime" >&2
          exit 1
        fi
      fi
      if [[ -n "$needs_root" ]]; then
        "$WAVM" run --abi=wasi --enable memory64 --mount-root "$WASI_ROOT" "$wasm"
      else
        "$WAVM" run --abi=wasi --enable memory64 "$wasm"
      fi
      ;;
    wasmtime)
      if [[ -n "$needs_root" ]]; then
        (cd "$WASI_ROOT" && python3 "$RUNNER" "$wasm")
      else
        python3 "$RUNNER" "$wasm"
      fi
      ;;
    *)
      echo "error: unknown WASM64_RUNTIME '$RUNTIME' (expected wavm or wasmtime)" >&2
      exit 1
      ;;
  esac
}

declare -a HELLO_EXPECT=(
  "Hello from Mojo on wasm64!"
)

declare -a SMOKE_EXPECT=(
  "Hello, wasm64!"
  "upper: HELLO, WASM64!"
  "byte_length: 14"
  "starts_with: True"
  "sum 1..10: 55"
  "sqrt(2): 1.4142135623730951"
  "floor(2.7): 2.0"
  "ceil(2.1): 3.0"
  "list: [1, 2, 3, 4]"
  "list len: 4"
  "list sum: 10"
  "dict: {a: 1, b: 2}"
  "dict a: 1"
)

hello_out="" smoke_out="" wasi_features_out=""
# The driver auto-links the glue from the sysroot; pass it explicitly when
# the shim was built outside the sysroot.
declare -a LINK_ARGS=()
if [[ ! -f "$SYSROOT/lib/wasm64-wasip1/libkgen_compilerrt_wasi.a" ]]; then
  LINK_ARGS=(-Xlinker "$SHIM" -Xlinker --undefined=__main_argc_argv)
fi
for src in hello.mojo smoke.mojo wasi_features.mojo; do
  wasm="$WORK_DIR/${src%.mojo}.wasm"
  echo "==> compiling $src"
  "$MOJO" build \
    --target-triple wasm64-unknown-wasip1 \
    --wasi-sysroot "$SYSROOT" \
    "$TESTS_DIR/$src" \
    -o "$wasm" \
    ${LINK_ARGS[@]+"${LINK_ARGS[@]}"} \
    -mojo-search-paths "$STDLIB"

  echo "==> running $src ($RUNTIME)"
  if [[ "$src" == "wasi_features.mojo" ]]; then
    out="$(run_wasm "$wasm" needs_root)"
  else
    out="$(run_wasm "$wasm")"
  fi
  echo "$out"
  case "$src" in
    hello.mojo) hello_out="$out" ;;
    smoke.mojo) smoke_out="$out" ;;
    *) wasi_features_out="$out" ;;
  esac
done

# The driver must reject non-executable wasm output and non-WASI sysroot use.
if "$MOJO" build --target-triple wasm64-unknown-wasip1 --wasi-sysroot "$SYSROOT" \
    --emit shared-lib "$TESTS_DIR/hello.mojo" -o "$WORK_DIR/bad.so" \
    ${LINK_ARGS[@]+"${LINK_ARGS[@]}"} -mojo-search-paths "$STDLIB" >/dev/null 2>&1; then
  echo "error: shared-lib wasm build unexpectedly succeeded" >&2
  exit 1
fi
# Assert the rejection (not just failure): the driver must refuse
# --wasi-sysroot for a freestanding triple at option validation, so the
# build never even reaches codegen or linking.
bad_out="$("$MOJO" build --target-triple wasm64-unknown-unknown \
    --wasi-sysroot "$SYSROOT" "$TESTS_DIR/hello.mojo" -o "$WORK_DIR/bad.wasm" \
    -mojo-search-paths "$STDLIB" 2>&1 >/dev/null)" && {
  echo "error: sysroot with freestanding triple unexpectedly succeeded" >&2
  exit 1
}
if ! grep -qF -- "--wasi-sysroot requires a WASI target triple" <<<"$bad_out"; then
  echo "error: freestanding sysroot build failed for the wrong reason:" >&2
  echo "$bad_out" >&2
  exit 1
fi

# Wasm targets must default +simd128: LLVM's wasm defaults leave it off,
# which scalarizes every Mojo SIMD op on the target. They must also default
# +relaxed-simd so LLVM can contract mul+add into fused relaxed ops
# (engines lower these to FMA where available).
effective_out="$("$MOJO" build --print-effective-target \
  --target-triple wasm64-unknown-wasip1 --wasi-sysroot "$SYSROOT" \
  "$TESTS_DIR/hello.mojo" -mojo-search-paths "$STDLIB" 2>/dev/null)"
if ! grep -q -- "--target-features +simd128" <<<"$effective_out"; then
  echo "error: wasm triple does not default +simd128" >&2
  echo "$effective_out" >&2
  exit 1
fi
if ! grep -q -- "+relaxed-simd" <<<"$effective_out"; then
  echo "error: wasm triple does not default +relaxed-simd" >&2
  echo "$effective_out" >&2
  exit 1
fi

# Verify the expected lines appear in each program's output.
check_output() {
  local label="$1"
  local out="$2"
  shift 2
  for needle in "$@"; do
    if ! grep -qF "$needle" <<<"$out"; then
      echo "error: $label output missing: $needle" >&2
      exit 1
    fi
  done
}

check_output hello "$hello_out" "${HELLO_EXPECT[@]}"
check_output smoke "$smoke_out" "${SMOKE_EXPECT[@]}"

# wasi_features prints PASS/FAIL per check and exits nonzero on failure;
# require the all-pass marker (it needs the mounted WASI root for file I/O).
if ! grep -qF "wasi_features: ALL PASS" <<<"$wasi_features_out"; then
  echo "error: wasi_features reported failures" >&2
  exit 1
fi

echo "wasm64 smoke test: PASS ($RUNTIME)"

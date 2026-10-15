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
# Sourced library (do not execute): locates the repo's pinned clang-22.
# Shared by the hand build scripts and the smoke test. The Bazel sysroot
# action does NOT use it — bazel passes the toolchain dir as a declared
# action input, so it needs no discovery.

# Print the path of the repo's pinned clang-22, or nothing.
#
# Each argument is tried in order: a directory is searched for the pinned
# host toolchain like the Bazel cache (Bazel's `external/` is a symlink, so
# `find -L`); an executable file is used directly. The standard Bazel caches
# and PATH are the final fallbacks.
kgen_locate_clang() {
  local glob hit cand cache
  case "$(uname -s)-$(uname -m)" in
    Darwin-*)
      glob='*clang-macos*/bin/clang-22' ;;
    Linux-aarch64|Linux-arm64)
      glob='*clang-linux-aarch64*/bin/clang-22' ;;
    Linux-*)
      glob='*clang-linux-x86_64*/bin/clang-22' ;;
    *)
      glob='*clang*/bin/clang-22' ;;
  esac
  for cand in "$@"; do
    if [[ -d "$cand" ]]; then
      hit="$(find -L "$cand" -maxdepth 5 -path "$glob" -type f 2>/dev/null | head -1 || true)"
      if [[ -n "$hit" && -x "$hit" ]]; then
        printf '%s\n' "$hit"
        return 0
      fi
    elif [[ -x "$cand" ]]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  for cache in \
    "${HOME}/Library/Caches/bazel/_bazel_$(whoami)" \
    "${HOME}/.cache/bazel/_bazel_$(whoami)"; do
    [[ -d "$cache" ]] || continue
    hit="$(find -L "$cache" -maxdepth 5 -path "$glob" -type f 2>/dev/null | head -1 || true)"
    if [[ -n "$hit" && -x "$hit" ]]; then
      printf '%s\n' "$hit"
      return 0
    fi
  done
  command -v clang-22 2>/dev/null || true
}

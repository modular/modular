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

# Builds the Intel Mac toolchain (community-supported): LLVM and tcmalloc.
# Usage: bootstrap.sh [WORK_DIR]   (or set WORK_DIR; default build/intel-mac)
# The result is in WORK_DIR/toolchain. On a 6-core i7, LLVM took about an hour.
# At the end it prints the line to add to local.bazelrc.

set -euo pipefail
die() { echo "error: $*" >&2; exit 1; }

# uname -m also says x86_64 under Rosetta; hw.optional.arm64 is 1 on Apple silicon.
if [[ $(uname -s) != Darwin || $(uname -m) != x86_64 ||
  $(sysctl -n hw.optional.arm64 2>/dev/null || echo 0) == 1 ]]; then
  echo "error: this script is only for Intel Macs (darwin x86_64)" >&2
  exit 1
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
work=${1:-${WORK_DIR:-$repo/build/intel-mac}}
mkdir -p "$work"
work=$(cd "$work" && pwd)

# Build the LLVM and gperftools versions the repo pins for Apple silicon.
pins=$repo/bazel/common.MODULE.bazel
pin() { # pin NAME PATTERN: the version in the first URL matching PATTERN
  local v
  v=$(sed -n "s|.*$2.*|\\1|p" "$pins" | head -n1)
  [[ -n $v ]] || die "no $1 version in $pins (looked for a URL like $2)"
  echo "$v"
}
LLVM_VERSION=$(pin clang-macos '/llvm-macos/\([0-9][0-9.]*\)/[0-9a-f]*/llvm-macos\.tar\.xz')
GPERFTOOLS_VERSION=$(pin gperftools-macos '/gperftools/\([0-9][0-9.]*\)/[0-9a-f]*/tcmalloc\.tar\.gz')
# GitHub publishes no checksum for gperftools 2.15 (uploaded before it started), so pin it here.
GPERFTOOLS_SHA256=c69fef855628c81ef56f12e3c58f2b7ce1f326c0a1fe783e5cae0b88cbbe9a80
[[ $GPERFTOOLS_VERSION == 2.15 ]] ||
  die "$pins pins gperftools $GPERFTOOLS_VERSION; update GPERFTOOLS_SHA256 in this script"

# digest REPO TAG ASSET: the sha256 GitHub publishes for a release asset.
digest() {
  local d
  d=$(curl -fsS "https://api.github.com/repos/$1/releases/tags/$2" | python3 -I -c '
import json, sys
for asset in json.load(sys.stdin)["assets"]:
    if asset["name"] == sys.argv[1] and asset.get("digest"):
        print(asset["digest"].removeprefix("sha256:"))
' "$3")
  [[ $d =~ ^[0-9a-f]{64}$ ]] || die "GitHub publishes no sha256 digest for $3 in $1 $2"
  echo "$d"
}

# fetch URL SHA256 FILE: download, check the checksum, unpack next to FILE.
fetch() {
  curl -fL --retry 3 -o "$3" "$1"
  echo "$2  $3" | shasum -a 256 -c -
  tar -xf "$3" -C "$(dirname "$3")"
}

# --- LLVM ---

# Only what the toolchain, sanitizers and coverage need. "builtins;runtimes"
# installs libclang_rt.osx.a and the runtime headers. Do not add clang-cpp:
# it installs libclang-cpp.dylib.
components="clang;clang-resource-headers;lld;dsymutil;llvm-ar;llvm-as;llvm-bitcode-strip;llvm-cov;llvm-dlltool;llvm-dwp;llvm-extract-bundle-entry;llvm-install-name-tool;llvm-lib;llvm-nm;llvm-objcopy;llvm-objdump;llvm-otool;llvm-profdata;llvm-ranlib;llvm-strip;builtins;runtimes"

mkdir -p "$work/llvm"
llvm_src="$work/llvm/llvm-project-$LLVM_VERSION.src"
tag=llvmorg-$LLVM_VERSION
fetch "https://github.com/llvm/llvm-project/releases/download/$tag/${llvm_src##*/}.tar.xz" \
  "$(digest llvm/llvm-project "$tag" "${llvm_src##*/}.tar.xz")" "$llvm_src.tar.xz"

# DARWIN_osx_ARCHS=x86_64: without it compiler-rt also builds x86_64h, and
# linking the sanitizer libraries fails in lipo.
cmake -G Ninja -S "$llvm_src/llvm" -B "$work/llvm/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_APPEND_VC_REV=OFF \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DCMAKE_INSTALL_PREFIX="$work/llvm/stage" \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RUNTIMES=compiler-rt \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_Z3_SOLVER=OFF \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_ENABLE_CURL=OFF \
  -DLLVM_ENABLE_ZLIB=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DCLANG_INCLUDE_TESTS=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DCLANG_LINK_CLANG_DYLIB=OFF \
  -DLLVM_ENABLE_LIBCXX=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_INSTALL_UTILS=OFF \
  -DLLVM_PARALLEL_LINK_JOBS=3 \
  -DDARWIN_osx_ARCHS=x86_64 \
  -DCOMPILER_RT_ENABLE_IOS=OFF \
  -DCOMPILER_RT_ENABLE_WATCHOS=OFF \
  -DCOMPILER_RT_ENABLE_TVOS=OFF \
  -DLLVM_DISTRIBUTION_COMPONENTS="$components"
(cd "$work/llvm/build" && ninja distribution && ninja install-distribution-stripped)

# The toolchain directory that Bazel uses. It holds the LLVM install (bin, lib,
# ...) plus tcmalloc. No BUILD files: the Bazel repo rule supplies those.
out="$work/toolchain"
rm -rf "$out"
mkdir -p "$out"
cp -R "$work/llvm/stage/." "$out/"

# --- tcmalloc (gperftools) ---

mkdir -p "$work/gperftools"
gp_src="$work/gperftools/gperftools-$GPERFTOOLS_VERSION"
fetch "https://github.com/gperftools/gperftools/releases/download/gperftools-$GPERFTOOLS_VERSION/gperftools-$GPERFTOOLS_VERSION.tar.gz" \
  "$GPERFTOOLS_SHA256" "$gp_src.tar.gz"

rm -rf "$work/gperftools/build"
mkdir -p "$work/gperftools/build" "$out/gperftools"
(
  cd "$work/gperftools/build"
  export CC="clang -arch x86_64" CXX="clang++ -arch x86_64"
  export CFLAGS="-O2 -g -arch x86_64" CXXFLAGS="-O2 -g -arch x86_64"
  export MACOSX_DEPLOYMENT_TARGET=11.0
  "$gp_src/configure" --host=x86_64-apple-darwin --enable-minimal \
    --enable-static --disable-shared --disable-debugalloc --disable-libunwind
  make -j"$(sysctl -n hw.ncpu)" libtcmalloc_minimal.la
  cp .libs/libtcmalloc_minimal.a "$out/"
  # configure generates tcmalloc.h. The other headers are in the source tree.
  for h in tcmalloc malloc_hook malloc_extension malloc_hook_c malloc_extension_c nallocx stacktrace heap-profiler profiler heap-checker; do
    cp "src/gperftools/$h.h" "$out/gperftools/" 2>/dev/null ||
      cp "$gp_src/src/gperftools/$h.h" "$out/gperftools/"
  done
)

# --- Done ---

echo
echo "Done. Add this line to $repo/local.bazelrc:"
echo "common --repo_env=MODULAR_INTEL_MAC_TOOLCHAIN=$out"

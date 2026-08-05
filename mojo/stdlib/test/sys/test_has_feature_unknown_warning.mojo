# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #

# An unrecognized feature name always evaluates to false, hiding spelling bugs,
# so it warns. Real names -- including cross-arch ones that are false here (e.g.
# "neon" on x86_64) -- do not. The pinned triple keeps the check host-agnostic.

# RUN: %bare-mojo build --target-triple=x86_64-unknown-linux-gnu --emit=llvm %s -o - 2>&1 \
# RUN:   | FileCheck %s \
# RUN:       --implicit-check-not="'avx2' is not a recognized" \
# RUN:       --implicit-check-not="'rdrnd' is not a recognized" \
# RUN:       --implicit-check-not="'neon' is not a recognized"

from std.sys import CompilationTarget


def main() raises:
    # CHECK-DAG: warning: 'RDRAND' is not a recognized target feature name
    # CHECK-DAG: warning: 'AMX' is not a recognized target feature name
    print(CompilationTarget._has_feature["RDRAND"]())
    print(CompilationTarget._has_feature["AMX"]())

    # COM: None of these warn (asserted by --implicit-check-not above): the
    # COM: warning is about a name being *recognized*, not *enabled*. Under the
    # COM: pinned x86-64 triple they all fold to false (neon is aarch64-only;
    # COM: avx2/rdrnd need a newer CPU), but the value tracks the target -- e.g.
    # COM: neon is true on a native aarch64 host -- while the warning does not.
    print(CompilationTarget._has_feature["avx2"]())
    print(CompilationTarget._has_feature["rdrnd"]())
    print(CompilationTarget._has_feature["neon"]())

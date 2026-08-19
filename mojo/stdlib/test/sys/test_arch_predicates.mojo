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

# RUN: %bare-mojo build --target-triple=aarch64-unknown-linux-gnu --target-cpu=neoverse-n1 -D EXPECT_ARM --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=arm64-unknown-linux-gnu --target-cpu=neoverse-n1 -D EXPECT_ARM --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=x86_64-unknown-linux-gnu --target-cpu=x86-64 -D EXPECT_X86 --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=x86_64-unknown-linux-gnu --target-cpu=znver3 -D EXPECT_X86 -D EXPECT_SSE4 --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=i686-unknown-linux-gnu --target-cpu=i686 -D EXPECT_X86 --emit=llvm %s -o /dev/null

# `is_arm()` and `is_x86()` describe the architecture, so they must not vary
# with the target CPU. The baseline `x86-64` CPU is the interesting case: it has
# no SSE4.1, which `is_x86()` used to be an alias for. The `arm64` triple covers
# a spelling that only matches after canonicalization.

from std.sys import is_defined
from std.sys.info import CompilationTarget


def main():
    comptime expect_arm = is_defined["EXPECT_ARM"]()
    comptime expect_x86 = is_defined["EXPECT_X86"]()
    comptime expect_sse4 = is_defined["EXPECT_SSE4"]()

    comptime assert (
        CompilationTarget.is_arm() == expect_arm
    ), "is_arm() disagrees with the target triple"
    comptime assert (
        CompilationTarget.is_x86() == expect_x86
    ), "is_x86() disagrees with the target triple"
    comptime assert (
        CompilationTarget.has_sse4() == expect_sse4
    ), "has_sse4() must track the target CPU, not the architecture"

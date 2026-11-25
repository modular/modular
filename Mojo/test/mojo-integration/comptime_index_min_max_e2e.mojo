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

# RUN: mojo build --target-triple x86_64-unknown-linux-gnu --emit=llvm -o %t %s
# RUN: mojo build --target-triple aarch64-unknown-linux-gnu --emit=llvm -o %t %s
# RUN: mojo build --target-triple arm64-apple-macosx --emit=llvm -o %t %s

# Compile-time `min`/`max` of `Int`/`UInt` must work even when 32-bit and
# 64-bit arithmetic disagree. The values need a 64-bit `Int`, so this builds
# for each supported 64-bit target.


def smaller(a: Int, b: Int) -> Int:
    return min(a, b)


def main():
    comptime assert smaller(11, 1 << 40) == 11
    comptime assert max(11, 1 << 40) == 1 << 40
    comptime assert min(UInt(1 << 40), UInt(3)) == 3

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

# RUN: %mojo %s | FileCheck %s

# Compile-time `min`/`max` of `Int`/`UInt` must work even when 32-bit and
# 64-bit arithmetic disagree.


def smaller(a: Int, b: Int) -> Int:
    return min(a, b)


def main():
    # CHECK: min: 11
    comptime a = smaller(11, 1 << 40)
    print("min:", a)
    # CHECK: max: 1099511627776
    comptime b = max(11, 1 << 40)
    print("max:", b)
    # CHECK: umin: 3
    comptime c = min(UInt(1 << 40), UInt(3))
    print("umin:", c)

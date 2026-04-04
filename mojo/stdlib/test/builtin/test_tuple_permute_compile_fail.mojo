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

# Test that Tuple.permute rejects invalid permutations at compile time.

# RUN: not %mojo -D test=1 %s 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not %mojo -D test=2 %s 2>&1 | FileCheck --check-prefix CHECK_2 %s

from std.sys import get_defined_int


def main():
    comptime if get_defined_int["test"]() == 1:
        # Duplicate indices: [0, 0, 1] is not a valid permutation of [0, 1, 2].
        # CHECK_1: perm must be a valid permutation of [0, len-1]
        var t = (1, "hello", 3.14)
        var p = t.permute[Variadic.values[0, 0, 1]]()

    elif get_defined_int["test"]() == 2:
        # Out-of-range index: [0, 1, 5] is not a valid permutation of [0, 1, 2].
        # CHECK_2: perm must be a valid permutation of [0, len-1]
        var t = (1, "hello", 3.14)
        var p = t.permute[Variadic.values[0, 1, 5]]()

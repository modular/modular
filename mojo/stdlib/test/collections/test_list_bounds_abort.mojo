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
#
# Verifies that invalid `ContiguousSlice` indexing (out of bounds, reversed,
# or negative) aborts on `List` instead of silently clamping.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo -D test=1 %s 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not %mojo -D test=2 %s 2>&1 | FileCheck --check-prefix CHECK_2 %s
# RUN: not %mojo -D test=3 %s 2>&1 | FileCheck --check-prefix CHECK_3 %s
# RUN: not %mojo -D test=4 %s 2>&1 | FileCheck --check-prefix CHECK_4 %s
# RUN: not %mojo -D test=5 %s 2>&1 | FileCheck --check-prefix CHECK_5 %s

from std.sys import get_defined_int


def main() raises:
    var l: List = [1, 2, 3]

    comptime if get_defined_int["test"]() == 1:
        # CHECK_1: {{.*}}: Assert Error: slice end index 100 is out of bounds, valid range is 0 to 3
        _ = l[0:100]
    elif get_defined_int["test"]() == 2:
        # CHECK_2: {{.*}}: Assert Error: slice start index 100 is out of bounds, valid range is 0 to 3
        _ = l[100:200]
    elif get_defined_int["test"]() == 3:
        # CHECK_3: {{.*}}: Assert Error: slice start index 3 is greater than slice end index 1
        _ = l[3:1]
    elif get_defined_int["test"]() == 4:
        # CHECK_4: {{.*}}: Assert Error: slice start index -1 is out of bounds, valid range is 0 to 3
        _ = l[-1:]
    elif get_defined_int["test"]() == 5:
        # CHECK_5: {{.*}}: Assert Error: slice end index -1 is out of bounds, valid range is 0 to 3
        _ = l[:-1]
    else:
        comptime assert False, "unreachable!"

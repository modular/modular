# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from algorithm._sorting_network import _sort
from testing import assert_true
from testing import TestSuite

alias sizes = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    64,
]


def test_all_sizes():
    @parameter
    for size in sizes:
        var lst = List[Int64](capacity=size)

        for i in range(size):
            lst.append(i)

        _sort[size](lst)

        for i in range(size - 1):
            assert_true(lst[i] < lst[i + 1])


def test_all_sizes_reverse():
    @parameter
    for size in sizes:
        var lst = List[Int64](capacity=size)

        for i in reversed(range(size)):
            lst.append(i)

        _sort[size](lst)

        for i in range(size - 1):
            assert_true(lst[i] < lst[i + 1])


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

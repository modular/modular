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

from std.testing import assert_equal, TestSuite


def test_append_across_regrow_in_while_loop() raises:
    # data-dependent trip count, every power-of-two append regrows the list
    var list = List[UInt8]()
    var u = UInt64(1) << 60
    while True:
        list.append(55)
        u >>= 3
        if u == 0:
            break
    assert_equal(len(list), 21)
    for i in range(len(list)):
        assert_equal(list[i], 55, msg=String("index ", i))


def test_append_digits_while_loop() raises:
    var list = List[UInt8]()
    var u = UInt64(9223372036854775807)
    while True:
        list.append(UInt8(48 + Int(u % 10)))
        u //= 10
        if u == 0:
            break
    # digits come out least significant first
    assert_equal(String(unsafe_from_utf8=Span(list)), "7085774586302733229")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

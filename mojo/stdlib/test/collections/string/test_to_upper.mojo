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

from std.collections.string._to_upper import upper_utf8

from std.testing import assert_equal, assert_false, assert_true, assert_raises
from std.testing import TestSuite


def test_upper() raises:
    assert_equal(upper_utf8("hello"), "HELLO")
    assert_equal(upper_utf8("HELLO"), "HELLO")
    assert_equal(upper_utf8("FoOBaR"), "FOOBAR")

    assert_equal(upper_utf8("mojo🔥"), "MOJO🔥")

    assert_equal(upper_utf8("É"), "É")
    assert_equal(upper_utf8("é"), "É")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

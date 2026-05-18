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

from std.collections.string._to_lower import lower_utf8

from std.testing import assert_equal, assert_false, assert_true, assert_raises
from std.testing import TestSuite


def test_lower() raises:
    assert_equal(lower_utf8("HELLO"), "hello")
    assert_equal(lower_utf8("hello"), "hello")
    assert_equal(lower_utf8("FoOBaR"), "foobar")

    assert_equal(lower_utf8("MOJO🔥"), "mojo🔥")

    assert_equal(lower_utf8("É"), "é")
    assert_equal(lower_utf8("é"), "é")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

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

from testing import assert_equal, TestSuite
from test_utils import check_write_to


def raise_an_error():
    raise Error("MojoError: This is an error!")


def test_error_raising():
    try:
        raise_an_error()
    except e:
        assert_equal(String(e), "MojoError: This is an error!")


def test_from_and_to_string():
    var my_string: String = "FOO"
    var error = Error(my_string)
    assert_equal(String(error), "FOO")

    assert_equal(String(Error("bad")), "bad")
    assert_equal(repr(Error("err")), "Error('err')")


def test_error_write_to():
    check_write_to(Error("err"), expected="err", is_repr=False)
    check_write_to(Error("err"), expected="Error('err')", is_repr=True)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

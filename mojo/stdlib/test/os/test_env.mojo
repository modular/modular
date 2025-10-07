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

from os import getenv, setenv, unsetenv

from testing import TestSuite, assert_equal


def test_getenv():
    assert_equal(getenv("TEST_MYVAR"), "MyValue")

    assert_equal(getenv("TEST_MYVAR", "DefaultValue"), "MyValue")

    assert_equal(getenv("NON_EXISTENT_VAR", "DefaultValue"), "DefaultValue")


def test_setenv():
    assert_equal(setenv("NEW_VAR", "FOO", True), True)
    assert_equal(getenv("NEW_VAR"), "FOO")

    assert_equal(setenv("NEW_VAR", "BAR", False), True)
    assert_equal(getenv("NEW_VAR"), "FOO")

    assert_equal(setenv("NEW_VAR", "BAR", True), True)
    assert_equal(getenv("NEW_VAR", "BAR"), "BAR")

    assert_equal(setenv("=", "INVALID", True), False)


def test_unsetenv():
    assert_equal(setenv("NEW_VAR", "FOO", True), True)
    assert_equal(getenv("NEW_VAR"), "FOO")
    assert_equal(unsetenv("NEW_VAR"), True)
    assert_equal(getenv("NEW_VAR"), "")


def main():
    var suite = TestSuite()

    suite.test[test_getenv]()
    suite.test[test_setenv]()
    suite.test[test_unsetenv]()

    suite^.run()

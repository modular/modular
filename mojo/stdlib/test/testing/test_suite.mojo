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

from os import abort
from testing import assert_raises, assert_equal, assert_false, TestSuite


def test_nonconforming_signature(x: Int):
    raise Error("should not be run")


def nonconforming_name():
    raise Error("should not be run")


def test_failing():
    raise Error("should be raised")


def test_passing():
    pass


def main():
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    var report = suite.generate_report()
    suite^.disable()

    assert_equal(report.failures, 1)
    assert_equal(len(report.reports), 2)

    assert_equal(report.reports[0].name, "test_failing")
    assert_equal(String(report.reports[0].error), "should be raised")

    assert_equal(report.reports[1].name, "test_passing")
    assert_false(report.reports[1].error)

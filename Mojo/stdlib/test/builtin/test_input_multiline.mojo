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
# RUN: printf 'first\nsecond\nthird\n' | %mojo %s

from std.testing import testing, TestSuite


def test_input_reads_consecutive_piped_lines() raises:
    # Regression test for modular/modular#7198: each `input()` call must not
    # consume more than one line from a piped stdin.
    testing.assert_equal(input(), "first")
    testing.assert_equal(input(), "second")
    testing.assert_equal(input(), "third")
    with testing.assert_raises(contains="EOF"):
        _ = input()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

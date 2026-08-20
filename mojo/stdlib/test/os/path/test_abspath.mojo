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

from std.os.path import abspath, join, normpath
from std.pathlib import cwd

from std.testing import TestSuite, assert_equal, assert_true


def test_absolute_path_is_normalized_and_unchanged() raises:
    assert_equal(abspath("/a/b/../c"), "/a/c")
    assert_equal(abspath("/"), "/")


def test_relative_path_resolves_against_cwd() raises:
    var expected = normpath(join(String(cwd()), "foo/bar"))
    assert_equal(abspath("foo/bar"), expected)


def test_starts_with_root() raises:
    assert_true(abspath(".").startswith("/"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

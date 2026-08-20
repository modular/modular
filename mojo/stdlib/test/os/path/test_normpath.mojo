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

from std.os.path import normpath

from std.testing import TestSuite, assert_equal


def test_empty_path() raises:
    assert_equal(normpath(""), ".")


def test_dot_and_double_slashes() raises:
    assert_equal(normpath("."), ".")
    assert_equal(normpath("a//b/./c"), "a/b/c")
    assert_equal(normpath("a/./b/"), "a/b")


def test_parent_references() raises:
    assert_equal(normpath("a//b/./c/../d"), "a/b/d")
    assert_equal(normpath("a/b/../../../c"), "../c")
    assert_equal(normpath(".."), "..")
    assert_equal(normpath("../a/b"), "../a/b")


def test_absolute_paths() raises:
    assert_equal(normpath("/a/b/../c"), "/a/c")

    # A `..` above the root is clamped, matching Python's `posixpath`.
    assert_equal(normpath("/../a"), "/a")
    assert_equal(normpath("/.."), "/")


def test_leading_slashes() raises:
    # Per POSIX, exactly two leading slashes is implementation-defined and
    # preserved; one, or three-or-more, collapse to a single slash.
    assert_equal(normpath("/foo/bar"), "/foo/bar")
    assert_equal(normpath("//foo/bar"), "//foo/bar")
    assert_equal(normpath("///foo/bar"), "/foo/bar")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

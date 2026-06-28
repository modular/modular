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


def test_empty_and_dot() raises:
    assert_equal(".", normpath(""))
    assert_equal(".", normpath("."))
    assert_equal("..", normpath(".."))


def test_simple_relative() raises:
    assert_equal("foo", normpath("foo"))
    assert_equal("foo/bar", normpath("foo/bar"))


def test_trailing_separator() raises:
    assert_equal("foo", normpath("foo/"))
    assert_equal("foo/bar", normpath("foo/bar/"))


def test_collapse_separators() raises:
    assert_equal("foo/bar", normpath("foo//bar"))
    assert_equal("foo/bar", normpath("foo///bar"))
    assert_equal("foo/bar/baz", normpath("foo//bar///baz"))


def test_dot_components() raises:
    assert_equal("foo/bar", normpath("foo/./bar"))
    assert_equal("foo/bar", normpath("./foo/bar"))
    assert_equal("foo", normpath("foo/."))
    assert_equal("foo", normpath("./foo"))


def test_dotdot_components() raises:
    assert_equal("foo/baz", normpath("foo/bar/../baz"))
    assert_equal("baz", normpath("foo/bar/../../baz"))
    assert_equal(".", normpath("foo/.."))
    assert_equal("..", normpath("foo/../.."))
    assert_equal("../..", normpath("foo/../../.."))
    assert_equal("foo", normpath("foo/bar/.."))


def test_leading_dotdot() raises:
    assert_equal("../foo", normpath("../foo"))
    assert_equal("../../foo", normpath("../../foo"))
    assert_equal("../../../foo", normpath("../../../foo"))
    assert_equal("../bar", normpath("../foo/../bar"))
    assert_equal("..", normpath("../foo/.."))


def test_absolute_paths() raises:
    assert_equal("/foo/bar", normpath("/foo/bar"))
    assert_equal("/bar", normpath("/foo/../bar"))
    assert_equal("/", normpath("/foo/bar/../.."))
    assert_equal("/", normpath("/foo/bar/../../.."))
    assert_equal("/", normpath("/.."))
    assert_equal("/", normpath("/../.."))
    assert_equal("/foo", normpath("/../../foo"))


def test_root_directory() raises:
    assert_equal("/", normpath("/"))
    assert_equal("/", normpath("/."))
    assert_equal("/", normpath("/./"))
    assert_equal("/", normpath("/./."))
    assert_equal("/foo", normpath("/./foo"))


def test_two_leading_slashes() raises:
    assert_equal("//", normpath("//"))
    assert_equal("//foo", normpath("//foo"))
    assert_equal("//foo/bar", normpath("//foo/bar"))
    assert_equal("//foo/bar", normpath("//foo//bar"))
    assert_equal("//", normpath("//."))
    assert_equal("//", normpath("//foo/.."))


def test_three_plus_leading_slashes() raises:
    assert_equal("/", normpath("///"))
    assert_equal("/foo", normpath("///foo"))
    assert_equal("/foo", normpath("////foo"))


def test_complex_paths() raises:
    assert_equal("foo/bar/qux", normpath("foo//bar/./baz/../qux"))
    assert_equal("/a/b", normpath("/a//b/./c/.././d/.."))


def test_special_names() raises:
    assert_equal("foo/...", normpath("foo/..."))
    assert_equal("foo/....", normpath("foo/...."))
    assert_equal("foo/..bar", normpath("foo/..bar"))
    assert_equal("foo/.bar", normpath("foo/.bar"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

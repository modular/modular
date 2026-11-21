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

from std.os import listdir, mkdir
from std.pathlib import Path
from std.tempfile import TemporaryDirectory

from std.testing import TestSuite, assert_equal, assert_true


def test_listdir() raises:
    var ls = listdir(Path())
    assert_true(len(ls) > 0)


def test_listdir_names() raises:
    # listdir must return each entry's exact name, without "." and "..".
    # On Intel macOS it once read directory entries with the wrong layout and
    # returned names with their first bytes cut off, plus "." and "..", which
    # a check on the count alone would not catch. The long name makes sure the
    # whole name is read, not just its first few bytes.
    with TemporaryDirectory() as tmp_dir:
        var dir = Path(tmp_dir)
        _ = open(dir / "a", "w")
        _ = open(dir / "a_name_longer_than_sixteen_bytes.txt", "w")
        mkdir(dir / "subdir")

        var names = listdir(dir)
        sort(names)
        assert_equal(len(names), 3)
        assert_equal(names[0], "a")
        assert_equal(names[1], "a_name_longer_than_sixteen_bytes.txt")
        assert_equal(names[2], "subdir")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

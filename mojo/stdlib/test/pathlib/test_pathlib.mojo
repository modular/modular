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

import os
from pathlib import DIR_SEPARATOR, Path, cwd
from sys import CompilationTarget
from tempfile import NamedTemporaryFile

from builtin._location import __source_location
from testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
    TestSuite,
)


def test_cwd():
    assert_true(String(cwd()).startswith("/"))


def test_path():
    assert_true(String(Path() / "some" / "dir").endswith("/some/dir"))

    assert_equal(String(Path("/foo") / "bar" / "jar"), "/foo/bar/jar")

    assert_equal(
        String(Path("/foo" + DIR_SEPARATOR) / "bar" / "jar"), "/foo/bar/jar"
    )

    assert_not_equal(Path().stat().st_mode, 0)

    assert_true(len(Path().listdir()) > 0)


def test_path_exists():
    assert_true(
        Path(__source_location().file_name).exists(), msg="does not exist"
    )

    assert_false(
        (Path() / "this_path_does_not_exist.mojo").exists(), msg="exists"
    )


def test_path_isdir():
    assert_true(Path().is_dir())
    assert_false((Path() / "this_path_does_not_exist").is_dir())


def test_path_isfile():
    assert_true(Path(__source_location().file_name).is_file())
    assert_false(Path("this/file/does/not/exist").is_file())


def test_suffix():
    # Common filenames.
    assert_equal(Path("/file.txt").suffix(), ".txt")
    assert_equal(Path("file.txt").suffix(), ".txt")
    assert_equal(Path("file").suffix(), "")
    assert_equal(Path("my.file.txt").suffix(), ".txt")

    # Dot Files and Directories
    assert_equal(Path(".bashrc").suffix(), "")
    assert_equal(Path("my.folder/file").suffix(), "")
    assert_equal(Path("my.folder/.file").suffix(), "")

    # Special Characters in File Names
    assert_equal(Path("my file@2023.pdf").suffix(), ".pdf")
    assert_equal(Path("résumé.doc").suffix(), ".doc")


def test_joinpath():
    assert_equal(Path(), Path().joinpath())
    assert_equal(Path() / "some" / "dir", Path().joinpath("some", "dir"))


def test_read_write():
    var temp_file = Path(os.getenv("TEST_TMPDIR")) / "foo.txt"
    temp_file.write_text("hello")
    assert_equal(temp_file.read_text(), "hello")


def test_read_write_bytes():
    alias data = "hello world".as_bytes()
    with NamedTemporaryFile() as tmp:
        var file = Path(tmp.name)
        file.write_bytes(data)
        assert_equal(List[Byte](data), file.read_bytes())


fn get_user_path() -> Path:
    return Path("/home/user")


fn get_current_home() -> String:
    return os.env.getenv("HOME")


def set_home(path: Path):
    path_str = String(path)
    _ = os.env.setenv("HOME", path_str)


# More elaborate tests in `os/path/test_expanduser.mojo`
def test_expand_user():
    var user_path = get_user_path()
    var original_home = get_current_home()
    set_home(user_path)

    path = Path("~") / "test"
    test_path = user_path / "test"
    assert_equal(test_path, os.path.expanduser(path))
    # Original path should remain unmodified
    assert_equal(path, os.path.join("~", "test"))

    # Make sure this process doesn't break other tests by changing the home dir.
    set_home(original_home)


def test_home():
    var user_path = get_user_path()
    var original_home = get_current_home()
    set_home(user_path)

    assert_equal(user_path, Path.home())
    # Match Python behavior allowing `home()` to overwrite existing path.
    assert_equal(user_path, Path("test").home())

    # Ensure other tests in this process aren't broken by changing the home dir.
    set_home(original_home)


def test_stat():
    var path = Path(__source_location().file_name)
    var stat = path.stat()
    assert_equal(
        String(stat),
        StaticString(
            "os.stat_result(st_mode={}, st_ino={}, st_dev={}, st_nlink={},"
            " st_uid={}, st_gid={}, st_size={}, st_atime={}, st_mtime={},"
            " st_ctime={}, st_birthtime={}, st_blocks={}, st_blksize={},"
            " st_rdev={}, st_flags={})"
        ).format(
            stat.st_mode,
            stat.st_ino,
            stat.st_dev,
            stat.st_nlink,
            stat.st_uid,
            stat.st_gid,
            stat.st_size,
            String(stat.st_atimespec),
            String(stat.st_mtimespec),
            String(stat.st_ctimespec),
            String(stat.st_birthtimespec),
            stat.st_blocks,
            stat.st_blksize,
            stat.st_rdev,
            stat.st_flags,
        ),
    )


# More elaborate tests in `os/path/test_basename.mojo`
def test_name():
    # Root directories
    assert_equal("", Path("/").name())

    # Empty strings
    assert_equal("", Path("").name())

    # Current directory (matching behavior of python, doesn't resolve `..` etc.)
    assert_equal(".", Path(".").name())

    # Parent directory
    assert_equal("..", Path("..").name())

    # Absolute paths
    assert_equal("file", Path("/file").name())
    assert_equal("file.txt", Path("/file.txt").name())
    assert_equal("file", Path("/dir/file").name())
    assert_equal("file", Path("/dir/subdir/file").name())

    # Relative paths
    assert_equal("file", Path("dir/file").name())
    assert_equal("file", Path("dir/subdir/file").name())
    assert_equal("file", Path("file").name())


def test_parts():
    var path_to_file = Path("/path/to/file")
    assert_equal(path_to_file.parts(), path_to_file.path.split("/"))

    var rel_path = Path("path/to/file")
    assert_equal(rel_path.parts(), rel_path.path.split("/"))

    var path_no_slash = Path("path")
    assert_equal(path_no_slash.parts(), path_no_slash.path.split("/"))

    var path_with_tail_slash = Path("path/")
    assert_equal(
        path_with_tail_slash.parts(), path_with_tail_slash.path.split("/")
    )

    var root_path = Path("/")
    assert_equal(root_path.parts(), root_path.path.split("/"))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

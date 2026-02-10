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

from pathlib import cwd, Path
from testing import *


fn test_cwd() raises:
    from pathlib import cwd

    var string_path = cwd()
    print(string_path)


fn test_stat() raises:
    from pathlib import Path

    var p = Path()  # Path to cwd
    print(p.stat())  # os.stat_result(...)


fn test_exists() raises:
    from pathlib import Path

    var p = Path("./path/to/nowhere/does-not-exist")
    var result = "Exists" if p.exists() else "Does not exist"
    print(result)  # Should be "Does not exist" but not guaranteed


fn test_expanding_user() raises:
    from pathlib import Path
    from testing import assert_true

    var p = Path("~")
    assert_true(p.expanduser() == Path.home())


fn test_isdir() raises:
    from pathlib import Path
    from testing import assert_true

    assert_true(Path.home().is_dir())


fn test_isfile() raises:
    from pathlib import Path
    from testing import assert_false

    assert_false(Path.home().is_file())


fn test_read() raises:
    from pathlib import Path
    from testing import assert_true

    var p = Path("testfile.txt")
    p.write_text("test")
    if p.exists():
        var contents = p.read_text()
        assert_true(contents == "test")


fn test_read_bytes() raises:
    from pathlib import Path
    from testing import assert_true

    var p = Path("testfile.txt")
    p.write_text("test")
    if p.exists():
        var contents = p.read_bytes()
        assert_true(contents[0] == 116)


fn test_write_text() raises:
    from pathlib import Path
    from testing import assert_true

    var p = Path("testfile.txt")
    p.write_text("Hello")
    if p.exists():
        var contents = p.read_text()
        assert_true(contents == "Hello")


fn test_write_bytes() raises:
    from pathlib import Path
    from testing import assert_true

    var p = Path("testfile.txt")
    var s = "Hello"
    p.write_bytes(s.as_bytes())
    if p.exists():
        var contents = p.read_text()
        assert_true(contents == "Hello")


fn test_suffix() raises:
    from pathlib import Path
    from testing import assert_true

    var p = Path("testfile.txt")
    print(p.suffix())
    assert_true(p.suffix() == ".txt")

    p = Path(".hiddenfile")
    assert_true(p.suffix() == "")  # No suffix


fn test_joinpath() raises:
    from pathlib import Path

    # gettmpdir() has no guarantee of trailing /
    # Use joinpath to ensure path construction
    var p = Path("/tmp")
    p = p.joinpath("testdir")  # No trailing /
    p = p.joinpath("testfile.txt")
    assert_true(p == Path("/tmp/testdir/testfile.txt"))

    p = Path("/tmp/")
    p = p.joinpath("testdir/")  # Trailing /
    p = p.joinpath("testfile.txt")
    assert_true(p == Path("/tmp/testdir/testfile.txt"))


fn test_listdir() raises:
    from pathlib import Path

    for item in cwd().listdir():
        print(item)


fn test_name() raises:
    from pathlib import Path
    from testing import assert_true

    var p = Path("a/path/foo.txt")
    assert_true(p.name() == "foo.txt")


fn test_parts() raises:
    from pathlib import Path
    from testing import assert_true

    for p, q in zip(Path("a/path/foo.txt").parts(), ["a", "path", "foo.txt"]):
        assert_true(p == q)


fn main() raises:
    test_cwd()
    test_stat()
    test_exists()
    test_expanding_user()
    test_isdir()
    test_isfile()
    test_read()
    test_read_bytes()
    test_write_text()
    test_write_bytes()
    test_suffix()
    test_joinpath()
    test_listdir()
    test_name()
    test_parts()

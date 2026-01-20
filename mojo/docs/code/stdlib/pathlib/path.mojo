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


fn try_cwd():
    from pathlib import Path

    try:
        var string_path = cwd()
        print(string_path)
    except e:
        print(e)


fn try_stat():
    from pathlib import Path

    try:
        var p = Path()  # Path to cwd
        print(p.stat())  # os.stat_result(...)
    except e:
        print(e)


fn doesnt_exist():
    from pathlib import Path

    var p = Path("./path-to-nowhere")
    print("Exists" if p.exists() else "Does not exist")


fn expanding_user():
    from pathlib import Path

    try:
        var p = Path("~")
        assert_true(p.expanduser() == Path.home())
    except e:
        print(e)


fn test_isdir():
    from pathlib import Path

    try:
        assert_true(Path.home().is_dir())
    except e:
        print(e)


fn test_isfile():
    from pathlib import Path

    try:
        assert_false(Path.home().is_file())
    except e:
        print(e)


fn test_read():
    from pathlib import Path

    try:
        var p = Path("testfile.txt")
        p.write_text("test passes")
        if p.exists():
            var contents = p.read_text()
            print(contents)
    except e:
        print(e)


fn test_read_bytes():
    from pathlib import Path

    try:
        var p = Path("testfile.txt")
        p.write_text("test passes")
        if p.exists():
            var contents = p.read_bytes()
            assert_true(contents[0] == 116)
            print("Byte was 116")
    except e:
        print(e)


fn test_write_bytes():
    from pathlib import Path

    try:
        var p = Path("testfile.txt")
        var s = "Hello"
        p.write_bytes(s.as_bytes())
        if p.exists():
            var contents = p.read_text()
            print(contents)  # Hello
    except e:
        print(e)


fn test_extension():
    from pathlib import Path

    try:
        var p = Path("testfile.txt")
        print(p.suffix())
        assert_true(p.suffix() == ".txt")
    except e:
        print(e)


fn test_join():
    from pathlib import Path

    from tempfile import gettempdir

    var p = Path(gettempdir().or_else("/tmp/"))  # Both end with trailing /
    var path = p.joinpath("intermediate/")  # Trailing / for intermediate
    path = path.joinpath("tempfile.txt")  # No / for file name
    print(path)  # legal path


fn test_list():
    from pathlib import Path

    try:
        for item in cwd().listdir():
            print(item)
    except e:
        print(e)


fn test_parts():
    from pathlib import Path

    for p in Path("a/path/foo.txt").parts():
        print(p)  # a, path, foo.txt


fn main():
    try_cwd()
    try_stat()
    doesnt_exist()
    expanding_user()
    test_isdir()
    test_isfile()
    test_read()
    test_read_bytes()
    test_write_bytes()
    test_extension()
    test_join()
    test_list()
    test_parts()

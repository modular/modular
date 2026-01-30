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


from testing import *


fn test_gettempdir() raises:
    from tempfile import gettempdir

    var temp_dir = gettempdir()
    assert_true(temp_dir != None)
    for dir in temp_dir:
        assert_true(dir.startswith("/", 0, -1))


fn test_mkdtemp() raises:
    from tempfile import mkdtemp
    import os

    dir_path = mkdtemp()
    assert_true(os.path.exists(dir_path))
    assert_true(os.path.isdir(dir_path))
    os.rmdir(dir_path)
    assert_false(os.path.exists(dir_path))


fn test_rmtree() raises:
    from tempfile import mkdtemp
    import os

    dir_path = mkdtemp()
    assert_true(os.path.exists(dir_path))

    os.rmdir(dir_path)
    assert_false(os.path.exists(dir_path))


fn test_temporary_directory_context_manager() raises:
    from tempfile import TemporaryDirectory
    import os

    var temp_dir_copy: String
    with TemporaryDirectory() as temp_dir:
        temp_dir_copy = temp_dir.copy()
        assert_true(os.path.exists(temp_dir))
        assert_true(os.path.isdir(temp_dir))

    assert_false(os.path.exists(temp_dir_copy))


fn test_named_temporary_file_context_manager() raises:
    from tempfile import NamedTemporaryFile
    from pathlib import Path

    var p: Path
    with NamedTemporaryFile(mode="rw") as f:
        p = f.name
        f.write("Hello world!")
        _ = f.seek(0)
        assert_true(f.read() == "Hello world!")
    assert_false(p.exists())


fn test_file_close() raises:
    from tempfile import NamedTemporaryFile
    import os

    var temp_file = NamedTemporaryFile()  # delete=True by default
    temp_file.write("Temporary data")
    temp_file.close()  # File is deleted if delete=True
    assert_false(os.path.exists(temp_file.name))

    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write("Temporary data")
    temp_file.close()  # File is not deleted
    assert_true(os.path.exists(temp_file.name))
    os.remove(temp_file.name)  # Clean up manually


fn test_file_read() raises:
    from tempfile import NamedTemporaryFile
    from pathlib import Path

    var p: Path
    with NamedTemporaryFile(mode="rw") as f:
        p = f.name.copy()
        f.write("Sample data")
        _ = f.seek(0)
        content = f.read()
        assert_true(content == "Sample data")
    assert_false(p.exists())


fn test_read_bytes() raises:
    from tempfile import NamedTemporaryFile
    from pathlib import Path

    var p: Path
    var bytes = [Byte(0x48), 0x65, 0x6C, 0x6C, 0x6F]

    with NamedTemporaryFile(mode="rw") as f:
        p = f.name.copy()
        f.write_bytes(bytes[:])
        _ = f.seek(0)

        var b = f.read_bytes()
        assert_true(b == bytes)

        var s = String(from_utf8_lossy=b)
        assert_true(s == "Hello")

    assert_false(p.exists())


fn test_write() raises:
    from tempfile import NamedTemporaryFile
    from pathlib import Path

    with NamedTemporaryFile(mode="rw") as f:
        f.write("Hello world!")
        _ = f.seek(0)
        assert_true(f.read() == "Hello world!")


fn test_write_bytes() raises:
    from tempfile import NamedTemporaryFile

    var bytes = [Byte(0x48), 0x65, 0x6C, 0x6C, 0x6F]  # "Hello" in ASCII

    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(bytes[:])
        _ = f.seek(0)

        var b = f.read_bytes()
        assert_true(b == bytes)

        var s = String(from_utf8_lossy=b)
        assert_true(s == "Hello")


fn main() raises:
    test_gettempdir()
    test_mkdtemp()
    test_rmtree()
    test_temporary_directory_context_manager()
    test_named_temporary_file_context_manager()
    test_file_close()
    test_file_read()
    test_read_bytes()
    test_write()
    test_write_bytes()

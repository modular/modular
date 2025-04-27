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
# RUN: %mojo -debug-level full %s


from pathlib import Path, _dir_of_current_file
from sys import os_is_windows
from tempfile import gettempdir

from memory import UnsafePointer
from testing import assert_equal, assert_true


def test_file_read():
    var path = _dir_of_current_file() / "test_file_dummy_input.txt"
    with open(path, "r") as f:
        assert_true(
            f.read().startswith(
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            )
        )


def test_file_read_multi():
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        assert_equal(f.read(12), "Lorem ipsum ")
        assert_equal(f.read(6), "dolor ")
        assert_true(
            f.read().startswith("sit amet, consectetur adipiscing elit.")
        )


def test_file_read_bytes_multi():
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var bytes1 = f.read_bytes(12)
        assert_equal(len(bytes1), 12, "12 bytes")
        var string1 = String(bytes1)
        assert_equal(len(string1), 12, "12 chars")
        assert_equal(string1, String("Lorem ipsum "))

        var bytes2 = f.read_bytes(6)
        assert_equal(len(bytes2), 6, "6 bytes")
        var string2 = String(bytes2)
        assert_equal(len(string2), 6, "6 chars")
        assert_equal(string2, "dolor ")

        # Read where N is greater than the number of bytes in the file.
        var s: String = f.read(1_000_000_000)

        assert_equal(len(s), 936)
        assert_true(s.startswith("sit amet, consectetur adipiscing elit."))


def test_file_read_path():
    var file_path = _dir_of_current_file() / "test_file_dummy_input.txt"

    with open(file_path, "r") as f:
        assert_true(
            f.read().startswith(
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            )
        )


def test_file_path_direct_read():
    var file_path = _dir_of_current_file() / "test_file_dummy_input.txt"
    assert_true(
        file_path.read_text().startswith(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        )
    )


def test_file_read_context():
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        assert_true(
            f.read().startswith(
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            )
        )


def test_file_seek():
    import os

    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var pos = f.seek(6)
        assert_equal(pos, 6)

        alias expected_msg1 = "ipsum dolor sit amet, consectetur adipiscing elit."
        assert_equal(f.read(len(expected_msg1)), expected_msg1)

        # Seek from the end of the file
        pos = f.seek(-16, os.SEEK_END)
        assert_equal(pos, 938)

        _ = f.read(6)

        # Seek from current position, skip the space
        pos = f.seek(1, os.SEEK_CUR)
        assert_equal(pos, 945)
        assert_equal(f.read(7), "rhoncus")

        try:
            _ = f.seek(-12)
        except e:
            alias expected_msg = "seek error"
            assert_equal(String(e)[: len(expected_msg)], expected_msg)


def test_file_open_nodir():
    var f = open(Path("test_file_open_nodir"), "w")
    f.close()


def test_file_write():
    var content: String = "The quick brown fox jumps over the lazy dog"
    var TEMP_FILE = Path(gettempdir().value()) / "test_file_write"
    with open(TEMP_FILE, "w") as f:
        f.write(content)

    with open(TEMP_FILE, "r") as read_file:
        assert_equal(read_file.read(), content)


def test_file_write_span():
    var content: String = "The quick brown fox jumps over the lazy dog"
    var TEMP_FILE = Path(gettempdir().value()) / "test_file_write_span"
    with open(TEMP_FILE, "w") as f:
        f.write_bytes(content.as_bytes())

    with open(TEMP_FILE, "r") as read_file:
        assert_equal(read_file.read(), content)


def test_file_write_again():
    var unexpected_content: String = "foo bar baz"
    var expected_content: String = "foo bar"
    var TEMP_FILE = Path(gettempdir().value()) / "test_file_write_again"
    with open(TEMP_FILE, "w") as f:
        f.write(unexpected_content)

    with open(TEMP_FILE, "w") as f:
        f.write(expected_content)

    with open(TEMP_FILE, "r") as read_file:
        assert_equal(read_file.read(), expected_content)


def test_file_get_raw_fd():
    # since JIT and build give different file descriptors, we test by checking
    # if we printed to the right file.
    var f1 = open(Path(gettempdir().value()) / "test_file_dummy_1", "rw")
    var f2 = open(Path(gettempdir().value()) / "test_file_dummy_2", "rw")
    var f3 = open(Path(gettempdir().value()) / "test_file_dummy_3", "rw")

    print(
        "test from file 1",
        file=FileDescriptor(f1._get_raw_fd()),
        flush=True,
        end="",
    )
    _ = f1.seek(0)
    assert_equal(f1.read(), "test from file 1")
    assert_equal(f2.read(), "")
    assert_equal(f3.read(), "")

    _ = f1.seek(0)
    _ = f2.seek(0)
    _ = f3.seek(0)

    print(
        "test from file 2",
        file=FileDescriptor(f2._get_raw_fd()),
        flush=True,
        end="",
    )
    print(
        "test from file 3",
        file=FileDescriptor(f3._get_raw_fd()),
        flush=True,
        end="",
    )

    _ = f2.seek(0)
    _ = f3.seek(0)

    assert_equal(f3.read(), "test from file 3")
    assert_equal(f2.read(), "test from file 2")
    assert_equal(f1.read(), "test from file 1")

    f1.close()
    f2.close()
    f3.close()


def main():
    test_file_read()
    test_file_read_multi()
    test_file_read_bytes_multi()
    test_file_read_path()
    test_file_path_direct_read()
    test_file_read_context()
    test_file_seek()
    test_file_open_nodir()
    test_file_write()
    test_file_write_span()
    test_file_write_again()
    test_file_get_raw_fd()

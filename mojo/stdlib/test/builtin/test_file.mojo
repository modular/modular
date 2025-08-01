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

from pathlib import Path, _dir_of_current_file
from tempfile import gettempdir

from testing import assert_equal, assert_true

alias DUMMY_FILE_SIZE: UInt = 954


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
        var string1 = String(bytes=bytes1)
        assert_equal(len(string1), 12, "12 chars")
        assert_equal(string1, "Lorem ipsum ")

        var bytes2 = f.read_bytes(6)
        assert_equal(len(bytes2), 6, "6 bytes")
        var string2 = String(bytes=bytes2)
        assert_equal(len(string2), 6, "6 chars")
        assert_equal(string2, "dolor ")

        # Read where N is greater than the number of bytes in the file.
        var s: String = f.read(1_000_000_000)

        assert_equal(len(s), 936)
        assert_true(s.startswith("sit amet, consectetur adipiscing elit."))


def test_file_read_bytes_all():
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var bytes_all = f.read_bytes(-1)
        assert_equal(len(bytes_all), Int(DUMMY_FILE_SIZE))


def test_file_read_all():
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var all = f.read(-1)
        assert_equal(len(all), Int(DUMMY_FILE_SIZE))


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


def test_file_read_to_address():
    alias DUMMY_FILE_SIZE = 954
    # Test buffer size > file size
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer = InlineArray[UInt8, size=1000](fill=0)
        assert_equal(f.read(buffer), DUMMY_FILE_SIZE)
        assert_equal(buffer[0], 76)  # L
        assert_equal(buffer[1], 111)  # o
        assert_equal(buffer[2], 114)  # r
        assert_equal(buffer[3], 101)  # e
        assert_equal(buffer[4], 109)  # m
        assert_equal(buffer[5], 32)  # <space>
        assert_equal(buffer[56], 10)  # <LF>

    # Test buffer size < file size
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer = InlineArray[UInt8, size=500](fill=0)
        assert_equal(f.read(buffer), 500)

    # Test buffer size == file size
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer = InlineArray[UInt8, size=DUMMY_FILE_SIZE](fill=0)
        assert_equal(f.read(buffer), DUMMY_FILE_SIZE)

    # Test buffer size 0
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer = List[UInt8]()
        assert_equal(f.read(buffer), 0)

    # Test sequential reads of different sizes
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer_30 = InlineArray[UInt8, size=30](fill=0)
        var buffer_1 = InlineArray[UInt8, size=1](fill=0)
        var buffer_2 = InlineArray[UInt8, size=2](fill=0)
        var buffer_100 = InlineArray[UInt8, size=100](fill=0)
        var buffer_1000 = InlineArray[UInt8, size=1000](fill=0)
        assert_equal(f.read(buffer_30), 30)
        assert_equal(f.read(buffer_1), 1)
        assert_equal(f.read(buffer_2), 2)
        assert_equal(f.read(buffer_100), 100)
        assert_equal(f.read(buffer_1000), DUMMY_FILE_SIZE - (30 + 1 + 2 + 100))

    # Test read after EOF
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer_1000 = InlineArray[UInt8, size=1000](fill=0)
        assert_equal(f.read(buffer_1000), DUMMY_FILE_SIZE)
        assert_equal(f.read(buffer_1000), 0)


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
    test_file_read_bytes_all()
    test_file_read_all()
    test_file_read_path()
    test_file_path_direct_read()
    test_file_read_context()
    test_file_read_to_address()
    test_file_seek()
    test_file_open_nodir()
    test_file_write()
    test_file_write_span()
    test_file_write_again()
    test_file_get_raw_fd()

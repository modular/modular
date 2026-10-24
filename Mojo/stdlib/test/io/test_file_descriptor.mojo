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

from std.collections import Optional
from std.ffi import c_int, get_errno
from std.io import BorrowedFileDescriptor, FileDescriptor, RawFileDescriptor
from std.sys import size_of
from std.sys._io import stdin, stdout, stderr
from std.sys._libc import FcntlCommands, fcntl, pipe

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)


def test_value_is_c_int() raises:
    assert_true(type_of(stdout._value) == c_int)
    assert_equal(stdout._value, 1)
    assert_equal(stderr._value, 2)
    assert_equal(RawFileDescriptor()._value, 1)


def test_isatty_with_standard_descriptors() raises:
    assert_true(type_of(stdin.isatty()) == Bool)
    assert_true(type_of(stdout.isatty()) == Bool)
    assert_true(type_of(stderr.isatty()) == Bool)


def test_optional_uses_fd_niche() raises:
    comptime assert (
        size_of[Optional[RawFileDescriptor]]() == size_of[RawFileDescriptor]()
    )
    comptime assert size_of[RawFileDescriptor]() == size_of[c_int]()

    var fd: Optional[RawFileDescriptor] = RawFileDescriptor(7)
    assert_true(fd)
    assert_equal(fd[]._value, 7)

    fd = None
    assert_false(fd)

    fd = RawFileDescriptor(3)
    assert_true(fd)
    assert_equal(fd[]._value, 3)


def test_isatty_with_invalid_fd() raises:
    # A descriptor that is not open is not a tty.
    assert_false(RawFileDescriptor(9999).isatty())


def _make_pipe() raises -> Tuple[c_int, c_int]:
    var fds = [c_int(-1), c_int(-1)]
    if pipe(fds.unsafe_ptr()) != 0:
        raise Error("pipe failed err: ", String(get_errno()))
    return (fds[0], fds[1])


def _is_open(fd: c_int) -> Bool:
    return fcntl(fd, FcntlCommands.F_GETFD) != -1


def test_close_closes_fd() raises:
    var read_fd, write_fd = _make_pipe()
    var owned = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(write_fd))
    var was_open = _is_open(write_fd)
    owned^.close()
    assert_true(was_open)
    assert_false(_is_open(write_fd))
    # A second close fails with EBADF, so `close()` closed it exactly once.
    with assert_raises():
        RawFileDescriptor(write_fd).close()
    RawFileDescriptor(read_fd).close()


def test_move_only() raises:
    assert_true(conforms_to(FileDescriptor, Movable))
    assert_false(conforms_to(FileDescriptor, Copyable))


def test_optional_owned_fd_uses_niche() raises:
    comptime assert (
        size_of[Optional[FileDescriptor]]() == size_of[FileDescriptor]()
    )


def test_borrowed_does_not_close_owner() raises:
    var read_fd, write_fd = _make_pipe()
    var owner = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(write_fd))
    var view = BorrowedFileDescriptor(owner)
    var copy = view
    # Destroying the views leaves the descriptor open; only the owner closes.
    _ = view
    _ = copy
    var was_open = _is_open(write_fd)
    owner^.close()
    RawFileDescriptor(read_fd).close()
    assert_true(was_open)
    assert_false(_is_open(write_fd))


def test_optional_borrowed_fd_uses_niche() raises:
    comptime assert (
        size_of[Optional[BorrowedFileDescriptor[ImmStaticOrigin]]]()
        == size_of[BorrowedFileDescriptor[ImmStaticOrigin]]()
    )

    var read_fd, write_fd = _make_pipe()
    var owner = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(write_fd))
    var view: Optional[
        BorrowedFileDescriptor[origin_of(owner)]
    ] = BorrowedFileDescriptor(owner)
    var was_some = Bool(view)
    view = None
    var was_none = not view
    var still_open = _is_open(write_fd)
    owner^.close()
    RawFileDescriptor(read_fd).close()
    assert_true(was_some)
    assert_true(was_none)
    assert_true(still_open)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

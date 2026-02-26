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

from collections import List
from io import FileDescriptor
from os.process import Pipe
from sys import CompilationTarget
from sys._io import stdin, stdout, stderr

from ffi import c_int
from sys._libc import fcntl
from sys.intrinsics import _type_is_eq
from testing import TestSuite, assert_equal, assert_false, assert_true


def test_isatty_with_standard_descriptors():
    assert_true(_type_is_eq[type_of(stdin.isatty()), Bool]())
    assert_true(_type_is_eq[type_of(stdout.isatty()), Bool]())
    assert_true(_type_is_eq[type_of(stderr.isatty()), Bool]())


def test_isatty_with_invalid_fd():
    # Test with an invalid file descriptor
    # isatty should return False for invalid file descriptors
    assert_false(FileDescriptor(-1).isatty())

    # Test with a very large invalid file descriptor
    assert_false(FileDescriptor(9999).isatty())


def test_write_bytes_large_to_nonblock_pipe():
    # Test writing more than PIPE_BUF bytes to a non-blocking pipe to exercise
    # the partial-write retry loop in FileDescriptor.write_bytes().
    comptime F_GETFL: c_int = 3
    comptime F_SETFL: c_int = 4
    # O_NONBLOCK is 2048 on Linux, 4 on macOS.
    comptime O_NONBLOCK: c_int = 2048 if CompilationTarget.is_linux() else 4

    # PIPE_BUF is 4096 on Linux, 512 on macOS. Write 2x to exceed it.
    comptime WRITE_SIZE = 8192

    var p = Pipe()

    # Set O_NONBLOCK on the write end of the pipe.
    var write_fd = c_int(p.fd_out.value().value)
    var flags = fcntl(write_fd, F_GETFL, c_int(0))
    _ = fcntl(write_fd, F_SETFL, flags | O_NONBLOCK)

    # Build a deterministic payload larger than PIPE_BUF.
    var data = List[Byte](capacity=WRITE_SIZE)
    for i in range(WRITE_SIZE):
        data.append(Byte(i & 0xFF))

    # Write all bytes through FileDescriptor.write_bytes() via Pipe.
    p.write_bytes(Span(data))

    # Read all bytes back from the read end.
    var buf = List[Byte](capacity=WRITE_SIZE)
    buf.resize(WRITE_SIZE, 0)
    var total_read = 0
    while total_read < WRITE_SIZE:
        var n = p.read_bytes(Span(buf)[total_read:])
        total_read += Int(n)

    assert_equal(total_read, WRITE_SIZE)
    for i in range(WRITE_SIZE):
        assert_equal(buf[i], data[i])


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

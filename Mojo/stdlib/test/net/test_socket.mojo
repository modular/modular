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

from std.ffi import c_int
from std.sys._libc import FcntlCommands, FcntlFDFlags, fcntl
from std.testing import TestSuite, assert_equal, assert_not_equal, assert_true

from std.net._socket import Socket
from std.net._sys.unix import (
    AF_INET,
    SOCK_DGRAM,
    SOCK_STREAM,
    SOL_SOCKET,
    SO_REUSEADDR,
    SO_TYPE,
)


def test_close_on_exec() raises:
    var sock = Socket(domain=AF_INET, type=SOCK_STREAM)
    var flags = fcntl(sock.unsafe_fd(), FcntlCommands.F_GETFD)
    sock^.close()
    assert_not_equal(flags, -1)
    assert_true((flags & FcntlFDFlags.FD_CLOEXEC) != 0)


def test_option_round_trip() raises:
    var sock = Socket(domain=AF_INET, type=SOCK_STREAM)
    var before: c_int
    var after: c_int
    try:
        before = sock.option(SOL_SOCKET, SO_REUSEADDR)
        sock.set_option(SOL_SOCKET, SO_REUSEADDR, 1)
        after = sock.option(SOL_SOCKET, SO_REUSEADDR)
    finally:
        sock^.close()
    assert_equal(before, 0)
    # macOS reports a set boolean option as its flag value, not 1.
    assert_not_equal(after, 0)


def socket_type(type: c_int) raises -> c_int:
    var sock = Socket(domain=AF_INET, type=type)
    try:
        return sock.option(SOL_SOCKET, SO_TYPE)
    finally:
        sock^.close()


def test_socket_type() raises:
    assert_equal(socket_type(SOCK_STREAM), SOCK_STREAM)
    assert_equal(socket_type(SOCK_DGRAM), SOCK_DGRAM)


def test_invalid_domain_raises() raises:
    var message = String()
    try:
        var sock = Socket(domain=c_int(-1), type=SOCK_STREAM)
        sock^.close()
    except e:
        message = String(e)
    assert_true(message.startswith("socket: "))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

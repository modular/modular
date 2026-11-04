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
from std.io import IOError, IOErrorKind
from std.sys._libc import FcntlCommands, FcntlFDFlags, fcntl
from std.testing import TestSuite, assert_equal, assert_not_equal, assert_true

from std.net._socket import Socket
from std.net.addr import SocketAddr, SocketAddrV4
from std.net._sys.unix import (
    AF_INET,
    AF_UNIX,
    MSG_PEEK,
    SOCK_DGRAM,
    SOCK_STREAM,
    SOL_SOCKET,
    SO_REUSEADDR,
    SO_TYPE,
)


@fieldwise_init
struct AutoClose:
    var socket: Socket

    def __deinit__(deinit self):
        try:
            self.socket^.close()
        except:
            pass


def test_close_on_exec() raises:
    var sock = Socket(domain=AF_INET, type=SOCK_STREAM)
    var flags = fcntl(sock.unsafe_fd(), FcntlCommands.F_GETFD)
    sock^.close()
    assert_not_equal(flags, -1)
    assert_true((flags & FcntlFDFlags.FD_CLOEXEC) != 0)


def with_connected_pair[F: def(Socket, Socket) raises](body: F) raises:
    var listener = AutoClose(Socket(domain=AF_INET, type=SOCK_STREAM))
    listener.socket.bind(.v4(SocketAddrV4(.localhost, 0)))
    listener.socket.listen(backlog=1)
    var client = AutoClose(Socket(domain=AF_INET, type=SOCK_STREAM))
    client.socket.connect(listener.socket.local_addr())
    match listener.socket.accept():
        case (var accepted, _):
            var server = AutoClose(accepted^)
            body(client.socket, server.socket)


def test_accept_returns_peer_addr() raises:
    var listener = AutoClose(Socket(domain=AF_INET, type=SOCK_STREAM))
    listener.socket.bind(.v4(SocketAddrV4(.localhost, 0)))
    listener.socket.listen(backlog=1)
    var client = AutoClose(Socket(domain=AF_INET, type=SOCK_STREAM))
    client.socket.connect(listener.socket.local_addr())
    match listener.socket.accept():
        case (var accepted, var peer):
            var server = AutoClose(accepted^)
            assert_equal(peer, client.socket.local_addr())
            assert_equal(peer, server.socket.peer_addr())


def recv_string(sock: Socket, flags: c_int = 0) raises IOError -> String:
    var buffer = List[Byte](length=16, fill=0)
    var count = sock.recv(buffer, flags=flags)
    return String(from_utf8_lossy=buffer[:count])


def test_accepted_socket_close_on_exec() raises:
    def body(client: Socket, server: Socket) raises:
        var flags = fcntl(server.unsafe_fd(), FcntlCommands.F_GETFD)
        assert_not_equal(flags, -1)
        assert_true((flags & FcntlFDFlags.FD_CLOEXEC) != 0)

    with_connected_pair(body)


def test_recv_peek() raises:
    def body(client: Socket, server: Socket) raises:
        _ = client.send("ping".as_bytes())
        assert_equal(recv_string(server, MSG_PEEK), "ping")
        assert_equal(recv_string(server), "ping")

    with_connected_pair(body)


def test_non_ip_address_raises() raises:
    var sock = AutoClose(Socket(domain=AF_UNIX, type=SOCK_STREAM))
    var kind = IOErrorKind.uncategorized
    try:
        _ = sock.socket.local_addr()
    except e:
        kind = e.kind()
    assert_equal(kind, .invalid_input)


def test_option_round_trip() raises:
    var sock = AutoClose(Socket(domain=AF_INET, type=SOCK_STREAM))
    var before = sock.socket.option(level=SOL_SOCKET, name=SO_REUSEADDR)
    sock.socket.set_option(level=SOL_SOCKET, name=SO_REUSEADDR, value=1)
    var after = sock.socket.option(level=SOL_SOCKET, name=SO_REUSEADDR)
    assert_equal(before, 0)
    # macOS reports a set boolean option as its flag value, not 1.
    assert_not_equal(after, 0)


def socket_type(type: c_int) raises -> c_int:
    var sock = AutoClose(Socket(domain=AF_INET, type=type))
    return sock.socket.option(level=SOL_SOCKET, name=SO_TYPE)


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

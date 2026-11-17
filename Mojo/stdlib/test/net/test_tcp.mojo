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

from std.io import IOError
from std.net import IpAddr, SocketAddr, SocketAddrV4, SocketAddrV6
from std.net.tcp import TCPListener, TCPSocket, TCPStream
from std.net.tcp.option import KeepAlive, NoDelay, ReuseAddress, ReusePort
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)


def or_raise[
    T: Movable & Deinitable
](var o: Optional[T], error: Error) raises -> T:
    # TODO(https://github.com/modular/modular/issues/7418): `match o^` with
    # `case .Some(var value)` destroys the value it moves out.
    if not o:
        raise error
    return o.take()


def loopback_addr() -> SocketAddr:
    return SocketAddr.v4(SocketAddrV4(.localhost, 0))


def ipv6_loopback_addr() -> SocketAddr:
    return SocketAddr.v6(SocketAddrV6(.localhost, 0))


def read_string(stream: TCPStream, count: Int) raises -> String:
    var buffer = List[Byte](length=count, fill=0)
    var n = 0
    while n < count:
        var got = stream.read(Span(buffer)[n:])
        if got == 0:
            break
        n += got
    return String(from_utf8_lossy=Span(buffer)[:n])


def test_socket_options_round_trip() raises:
    var socket = TCPSocket.v4()

    assert_false(socket.get_option[ReuseAddress]())
    socket.set_option[ReuseAddress](True)
    assert_true(socket.get_option[ReuseAddress]())
    socket.set_option[ReuseAddress](False)
    assert_false(socket.get_option[ReuseAddress]())

    assert_false(socket.get_option[ReusePort]())
    socket.set_option[ReusePort](True)
    assert_true(socket.get_option[ReusePort]())
    socket.set_option[ReusePort](False)
    assert_false(socket.get_option[ReusePort]())

    assert_false(socket.get_option[NoDelay]())
    socket.set_option[NoDelay](True)
    assert_true(socket.get_option[NoDelay]())
    socket.set_option[NoDelay](False)
    assert_false(socket.get_option[NoDelay]())

    assert_false(socket.get_option[KeepAlive]())
    socket.set_option[KeepAlive](True)
    assert_true(socket.get_option[KeepAlive]())
    socket.set_option[KeepAlive](False)
    assert_false(socket.get_option[KeepAlive]())


def test_ipv6_socket_options() raises:
    var socket = TCPSocket.v6()
    assert_false(socket.get_option[NoDelay]())
    socket.set_option[NoDelay](True)
    assert_true(socket.get_option[NoDelay]())
    socket.set_option[NoDelay](False)
    assert_false(socket.get_option[NoDelay]())


def test_listener_bind_picks_a_port() raises:
    var addr = TCPListener(bind=loopback_addr()).local_addr()
    assert_equal(addr.ip(), IpAddr.v4(.localhost))
    assert_not_equal(addr.port(), 0)


def test_socket_bind_picks_a_port() raises:
    var socket = TCPSocket.v4()
    socket.bind(loopback_addr())
    var addr = socket^.listen().local_addr()
    assert_equal(addr.ip(), IpAddr.v4(.localhost))
    assert_not_equal(addr.port(), 0)


def test_ipv6_bind_picks_a_port() raises:
    var addr = TCPListener(bind=ipv6_loopback_addr()).local_addr()
    assert_equal(addr.ip(), IpAddr.v6(.localhost))
    assert_not_equal(addr.port(), 0)

    var socket = TCPSocket.v6()
    socket.bind(ipv6_loopback_addr())
    addr = socket^.listen().local_addr()
    assert_equal(addr.ip(), IpAddr.v6(.localhost))
    assert_not_equal(addr.port(), 0)


def test_only_listener_bind_turns_on_reuse_address() raises:
    var listener = TCPListener(bind=loopback_addr())
    assert_true(listener.get_option[ReuseAddress]())
    var socket = TCPSocket.v4()
    socket.bind(loopback_addr())
    assert_false(socket^.listen().get_option[ReuseAddress]())


def test_listen_keeps_socket_options() raises:
    var socket = TCPSocket.v4()
    socket.set_option[ReuseAddress](True)
    socket.bind(loopback_addr())
    assert_true(socket^.listen().get_option[ReuseAddress]())


def test_bind_address_in_use_raises() raises:
    var listener = TCPListener(bind=loopback_addr())
    var addr = listener.local_addr()
    var error: Optional[IOError] = None
    try:
        _ = TCPListener(bind=addr)
    except e:
        error = e^
    # Keeps the first listener open until the second bind has failed.
    _ = listener^
    var e = or_raise(error^, "the second bind succeeded")
    assert_equal(e.kind(), .address_in_use)


def test_bind_wrong_version_raises() raises:
    # The error's kind varies by OS and sandbox.
    with assert_raises():
        TCPSocket.v4().bind(ipv6_loopback_addr())
    with assert_raises():
        TCPSocket.v6().bind(loopback_addr())


def test_listener_bind_reuses_an_address_connections_hold() raises:
    var listener = TCPListener(bind=loopback_addr())
    var addr = listener.local_addr()
    var client = TCPStream(connect=addr)
    var server, _ = listener.accept()
    listener^.close()

    # `server` still holds `addr`, so binding it needs `ReuseAddress`, which
    # only `TCPListener(bind=)` turns on.
    var error: Optional[IOError] = None
    try:
        TCPSocket.v4().bind(addr)
    except e:
        error = e^
    var e = or_raise(error^, "the bind without `ReuseAddress` succeeded")
    assert_equal(e.kind(), .address_in_use)
    assert_equal(TCPListener(bind=addr).local_addr(), addr)

    _ = server^
    _ = client^


def test_connect_and_accept() raises:
    var listener = TCPListener(bind=loopback_addr())
    var client = TCPStream(connect=listener.local_addr())
    var server, peer = listener.accept()
    assert_equal(peer, client.local_addr())
    assert_equal(server.peer_addr(), client.local_addr())
    assert_equal(client.peer_addr(), listener.local_addr())
    assert_equal(server.local_addr(), listener.local_addr())


def test_bind_then_connect() raises:
    var listener = TCPListener(bind=loopback_addr())
    var socket = TCPSocket.v4()
    socket.bind(loopback_addr())
    # This checks only that `connect()` works after `bind()`. `TCPSocket` has
    # no `local_addr()`, so the test can't compare the bound address.
    var client = socket^.connect(listener.local_addr())
    assert_equal(listener.accept()[1], client.local_addr())


def test_connect_keeps_socket_options() raises:
    var listener = TCPListener(bind=loopback_addr())
    var socket = TCPSocket.v4()
    socket.set_option[NoDelay](True)
    assert_true(socket^.connect(listener.local_addr()).get_option[NoDelay]())
    _ = listener^


def test_ipv6_connect_and_accept() raises:
    var listener = TCPListener(bind=ipv6_loopback_addr())
    var client = TCPStream(connect=listener.local_addr())
    assert_equal(client.peer_addr(), listener.local_addr())
    assert_equal(listener.accept()[1], client.local_addr())


def test_write_then_read() raises:
    var listener = TCPListener(bind=loopback_addr())
    var client = TCPStream(connect=listener.local_addr())
    var server, _ = listener.accept()
    client.write_all("ping".as_bytes())
    assert_equal(read_string(server, 4), "ping")
    # The send buffer is empty, so a write this small isn't cut short.
    assert_equal(server.write("pong".as_bytes()), 4)
    assert_equal(read_string(client, 4), "pong")


def test_read_after_peer_closes_returns_zero() raises:
    var listener = TCPListener(bind=loopback_addr())
    var client = TCPStream(connect=listener.local_addr())
    var server, _ = listener.accept()
    server^.close()
    var buffer = List[Byte](length=4, fill=0)
    assert_equal(client.read(buffer), 0)
    assert_equal(client.read(buffer), 0)


def test_read_after_peer_resets_raises() raises:
    var listener = TCPListener(bind=loopback_addr())
    var client = TCPStream(connect=listener.local_addr())
    var server, _ = listener.accept()
    client.write_all("ab".as_bytes())
    # Reading one of the two bytes waits for them to arrive. Closing with the
    # other one unread resets the connection instead of closing it cleanly.
    var buffer = List[Byte](length=1, fill=0)
    assert_equal(server.read(buffer), 1)
    server^.close()
    var error: Optional[IOError] = None
    try:
        _ = client.read(buffer)
    except e:
        error = e^
    var e = or_raise(error^, "the read after the reset succeeded")
    assert_equal(e.kind(), .connection_reset)


def test_destroying_a_stream_closes_it() raises:
    var listener = TCPListener(bind=loopback_addr())
    var client = TCPStream(connect=listener.local_addr())
    _ = listener.accept()
    var buffer = List[Byte](length=4, fill=0)
    assert_equal(client.read(buffer), 0)


def test_connect_to_closed_port_raises() raises:
    # The listener is destroyed, and so closed, right after `local_addr()`,
    # its last use, which leaves a port that nothing listens on.
    var addr = TCPListener(bind=loopback_addr()).local_addr()
    var error: Optional[IOError] = None
    try:
        _ = TCPStream(connect=addr)
    except e:
        error = e^
    var e = or_raise(error^, "the connect succeeded")
    assert_equal(e.kind(), .connection_refused)


def test_connect_wrong_version_raises() raises:
    # A listener at the address, so the connect can only fail on the version.
    # The error's kind varies by OS and sandbox.
    var listener = TCPListener(bind=loopback_addr())
    with assert_raises():
        _ = TCPSocket.v6().connect(listener.local_addr())
    _ = listener^


def test_stream_options_round_trip() raises:
    var listener = TCPListener(bind=loopback_addr())
    var stream = TCPStream(connect=listener.local_addr())
    assert_false(stream.get_option[NoDelay]())
    stream.set_option[NoDelay](True)
    assert_true(stream.get_option[NoDelay]())
    stream.set_option[NoDelay](False)
    assert_false(stream.get_option[NoDelay]())
    _ = listener^


def test_listener_options_round_trip() raises:
    var listener = TCPListener(bind=loopback_addr())
    assert_false(listener.get_option[KeepAlive]())
    listener.set_option[KeepAlive](True)
    assert_true(listener.get_option[KeepAlive]())
    listener.set_option[KeepAlive](False)
    assert_false(listener.get_option[KeepAlive]())


def test_explicit_close() raises:
    TCPSocket.v4().close()
    TCPListener(bind=loopback_addr()).close()


def has_ipv6_loopback() raises -> Bool:
    try:
        _ = TCPListener(bind=ipv6_loopback_addr())
        return True
    except e:
        if e.kind() == .address_not_available:
            return False
        raise e


def main() raises:
    # Some CI containers have no IPv6 loopback address.
    var ipv6 = has_ipv6_loopback()
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    if not ipv6:
        suite.skip[test_ipv6_bind_picks_a_port]()
        suite.skip[test_ipv6_connect_and_accept]()
    suite^.run()

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
from std.net.tcp import TCPListener, TCPSocket
from std.net.tcp.option import KeepAlive, NoDelay, ReuseAddress, ReusePort
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)


def loopback_addr() -> SocketAddr:
    return SocketAddr.v4(SocketAddrV4(.localhost, 0))


def ipv6_loopback_addr() -> SocketAddr:
    return SocketAddr.v6(SocketAddrV6(.localhost, 0))


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
    match error:
        case .Some(e):
            assert_equal(e.kind(), .address_in_use)
        case .None:
            raise Error("the second bind succeeded")


def test_bind_wrong_version_raises() raises:
    # The error's kind varies by OS and sandbox.
    with assert_raises():
        TCPSocket.v4().bind(ipv6_loopback_addr())
    with assert_raises():
        TCPSocket.v6().bind(loopback_addr())


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
    suite^.run()

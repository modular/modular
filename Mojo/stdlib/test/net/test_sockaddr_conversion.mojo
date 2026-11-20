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

from std.testing import TestSuite, assert_equal

from std.net._sys.socket_addr import (
    from_sockaddr_in,
    from_sockaddr_in6,
    to_sockaddr_in,
    to_sockaddr_in6,
)
from std.net.addr import Ipv4Addr, Ipv6Addr, SocketAddrV4, SocketAddrV6


def test_sockaddr_in_network_order() raises:
    var addr = to_sockaddr_in(SocketAddrV4(Ipv4Addr(127, 0, 0, 1), 8080))
    assert_equal(UInt16.from_bytes[endian=.big](addr.port().as_bytes()), 8080)
    assert_equal(
        UInt32.from_bytes[endian=.big](addr.addr().s_addr.as_bytes()),
        0x7F000001,
    )


def test_sockaddr_in_round_trip() raises:
    for addr in [
        SocketAddrV4(Ipv4Addr.localhost, 8080),
        SocketAddrV4(Ipv4Addr.unspecified, 0),
        SocketAddrV4(Ipv4Addr(192, 168, 1, 200), 65535),
    ]:
        assert_equal(from_sockaddr_in(to_sockaddr_in(addr)), addr)


def test_sockaddr_in6_network_order() raises:
    var addr = to_sockaddr_in6(
        SocketAddrV6(
            Ipv6Addr(0x2001, 0xDB8, 0, 0, 0, 0, 0, 1),
            8080,
            flowinfo=7,
            scope_id=3,
        )
    )
    assert_equal(UInt16.from_bytes[endian=.big](addr.port().as_bytes()), 8080)
    assert_equal(
        addr.addr().s6_addr,
        [0x20, 0x01, 0x0D, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01],
    )
    assert_equal(addr.flowinfo(), 7)
    assert_equal(addr.scope_id(), 3)


def test_sockaddr_in6_round_trip() raises:
    for addr in [
        SocketAddrV6(Ipv6Addr.localhost, 8080),
        SocketAddrV6(Ipv6Addr.unspecified, 0),
        SocketAddrV6(
            Ipv6Addr(0xFE80, 0, 0, 0, 0, 0, 0, 1),
            65535,
            flowinfo=0x12345,
            scope_id=2,
        ),
    ]:
        assert_equal(from_sockaddr_in6(to_sockaddr_in6(addr)), addr)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

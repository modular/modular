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

from std.testing import TestSuite, assert_equal, assert_not_equal

from std.net.addr import (
    IpAddr,
    Ipv4Addr,
    Ipv6Addr,
    SocketAddr,
    SocketAddrV4,
    SocketAddrV6,
)


def test_ipv4_initializers_agree() raises:
    var addr = Ipv4Addr(octets=[0x12, 0x34, 0x56, 0x78])
    assert_equal(Ipv4Addr(0x12, 0x34, 0x56, 0x78), addr)
    assert_equal(Ipv4Addr(bits=0x12345678), addr)
    assert_not_equal(Ipv4Addr(bits=0x78563412), addr)


def test_ipv4_bits() raises:
    assert_equal(Ipv4Addr(0x12, 0x34, 0x56, 0x78).bits(), 0x12345678)
    assert_equal(Ipv4Addr(bits=0xC0A80001).bits(), 0xC0A80001)


def test_ipv4_write_to() raises:
    assert_equal(String(Ipv4Addr(192, 168, 0, 1)), "192.168.0.1")
    assert_equal(String(Ipv4Addr.unspecified), "0.0.0.0")
    assert_equal(String(Ipv4Addr.broadcast), "255.255.255.255")


def test_ipv4_constants() raises:
    assert_equal(Ipv4Addr.localhost, Ipv4Addr(bits=0x7F000001))
    assert_equal(Ipv4Addr.broadcast, Ipv4Addr(bits=0xFFFFFFFF))
    assert_equal(Ipv4Addr.unspecified, Ipv4Addr(bits=0))


def test_ipv6_initializers_agree() raises:
    var addr = Ipv6Addr(
        octets=[
            0x10,
            0x20,
            0x30,
            0x40,
            0x50,
            0x60,
            0x70,
            0x80,
            0x90,
            0xA0,
            0xB0,
            0xC0,
            0xD0,
            0xE0,
            0xF0,
            0x0D,
        ]
    )
    assert_equal(
        Ipv6Addr(
            0x1020, 0x3040, 0x5060, 0x7080, 0x90A0, 0xB0C0, 0xD0E0, 0xF00D
        ),
        addr,
    )
    assert_equal(
        Ipv6Addr(
            segments=[
                0x1020,
                0x3040,
                0x5060,
                0x7080,
                0x90A0,
                0xB0C0,
                0xD0E0,
                0xF00D,
            ]
        ),
        addr,
    )
    assert_equal(Ipv6Addr(bits=0x102030405060708090A0B0C0D0E0F00D), addr)
    assert_not_equal(Ipv6Addr(bits=0x0DF0E0D0C0B0A0908070605040302010), addr)


def test_ipv6_bits() raises:
    assert_equal(
        Ipv6Addr(
            0x1020, 0x3040, 0x5060, 0x7080, 0x90A0, 0xB0C0, 0xD0E0, 0xF00D
        ).bits(),
        0x102030405060708090A0B0C0D0E0F00D,
    )
    assert_equal(Ipv6Addr.localhost.bits(), 1)


def test_ipv6_write_to() raises:
    assert_equal(String(Ipv6Addr.unspecified), "::")
    assert_equal(String(Ipv6Addr.localhost), "::1")
    assert_equal(String(Ipv6Addr(1, 2, 3, 4, 5, 6, 7, 8)), "1:2:3:4:5:6:7:8")
    assert_equal(
        String(Ipv6Addr(0xFE80, 0, 0, 0, 0, 0, 0, 0xABCD)), "fe80::abcd"
    )
    assert_equal(
        String(Ipv6Addr(0x2001, 0xDB8, 0, 0, 0, 0, 0, 1)), "2001:db8::1"
    )
    assert_equal(
        String(Ipv6Addr(0x2001, 0xDB8, 0, 0, 0, 0, 0, 0)), "2001:db8::"
    )
    # RFC 5952 leaves a lone zero segment uncompressed.
    assert_equal(
        String(Ipv6Addr(0x2001, 0xDB8, 0, 1, 1, 1, 1, 1)),
        "2001:db8:0:1:1:1:1:1",
    )
    # The longest zero run is compressed, and the first one on a tie.
    assert_equal(String(Ipv6Addr(0x2001, 0, 0, 1, 0, 0, 0, 1)), "2001:0:0:1::1")
    assert_equal(
        String(Ipv6Addr(0x2001, 0xDB8, 0, 0, 1, 0, 0, 1)), "2001:db8::1:0:0:1"
    )
    assert_equal(
        String(Ipv6Addr(0, 0, 0, 0, 0, 0xFFFF, 0xC000, 0x0201)),
        "::ffff:192.0.2.1",
    )


def test_ipv6_constants() raises:
    assert_equal(Ipv6Addr.localhost, Ipv6Addr(bits=1))
    assert_equal(Ipv6Addr.unspecified, Ipv6Addr(bits=0))


def test_ipaddr_equality() raises:
    assert_equal(IpAddr.v4(.localhost), IpAddr.v4(Ipv4Addr(127, 0, 0, 1)))
    assert_not_equal(IpAddr.v4(.localhost), IpAddr.v4(.broadcast))
    assert_equal(IpAddr.v6(.localhost), IpAddr.v6(Ipv6Addr(bits=1)))
    assert_not_equal(IpAddr.v6(.localhost), IpAddr.v6(.unspecified))
    # Different families never compare equal, even for an IPv4-mapped address.
    assert_not_equal(
        IpAddr.v4(.localhost),
        IpAddr.v6(Ipv6Addr(0, 0, 0, 0, 0, 0xFFFF, 0x7F00, 1)),
    )


def test_ipaddr_hash() raises:
    assert_equal(hash(Ipv4Addr(127, 0, 0, 1)), hash(Ipv4Addr.localhost))
    assert_not_equal(hash(Ipv4Addr.localhost), hash(Ipv4Addr.broadcast))
    assert_equal(hash(Ipv6Addr(bits=1)), hash(Ipv6Addr.localhost))
    assert_not_equal(hash(Ipv6Addr.localhost), hash(Ipv6Addr.unspecified))
    assert_equal(
        hash(IpAddr.v4(Ipv4Addr(127, 0, 0, 1))), hash(IpAddr.v4(.localhost))
    )
    assert_equal(hash(IpAddr.v6(Ipv6Addr(bits=1))), hash(IpAddr.v6(.localhost)))
    assert_not_equal(hash(IpAddr.v4(.localhost)), hash(IpAddr.v6(.localhost)))


def test_ipaddr_write_to() raises:
    assert_equal(String(IpAddr.v4(.localhost)), "127.0.0.1")
    assert_equal(String(IpAddr.v6(.localhost)), "::1")


def test_socket_addr_v4() raises:
    var addr = SocketAddrV4(Ipv4Addr(192, 168, 0, 1), 8080)
    assert_equal(addr.ip(), Ipv4Addr(192, 168, 0, 1))
    assert_equal(addr.port(), 8080)
    assert_equal(addr, SocketAddrV4(ip=Ipv4Addr(192, 168, 0, 1), port=8080))
    assert_not_equal(addr, SocketAddrV4(ip=addr.ip(), port=8081))


def test_socket_addr_v6() raises:
    var addr = SocketAddrV6(.localhost, 8080, flowinfo=7, scope_id=3)
    assert_equal(addr.ip(), .localhost)
    assert_equal(addr.port(), 8080)
    assert_equal(addr.flowinfo(), 7)
    assert_equal(addr.scope_id(), 3)
    assert_equal(SocketAddrV6(.localhost, 8080).flowinfo(), 0)
    assert_equal(SocketAddrV6(.localhost, 8080).scope_id(), 0)
    assert_not_equal(
        SocketAddrV6(.localhost, 8080, flowinfo=1),
        SocketAddrV6(.localhost, 8080),
    )
    assert_not_equal(
        SocketAddrV6(.localhost, 8080, scope_id=1),
        SocketAddrV6(.localhost, 8080),
    )


def test_socket_addr_v4_write_to() raises:
    assert_equal(String(SocketAddrV4(.localhost, 8080)), "127.0.0.1:8080")


def test_socket_addr_v6_write_to() raises:
    assert_equal(String(SocketAddrV6(.localhost, 8080)), "[::1]:8080")
    assert_equal(
        String(
            SocketAddrV6(
                Ipv6Addr(0xFE80, 0, 0, 0, 0, 0, 0, 1), 8080, scope_id=2
            )
        ),
        "[fe80::1%2]:8080",
    )
    assert_equal(
        String(SocketAddrV6(.localhost, 8080, flowinfo=7)),
        "[::1]:8080",
    )


def test_socket_addr_from_ip() raises:
    var v4 = SocketAddr(.v4(.localhost), 8080)
    assert_equal(v4, .v4(SocketAddrV4(.localhost, 8080)))
    assert_equal(v4.ip(), .v4(.localhost))
    assert_equal(v4.port(), 8080)

    var v6 = SocketAddr(.v6(.localhost), 8080)
    assert_equal(v6, .v6(SocketAddrV6(.localhost, 8080)))
    assert_equal(v6.ip(), .v6(.localhost))
    assert_equal(v6.port(), 8080)


def test_socket_addr_equality() raises:
    assert_not_equal(
        SocketAddr(.v4(.localhost), 80),
        SocketAddr(.v4(.localhost), 81),
    )
    assert_not_equal(
        SocketAddr(.v4(.unspecified), 80),
        SocketAddr(.v6(.unspecified), 80),
    )


def test_socket_addr_hash() raises:
    assert_equal(
        hash(SocketAddrV4(Ipv4Addr(127, 0, 0, 1), 80)),
        hash(SocketAddrV4(.localhost, 80)),
    )
    assert_not_equal(
        hash(SocketAddrV4(.localhost, 80)), hash(SocketAddrV4(.localhost, 81))
    )
    assert_equal(
        hash(SocketAddrV6(Ipv6Addr(bits=1), 80, flowinfo=7, scope_id=3)),
        hash(SocketAddrV6(.localhost, 80, flowinfo=7, scope_id=3)),
    )
    assert_not_equal(
        hash(SocketAddrV6(.localhost, 80, flowinfo=1)),
        hash(SocketAddrV6(.localhost, 80)),
    )
    assert_not_equal(
        hash(SocketAddrV6(.localhost, 80, scope_id=1)),
        hash(SocketAddrV6(.localhost, 80)),
    )
    assert_equal(
        hash(SocketAddr(.v4(.localhost), 80)),
        hash(SocketAddr.v4(SocketAddrV4(.localhost, 80))),
    )
    assert_not_equal(
        hash(SocketAddr(.v4(.unspecified), 80)),
        hash(SocketAddr(.v6(.unspecified), 80)),
    )


def test_socket_addr_write_to() raises:
    assert_equal(String(SocketAddr(.v4(.localhost), 80)), "127.0.0.1:80")
    assert_equal(String(SocketAddr(.v6(.localhost), 80)), "[::1]:80")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

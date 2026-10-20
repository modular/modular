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

from std.net.ipaddr import IpAddr, Ipv4Addr, Ipv6Addr


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
    assert_equal(
        IpAddr.v4(Ipv4Addr.localhost), IpAddr.v4(Ipv4Addr(127, 0, 0, 1))
    )
    assert_not_equal(
        IpAddr.v4(Ipv4Addr.localhost), IpAddr.v4(Ipv4Addr.broadcast)
    )
    assert_equal(IpAddr.v6(Ipv6Addr.localhost), IpAddr.v6(Ipv6Addr(bits=1)))
    assert_not_equal(
        IpAddr.v6(Ipv6Addr.localhost), IpAddr.v6(Ipv6Addr.unspecified)
    )
    # Different families never compare equal, even for an IPv4-mapped address.
    assert_not_equal(
        IpAddr.v4(Ipv4Addr.localhost),
        IpAddr.v6(Ipv6Addr(0, 0, 0, 0, 0, 0xFFFF, 0x7F00, 1)),
    )


def test_ipaddr_write_to() raises:
    assert_equal(String(IpAddr.v4(Ipv4Addr.localhost)), "127.0.0.1")
    assert_equal(String(IpAddr.v6(Ipv6Addr.localhost)), "::1")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

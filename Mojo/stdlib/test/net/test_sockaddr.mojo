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

from std.net._sys.inet import in6_addr, in_addr
from std.net._sys.unix import sockaddr_in, sockaddr_in6


def ipv6_loopback() -> in6_addr:
    var bytes = Array[UInt8, 16](fill=0)
    bytes[15] = 1
    return in6_addr(bytes^)


def test_sockaddr_in() raises:
    var addr = sockaddr_in(port=8080, addr=in_addr(0x0100007F))
    assert_equal(addr.port(), 8080)
    assert_equal(addr.addr().s_addr, 0x0100007F)


def test_sockaddr_in6() raises:
    var addr = sockaddr_in6(
        port=8080, addr=ipv6_loopback(), flowinfo=7, scope_id=3
    )
    assert_equal(addr.port(), 8080)
    assert_equal(addr.flowinfo(), 7)
    assert_equal(addr.scope_id(), 3)
    assert_equal(addr.addr().s6_addr, ipv6_loopback().s6_addr)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

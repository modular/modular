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
"""C ABI manifest: the `std.net._sys` mirrors shared by Linux and macOS.

Struct mirrors carry their C names verbatim, so their type list needs no
name column; type aliases carry their C name in an `AbiTypedef` row,
because reflection sees through an alias to the underlying type.
"""

from test_utils.cabi_check import AbiConstant, AbiTypedef, AbiTypedefLike

from std.net._sys.inet import (
    INADDR_ANY,
    INADDR_LOOPBACK,
    IPPROTO_IP,
    IPPROTO_IPV6,
    IPPROTO_TCP,
    IPPROTO_UDP,
    in6_addr,
    in_addr,
    in_port_t,
)

comptime CABI_INCLUDES: List[StaticString] = ["<netinet/in.h>"]
"""The headers that declare the C side."""

comptime CABI_STRUCTS = TypeList.of[Trait=AnyType, in_addr, in6_addr]()
"""The struct mirrors to check."""

comptime CABI_TYPEDEFS = TypeList.of[
    Trait=AbiTypedefLike,
    AbiTypedef["in_port_t", in_port_t],
]()
"""The type aliases to check."""

comptime CABI_CONSTANTS: List[AbiConstant] = [
    AbiConstant("IPPROTO_IP", IPPROTO_IP),
    AbiConstant("IPPROTO_TCP", IPPROTO_TCP),
    AbiConstant("IPPROTO_UDP", IPPROTO_UDP),
    AbiConstant("IPPROTO_IPV6", IPPROTO_IPV6),
    AbiConstant("INADDR_ANY", INADDR_ANY),
    AbiConstant("INADDR_LOOPBACK", INADDR_LOOPBACK),
]
"""The constants to check."""

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
because reflection sees through an alias to the underlying type. A constant
whose value differs between the platforms resolves for the target, so each
platform checks its own value.
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
    in_addr_t,
    in_port_t,
)
from std.net._sys.unix import (
    AF_INET,
    AF_INET6,
    AF_UNIX,
    AF_UNSPEC,
    MSG_CTRUNC,
    MSG_DONTROUTE,
    MSG_DONTWAIT,
    MSG_EOR,
    MSG_NOSIGNAL,
    MSG_OOB,
    MSG_PEEK,
    MSG_TRUNC,
    MSG_WAITALL,
    SCM_RIGHTS,
    O_NONBLOCK,
    SHUT_RD,
    SHUT_RDWR,
    SHUT_WR,
    SOCK_DGRAM,
    SOCK_RAW,
    SOCK_SEQPACKET,
    SOCK_STREAM,
    SOL_SOCKET,
    SO_ACCEPTCONN,
    SO_BROADCAST,
    SO_DEBUG,
    SO_DONTROUTE,
    SO_ERROR,
    SO_KEEPALIVE,
    SO_LINGER,
    SO_OOBINLINE,
    SO_RCVBUF,
    SO_RCVTIMEO,
    SO_REUSEADDR,
    SO_REUSEPORT,
    SO_SNDBUF,
    SO_SNDTIMEO,
    SO_TYPE,
    TCP_NODELAY,
    socklen_t,
)

comptime CABI_INCLUDES: List[StaticString] = [
    "<fcntl.h>",
    "<netinet/in.h>",
    "<netinet/tcp.h>",
    "<sys/socket.h>",
]
"""The headers that declare the C side."""

comptime CABI_STRUCTS = TypeList.of[Trait=AnyType, in_addr, in6_addr]()
"""The struct mirrors to check."""

comptime CABI_TYPEDEFS = TypeList.of[
    Trait=AbiTypedefLike,
    AbiTypedef["in_port_t", in_port_t],
    AbiTypedef["in_addr_t", in_addr_t],
    AbiTypedef["socklen_t", socklen_t],
]()
"""The type aliases to check."""

comptime CABI_CONSTANTS = [
    AbiConstant("IPPROTO_IP", IPPROTO_IP),
    AbiConstant("IPPROTO_TCP", IPPROTO_TCP),
    AbiConstant("IPPROTO_UDP", IPPROTO_UDP),
    AbiConstant("IPPROTO_IPV6", IPPROTO_IPV6),
    AbiConstant("INADDR_ANY", INADDR_ANY),
    AbiConstant("INADDR_LOOPBACK", INADDR_LOOPBACK),
    AbiConstant("AF_UNSPEC", AF_UNSPEC),
    AbiConstant("AF_UNIX", AF_UNIX),
    AbiConstant("AF_INET", AF_INET),
    AbiConstant("AF_INET6", AF_INET6),
    AbiConstant("SOCK_STREAM", SOCK_STREAM),
    AbiConstant("SOCK_DGRAM", SOCK_DGRAM),
    AbiConstant("SOCK_RAW", SOCK_RAW),
    AbiConstant("SOCK_SEQPACKET", SOCK_SEQPACKET),
    AbiConstant("SOL_SOCKET", SOL_SOCKET),
    AbiConstant("SO_DEBUG", SO_DEBUG),
    AbiConstant("SO_REUSEADDR", SO_REUSEADDR),
    AbiConstant("SO_TYPE", SO_TYPE),
    AbiConstant("SO_ERROR", SO_ERROR),
    AbiConstant("SO_DONTROUTE", SO_DONTROUTE),
    AbiConstant("SO_BROADCAST", SO_BROADCAST),
    AbiConstant("SO_SNDBUF", SO_SNDBUF),
    AbiConstant("SO_RCVBUF", SO_RCVBUF),
    AbiConstant("SO_KEEPALIVE", SO_KEEPALIVE),
    AbiConstant("SO_OOBINLINE", SO_OOBINLINE),
    AbiConstant("SO_LINGER", SO_LINGER),
    AbiConstant("SO_REUSEPORT", SO_REUSEPORT),
    AbiConstant("SO_RCVTIMEO", SO_RCVTIMEO),
    AbiConstant("SO_SNDTIMEO", SO_SNDTIMEO),
    AbiConstant("SO_ACCEPTCONN", SO_ACCEPTCONN),
    AbiConstant("TCP_NODELAY", TCP_NODELAY),
    AbiConstant("MSG_OOB", MSG_OOB),
    AbiConstant("MSG_PEEK", MSG_PEEK),
    AbiConstant("MSG_DONTROUTE", MSG_DONTROUTE),
    AbiConstant("MSG_CTRUNC", MSG_CTRUNC),
    AbiConstant("MSG_TRUNC", MSG_TRUNC),
    AbiConstant("MSG_DONTWAIT", MSG_DONTWAIT),
    AbiConstant("MSG_EOR", MSG_EOR),
    AbiConstant("MSG_WAITALL", MSG_WAITALL),
    AbiConstant("MSG_NOSIGNAL", MSG_NOSIGNAL),
    AbiConstant("SCM_RIGHTS", SCM_RIGHTS),
    AbiConstant("SHUT_RD", SHUT_RD),
    AbiConstant("SHUT_WR", SHUT_WR),
    AbiConstant("SHUT_RDWR", SHUT_RDWR),
    AbiConstant("O_NONBLOCK", O_NONBLOCK),
]
"""The constants to check."""

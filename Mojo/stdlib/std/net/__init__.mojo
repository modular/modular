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
"""Provides networking: IP and socket addresses, and TCP sockets.

Examples:

```mojo
from std.net.tcp import TCPSocket
from std.net.tcp.option import NoDelay

def main() raises:
    var socket = TCPSocket.v4()
    socket.set_option[NoDelay](True)
```
"""

from .addr import (
    IpAddr,
    Ipv4Addr,
    Ipv6Addr,
    SocketAddr,
    SocketAddrV4,
    SocketAddrV6,
)
from .tcp import TCPListener, TCPSocket

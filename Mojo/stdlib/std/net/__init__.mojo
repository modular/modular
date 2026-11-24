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
from std.net import SocketAddrV4, TCPListener, TCPStream

def main() raises:
    var listener = TCPListener(bind=.v4(SocketAddrV4(.localhost, 0)))
    var client = TCPStream(connect=listener.local_addr())
    var server, _ = listener.accept()
    client.write_all("ping".as_bytes())
    var buffer = List[Byte](length=4, fill=0)
    var n = server.read(buffer)
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
from .tcp import TCPListener, TCPSocket, TCPStream

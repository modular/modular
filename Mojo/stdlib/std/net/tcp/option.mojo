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
"""Provides the options that TCP sockets accept.

This includes the options from `std.net.option` that other kinds of socket
accept too, such as `ReuseAddress`.
"""

from .._sys.inet import IPPROTO_TCP as _IPPROTO_TCP
from .._sys.unix import (
    SOL_SOCKET as _SOL_SOCKET,
    SO_KEEPALIVE as _SO_KEEPALIVE,
    TCP_NODELAY as _TCP_NODELAY,
)
from ..option import FlagOption, ReuseAddress, ReusePort

comptime NoDelay = FlagOption["NoDelay", _IPPROTO_TCP, _TCP_NODELAY]
"""Represents `TCP_NODELAY`: whether small writes are sent right away instead
of being batched.

By default, TCP holds back a small write until the data sent before it is
acknowledged, so that it can combine small writes into fewer packets. Turning
this on sends each write right away, which lowers latency for small messages,
such as requests and replies, at the cost of sending more packets.
"""

comptime KeepAlive = FlagOption["KeepAlive", _SOL_SOCKET, _SO_KEEPALIVE]
"""Represents `SO_KEEPALIVE`: whether an idle connection sends probes to check
that the peer is still there."""

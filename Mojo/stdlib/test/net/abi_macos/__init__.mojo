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
"""C ABI manifest: the macOS-specific `std.net._sys` mirrors."""

from test_utils.cabi_check import AbiConstant, AbiTypedef, AbiTypedefLike

from std.net._sys.macos import (
    SO_NOSIGPIPE,
    sa_family_t,
    sockaddr,
    sockaddr_in,
    sockaddr_in6,
    sockaddr_storage,
    sockaddr_un,
)

comptime CABI_INCLUDES: List[StaticString] = [
    "<netinet/in.h>",
    "<sys/socket.h>",
    "<sys/un.h>",
]
"""The headers that declare the C side."""

comptime CABI_STRUCTS = TypeList.of[
    Trait=AnyType,
    sockaddr,
    sockaddr_in,
    sockaddr_in6,
    sockaddr_un,
    sockaddr_storage,
]()
"""The struct mirrors to check."""

comptime CABI_TYPEDEFS = TypeList.of[
    Trait=AbiTypedefLike,
    AbiTypedef["sa_family_t", sa_family_t],
]()
"""The type aliases to check."""

comptime CABI_CONSTANTS = [
    AbiConstant("SO_NOSIGPIPE", SO_NOSIGPIPE),
]
"""The constants to check."""

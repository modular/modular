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
"""C ABI manifest: the Linux-specific `std.net._sys` mirrors."""

from test_utils.cabi_check import AbiConstant, AbiTypedefLike

from std.net._sys.linux import (
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

comptime CABI_TYPEDEFS = TypeList.of[Trait=AbiTypedefLike]()
"""The type aliases to check."""

comptime CABI_CONSTANTS: List[AbiConstant] = []
"""The constants to check."""

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
"""C ABI manifest: the `std.ffi` C type aliases.

The checker names each C type in a single identifier, so multi-word types
such as `unsigned long` aren't listed; they share the size and alignment of
their signed counterparts.
"""

from test_utils.cabi_check import AbiConstant, AbiTypedef, AbiTypedefLike

from std.ffi import (
    c_char,
    c_double,
    c_float,
    c_int,
    c_long,
    c_pid_t,
    c_short,
    c_size_t,
    c_ssize_t,
)

comptime CABI_INCLUDES: List[StaticString] = ["<stddef.h>", "<sys/types.h>"]
"""The headers that declare the C side."""

comptime CABI_STRUCTS = TypeList.of[Trait=AnyType]()
"""The struct mirrors to check."""

comptime CABI_TYPEDEFS = TypeList.of[
    Trait=AbiTypedefLike,
    AbiTypedef["char", c_char],
    AbiTypedef["short", c_short],
    AbiTypedef["int", c_int],
    AbiTypedef["long", c_long],
    AbiTypedef["float", c_float],
    AbiTypedef["double", c_double],
    AbiTypedef["size_t", c_size_t],
    AbiTypedef["ssize_t", c_ssize_t],
    AbiTypedef["pid_t", c_pid_t],
]()
"""The type aliases to check."""

comptime CABI_CONSTANTS: List[AbiConstant] = []
"""The constants to check."""

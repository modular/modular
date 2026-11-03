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
"""C ABI manifest: the `std.sys._libc` mirrors shared by Linux and macOS."""

from test_utils.cabi_check import AbiConstant, AbiTypedef, AbiTypedefLike

from std.sys._libc import intptr_t, time_t, timespec, uintptr_t

comptime CABI_INCLUDES: List[StaticString] = [
    "<stdint.h>",
    "<time.h>",
]
"""The headers that declare the C side."""

comptime CABI_STRUCTS = TypeList.of[Trait=AnyType, timespec]()
"""The struct mirrors to check."""

comptime CABI_TYPEDEFS = TypeList.of[
    Trait=AbiTypedefLike,
    AbiTypedef["time_t", time_t],
    AbiTypedef["uintptr_t", uintptr_t],
    AbiTypedef["intptr_t", intptr_t],
]()
"""The type aliases to check."""

comptime CABI_CONSTANTS: List[AbiConstant] = []
"""The constants to check."""

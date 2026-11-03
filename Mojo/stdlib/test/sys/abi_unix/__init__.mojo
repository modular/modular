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

from std.sys._libc import (
    MAP_ANONYMOUS,
    MAP_FIXED,
    MAP_PRIVATE,
    MAP_SHARED,
    MS_ASYNC,
    MS_INVALIDATE,
    MS_SYNC,
    PROT_EXEC,
    PROT_NONE,
    PROT_READ,
    PROT_WRITE,
    intptr_t,
    off_t,
    time_t,
    timespec,
    uintptr_t,
)

comptime CABI_INCLUDES: List[StaticString] = [
    "<stdint.h>",
    "<sys/mman.h>",
    "<sys/types.h>",
    "<time.h>",
]
"""The headers that declare the C side."""

comptime CABI_STRUCTS = TypeList.of[Trait=AnyType, timespec]()
"""The struct mirrors to check."""

comptime CABI_TYPEDEFS = TypeList.of[
    Trait=AbiTypedefLike,
    AbiTypedef["off_t", off_t],
    AbiTypedef["time_t", time_t],
    AbiTypedef["uintptr_t", uintptr_t],
    AbiTypedef["intptr_t", intptr_t],
]()
"""The type aliases to check."""

comptime CABI_CONSTANTS = [
    AbiConstant("PROT_NONE", PROT_NONE),
    AbiConstant("PROT_READ", PROT_READ),
    AbiConstant("PROT_WRITE", PROT_WRITE),
    AbiConstant("PROT_EXEC", PROT_EXEC),
    AbiConstant("MAP_SHARED", MAP_SHARED),
    AbiConstant("MAP_PRIVATE", MAP_PRIVATE),
    AbiConstant("MAP_FIXED", MAP_FIXED),
    AbiConstant("MAP_ANONYMOUS", MAP_ANONYMOUS),
    AbiConstant("MS_ASYNC", MS_ASYNC),
    AbiConstant("MS_INVALIDATE", MS_INVALIDATE),
    AbiConstant("MS_SYNC", MS_SYNC),
]
"""The constants to check."""

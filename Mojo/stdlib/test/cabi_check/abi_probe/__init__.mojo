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
"""C ABI manifest: `<stdint.h>` limits, for testing the constant checks."""

from test_utils.cabi_check import AbiConstant, AbiTypedefLike

comptime CABI_INCLUDES: List[StaticString] = ["<stdint.h>"]
"""The headers that declare the C side."""

comptime CABI_STRUCTS = TypeList.of[Trait=AnyType]()
"""The struct mirrors to check."""

comptime CABI_TYPEDEFS = TypeList.of[Trait=AbiTypedefLike]()
"""The type aliases to check."""

# C gives `INT8_MIN` and `UINT16_MAX` the type `int`, so their mirrors are
# narrower, like a constant stored in a struct field. `UINT64_MAX` is
# negative as an `Int64`.
comptime CABI_CONSTANTS = [
    AbiConstant("INT8_MIN", Int8.MIN),
    AbiConstant("UINT16_MAX", UInt16.MAX),
    AbiConstant("UINT32_MAX", UInt32.MAX),
    AbiConstant("INT64_MIN", Int64.MIN),
    AbiConstant("UINT64_MAX", UInt64.MAX),
]
"""The constants to check."""

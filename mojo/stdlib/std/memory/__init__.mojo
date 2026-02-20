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
"""Low-level memory management: pointers, allocations, address spaces.

The `memory` package provides primitives for direct memory manipulation and
pointer operations. It offers multiple pointer types with varying safety
guarantees, from reference-counted smart pointers to raw unsafe pointers, along
with functions for memory operations and allocation. This package enables
systems programming and interfacing with external code requiring explicit
memory control.

Use this package for performance-critical code requiring manual memory control,
interfacing with C libraries, implementing custom data structures, or accessing
specialized memory. Most code should prefer higher-level collections and
automatic memory management.
"""

from .arc_pointer import ArcPointer
from .legacy_unsafe_pointer import (
    LegacyOpaquePointer,
    LegacyUnsafePointer,
)
from .memory import (
    memcmp,
    memcpy,
    memmove,
    memset,
    memset_zero,
    stack_allocation,
    destroy_n,
    uninit_copy_n,
    uninit_move_n,
)
from .owned_pointer import OwnedPointer
from .pointer import (
    AddressSpace,
    _GPUAddressSpace,
    GPUAddressSpace,
    ImmutPointer,
    MutPointer,
    Pointer,
)
from .span import ImmutSpan, MutSpan, Span
from .unsafe import bitcast, pack_bits
from .unsafe_pointer import (
    alloc,
    ImmutOpaquePointer,
    MutOpaquePointer,
    OpaquePointer,
    ImmutUnsafePointer,
    MutUnsafePointer,
    UnsafePointer,
)
from .unsafe_maybe_uninit import UnsafeMaybeUninit

# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""The memory package provides several pointer types, as well
as utility functions for dealing with memory."""

from .arc import ArcPointer
from .legacy_unsafe_pointer import (
    LegacyOpaquePointer,
    LegacyUnsafePointer,
)
from .memory import memcmp, memcpy, memset, memset_zero, stack_allocation
from .owned_pointer import OwnedPointer
from .pointer import (
    AddressSpace,
    _GPUAddressSpace,
    GPUAddressSpace,
    Pointer,
)
from .span import Span
from .unsafe import bitcast, pack_bits
from .unsafe_pointer import (
    alloc,
    OpaqueImmutPointer,
    OpaqueMutPointer,
    OpaquePointer,
    UnsafeImmutPointer,
    UnsafeMutPointer,
    UnsafePointer,
)
from .maybe_uninitialized import UnsafeMaybeUninitialized

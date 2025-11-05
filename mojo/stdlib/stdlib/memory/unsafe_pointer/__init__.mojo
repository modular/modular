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
from ._legacy_unsafe_pointer import (
    LegacyOpaquePointer,
    LegacyOpaquePointer as OpaquePointer,
    LegacyUnsafePointer,
    LegacyUnsafePointer as UnsafePointer,
)
from ._unsafe_pointer_v2 import (
    alloc,
    ExternalImmutPointer,
    ExternalMutPointer,
    ExternalPointer,
    OpaqueImmutPointer,
    OpaqueMutPointer,
    OpaquePointerV2,
    UnsafeImmutPointer,
    UnsafeMutPointer,
    UnsafePointerV2,
)

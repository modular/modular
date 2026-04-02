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
"""HAL Context — per-device context for memory and queue operations."""

from .plugin import (
    Plugin,
    ContextHandle,
    DeviceHandle,
    QueueHandle,
    EventHandle,
)
from .status import STATUS_SUCCESS, HALError


struct Context:
    """A context loaded on a specific device.

    Represents a runtime handle to an initialized
    and usable device.

    This type is potentially expensive to construct,
    so information gathering for device selection
    should ideally be done on Device before creating
    a Context.
    """

    var _handle: ContextHandle
    # TODO: Proper lifetime management — for now the caller must ensure
    # the Plugin outlives the Context.

    def __init__(out self, handle: ContextHandle):
        self._handle = handle


struct Buffer:
    """A device memory allocation.

    Tracks the allocation mode and byte size.
    """

    var memory: UnsafePointer[NoneType, MutAnyOrigin]
    var byte_size: UInt64
    var _context_handle: ContextHandle


struct Queue:
    """A command queue bound to a context."""

    var _handle: QueueHandle
    var _context_handle: ContextHandle


struct Event:
    """A synchronisation event."""

    var _handle: EventHandle
    var _context_handle: ContextHandle

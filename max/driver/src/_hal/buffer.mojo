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
"""HAL Buffer — device allocations and non-owning views into them."""

from .plugin import M_driver_memory_view, MemoryHandle
from .device import DeviceSpec
from .context import Context

from std.memory import ArcPointer


struct BufferView(Copyable, Movable):
    """A non-owning view over a (sub-range of a) device allocation."""

    var _view: M_driver_memory_view

    def __init__(
        out self,
        handle: MemoryHandle,
        byte_offset: UInt64,
        byte_size: UInt64,
    ):
        self._view = M_driver_memory_view(handle, byte_offset, byte_size)

    def byte_offset(self) -> UInt64:
        return self._view.offset

    def byte_size(self) -> UInt64:
        return self._view.size


@fieldwise_init
struct Buffer[device_spec: DeviceSpec](Movable):
    """A device memory allocation.

    Tracks the allocation handle, its byte size, and a strong reference to
    the owning `Context`.

    Holding the context `ArcPointer` keeps the context (and its driver) alive
    for the buffer's lifetime, and lets transport-dispatching APIs such as
    `copy` recover the buffer's residency (via the context's device) and
    obtain a queue for synchronous transfers.

    Parameters:
        device_spec: The compilation target whose memory this buffer lives on.
    """

    # TODO(Sawyer): decide Buffer ownership. Currently leaks unless the user
    # calls `Context.free_sync`; `_context` is carried for residency only, not
    # to drive an auto-freeing destructor. Either give Buffer a destructor or
    # document it as a non-owning view and remove `Movable`.

    var _handle: MemoryHandle
    var byte_size: UInt64
    var _context: ArcPointer[Context[Self.device_spec]]

    def view(self) -> BufferView:
        """Returns a view over the whole allocation."""
        return BufferView(self._handle, 0, self.byte_size)

    def view(self, *, byte_offset: UInt64, byte_size: UInt64) -> BufferView:
        """Returns a view over a sub-range of the allocation."""
        debug_assert(
            byte_offset + byte_size <= self.byte_size,
            "BufferView range exceeds Buffer bounds",
        )
        return BufferView(self._handle, byte_offset, byte_size)

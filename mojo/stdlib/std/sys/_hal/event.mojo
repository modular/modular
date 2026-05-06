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
"""HAL Event — a synchronization primitive for queue and stream operations."""

from .plugin import (
    RawDriver,
    EventHandle,
    ContextHandle,
)
from .status import HALError
from std.memory import ArcPointer, ImmutPointer


@fieldwise_init
struct _EventInner[context_origin: ImmutOrigin](Movable):
    """Internal event state, ref-counted via ArcPointer.

    Event needs to be ImplicitlyCopyable so it can be
    stored in a `List` (e.g. for `Queue.wait_for_events`) and shared across
    multiple queues — the same underlying handle is signaled once but waited
    on from many places.
    """

    var _handle: EventHandle
    var _context_handle: ContextHandle
    var _raw: ImmutPointer[RawDriver, Self.context_origin]

    def __del__(deinit self):
        """Destroys the underlying plugin event handle."""
        try:
            self._raw[].destroy_event(self._context_handle, self._handle)
        except e:
            # Mojo destructors are required to be non-raising
            # and aborting on cleanup failure is too aggressive.
            print("warning: event_destroy failed:", e)


struct Event[context_origin: ImmutOrigin](ImplicitlyCopyable, Movable):
    """A synchronization event tied to a context.

    Created via `Queue.record_event()` or `Stream.record_event()`.
    Ref-counted via ArcPointer — copies share the same underlying event.
    The plugin event handle is destroyed exactly once when the last
    reference goes out of scope.

    Parameters:
        context_origin: The origin of the parent Context pointer.
    """

    var _inner: ArcPointer[_EventInner[Self.context_origin]]

    def __init__(
        out self,
        var inner: _EventInner[Self.context_origin],
    ):
        self._inner = ArcPointer(inner^)

    def synchronize(self) raises HALError:
        """Blocks the host until this event has been signaled on the device."""
        ref inner = self._inner[]
        inner._raw[].synchronize_event(inner._context_handle, inner._handle)

    def is_ready(self) raises HALError -> Bool:
        """Checks whether the event has completed."""
        ref inner = self._inner[]
        return inner._raw[].is_event_ready(inner._context_handle, inner._handle)

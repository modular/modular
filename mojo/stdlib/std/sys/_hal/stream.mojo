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

from .plugin import OpaquePointer, FunctionHandle, OutParam
from .context import Context, Buffer
from .queue import Queue
from .event import Event, EventFlags, EVENT_FLAG_NONE, Waitable
from .device import DeviceSpec
from .status import STATUS_SUCCESS, HALError
from std.memory import (
    ArcPointer,
    UnsafePointer,
    UnsafeMaybeUninit,
)
from std.memory.arc_pointer import WeakPointer


@fieldwise_init
struct Stream[device_spec: DeviceSpec](ImplicitlyDestructible, Movable):
    """An in-order command stream bound to a context.

    Operations submitted to a Stream complete in submission order. Each op
    implicitly waits for the previous one to finish.

    Parameters:
        device_spec: The compilation target this stream is set up for.
    """

    var _queue: ArcPointer[Queue[Self.device_spec]]
    var _chain_event: Optional[Event[EVENT_FLAG_NONE]]
    var _queue_is_stream: Bool
    var _self_ref: WeakPointer[Self]

    @staticmethod
    def _create(
        out _self: ArcPointer[Self], context: Context[Self.device_spec]
    ) raises HALError:
        _self = ArcPointer(Self(context))
        _self[]._self_ref = WeakPointer(downgrade=_self)

    @doc_hidden
    def __init__(
        out self: Stream[Self.device_spec],
        ref context: Context[Self.device_spec],
    ) raises HALError:
        self._queue = context.create_queue()

        var is_stream = UnsafeMaybeUninit[Bool]()
        var status = context._raw[]._raw.queue_is_stream.f(
            self._queue[]._handle, OutParam[Bool](to=is_stream)
        )
        if status != STATUS_SUCCESS:
            var err = context._raw[].get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to query queue_is_stream: {err.message}"
                ),
            )
        self._queue_is_stream = is_stream.unsafe_assume_init_ref()
        self._chain_event = None
        self._self_ref = WeakPointer[Self]()

    # ===-------------------------------------------------------------------===#
    # Internal helpers
    # ===-------------------------------------------------------------------===#

    def _chain_wait(mut self) raises HALError:
        """Waits on the chain event if a previous op has recorded one."""
        if self._queue_is_stream or not self._chain_event:
            return
        self._queue[].wait_for_events(self._chain_event.value())

    def _chain_signal(mut self) raises HALError:
        """Records a fresh chain event after the just-submitted op."""
        if self._queue_is_stream:
            return
        self._chain_event = Optional(self._queue[].record_event())

    # ===-------------------------------------------------------------------===#
    # Stream operations
    # ===-------------------------------------------------------------------===#

    def execute(
        mut self,
        func: FunctionHandle,
        grid: Tuple[UInt32, UInt32, UInt32],
        block: Tuple[UInt32, UInt32, UInt32],
        args: UnsafePointer[OpaquePointer[MutExternalOrigin], MutAnyOrigin],
        arg_sizes: UnsafePointer[UInt64, MutAnyOrigin],
        num_args: UInt32,
    ) raises HALError:
        """Enqueues a function execution. Runs after all previous Stream ops."""
        self._chain_wait()
        self._queue[].execute(func, grid, block, args, arg_sizes, num_args)
        self._chain_signal()

    def copy_to_device(
        mut self,
        dst: Buffer,
        src: UnsafePointer[UInt8, MutAnyOrigin],
        size: UInt64,
    ) raises HALError:
        """Host-to-device copy. Runs after all previous Stream ops."""
        self._chain_wait()
        self._queue[].copy_to_device(dst, src, size)
        self._chain_signal()

    def copy_from_device(
        mut self,
        dst: UnsafePointer[UInt8, MutAnyOrigin],
        src: Buffer,
        size: UInt64,
    ) raises HALError:
        """Device-to-host copy. Runs after all previous Stream ops."""
        self._chain_wait()
        self._queue[].copy_from_device(dst, src, size)
        self._chain_signal()

    def copy_intra_device(
        mut self,
        dst: Buffer,
        src: Buffer,
        size: UInt64,
    ) raises HALError:
        """Same-device buffer copy. Runs after all previous Stream ops."""
        self._chain_wait()
        self._queue[].copy_intra_device(dst, src, size)
        self._chain_signal()

    def record_event[
        flags: EventFlags = EVENT_FLAG_NONE,
    ](mut self,) raises HALError -> Event[flags]:
        """Returns an event signaled when all previous stream ops complete.

        Parameters:
            flags: Capability bitmask. Default `EVENT_FLAG_NONE` is intra-GPU
                only. Pass `EVENT_FLAG_CPU_VISIBLE` to enable host-side
                synchronization on the returned event.
        """
        return self._queue[].record_event[flags]()

    def wait_for_events[
        *EventTypes: Waitable,
    ](mut self, *events: *EventTypes,) raises HALError:
        """Inserts a wait for cross-stream events into the in-order chain.

        Accepts any combination of events with different flag combos.
        """
        self._queue[].wait_for_events(*events)
        self._chain_signal()

    def synchronize(self) raises HALError:
        """Blocks the host until all submitted ops on this stream complete."""
        self._queue[].synchronize()

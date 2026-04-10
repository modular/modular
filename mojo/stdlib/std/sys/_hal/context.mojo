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
    OutParam,
    ContextHandle,
    DeviceHandle,
    QueueHandle,
    EventHandle,
    BundleHandle,
)
from .status import STATUS_SUCCESS, HALError

from std.memory import (
    MutPointer,
    ArcPointer,
    UnsafeMaybeUninit,
)


@fieldwise_init
struct Context[device_origin: MutOrigin](Movable):
    """A context loaded on a specific device.

    Represents a runtime handle to an initialized
    and usable device.

    This type is potentially expensive to construct,
    so information gathering for device selection
    should ideally be done on Device before creating
    a Context.

    Parameters:
        device_origin: The origin of the parent Device pointer.
    """

    var _handle: ContextHandle
    var device: MutPointer[Device[Self.device_origin], Self.device_origin]

    def __init__[
        o1: MutOrigin, o2: MutOrigin
    ](
        out self: Context[origin_of(o1, o2)], ref[o1] device: Device[o2]
    ) raises HALError:
        # This is a horrible hack that should be revisited as soon as
        # we can express subtyping relations between origins and/or
        # inferred/unbound inner origin params for arguments
        self.device = rebind[type_of(self.device)](Pointer(to=device))

        ref driver = device.driver[]
        ref plugin = driver._plugin
        var context_handle_uninit = UnsafeMaybeUninit[ContextHandle]()
        var status = plugin.context_create.f(
            device._handle, OutParam[ContextHandle](to=context_handle_uninit)
        )

        if status != STATUS_SUCCESS:
            var err = plugin.get_status_message(driver._handle, status)
            raise HALError(
                err.status,
                message="failed to create context from device: " + err.message,
            )

        self._handle = context_handle_uninit.unsafe_assume_init_ref()


struct Buffer:
    """A device memory allocation.

    Tracks the allocation mode and byte size.
    """

    var memory: UnsafePointer[NoneType, MutAnyOrigin]
    var byte_size: UInt64


struct Queue[context_origin: MutOrigin]:
    """A command queue bound to a context.

    Parameters:
        context_origin: The origin of the parent Context pointer.
    """

    var _handle: QueueHandle
    var context: MutPointer[Context[Self.context_origin], Self.context_origin]


struct Event:
    """A synchronisation event."""

    var _handle: EventHandle


struct Bundle:
    var _handle: BundleHandle

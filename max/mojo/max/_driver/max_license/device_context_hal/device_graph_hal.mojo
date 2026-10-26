# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""HAL stand-in for the device-graph API.

The HAL DeviceContext has no device-graph support: graph capture fails in the
C++ runtime shim before any of these values is ever created. The types carry
just enough surface for `max.gpu.host`'s re-exports and the graph-aware
graph-compiler primitives to compile; none of them can be meaningfully
constructed or used at runtime.
"""

from .device_context_hal import DeviceBuffer, DeviceContext


struct _DeviceGraphBuilderCpp:
    pass


struct _DeviceGraphCpp:
    pass


comptime _DeviceGraphBuilderPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceGraphBuilderCpp, origin]

comptime _DeviceGraphPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceGraphCpp, origin]


trait DeviceGraphInput:
    """Unsupported: the HAL DeviceContext has no device-graph support."""

    ...


@fieldwise_init
struct StableAddr(TrivialRegisterPassable):
    """Identifies a stable address slot; same layout as the real type so the
    graph-compiler primitives that pass one through compile. No slot is ever
    created on the HAL backend."""

    var ptr: Pointer[Pointer[NoneType, MutUntrackedOrigin], ImmUntrackedOrigin]
    """Device address of the slot; the slot holds the device data pointer."""


@fieldwise_init
struct DeviceGraphCache:
    """Unsupported: the HAL DeviceContext has no device-graph support."""

    pass


struct DeviceGraph(ImplicitlyCopyable):
    """Unsupported: the HAL DeviceContext has no device-graph support."""

    var _handle: _DeviceGraphPtr[mut=True]

    def take_handle(deinit self) -> _DeviceGraphPtr[mut=True]:
        """Surrenders the owning handle net-zero, suppressing the destructor.

        Returns:
            The owning `DeviceGraph*`; the caller must hand it to a runtime owner
            that adopts it without an extra reference.
        """
        return self._handle


struct DeviceGraphBuilder(Movable):
    """Unsupported: the HAL DeviceContext has no device-graph support."""

    var _ctx: DeviceContext

    def context(self) -> DeviceContext:
        """Returns the device context this builder records against.

        Returns:
            The `DeviceContext` the builder was created from.
        """
        return self._ctx

    def recording_context(self) raises -> DeviceContext:
        """Unsupported: returns a `DeviceContext` that records into this builder.

        Returns:
            Never returns; the HAL `DeviceContext` cannot record device graphs.

        Raises:
            Always, since the HAL backend has no recording context.
        """
        raise Error(
            "device-graph recording is unsupported by the HAL DeviceContext"
        )

    def create_buffer[
        dtype: DType
    ](self, size: Int, is_host: Bool) raises -> DeviceBuffer[dtype]:
        """Unsupported: allocates a buffer owned by the current device graph.

        Parameters:
            dtype: The element type of the resulting buffer.

        Args:
            size: The number of elements to allocate for the buffer.
            is_host: Allocate host-addressable memory instead of device memory.

        Returns:
            Never returns; the HAL `DeviceContext` has no device-graph pool.

        Raises:
            Always, since the HAL backend cannot allocate from a device graph.
        """
        raise Error(
            "device-graph allocation is unsupported by the HAL DeviceContext"
        )

    def create_input_buffer[
        dtype: DType
    ](self, size: Int, is_host: Bool, out result: DeviceBuffer[dtype]) raises:
        """Unsupported: allocates an input buffer by the current device graph.

        Allocation and registration are one step so a `DeviceGraphInput` cannot
        allocate a stable location without the graph learning about it, which
        would leave replay with nothing to copy into.

        Parameters:
            dtype: The element type of the resulting buffer.

        Args:
            size: The number of elements to allocate for the buffer.
            is_host: Allocate host-addressable memory instead of device memory.

        Returns:
            The newly allocated buffer, already registered with the graph.

        Raises:
            If memory allocation fails.
        """
        raise Error(
            "device-graph allocation unuspported by the HAL DeviceContext"
        )

    def register_in_place_input(mut self):
        """Unsupported: registers an in-place input marker.

        A no-op stand-in; the HAL `DeviceContext` cannot build device graphs,
        so no marker is ever consulted.
        """
        pass

    def get_stable(mut self, index: Int) raises -> StableAddr:
        """Unsupported: returns stable address slot `index` of the collective.

        Args:
            index: Position of the slot in the collective's slot table.

        Returns:
            Never returns; the HAL `DeviceContext` installs no slot table.

        Raises:
            Always, since the HAL backend has no stable address slots.
        """
        raise Error(
            "stable address slots are unsupported by the HAL DeviceContext"
        )

    def place_stable(
        mut self,
        index: Int,
        source: Pointer[mut=True, NoneType, _],
    ) raises -> StableAddr:
        """Unsupported: publishes `source` into stable address slot `index`.

        Args:
            index: Position of the slot in the collective's slot table.
            source: The device data pointer to publish.

        Returns:
            Never returns; the HAL `DeviceContext` installs no slot table.

        Raises:
            Always, since the HAL backend has no stable address slots.
        """
        raise Error(
            "stable address slots are unsupported by the HAL DeviceContext"
        )


struct DeviceGraphNode:
    """Unsupported: the HAL DeviceContext has no device-graph support."""

    pass


@doc_hidden
struct DeviceGraphInputKind(Equatable, TrivialRegisterPassable):
    """How a tensor participates as a device graph input.

    Mirrors the real type value for value: the graph-compiler primitives
    select their `add_input` behavior by matching on the kind at compile
    time, so the same cases must exist here even though no graph is ever
    recorded on the HAL backend.
    """

    var value: Int

    comptime data = DeviceGraphInputKind(0)
    comptime mutable = DeviceGraphInputKind(1)
    comptime host = DeviceGraphInputKind(2)
    comptime shape_only = DeviceGraphInputKind(3)

    def __init__(out self, value: Int):
        self.value = value

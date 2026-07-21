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

"""HAL stand-in for the device-graph API.

The HAL DeviceContext has no device-graph support: graph capture fails in the
C++ runtime shim before any of these values is ever created. The types carry
just enough surface for `std.gpu.host`'s re-exports and the graph-aware
graph-compiler primitives to compile; none of them can be meaningfully
constructed or used at runtime.
"""

from std.ffi import _CPointer

from .device_context import DeviceContext


struct _DeviceGraphBuilderCpp:
    pass


struct _DeviceGraphCpp:
    pass


comptime _DeviceGraphBuilderPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceGraphBuilderCpp, origin]

comptime _DeviceGraphPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceGraphCpp, origin]


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


struct DeviceGraphNode:
    """Unsupported: the HAL DeviceContext has no device-graph support."""

    pass

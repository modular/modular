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

from std.memory import Allocation, Layout, dealloc

from max.gpu.host import DeviceBuffer, DeviceContext

from .tile_layout import TensorLayout
from .tile_tensor import TileTensor


struct HostDeviceTileTensor[dtype: DType, LayoutType: TensorLayout](Movable):
    """Owns a host allocation and a device buffer that share a layout.

    Call `to_device()` and `to_host()` to move data between the two buffers.
    Without a `DeviceContext` there's no device buffer so transfers do nothing.
    Views `host_tensor()` and `device_tensor()` never copy data.

    Parameters:
        dtype: The element type.
        LayoutType: The layout type shared by both buffers.
    """

    var _layout: Self.LayoutType
    var _host: Allocation[Scalar[Self.dtype]]
    var _device: Optional[DeviceBuffer[Self.dtype]]
    var _ctx: Optional[DeviceContext]

    def __init__(out self, layout: Self.LayoutType):
        """Allocates host memory only.

        Args:
            layout: The layout of the tensor.
        """
        self._layout = layout
        self._host = alloc(Layout[Scalar[Self.dtype]](count=_cosize(layout)))
        self._device = None
        self._ctx = None

    def __init__(out self, layout: Self.LayoutType, ctx: DeviceContext) raises:
        """Allocates host memory and a device buffer on `ctx`.

        Args:
            layout: The layout of the tensor.
            ctx: The device context that owns the device buffer.
        """
        var size = _cosize(layout)
        var device = ctx.enqueue_create_buffer[Self.dtype](size)
        self._layout = layout
        self._host = alloc(Layout[Scalar[Self.dtype]](count=size))
        self._device = device
        self._ctx = ctx

    def __deinit__(deinit self):
        dealloc(self._host^)

    def host_tensor(
        ref self,
    ) -> TileTensor[Self.dtype, Self.LayoutType, origin_of(self)]:
        """Returns a view of the host memory without copying.

        Returns:
            A `TileTensor` over the host allocation.
        """
        return TileTensor[Self.dtype, Self.LayoutType, origin_of(self)](
            ptr=self._host.unsafe_ptr().unsafe_origin_cast[origin_of(self)](),
            layout=self._layout,
        )

    def device_tensor(
        ref self,
    ) raises -> TileTensor[Self.dtype, Self.LayoutType, origin_of(self)]:
        """Returns a view of the device buffer without copying.

        Returns:
            A `TileTensor` over the device buffer.

        Raises:
            If the tensor was created without a `DeviceContext`.
        """
        if not self._device:
            raise Error("HostDeviceTileTensor has no device buffer")
        return TileTensor[Self.dtype, Self.LayoutType, origin_of(self)](
            ptr=self._device.value()
            .unsafe_ptr()
            .unsafe_mut_cast[origin_of(self).mut]()
            .unsafe_origin_cast[origin_of(self)](),
            layout=self._layout,
        )

    def to_device(self) raises:
        """Copies host memory to the device buffer and waits for the copy.

        Does nothing if there is no device buffer.
        """
        if self._ctx:
            self._ctx.value().enqueue_copy(
                self._device.value(), self._host.unsafe_ptr().as_imm()
            )
            self._ctx.value().synchronize()

    def to_host(mut self) raises:
        """Copies the device buffer to host memory and waits for the copy.

        Does nothing if there is no device buffer.
        """
        if self._ctx:
            self._ctx.value().enqueue_copy(
                self._host.unsafe_ptr(), self._device.value()
            )
            self._ctx.value().synchronize()


def _cosize[L: TensorLayout](layout: L) -> Int:
    var n = layout.product()
    if n == 0:
        return 0
    return Int(layout(n - 1)) + 1

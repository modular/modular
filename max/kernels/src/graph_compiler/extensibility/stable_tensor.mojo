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
"""Provides `StableTensor`, a tensor view over a stable address slot."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.sys.info import is_gpu
from max.gpu.host import StableAddr
from layout import TileTensor
from layout.tile_layout import Layout as TileLayout

from .managed_tensor_slice import (
    IO,
    IOSpec,
    ManagedTensorSlice,
    StaticTensorSpec,
)


@fieldwise_init
struct StableTensor[
    mut: Bool,
    input: IO,
    dtype: DType,
    rank: Int,
    //,
    io_spec: IOSpec[mut, input],
    *,
    static_spec: StaticTensorSpec[dtype, rank, _],
](DevicePassable, TrivialRegisterPassable):
    """A tensor view whose data pointer lives in a stable address slot.

    The view holds the slot's address and the tensor's layout; the data pointer
    is read out of the slot on each access. A device graph that records this
    view bakes in the slot, so
    [`DeviceGraphBuilder.place_stable()`](/api/mojo/max/gpu/host/device_graph/DeviceGraphBuilder/#place_stable)
    can point the slot at a different tensor before each replay without the
    graph being rebuilt.

    The slot and the pointer it holds are both device addresses, so the view
    is built on the host but read only on the device: pass it to a kernel and
    convert to a `ManagedTensorSlice` or `TileTensor` there. The conversions
    are compile errors on the host.

    Parameters:
        mut: Whether the tensor is mutable.
        input: The IO kind of the tensor.
        dtype: The element type of the tensor.
        rank: The rank of the tensor.
        io_spec: The IO specification of the tensor.
        static_spec: The static specification of the tensor, including its
            layout.
    """

    comptime RuntimeLayout = TileLayout[
        shape_types=Self.static_spec.static_layout._shape_types,
        stride_types=Self.static_spec.static_layout._stride_types,
    ]

    var _addr: StableAddr
    var _runtime_layout: Self.RuntimeLayout

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return String(
            t"StableTensor[mut = {Self.mut}, dtype = {Self.dtype}, rank ="
            t" {Self.rank}]"
        )

    @inline(.always)
    def unsafe_ptr(self) -> Pointer[Scalar[Self.dtype], MutUntrackedOrigin]:
        """Reads the data pointer the slot holds at the time of the call.

        Device only: the slot is device memory.

        Returns:
            The tensor's current data pointer.
        """
        comptime assert is_gpu(), (
            "StableTensor reads its slot on the device; convert it inside the"
            " kernel that consumes it"
        )
        return self._addr.ptr[].unsafe_bitcast[Scalar[Self.dtype]]()

    @inline(.always)
    def to_managed_tensor_slice(
        self,
        out result: ManagedTensorSlice[
            io_spec=Self.io_spec, static_spec=Self.static_spec
        ],
    ):
        """Builds a `ManagedTensorSlice` over the pointer the slot holds now.

        Device only. The result no longer goes through the slot, which is why
        the conversion belongs inside the consuming kernel.

        Returns:
            A slice over the current data pointer with this view's layout.
        """
        return {
            self.unsafe_ptr(),
            self._runtime_layout.shape_coord(),
            self._runtime_layout.stride_coord(),
        }

    @inline(.always)
    def to_tile_tensor(
        self,
        out result: TileTensor[
            Self.dtype,
            origin=MutUntrackedOrigin,
            LayoutType=Self.RuntimeLayout,
        ],
    ):
        """Builds a `TileTensor` over the pointer the slot holds now.

        Device only. Like `to_managed_tensor_slice()`, the result no longer
        goes through the slot.

        Returns:
            A tile tensor over the current data pointer with this view's
            layout.
        """
        return {self.unsafe_ptr(), self._runtime_layout}

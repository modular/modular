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
"""Provides `StableTensor`, a tensor view over a stable address slot, and
`StableVariadicTensors`, the variadic pack of such views."""

from layout import coord_to_index_list
from layout.tile_layout import Layout as TileLayout
from max.gpu.host import StableAddr
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.sys.info import is_gpu
from std.utils import IndexList, StaticTuple

from .managed_tensor_slice import (
    IO,
    IOSpec,
    ManagedTensorSlice,
    StaticTensorSpec,
    StaticTensorSpecList,
    _index_list_to_static_coord,
)
from .stable_addr_engine import StableTileTensor, stable_tile_tensor


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

    @staticmethod
    @inline(.always)
    def _make_runtime_layout(
        shape: IndexList[Self.rank], strides: IndexList[Self.rank]
    ) -> Self.RuntimeLayout:
        """Builds the runtime layout from a runtime shape and strides.

        Static dimensions keep their compile-time values; dynamic dimensions are
        filled from `shape`/`strides`.
        """
        return Self.RuntimeLayout(
            _index_list_to_static_coord[
                Self.static_spec.static_layout._shape_types
            ](shape),
            _index_list_to_static_coord[
                Self.static_spec.static_layout._stride_types
            ](strides),
        )

    @inline(.always)
    def shape(self) -> IndexList[Self.rank]:
        """Gets the shape of this tensor, as an `IndexList`.

        Host-callable: the shape lives in the view, not behind the slot.

        Returns:
            The shape of this tensor.
        """
        return rebind[IndexList[Self.rank]](
            coord_to_index_list(self._runtime_layout.shape_coord())
        )

    @inline(.always)
    def size(self) -> Int:
        """Computes the tensor's number of elements.

        Host-callable: the size lives in the view, not behind the slot.

        Returns:
            The total number of elements in the tensor.
        """
        return Int(self._runtime_layout.size())

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
    def to_tile_tensor(
        self,
        out result: StableTileTensor[Self.dtype, Self.RuntimeLayout],
    ):
        """Builds a `TileTensor` that keeps reading through the stable address.

        The result stores the stable address rather than resolving it at
        conversion time, so the conversion may run on the host while the slot
        is still dereferenced only on the device. The cost is one additional
        pointer dereference per load or store.

        Returns:
            A slot-backed tile tensor with this view's layout.
        """
        return stable_tile_tensor[Self.dtype](self._addr, self._runtime_layout)


# ===----------------------------------------------------------------------=== #
# StableVariadicTensors
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct StableVariadicTensors[
    mut: Bool,
    input: IO,
    dtype: DType,
    rank: Int,
    //,
    size: Int,
    io_spec: IOSpec[mut, input],
    *,
    static_specs: StaticTensorSpecList[dtype=dtype, rank=rank, ...],
](Sized, TrivialRegisterPassable):
    """A tuple-like container of `StableTensor`s representing variadic
    arguments from the graph compiler.

    The graph compiler builds it from one stable address slot and one runtime
    shape per element; each element's data pointer is read out of its slot on
    the device, so a device graph recorded against the pack can be retargeted
    with [`DeviceGraphBuilder.place_stable()`](/api/mojo/max/gpu/host/device_graph/DeviceGraphBuilder/#place_stable)
    before each replay.

    Parameters:
        mut: Whether the tensors are mutable.
        input: The IO kind of the tensors.
        dtype: The element type of the tensors.
        rank: The rank of the tensors.
        size: The number of tensors in the pack.
        io_spec: The IO specification shared by every tensor.
        static_specs: The per-element static specifications.
    """

    var _addrs: StaticTuple[StableAddr, Self.size]
    var _shapes: StaticTuple[IndexList[Self.rank], Self.size]

    def __len__(self) -> Int:
        """Returns the number of variadic arguments in the pack.

        Returns:
            The number of variadic arguments.
        """
        return Self.size

    def __getitem_param__[
        index: Int
    ](
        self,
        out result: StableTensor[
            io_spec=Self.io_spec, static_spec=Self.static_specs[index]
        ],
    ):
        """Returns the tensor at the given position in the variadic argument
        pack.

        Parameters:
            index: The index into the variadic tensor arguments.

        Returns:
            The stable tensor view at the specified index.
        """
        comptime assert index < Self.size
        var shape = self._shapes[index]
        return type_of(result)(
            self._addrs[index],
            type_of(result)._make_runtime_layout(
                shape, shape.get_row_major_strides()
            ),
        )

    def to_tile_tensor(
        self,
        out result: Array[type_of(self[0].to_tile_tensor()), Self.size],
    ):
        result = {uninitialized = True}

        comptime TensorType = type_of(self[0].to_tile_tensor())

        comptime for i in range(Self.size):
            result[i] = rebind[TensorType](self[i].to_tile_tensor())


comptime StableInputTensor = StableTensor[io_spec=IOSpec.Input, ...]
comptime _StableMutableInputTensor = StableTensor[
    io_spec=IOSpec.MutableInput, ...
]

comptime StableInputVariadicTensors = StableVariadicTensors[
    io_spec=IOSpec.Input, ...
]
comptime _StableMutableInputVariadicTensors = StableVariadicTensors[
    io_spec=IOSpec.MutableInput, ...
]

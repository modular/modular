# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from sys import align_of

from builtin.variadics import VariadicOf, variadic_size

from ._mixed_layout import MixedLayout
from ._mixed_tuple import ComptimeInt, Idx, MixedTuple, MixedTupleLike


@fieldwise_init
struct MixedLayoutTensor[
    mut: Bool,
    dtype: DType,
    shape_types: VariadicOf[MixedTupleLike],
    stride_types: VariadicOf[MixedTupleLike], //,
    origin: Origin[mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    linear_idx_type: DType = DType.int64,
](Copyable, Movable):
    var ptr: UnsafePointer[
        Scalar[Self.dtype], Self.origin, address_space = Self.address_space
    ]

    var layout: MixedLayout[
        shape_types = Self.shape_types,
        stride_types = Self.stride_types,
    ]

    fn __init__(
        out self: MixedLayoutTensor[
            dtype = Self.dtype,
            shape_types = Self.shape_types,
            stride_types = Self.stride_types,
            origin = Self.origin,
            address_space = Self.address_space,
            linear_idx_type = Self.linear_idx_type,
        ],
        var span: Span[Scalar[Self.dtype], Self.origin],
        var layout: MixedLayout[Self.shape_types, Self.stride_types],
    ):
        self.ptr = span.unsafe_ptr().address_space_cast[Self.address_space]()
        self.layout = layout^

    @always_inline("nodebug")
    fn __getitem__(
        self, tuple: MixedTuple
    ) -> Scalar[Self.dtype] where variadic_size(
        tuple.element_types
    ) == variadic_size(Self.shape_types):
        return self.ptr[
            self.layout[linear_idx_type = Self.linear_idx_type](tuple)
        ]

    @always_inline("nodebug")
    fn __setitem__(
        self: MixedLayoutTensor[
            mut=True,
            dtype = Self.dtype,
            shape_types = Self.shape_types,
            stride_types = Self.stride_types,
            address_space = Self.address_space,
            linear_idx_type = Self.linear_idx_type,
        ],
        tuple: MixedTuple,
        value: Scalar[Self.dtype],
    ) where variadic_size(tuple.element_types) == variadic_size(
        Self.shape_types
    ):
        self.ptr[
            self.layout[linear_idx_type = Self.linear_idx_type](tuple)
        ] = value


fn distribute[
    thread_shape_0: Int,
    thread_shape_1: Int,
    thread_stride_0: Int,
    thread_stride_1: Int,
    data_shape_0: Int,
    data_shape_1: Int,
    data_stride_0: Int,
    data_stride_1: Int, //,
    dtype: DType,
    thread_layout: MixedLayout[
        Tuple[
            ComptimeInt[thread_shape_0], ComptimeInt[thread_shape_1]
        ].element_types,
        Tuple[
            ComptimeInt[thread_stride_0], ComptimeInt[thread_stride_1]
        ].element_types,
    ],
](
    data_layout_tensor: MixedLayoutTensor[
        dtype=dtype,
        shape_types = Tuple[
            ComptimeInt[data_shape_0], ComptimeInt[data_shape_1]
        ].element_types,
        stride_types = Tuple[
            ComptimeInt[data_stride_0], ComptimeInt[data_stride_1]
        ].element_types,
    ],
    thread_id: Int,
) -> MixedLayoutTensor[
    dtype = data_layout_tensor.dtype,
    shape_types = MixedTuple[
        ComptimeInt[data_shape_0 // thread_shape_0],
        ComptimeInt[data_shape_1 // thread_shape_1],
    ].element_types,
    stride_types = Tuple[
        ComptimeInt[data_stride_0 * thread_shape_0],
        ComptimeInt[data_stride_1 * thread_shape_1],
    ].element_types,
    data_layout_tensor.origin,
    address_space = data_layout_tensor.address_space,
    linear_idx_type = data_layout_tensor.linear_idx_type,
]:
    """A simplified implementation of LayoutTensor.distribute on MixedLayoutTensor.
    """

    var offset: UInt = 0

    @parameter
    for i in range(len(thread_layout.stride)):
        comptime stride_i = Int(thread_layout.stride[i].value())
        comptime shape_i = Int(thread_layout.shape[i].value())
        var thread_coord_i = (thread_id // stride_i) % shape_i
        offset += UInt(
            thread_coord_i * Int(data_layout_tensor.layout.stride[i].value())
        )

    comptime shape = MixedTuple(
        ComptimeInt[data_shape_0 // thread_shape_0](),
        ComptimeInt[data_shape_1 // thread_shape_1](),
    )

    comptime stride = MixedTuple(
        ComptimeInt[data_stride_0 * thread_shape_0](),
        ComptimeInt[data_stride_1 * thread_shape_1](),
    )

    var frag_layout = MixedLayout(
        shape=shape,
        stride=stride,
    )

    return MixedLayoutTensor[dtype = data_layout_tensor.dtype,](
        UnsafePointer(to=data_layout_tensor.ptr[offset]),
        rebind[
            MixedLayout[
                shape_types = type_of(shape._storage).element_types,
                stride_types = type_of(stride._storage).element_types,
            ]
        ](frag_layout),
    )


fn tile[
    dtype: DType,
    shape_types: VariadicOf[MixedTupleLike],
    stride_types: VariadicOf[MixedTupleLike],
    coord_types: VariadicOf[MixedTupleLike],
    tile_shape_types: VariadicOf[MixedTupleLike], //,
](
    data_layout_tensor: MixedLayoutTensor[
        dtype=dtype, shape_types=shape_types, stride_types=stride_types
    ],
    tile_shape: MixedTuple[*tile_shape_types],
    tile_coords: MixedTuple[*coord_types],
) -> MixedLayoutTensor[
    dtype=dtype,
    shape_types=tile_shape_types,
    stride_types=stride_types,
    data_layout_tensor.origin,
    address_space = data_layout_tensor.address_space,
    linear_idx_type = data_layout_tensor.linear_idx_type,
]:
    """Extract a tile (sub-tensor) from a MixedLayoutTensor at specified coordinates.

    This function creates a view into a specific rectangular region of the source tensor
    without copying data. It computes the memory offset for the tile and creates a new
    MixedLayoutTensor with the tile dimensions while preserving the original stride pattern.

    Difference from LayoutTensor.tile:
        This simplified implementation returns a tile with the original tensor's
        stride information rather than creating a hierarchical (blocked/tiled)
        layout with an appropriate stride.

        It is incorrect for non-divisible tile shapes (like dividing a 16x16 tensor
        into 3x3 tiles).

    Parameters:
        dtype: Data type of the tensor elements (inferred from tensor argument).
        shape_types: Shape types of the source tensor (inferred from tensor argument).
        stride_types: Stride types of the source tensor (inferred from tensor argument).
        coord_types: Types of the tile coordinates (inferred from coordinates argument).
        tile_shape_types: Types of the tile dimensions (inferred from tile_shape argument).

    Args:
        data_layout_tensor: The source tensor to extract the tile from.
        tile_shape: The shape that the layout should be tiled into.
        tile_coords: The index of the tile to extract as a MixedTuple.

    Returns:
        A MixedLayoutTensor representing a view into the specified tile region.
        The returned tensor has the tile_shape as its dimensions and shares memory
        with the original tensor.
    """

    var offset: UInt = 0

    @parameter
    for i in range(MixedTuple[*coord_types].__len__()):
        offset += UInt(
            tile_coords[i].value()
            * tile_shape[i].value()
            * Int(data_layout_tensor.layout.stride[i].value())
        )

    var tile_layout = MixedLayout(
        shape=tile_shape,
        stride=data_layout_tensor.layout.stride,
    )

    return MixedLayoutTensor[
        dtype=dtype,
        shape_types=tile_shape_types,
        stride_types=stride_types,
    ](
        UnsafePointer(to=data_layout_tensor.ptr[offset]),
        tile_layout,
    )

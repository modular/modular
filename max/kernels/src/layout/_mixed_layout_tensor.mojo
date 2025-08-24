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

from builtin.variadics import VariadicOf
from sys import alignof
from ._mixed_layout import MixedLayout
from ._mixed_tuple import MixedTuple, MixedTupleLike, ComptimeInt, Idx


struct MixedLayoutTensor[
    dtype: DType,
    shape_types: VariadicOf[MixedTupleLike],
    stride_types: VariadicOf[MixedTupleLike], //,
    alignment: Int = alignof[dtype](),
]:
    var ptr: UnsafePointer[Scalar[dtype]]

    var layout: MixedLayout[
        shape_types=shape_types,
        stride_types=stride_types,
    ]

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[dtype]],
        layout: MixedLayout[shape_types, stride_types],
    ):
        self.ptr = ptr
        self.layout = layout

    fn __getitem__[
        index_type: MixedTupleLike
    ](self, arg: index_type) -> SIMD[dtype, 1]:
        return self.ptr[self.layout(arg)]

    fn __setitem__[
        index_type: MixedTupleLike
    ](self, arg: index_type, value: SIMD[dtype, 1]):
        self.ptr[self.layout(arg)] = value


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
        MixedTuple[
            ComptimeInt[thread_shape_0], ComptimeInt[thread_shape_1]
        ]._get_variadic_pack(),
        MixedTuple[
            ComptimeInt[thread_stride_0], ComptimeInt[thread_stride_1]
        ]._get_variadic_pack(),
    ],
](
    data_layout_tensor: MixedLayoutTensor[
        dtype=dtype,
        shape_types = MixedTuple[
            ComptimeInt[data_shape_0], ComptimeInt[data_shape_1]
        ]._get_variadic_pack(),
        stride_types = MixedTuple[
            ComptimeInt[data_stride_0], ComptimeInt[data_stride_1]
        ]._get_variadic_pack(),
    ],
    thread_id: Int,
) -> MixedLayoutTensor[
    dtype = data_layout_tensor.dtype,
    shape_types = MixedTuple[
        ComptimeInt[data_shape_0 // thread_shape_0],
        ComptimeInt[data_shape_1 // thread_shape_1],
    ]._get_variadic_pack(),
    stride_types = MixedTuple[
        ComptimeInt[data_stride_0 * thread_shape_0],
        ComptimeInt[data_stride_1 * thread_shape_1],
    ]._get_variadic_pack(),
]:
    """A simplified implementation of LayoutTensor.distribute on MixedLayoutTensor.
    """

    var offset: UInt = 0

    @parameter
    for i in range(len(thread_layout.stride)):
        alias stride_i = Int(thread_layout.stride[i].value())
        alias shape_i = Int(thread_layout.shape[i].value())
        var thread_coord_i = (thread_id // stride_i) % shape_i
        offset += thread_coord_i * Int(
            data_layout_tensor.layout.stride[i].value()
        )

    alias shape = MixedTuple(
        ComptimeInt[data_shape_0 // thread_shape_0](),
        ComptimeInt[data_shape_1 // thread_shape_1](),
    )

    alias stride = MixedTuple(
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
                shape_types = __type_of(shape)._get_variadic_pack(),
                stride_types = __type_of(stride)._get_variadic_pack(),
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
        offset += (
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

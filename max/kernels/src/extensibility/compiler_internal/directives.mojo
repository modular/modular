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

from std.sys import align_of
from std.sys.intrinsics import _type_is_eq

from layout.coord import (
    ComptimeInt,
    CoordLike,
    RuntimeInt,
    _IntToComptimeInt,
)
from std.utils import IndexList

from layout.int_tuple import _IntTupleToCoordLike, coord_to_int_tuple
from layout.tile_layout import Layout as TileLayout, TensorLayout, _RowMajor
from layout import IntTuple, Layout


# ===----------------------------------------------------------------------=== #
# TileLayout helper aliases
# ===----------------------------------------------------------------------=== #

comptime _AllRuntimeInt[rank: Int] = TypeList.splat[
    Trait=CoordLike, count=rank, type=RuntimeInt[]
]()
"""A variadic of `rank` RuntimeInt types."""

comptime _UnknownTileLayout[rank: Int] = TileLayout[
    shape_types=_AllRuntimeInt[rank],
    stride_types=_AllRuntimeInt[rank],
]
"""A fully-dynamic TileLayout where all shape and stride dims are RuntimeInt."""


comptime _RowMajorTileLayout[
    shape_types: TypeList[Trait=CoordLike, ...]
] = TileLayout[
    shape_types=shape_types,
    stride_types=_RowMajor[*shape_types],
]
"""A TileLayout with row-major strides derived from the given shape types."""

comptime _IndexListToCoordLikeTabulator[
    list: IndexList,
    idx: SIMDSize,
]: CoordLike = ComptimeInt[list[idx]] if list[idx] >= 0 else RuntimeInt[]

"""Maps a single IndexList element to a CoordLike type.
Negative values (-1 = dynamic) become RuntimeInt, others become ComptimeInt."""


comptime _IndexListToCoordLike[list: IndexList] = TypeList.tabulate[
    list.size, _IndexListToCoordLikeTabulator[list, _]
]()
"""Converts a compile-time IndexList to a variadic of CoordLike types.
Negative values become RuntimeInt, non-negative become ComptimeInt."""


comptime _IndexListToTileLayout[
    shape: IndexList, strides: IndexList
] = TileLayout[
    shape_types=_IndexListToCoordLike[shape],
    stride_types=_IndexListToCoordLike[strides],
]
"""Convert a pair of compile-time IndexLists to a TileLayout.
Negative values (-1) become RuntimeInt, non-negative become ComptimeInt."""


comptime _RowMajorIntTupleTileLayout[
    shape: IntTuple,
] = _RowMajorTileLayout[_IntTupleToCoordLike[DType.int, shape]]
"""A TileLayout with row-major strides derived from an IntTuple shape."""


comptime _IntTupleShapeIndexListStridesToTileLayout[
    shape: IntTuple, strides: IndexList
] = TileLayout[
    shape_types=_IntTupleToCoordLike[DType.int, shape],
    stride_types=_IndexListToCoordLike[strides],
]
"""Convert an IntTuple shape and IndexList strides to a TileLayout."""


def __mogg_intrinsic_attr(intrin: StaticString):
    return


# Register a DPS Kernel
@__mogg_intrinsic_attr("mogg.intrinsic_register")
def register(name: StaticString):
    pass


# Indicates that a DPS Kernel is a view operation
@__mogg_intrinsic_attr("mogg.view_kernel")
def view_kernel():
    return


def get_row_major_tensor_spec_static[
    dtype: DType, rank: Int, *shape_dims: Int
]() -> StaticTensorSpec[
    dtype,
    rank,
    static_layout=_RowMajorTileLayout[_IntToComptimeInt[*shape_dims]],
]:
    """Returns a row-major StaticTensorSpec from compile-time Int dimensions.

    All dimensions must be static (known at compile time).

    Parameters:
        dtype: The element data type.
        rank: The tensor rank (must match `len(shape_dims)`).
        shape_dims: Compile-time integer dimensions of the tensor shape.
    """
    return {align_of[dtype](), AddressSpace.GENERIC, False}


def _get_unknown_tensor_spec[
    dtype: DType, rank: Int
]() -> StaticTensorSpec[dtype, rank, static_layout=_UnknownTileLayout[rank]]:
    """
    Returns a StaticTensorSpec with the specified type and rank with all
    fields dynamic or defaulted.
    """
    return {
        1,
        AddressSpace.GENERIC,
        True,
    }


# ===----------------------------------------------------------------------=== #
# Fusion traits and sentinel types
# ===----------------------------------------------------------------------=== #


trait InputFusion(TrivialRegisterPassable):
    """Trait for input fusion structs that provide custom load behavior."""

    def load[
        dtype: DType,
        rank: Int,
        simd_width: Int,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
        ...


trait OutputFusion(TrivialRegisterPassable):
    """Trait for output fusion structs that provide custom store behavior."""

    def store[
        dtype: DType,
        rank: Int,
        simd_width: SIMDSize,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]):
        ...


trait ComputeOutputFusion(TrivialRegisterPassable):
    """Trait for compute-output fusion structs that transform values before
    storing."""

    def compute[
        dtype: DType,
        rank: Int,
        simd_width: SIMDSize,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]) -> SIMD[
        dtype, simd_width
    ]:
        ...


trait ElementwiseFusion(TrivialRegisterPassable):
    """Trait for pure elementwise fusion structs emitted by the graph
    compiler."""

    def compute[
        dtype: DType,
        rank: Int,
        simd_width: Int,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
        ...


struct _NoFusionIn(InputFusion):
    """Sentinel type indicating no input fusion is active."""

    def __init__(out self):
        pass

    def load[
        dtype: DType,
        rank: Int,
        simd_width: Int,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
        comptime assert False, "load() not implemented for this InputFusion"


struct _NoFusionOut(OutputFusion):
    """Sentinel type indicating no output fusion is active."""

    def __init__(out self):
        pass

    def store[
        dtype: DType,
        rank: Int,
        simd_width: SIMDSize,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]):
        comptime assert False, "store() not implemented for this OutputFusion"


struct _NoComputeFusion(ComputeOutputFusion):
    """Sentinel type indicating no compute-output fusion is active."""

    def __init__(out self):
        pass

    def compute[
        dtype: DType,
        rank: Int,
        simd_width: SIMDSize,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]) -> SIMD[
        dtype, simd_width
    ]:
        comptime assert (
            False
        ), "compute() not implemented for this ComputeOutputFusion"


# Compile time Tensor information
struct StaticTensorSpec[
    dtype: DType,
    rank: Int,
    static_layout: TensorLayout,
    InFusion: InputFusion = _NoFusionIn,
    OutFusion: OutputFusion = _NoFusionOut,
    ComputeFusion: ComputeOutputFusion = _NoComputeFusion,
](ImplicitlyCopyable):
    # IntTuple aliases for static shape/strides.
    comptime shape_tuple = coord_to_int_tuple[
        *Self.static_layout._shape_types
    ]()
    comptime strides_tuple = coord_to_int_tuple[
        *Self.static_layout._stride_types
    ]()

    var alignment: Int
    var address_space: AddressSpace
    var exclusive: Bool

    def __init__(
        out self,
        alignment: Int,
        address_space: AddressSpace,
        exclusive: Bool,
    ):
        comptime assert Self.rank == Self.static_layout.rank, "rank mismatch"
        comptime _has_in = not _type_is_eq[Self.InFusion, _NoFusionIn]()
        comptime _has_out = not _type_is_eq[Self.OutFusion, _NoFusionOut]()
        comptime _has_compute = not _type_is_eq[
            Self.ComputeFusion, _NoComputeFusion
        ]()
        comptime assert (
            Int(_has_in) + Int(_has_out) + Int(_has_compute) <= 1
        ), "StaticTensorSpec can have at most one fusion type"
        self.alignment = alignment
        self.address_space = address_space
        self.exclusive = exclusive

    def __init__(
        out self, internals: StaticTensorSpecInternal[Self.dtype, Self.rank]
    ):
        """
        Returns a StaticTensorSpec from a StaticTensorSpecInternal.
        """
        comptime _has_in = not _type_is_eq[Self.InFusion, _NoFusionIn]()
        comptime _has_out = not _type_is_eq[Self.OutFusion, _NoFusionOut]()
        comptime _has_compute = not _type_is_eq[
            Self.ComputeFusion, _NoComputeFusion
        ]()
        comptime assert (
            Int(_has_in) + Int(_has_out) + Int(_has_compute) <= 1
        ), "StaticTensorSpec can have at most one fusion type"
        self.alignment = internals.alignment
        self.address_space = internals.address_space
        self.exclusive = internals.exclusive

    @always_inline
    def to_unfused(
        self,
    ) -> StaticTensorSpec[
        Self.dtype, Self.rank, static_layout=Self.static_layout
    ]:
        """Returns a copy with sentinel (no-op) fusion types.

        The runtime fields (alignment, etc.) are identical;
        only the compile-time fusion type parameters change.
        """
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
        }

    # This indirect approach to providing get_unknown is necessary because the
    # we don't want clients to have to bind rank and shapes to use this. Aliases
    # only require the parameters they USE to be bound, whereas static methods
    # require all parameters to be bound.
    comptime get_unknown = _get_unknown_tensor_spec[Self.dtype, Self.rank]

    @always_inline
    def with_tile_layout[
        new_layout: TensorLayout,
    ](
        self,
    ) -> StaticTensorSpec[
        Self.dtype,
        new_layout.rank,
        static_layout=new_layout,
    ]:
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def with_tile_layout[
        new_rank: Int,
        new_layout: TensorLayout,
    ](
        self,
    ) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=new_layout,
    ]:
        comptime assert new_rank == new_layout.rank, "rank mismatch"
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def with_tile_layout_and_alignment[
        new_layout: TensorLayout,
    ](self, new_alignment: Int) -> StaticTensorSpec[
        Self.dtype,
        new_layout.rank,
        static_layout=new_layout,
    ]:
        return {
            new_alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def with_tile_layout_and_alignment[
        new_rank: Int,
        new_layout: TensorLayout,
    ](self, new_alignment: Int) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=new_layout,
    ]:
        comptime assert new_rank == new_layout.rank, "rank mismatch"
        return {
            new_alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def with_int_tuple_layout[
        new_rank: Int, new_shape: IntTuple, new_strides: IndexList
    ](
        self,
    ) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=_IntTupleShapeIndexListStridesToTileLayout[
            new_shape, new_strides
        ],
    ]:
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def with_int_tuple_layout_and_alignment[
        new_rank: Int, new_shape: IntTuple, new_strides: IndexList
    ](self, new_alignment: Int) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=_IntTupleShapeIndexListStridesToTileLayout[
            new_shape, new_strides
        ],
    ]:
        return {
            new_alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def with_row_major_int_tuple_layout[
        new_rank: Int, new_shape: IntTuple
    ](
        self,
    ) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=_RowMajorIntTupleTileLayout[new_shape],
    ]:
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def with_input_fusion[
        F: InputFusion
    ](self) -> StaticTensorSpec[
        Self.dtype,
        Self.rank,
        Self.static_layout,
        F,
        Self.OutFusion,
        Self.ComputeFusion,
    ]:
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def with_output_fusion[
        F: OutputFusion
    ](self) -> StaticTensorSpec[
        Self.dtype,
        Self.rank,
        Self.static_layout,
        Self.InFusion,
        F,
        Self.ComputeFusion,
    ]:
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def with_compute_fusion[
        F: ComputeOutputFusion
    ](self) -> StaticTensorSpec[
        Self.dtype,
        Self.rank,
        Self.static_layout,
        Self.InFusion,
        Self.OutFusion,
        F,
    ]:
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def to_layout(self) -> Layout:
        return Layout(
            coord_to_int_tuple[*Self.static_layout._shape_types](),
            coord_to_int_tuple[*Self.static_layout._stride_types](),
        )

    comptime static_size: Int = Layout(
        coord_to_int_tuple[*Self.static_layout._shape_types](),
        coord_to_int_tuple[*Self.static_layout._stride_types](),
    ).size()

    def get_internals(self) -> StaticTensorSpecInternal[Self.dtype, Self.rank]:
        """
        Returns a StaticTensorSpecInternal from a StaticTensorSpec.
        """
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
        }


@fieldwise_init
struct StaticTensorSpecInternal[dtype: DType, rank: Int](ImplicitlyCopyable):
    var alignment: Int
    var address_space: AddressSpace
    var exclusive: Bool

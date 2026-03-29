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

from std.builtin.variadics import Variadic
from buffer.dimlist import DimList
from layout import Layout
from layout.coord import (
    ComptimeInt,
    CoordLike,
    RuntimeInt,
    _CoordToDimList,
    _DimsToCoordLike,
    _IntToComptimeInt,
    coord_to_int_tuple,
)
from layout.tile_layout import Layout as TileLayout, TensorLayout, _RowMajor

from std.utils import IndexList


# ===----------------------------------------------------------------------=== #
# TileLayout helper aliases
# ===----------------------------------------------------------------------=== #

comptime _AllRuntimeInt[rank: Int] = Variadic.splat_type[
    Trait=CoordLike, count=rank, type=RuntimeInt[]
]
"""A variadic of `rank` RuntimeInt types."""

comptime _UnknownTileLayout[rank: Int] = TileLayout[
    shape_types=_AllRuntimeInt[rank],
    stride_types=_AllRuntimeInt[rank],
]
"""A fully-dynamic TileLayout where all shape and stride dims are RuntimeInt."""


comptime _RowMajorTileLayout[
    shape_types: Variadic.TypesOfTrait[CoordLike]
] = TileLayout[
    shape_types=shape_types,
    stride_types=_RowMajor[*shape_types],
]
"""A TileLayout with row-major strides derived from the given shape types."""


comptime _DimListToTileLayout[shape: DimList, strides: DimList] = TileLayout[
    shape_types=_DimsToCoordLike[DType.int, shape],
    stride_types=_DimsToCoordLike[DType.int, strides],
]
"""Convert a pair of DimLists to a TileLayout, using the _DimsToCoordLike
bridge. Static Dim values become ComptimeInt, dynamic Dims become RuntimeInt."""


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


def get_row_major_tensor_spec[
    dtype: DType, rank: Int, shape: DimList
]() -> StaticTensorSpec[
    dtype,
    rank,
    static_layout=_RowMajorTileLayout[_DimsToCoordLike[DType.int, shape]],
]:
    """
    Returns a row-major StaticTensorSpec with the specified shape and dtype.
    """
    return {align_of[dtype](), AddressSpace.GENERIC, False}


def get_row_major_tensor_spec_static[
    dtype: DType, rank: Int, *shape_dims: Int
]() -> StaticTensorSpec[
    dtype,
    rank,
    static_layout=_RowMajorTileLayout[_IntToComptimeInt[*shape_dims]],
]:
    """Returns a row-major StaticTensorSpec from compile-time Int dimensions.

    This is the DimList-free alternative to `get_row_major_tensor_spec`.
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
        simd_width: Int,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]):
        ...


trait ComputeOutputFusion(TrivialRegisterPassable):
    """Trait for compute-output fusion structs that transform values before
    storing."""

    def compute[
        dtype: DType,
        rank: Int,
        simd_width: Int,
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
        simd_width: Int,
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
        simd_width: Int,
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
    # Backward-compatible DimList aliases derived from the TileLayout types.
    comptime shape = _CoordToDimList[*Self.static_layout._shape_types]
    comptime strides = _CoordToDimList[*Self.static_layout._stride_types]

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
    def with_layout[
        new_rank: Int, new_shape: DimList, new_strides: DimList
    ](
        self,
    ) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=_DimListToTileLayout[new_shape, new_strides],
    ]:
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def with_layout_and_alignment[
        new_rank: Int, new_shape: DimList, new_strides: DimList
    ](self, new_alignment: Int) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=_DimListToTileLayout[new_shape, new_strides],
    ]:
        return {
            new_alignment,
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

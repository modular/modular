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

from std.sys import align_of, size_of
from std.sys.intrinsics import _type_is_eq

from buffer.dimlist import DimList
from layout import IntTuple, Layout

from std.utils import IndexList


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
]() -> StaticTensorSpec[dtype, rank, shape, shape.get_row_major_strides()]:
    """
    Returns a row-major StaticTensorSpec with the specified shape and dtype.
    """
    return {align_of[dtype](), AddressSpace.GENERIC, False}


def _get_unknown_tensor_spec[
    dtype: DType, rank: Int
]() -> StaticTensorSpec[
    dtype, rank, DimList.create_unknown[rank](), DimList.create_unknown[rank]()
]:
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
    shape: DimList,
    strides: DimList,
    InFusion: InputFusion = _NoFusionIn,
    OutFusion: OutputFusion = _NoFusionOut,
    ComputeFusion: ComputeOutputFusion = _NoComputeFusion,
](ImplicitlyCopyable):
    var alignment: Int
    var address_space: AddressSpace
    var exclusive: Bool

    def __init__(
        out self,
        alignment: Int,
        address_space: AddressSpace,
        exclusive: Bool,
    ):
        comptime assert Self.rank == len(Self.shape), "rank mismatch"
        comptime assert Self.rank == len(Self.strides), "rank mismatch"
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
    ) -> StaticTensorSpec[Self.dtype, Self.rank, Self.shape, Self.strides]:
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
    ](self) -> StaticTensorSpec[Self.dtype, new_rank, new_shape, new_strides]:
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
        }

    @always_inline
    def with_layout_and_alignment[
        new_rank: Int, new_shape: DimList, new_strides: DimList
    ](self, new_alignment: Int) -> StaticTensorSpec[
        Self.dtype, new_rank, new_shape, new_strides
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
        Self.shape,
        Self.strides,
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
        Self.shape,
        Self.strides,
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
        Self.shape,
        Self.strides,
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
        return Layout(IntTuple(self.shape), IntTuple(self.strides))

    comptime static_size: Int = Layout(
        IntTuple(Self.shape), IntTuple(Self.strides)
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

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

from std.collections import OptionalReg
from std.sys import align_of

from buffer.dimlist import DimList
from layout import IntTuple, Layout

from std.utils import IndexList


fn __mogg_intrinsic_attr(intrin: StaticString):
    return


# Register a DPS Kernel
@__mogg_intrinsic_attr("mogg.intrinsic_register")
fn register(name: StaticString):
    pass


# Indicates that a DPS Kernel is a view operation
@__mogg_intrinsic_attr("mogg.view_kernel")
fn view_kernel():
    return


fn get_row_major_tensor_spec[
    dtype: DType, rank: Int, shape: DimList
]() -> StaticTensorSpec[
    dtype, rank, shape, DimList.get_row_major_strides[shape]()
]:
    """
    Returns a row-major StaticTensorSpec with the specified shape and dtype.
    """
    return {align_of[dtype](), AddressSpace.GENERIC, False, None, None, None}


fn _get_unknown_tensor_spec[
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
        None,
        None,
        None,
    }


# Compile time Tensor information
struct StaticTensorSpec[
    dtype: DType,
    rank: Int,  # TODO: Should just use len(shape).
    shape: DimList,
    strides: DimList,
](ImplicitlyCopyable):
    # Represents the DimList type (not accessible from KGEN tests).
    comptime in_lambda_t = fn[simd_width: Int, element_alignment: Int = 1](
        IndexList[Self.rank]
    ) capturing -> SIMD[Self.dtype, simd_width]
    comptime out_lambda_t = fn[simd_width: Int, element_alignment: Int = 1](
        IndexList[Self.rank], SIMD[Self.dtype, simd_width]
    ) capturing -> None

    comptime out_compute_lambda_t = fn[
        simd_width: Int, element_alignment: Int = 1
    ](IndexList[Self.rank], SIMD[Self.dtype, simd_width]) capturing -> SIMD[
        Self.dtype, simd_width
    ]

    var alignment: Int
    var address_space: AddressSpace
    var exclusive: Bool

    var in_lambda: OptionalReg[Self.in_lambda_t]
    var out_lambda: OptionalReg[Self.out_lambda_t]
    var out_compute_lambda: OptionalReg[Self.out_compute_lambda_t]

    fn __init__(
        out self,
        alignment: Int,
        address_space: AddressSpace,
        exclusive: Bool,
        in_lambda: OptionalReg[Self.in_lambda_t],
        out_lambda: OptionalReg[Self.out_lambda_t],
        out_compute_lambda: OptionalReg[Self.out_compute_lambda_t],
    ):
        comptime assert Self.rank == len(Self.shape), "rank mismatch"
        comptime assert Self.rank == len(Self.strides), "rank mismatch"
        self.alignment = alignment
        self.address_space = address_space
        self.exclusive = exclusive
        self.in_lambda = in_lambda
        self.out_lambda = out_lambda
        self.out_compute_lambda = out_compute_lambda

    fn __init__(
        out self, internals: StaticTensorSpecInternal[Self.dtype, Self.rank]
    ):
        """
        Returns a StaticTensorSpec from a StaticTensorSpecInternal.
        """
        self.alignment = internals.alignment
        self.address_space = internals.address_space
        self.exclusive = internals.exclusive
        self.in_lambda = internals.in_lambda
        self.out_lambda = internals.out_lambda
        self.out_compute_lambda = internals.out_compute_lambda

    # This indirect approach to providing get_unknown is necessary because the
    # we don't want clients to have to bind rank and shapes to use this. Aliases
    # only require the parameters they USE to be bound, whereas static methods
    # require all parameters to be bound.
    comptime get_unknown = _get_unknown_tensor_spec[Self.dtype, Self.rank]

    @always_inline
    fn with_layout[
        new_rank: Int, new_shape: DimList, new_strides: DimList
    ](self) -> StaticTensorSpec[Self.dtype, new_rank, new_shape, new_strides]:
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
            None,
            None,
            None,
        }

    @always_inline
    fn with_layout_and_alignment[
        new_rank: Int, new_shape: DimList, new_strides: DimList
    ](self, new_alignment: Int) -> StaticTensorSpec[
        Self.dtype, new_rank, new_shape, new_strides
    ]:
        return {
            new_alignment,
            self.address_space,
            self.exclusive,
            None,
            None,
            None,
        }

    @always_inline
    fn to_layout(self) -> Layout:
        return Layout(IntTuple(self.shape), IntTuple(self.strides))

    fn get_internals(self) -> StaticTensorSpecInternal[Self.dtype, Self.rank]:
        """
        Returns a StaticTensorSpecInternal from a StaticTensorSpec.
        """
        return {
            self.alignment,
            self.address_space,
            self.exclusive,
            self.in_lambda,
            self.out_lambda,
            self.out_compute_lambda,
        }


@fieldwise_init
struct StaticTensorSpecInternal[dtype: DType, rank: Int](ImplicitlyCopyable):
    # Represents the DimList type (not accessible from KGEN tests).
    comptime in_lambda_t = fn[simd_width: Int, element_alignment: Int = 1](
        IndexList[Self.rank]
    ) capturing -> SIMD[Self.dtype, simd_width]
    comptime out_lambda_t = fn[simd_width: Int, element_alignment: Int = 1](
        IndexList[Self.rank], SIMD[Self.dtype, simd_width]
    ) capturing -> None

    comptime out_compute_lambda_t = fn[
        simd_width: Int, element_alignment: Int = 1
    ](IndexList[Self.rank], SIMD[Self.dtype, simd_width]) capturing -> SIMD[
        Self.dtype, simd_width
    ]

    var alignment: Int
    var address_space: AddressSpace
    var exclusive: Bool

    var in_lambda: OptionalReg[Self.in_lambda_t]
    var out_lambda: OptionalReg[Self.out_lambda_t]
    var out_compute_lambda: OptionalReg[Self.out_compute_lambda_t]

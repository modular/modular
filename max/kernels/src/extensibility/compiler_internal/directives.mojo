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

from collections import OptionalReg
from sys import align_of

from buffer.dimlist import DimList
from layout import IntTuple, Layout

from utils import IndexList


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


@always_inline
fn _row_major_strides[rank: Int](shape: DimList) -> DimList:
    """Return a `DimList` of strides for data laid out in row-major order, from
    a `DimList` representing the shape."""

    comptime if rank == 1:
        return 1
    elif rank == 2:
        return DimList(shape.get[1](), 1)
    elif rank == 3:
        return DimList(shape.get[2]() * shape.get[1](), shape.get[2](), 1)
    else:
        return -1


# Compile time Tensor information
struct StaticTensorSpec[
    dtype: DType,
    rank: Int,
](TrivialRegisterPassable):
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

    var shape: DimList
    var strides: DimList

    var alignment: Int
    var address_space: AddressSpace
    var exclusive: Bool

    var in_lambda: OptionalReg[Self.in_lambda_t]
    var out_lambda: OptionalReg[Self.out_lambda_t]
    var out_compute_lambda: OptionalReg[Self.out_compute_lambda_t]

    fn __init__(
        out self,
        shape: DimList,
        strides: DimList,
        alignment: Int,
        address_space: AddressSpace,
        exclusive: Bool,
        in_lambda: OptionalReg[Self.in_lambda_t],
        out_lambda: OptionalReg[Self.out_lambda_t],
        out_compute_lambda: OptionalReg[Self.out_compute_lambda_t],
    ):
        self.shape = shape
        self.strides = strides
        self.alignment = alignment
        self.address_space = address_space
        self.exclusive = exclusive
        self.in_lambda = in_lambda
        self.out_lambda = out_lambda
        self.out_compute_lambda = out_compute_lambda

    fn __init__(out self, shape: DimList):
        comptime assert Self.rank > 0, (
            "initializing `StaticTensorSpec` with just a shape only"
            " supports rank 1 to 3"
        )
        debug_assert(
            len(shape) == Self.rank,
            (
                "initialized `StaticTensorSpec` with a shape length not equal"
                "to the `rank` parameter"
            ),
        )
        self.shape = shape
        self.strides = _row_major_strides[Self.rank](shape)
        self.alignment = align_of[Self.dtype]()
        self.address_space = AddressSpace.GENERIC
        self.exclusive = False
        self.in_lambda = None
        self.out_lambda = None
        self.out_compute_lambda = None

    @staticmethod
    fn create_unknown() -> Self:
        """
        Returns a StaticTensorSpec with the specified type and rank with all
        fields dynamic or defaulted.
        """
        return Self(
            DimList.create_unknown[Self.rank](),
            DimList.create_unknown[Self.rank](),
            1,
            AddressSpace.GENERIC,
            True,
            OptionalReg[Self.in_lambda_t](None),
            OptionalReg[Self.out_lambda_t](None),
            OptionalReg[Self.out_compute_lambda_t](None),
        )

    @always_inline
    fn with_layout[
        new_rank: Int
    ](self, new_shape: DimList, new_strides: DimList) -> StaticTensorSpec[
        Self.dtype, new_rank
    ]:
        return StaticTensorSpec[Self.dtype, new_rank](
            new_shape,
            new_strides,
            self.alignment,
            self.address_space,
            self.exclusive,
            None,
            None,
            None,
        )

    @always_inline
    fn with_layout_and_alignment[
        new_rank: Int
    ](
        self, new_shape: DimList, new_strides: DimList, new_alignment: Int
    ) -> StaticTensorSpec[Self.dtype, new_rank]:
        return StaticTensorSpec[Self.dtype, new_rank](
            new_shape,
            new_strides,
            new_alignment,
            self.address_space,
            self.exclusive,
            None,
            None,
            None,
        )

    @always_inline
    fn to_layout(self) -> Layout:
        return Layout(IntTuple(self.shape), IntTuple(self.strides))

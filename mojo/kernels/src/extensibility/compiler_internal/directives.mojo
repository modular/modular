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

from collections import OptionalReg

from buffer.dimlist import DimList
from layout import IntTuple, Layout

from utils import IndexList, StaticTuple


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


# Compile time Tensor informations
@value
@register_passable("trivial")
struct StaticTensorSpec[
    type: DType,
    rank: Int,
]:
    # Represents the DimList type (not accessible from KGEN tests).
    alias in_lambda_t = fn[simd_width: Int] (IndexList[rank]) capturing -> SIMD[
        type, simd_width
    ]
    alias out_lambda_t = fn[simd_width: Int, element_alignment: Int = 1] (
        IndexList[rank], SIMD[type, simd_width]
    ) capturing -> None

    var shape: DimList
    var strides: DimList

    var alignment: Int
    var address_space: AddressSpace
    var exclusive: Bool

    var in_lambda: OptionalReg[Self.in_lambda_t]
    var out_lambda: OptionalReg[Self.out_lambda_t]

    fn __init__(
        out self,
        shape: DimList,
        strides: DimList,
        alignment: Int,
        address_space: AddressSpace,
        exclusive: Bool,
        in_lambda: OptionalReg[Self.in_lambda_t],
        out_lambda: OptionalReg[Self.out_lambda_t],
    ):
        self.shape = shape
        self.strides = strides
        self.alignment = alignment
        self.address_space = address_space
        self.exclusive = exclusive
        self.in_lambda = in_lambda
        self.out_lambda = out_lambda

    @staticmethod
    fn create_unknown() -> Self:
        """
        Returns a StaticTensorSpec with the specified type and rank with all
        fields dynamic or defaulted.
        """
        return Self(
            DimList.create_unknown[rank](),
            DimList.create_unknown[rank](),
            1,
            AddressSpace.GENERIC,
            True,
            OptionalReg[Self.in_lambda_t](None),
            OptionalReg[Self.out_lambda_t](None),
        )

    @always_inline
    fn with_layout[
        new_rank: Int
    ](self, new_shape: DimList, new_strides: DimList) -> StaticTensorSpec[
        type, new_rank
    ]:
        return StaticTensorSpec[type, new_rank](
            new_shape,
            new_strides,
            self.alignment,
            self.address_space,
            self.exclusive,
            None,
            None,
        )

    @always_inline
    fn to_layout(self) -> Layout:
        return Layout(IntTuple(self.shape), IntTuple(self.strides))

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

# DOC: develop/build-custom-ops.mdx

from std.sys import align_of

import extensibility

from max.algorithm import elementwise
from max.gpu.host import DeviceContext

from extensibility import (
    ElementwiseUnaryOp,
    Input,
    Output,
    Tensor,
    TileTensorable,
    get_kernel_simd_width,
)

from std.utils.coord import Coord
from std.utils.index import IndexList


@extensibility.register("add_one")
struct AddOne:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        # the context is needed for some GPU calls
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime output_alignment = type_of(output).alignment
        comptime assert type_of(x).rank == type_of(output).rank
        var output_tile = output.to_tile_tensor[dtype]()
        var x_tile = x.to_tile_tensor[dtype]().as_imm()

        @inline(.always)
        def elementwise_add_one[
            width: Int, alignment: Int = 1
        ](idx: Coord) {var}:
            var result = (
                x_tile.load[width=width, alignment=align_of[dtype]()](idx) + 1
            )
            output_tile.store[
                width=width,
                alignment=min(output_alignment, alignment * align_of[dtype]()),
            ](idx, result)

        elementwise[simd_width=get_kernel_simd_width[dtype, ctx.target]()](
            elementwise_add_one, output.shape(), ctx
        )


# You only need to implement this if you do not manually annotate
# output shapes in the graph.
@extensibility.register_shape_function("add_one")
def add_one_shape(
    x: Some[Tensor],
) raises -> IndexList[type_of(x).rank]:
    raise Error("NotImplemented")


# The graph compiler can fuse this op with the ops around it.
@extensibility.register("add_one_elementwise")
struct AddOneElementwise(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return x + 1

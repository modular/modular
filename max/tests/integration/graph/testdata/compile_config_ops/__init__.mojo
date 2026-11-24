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

from std.sys import align_of, get_defined_int

import extensibility
from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from std.logger import Logger
from extensibility import (
    Input,
    Output,
    Tensor,
    TileTensorable,
    get_kernel_simd_width,
)


from std.utils.coord import Coord
from std.utils.index import IndexList

comptime logger = Logger()


@extensibility.register("use_splitk_reduction_scheme")
struct UseSplitkReductionScheme:
    @staticmethod
    def execute(output: Some[Output & Tensor]):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert type_of(output).dtype == .int32
        comptime assert type_of(output).rank == 1
        comptime split_k_reduction_scheme = get_defined_int[
            "SPLITK_REDUCTION_SCHEME", 2
        ]()
        output.to_tile_tensor[.int32]().store[width=1](
            Coord(0), Int32(split_k_reduction_scheme)
        )


@extensibility.register("use_logger")
struct UseLogger:
    @staticmethod
    def execute(output: Some[Output & Tensor]):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert type_of(output).dtype == .int32
        comptime assert type_of(output).rank == 1
        logger.error("I'm a custom Mojo function!")
        output.to_tile_tensor[.int32]().store[width=1](
            Coord(0), Int32(logger.level._value)
        )


@extensibility.register("add_one_custom")
struct AddOneCustom:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(x).rank == type_of(output).rank
        var output_tile = output.to_tile_tensor()
        var x_tile = x.to_tile_tensor[dtype]().as_imm()

        def add_one[
            width: Int, element_alignment: Int = 1
        ](idx: Coord) {var output_tile, var x_tile}:
            output_tile.store[
                width=width,
                alignment=min(
                    type_of(output).alignment,
                    element_alignment * align_of[dtype](),
                ),
            ](
                idx,
                x_tile.load[width=width, alignment=align_of[dtype]()](idx) + 1,
            )

        elementwise[simd_width=get_kernel_simd_width[dtype, ctx.target]()](
            add_one, output.shape(), ctx
        )


@extensibility.register_shape_function("add_one_custom")
def add_one_custom_shape(
    x: Some[Tensor],
) raises -> IndexList[type_of(x).rank]:
    raise "NotImplemented"

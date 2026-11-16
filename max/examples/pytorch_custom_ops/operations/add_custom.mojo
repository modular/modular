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

import extensibility

from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from extensibility import (
    Input,
    Output,
    Tensor,
    TileTensorable,
    get_kernel_simd_width,
)

from std.utils.coord import Coord


@extensibility.register("add_constant_custom")
struct AddConstantCustom[value: Int]:
    @staticmethod
    def execute(
        outp: Some[Output & Tensor],
        x: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(outp), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(outp).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(x).rank == type_of(outp).rank
        var out_tile = outp.to_tile_tensor[dtype]()
        var x_tile = x.to_tile_tensor[dtype]().as_imm()

        @inline(.always)
        def add_constant[
            width: Int, element_alignment: Int = 1
        ](idx: Coord) {var out_tile, var x_tile}:
            out_tile.store[
                width=width,
                alignment=min(
                    type_of(outp).alignment,
                    element_alignment * align_of[dtype](),
                ),
            ](
                idx,
                x_tile.load[width=width, alignment=align_of[dtype]()](idx)
                + Scalar[dtype](Self.value),
            )

        elementwise[simd_width=get_kernel_simd_width[dtype, ctx.target]()](
            add_constant, outp.shape(), ctx
        )

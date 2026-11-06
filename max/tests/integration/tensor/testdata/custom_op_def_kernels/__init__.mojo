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
"""Test-fixture custom ops for `max.functional.custom_op.declare`.

Only `mxf353_filter` registers a shape function: its output row count is
data-dependent, so no dim of a declared `CustomOp` signature can size it.
"""
# MXF-353

from std.sys import align_of

import extensibility
from extensibility import (
    Input,
    Output,
    Tensor,
    TileTensorable,
    get_kernel_simd_width,
)
from layout import ImmTileTensor, TensorEngine, TensorLayout
from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList


@extensibility.register("mxf353_downsample")
struct Mxf353Downsample:
    """Copies the first half of the input along dim 1: `[n, w, c] -> [n, w // 2, c]`.
    """

    @staticmethod
    def execute(output: Some[Output & Tensor], x: Some[Input & Tensor]):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(x).rank == 3
        var out_tile = output.to_tile_tensor[dtype]()
        var x_tile = x.to_tile_tensor[dtype]().as_imm()
        for n in range(Int(output.shape()[0].value())):
            for w in range(Int(output.shape()[1].value())):
                for c in range(Int(output.shape()[2].value())):
                    out_tile.store(
                        Coord(n, w, c), x_tile.load[width=1](Coord(n, w, c))
                    )


@extensibility.register("mxf353_scale")
struct Mxf353Scale[factor: Int]:
    """Elementwise identity-with-param: multiplies every element by `factor`."""

    @staticmethod
    def execute(output: Some[Output & Tensor], x: Some[Input & Tensor]):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(x).rank == type_of(output).rank
        var out_tile = output.to_tile_tensor[dtype]()
        var out_ptr = out_tile.ptr
        var x_ptr = x.to_tile_tensor[dtype]().as_imm().ptr
        for i in range(out_tile.num_elements()):
            out_ptr[unsafe_offset=i] = x_ptr[unsafe_offset=i] * Scalar[dtype](
                Self.factor
            )


@extensibility.register("mxf353_q8_like")
struct Mxf353Q8Like:
    """Contracts two rank-2 inputs over their shared inner dim.

    `x: [m, k]`, `y: [n, k] -> out: [m, n]`, `out[i, j] = sum_l x[i, l] * y[j, l]`.
    """

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        y: Some[Input & Tensor],
    ):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert conforms_to(type_of(y), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(y).dtype == dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(x).rank == 2
        comptime assert type_of(y).rank == 2
        var out_tile = output.to_tile_tensor[dtype]()
        var x_tile = x.to_tile_tensor[dtype]().as_imm()
        var y_tile = y.to_tile_tensor[dtype]().as_imm()
        for i in range(Int(output.shape()[0].value())):
            for j in range(Int(output.shape()[1].value())):
                var acc = Scalar[dtype](0)
                for l in range(Int(x.shape()[1].value())):
                    acc += x_tile.load[width=1](Coord(i, l)) * y_tile.load[
                        width=1
                    ](Coord(j, l))
                out_tile.store(Coord(i, j), acc)


@extensibility.register("mxf353_group_pack")
struct Mxf353GroupPack[group_size: Int]:
    """Downsamples dim 1 by a compile-time group size:
    `output[i, j] = w[i, j * group_size]`.
    """

    @staticmethod
    def execute(output: Some[Output & Tensor], w: Some[Input & Tensor]):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(w), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(w).dtype == dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(w).rank == 2
        var out_tile = output.to_tile_tensor[dtype]()
        var w_tile = w.to_tile_tensor[dtype]().as_imm()
        for i in range(Int(output.shape()[0].value())):
            for j in range(Int(output.shape()[1].value())):
                out_tile.store(
                    Coord(i, j),
                    w_tile.load[width=1](Coord(i, j * Self.group_size)),
                )


@extensibility.register("mxf353_filter")
struct Mxf353Filter:
    """Copies the rows of a rank-2 input whose first column is positive:
    `x: [rows, cols] -> out: [k, cols]`. `k` is data-dependent, hence the
    `mxf353_filter_shape` shape function below.
    """

    @staticmethod
    def execute(output: Some[Output & Tensor], x: Some[Input & Tensor]):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(x).rank == 2
        var out_tile = output.to_tile_tensor[dtype]()
        var x_tile = x.to_tile_tensor[dtype]().as_imm()
        var out_row = 0
        for row in range(Int(x.shape()[0].value())):
            if x_tile.load[width=1](Coord(row, 0)) > 0:
                for col in range(Int(output.shape()[1].value())):
                    out_tile.store(
                        Coord(out_row, col),
                        x_tile.load[width=1](Coord(row, col)),
                    )
                out_row += 1


@extensibility.register_shape_function("mxf353_filter")
def mxf353_filter_shape(x: Some[TileTensorable]) raises -> IndexList[2]:
    """Counts the rows of `x` whose first column is positive: the
    data-dependent output row count for `mxf353_filter`.
    """
    var input = x.to_tile_tensor()
    var shape = coord_to_index_list(x.shape())
    var count = 0
    for row in range(shape[0]):
        if input[row, 0] > 0:
            count += 1
    return IndexList[2](count, shape[1])


# --- Device-portable NVFP4 pipeline ops -------------------------------------
#
# Naive by construction: every output element recomputes its own block scale
# rather than sharing one, so `quantize` needs no ordering between its two
# outputs and each op is a pure function of its inputs. That keeps all of
# them expressible through `foreach`, which dispatches to whichever device
# the op was placed on.
#
# NVFP4 here is emulated, not real NVFP4: values quantized to the 8 E2M1
# magnitudes with a sign bit, two per byte along K, plus one fp32 scale per
# 16-element block. Real NVFP4 uses E4M3 block scales plus a global scale.

comptime _NVFP4_BLOCK = 16
comptime _NVFP4_MAX = 6.0
comptime _NVFP4_LEVELS = SIMD[DType.float32, 8](
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
)


def _nvfp4_encode(value: Float32) -> UInt8:
    """Returns *value* as a 4-bit E2M1 code: sign in bit 3, magnitude below.

    Branches on the midpoints between representable magnitudes, so the
    result is the nearest level with ties resolved upward.
    """
    var mag = abs(value)
    var code: UInt8
    if mag < 0.25:
        code = 0
    elif mag < 0.75:
        code = 1
    elif mag < 1.25:
        code = 2
    elif mag < 1.75:
        code = 3
    elif mag < 2.5:
        code = 4
    elif mag < 3.5:
        code = 5
    elif mag < 5.0:
        code = 6
    else:
        code = 7
    return code | 8 if value < 0.0 else code


def _nvfp4_decode(code: UInt8) -> Float32:
    """Returns the value a 4-bit E2M1 *code* stands for."""
    var mag = _NVFP4_LEVELS[Int(code & 7)]
    return -mag if (code & 8) != 0 else mag


def _block_absmax[
    LT: TensorLayout, E: TensorEngine
](x: ImmTileTensor[.float32, LT, _, Engine=E], row: Int, block: Int) -> Float32:
    """Returns the largest magnitude in *block* of *row*."""
    var hi: Float32 = 0.0
    for i in range(_NVFP4_BLOCK):
        var v = abs(x.load[width=1](Coord(row, block * _NVFP4_BLOCK + i)))
        if v > hi:
            hi = v
    return hi


@extensibility.register("mxf353_nvfp4_quantize")
struct Mxf353NVFP4Quantize:
    """`x: [rows, k] -> packed: [rows, k // 2], scales: [rows, k // 16]`.

    Used twice in the same pipeline: once on the fp32 activations and
    weights, once to requantize the activation output.
    """

    @staticmethod
    def execute(
        packed: Some[Output & Tensor],
        scales: Some[Output & Tensor],
        x: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(packed), TileTensorable)
        comptime assert conforms_to(type_of(scales), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert type_of(packed).dtype == .uint8
        comptime assert type_of(scales).dtype == .float32
        comptime assert type_of(x).dtype == .float32
        comptime assert type_of(packed).rank == 2
        comptime assert type_of(scales).rank == 2
        comptime assert type_of(x).rank == 2
        var packed_tile = packed.to_tile_tensor[.uint8]()
        var scales_tile = scales.to_tile_tensor[.float32]()
        var x_tile = x.to_tile_tensor[.float32]().as_imm()

        @always_inline
        def block_scale[
            width: Int, alignment: Int = 1
        ](idx: Coord) {var x_tile, var scales_tile}:
            var at = coord_to_index_list(idx)
            var hi = _block_absmax(x_tile, at[0], at[1])
            var scale = hi / _NVFP4_MAX
            scales_tile.store[alignment=alignment * align_of[DType.float32]()](
                idx,
                SIMD[DType.float32, width](1.0 if scale == 0.0 else scale),
            )

        elementwise[1](block_scale, scales.shape(), ctx)

        @always_inline
        def pack[
            width: Int, alignment: Int = 1
        ](idx: Coord) {var x_tile, var packed_tile}:
            var at = coord_to_index_list(idx)
            var row = at[0]
            var lo_k = at[1] * 2
            # Both halves of a byte sit in one block, so one scale covers them.
            var hi = _block_absmax(x_tile, row, lo_k // _NVFP4_BLOCK)
            var scale = hi / _NVFP4_MAX
            if scale == 0.0:
                scale = 1.0
            var lo = x_tile.load[width=1](Coord(row, lo_k)) / scale
            var up = x_tile.load[width=1](Coord(row, lo_k + 1)) / scale
            var byte = _nvfp4_encode(lo) | (_nvfp4_encode(up) << 4)
            packed_tile.store[alignment=alignment * align_of[DType.uint8]()](
                idx, SIMD[DType.uint8, width](byte)
            )

        elementwise[1](pack, packed.shape(), ctx)


@extensibility.register("mxf353_nvfp4_matmul")
struct Mxf353NVFP4Matmul:
    """`a: [m, k/2] x b: [n, k/2] -> out: [m, n]`, dequantizing as it goes.

    Contracts over K with B laid out row-per-output-column, which is what
    the quantize op produces for a weight matrix.
    """

    @staticmethod
    def execute(
        out_tensor: Some[Output & Tensor],
        a_packed: Some[Input & Tensor],
        a_scales: Some[Input & Tensor],
        b_packed: Some[Input & Tensor],
        b_scales: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(out_tensor), TileTensorable)
        comptime assert conforms_to(type_of(a_packed), TileTensorable)
        comptime assert conforms_to(type_of(a_scales), TileTensorable)
        comptime assert conforms_to(type_of(b_packed), TileTensorable)
        comptime assert conforms_to(type_of(b_scales), TileTensorable)
        comptime assert type_of(out_tensor).dtype == .float32
        comptime assert type_of(a_packed).dtype == .uint8
        comptime assert type_of(a_scales).dtype == .float32
        comptime assert type_of(b_packed).dtype == .uint8
        comptime assert type_of(b_scales).dtype == .float32
        comptime assert type_of(out_tensor).rank == 2
        comptime assert type_of(a_packed).rank == 2
        comptime assert type_of(a_scales).rank == 2
        comptime assert type_of(b_packed).rank == 2
        comptime assert type_of(b_scales).rank == 2
        var out_tile = out_tensor.to_tile_tensor[.float32]()
        var a_packed_tile = a_packed.to_tile_tensor[.uint8]().as_imm()
        var a_scales_tile = a_scales.to_tile_tensor[.float32]().as_imm()
        var b_packed_tile = b_packed.to_tile_tensor[.uint8]().as_imm()
        var b_scales_tile = b_scales.to_tile_tensor[.float32]().as_imm()
        var bytes = Int(a_packed.shape()[1].value())

        @always_inline
        def dot[
            width: Int, alignment: Int = 1
        ](idx: Coord) {
            var out_tile,
            var a_packed_tile,
            var a_scales_tile,
            var b_packed_tile,
            var b_scales_tile,
            var bytes,
        }:
            var at = coord_to_index_list(idx)
            var row = at[0]
            var col = at[1]
            var acc: Float32 = 0.0
            for j in range(bytes):
                var block = (j * 2) // _NVFP4_BLOCK
                var a_scale = a_scales_tile.load[width=1](Coord(row, block))
                var b_scale = b_scales_tile.load[width=1](Coord(col, block))
                var a_byte = a_packed_tile.load[width=1](Coord(row, j))
                var b_byte = b_packed_tile.load[width=1](Coord(col, j))
                acc += (
                    _nvfp4_decode(a_byte & 15)
                    * a_scale
                    * _nvfp4_decode(b_byte & 15)
                    * b_scale
                )
                acc += (
                    _nvfp4_decode(a_byte >> 4)
                    * a_scale
                    * _nvfp4_decode(b_byte >> 4)
                    * b_scale
                )
            out_tile.store[alignment=alignment * align_of[DType.float32]()](
                idx, SIMD[DType.float32, width](acc)
            )

        elementwise[1](dot, out_tensor.shape(), ctx)


@extensibility.register("mxf353_relu")
struct Mxf353Relu:
    """Shape- and dtype-preserving ReLU, the pipeline's activation."""

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
        var out_tile = output.to_tile_tensor[dtype]()
        var x_tile = x.to_tile_tensor[dtype]().as_imm()

        @always_inline
        def relu[
            width: Int, alignment: Int = 1
        ](idx: Coord) {var x_tile, var out_tile}:
            comptime align = alignment * align_of[dtype]()
            var v = x_tile.load[width=width, alignment=align](idx)
            out_tile.store[alignment=align](idx, max(v, 0))

        elementwise[get_kernel_simd_width[dtype, ctx.target]()](
            relu, output.shape(), ctx
        )

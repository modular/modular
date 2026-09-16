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
"""Test-fixture custom ops for `max.experimental.custom.declare`.

Only `mxf353_filter` registers a shape function: its output row count is
data-dependent, so no dim of a declared `CustomOp` signature can size it.
"""
# MXF-353

import extensibility
from extensibility import InputTensor, OutputTensor, foreach
from max.gpu.host import DeviceContext
from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList


@extensibility.register("mxf353_downsample")
struct Mxf353Downsample:
    """Copies the first half of the input along dim 1: `[n, w, c] -> [n, w // 2, c]`.
    """

    @staticmethod
    def execute(
        output: OutputTensor[rank=3, ...],
        x: InputTensor[dtype=output.dtype, rank=3, ...],
    ):
        for n in range(output.dim_size(0)):
            for w in range(output.dim_size(1)):
                for c in range(output.dim_size(2)):
                    output[n, w, c] = x[n, w, c]


@extensibility.register("mxf353_scale")
struct Mxf353Scale[factor: Int]:
    """Elementwise identity-with-param: multiplies every element by `factor`."""

    @staticmethod
    def execute(
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
    ):
        var out_ptr = output.unsafe_ptr()
        var x_ptr = x.unsafe_ptr()
        for i in range(output.size()):
            out_ptr[unsafe_offset=i] = x_ptr[unsafe_offset=i] * Scalar[
                output.dtype
            ](Self.factor)


@extensibility.register("mxf353_q8_like")
struct Mxf353Q8Like:
    """Contracts two rank-2 inputs over their shared inner dim.

    `x: [m, k]`, `y: [n, k] -> out: [m, n]`, `out[i, j] = sum_l x[i, l] * y[j, l]`.
    """

    @staticmethod
    def execute(
        output: OutputTensor[rank=2, ...],
        x: InputTensor[dtype=output.dtype, rank=2, ...],
        y: InputTensor[dtype=output.dtype, rank=2, ...],
    ):
        for i in range(output.dim_size(0)):
            for j in range(output.dim_size(1)):
                var acc = Scalar[output.dtype](0)
                for l in range(x.dim_size(1)):
                    acc += x[i, l] * y[j, l]
                output[i, j] = acc


@extensibility.register("mxf353_group_pack")
struct Mxf353GroupPack[group_size: Int]:
    """Downsamples dim 1 by a compile-time group size:
    `output[i, j] = w[i, j * group_size]`.
    """

    @staticmethod
    def execute(
        output: OutputTensor[rank=2, ...],
        w: InputTensor[dtype=output.dtype, rank=2, ...],
    ):
        for i in range(output.dim_size(0)):
            for j in range(output.dim_size(1)):
                output[i, j] = w[i, j * Self.group_size]


@extensibility.register("mxf353_filter")
struct Mxf353Filter:
    """Copies the rows of a rank-2 input whose first column is positive:
    `x: [rows, cols] -> out: [k, cols]`. `k` is data-dependent, hence the
    `mxf353_filter_shape` shape function below.
    """

    @staticmethod
    def execute(
        output: OutputTensor[rank=2, ...],
        x: InputTensor[dtype=output.dtype, rank=2, ...],
    ):
        var out_row = 0
        for row in range(x.dim_size(0)):
            if x[row, 0] > 0:
                for col in range(output.dim_size(1)):
                    output[out_row, col] = x[row, col]
                out_row += 1


@extensibility.register_shape_function("mxf353_filter")
def mxf353_filter_shape(x: InputTensor) raises -> IndexList[2]:
    """Counts the rows of `x` whose first column is positive: the
    data-dependent output row count for `mxf353_filter`.
    """
    var count = 0
    for row in range(x.dim_size(0)):
        if x[row, 0] > 0:
            count += 1
    return IndexList[2](count, x.dim_size(1))


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


def _block_absmax(
    x: InputTensor[dtype=DType.float32, rank=2, ...], row: Int, block: Int
) -> Float32:
    """Returns the largest magnitude in *block* of *row*."""
    var hi: Float32 = 0.0
    for i in range(_NVFP4_BLOCK):
        var v = abs(x.load[1](IndexList[2](row, block * _NVFP4_BLOCK + i))[0])
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
    def execute[
        target: StaticString
    ](
        packed: OutputTensor[dtype=DType.uint8, rank=2, ...],
        scales: OutputTensor[dtype=DType.float32, rank=2, ...],
        x: InputTensor[dtype=DType.float32, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        @always_inline
        def block_scale[
            width: Int
        ](idx: Coord) capturing -> SIMD[DType.float32, width]:
            var at = coord_to_index_list(idx)
            var hi = _block_absmax(x, at[0], at[1])
            var scale = hi / _NVFP4_MAX
            return SIMD[DType.float32, width](1.0 if scale == 0.0 else scale)

        foreach[block_scale, target=target, simd_width=1](scales, ctx)

        @always_inline
        def pack[width: Int](idx: Coord) capturing -> SIMD[DType.uint8, width]:
            var at = coord_to_index_list(idx)
            var row = at[0]
            var lo_k = at[1] * 2
            # Both halves of a byte sit in one block, so one scale covers them.
            var hi = _block_absmax(x, row, lo_k // _NVFP4_BLOCK)
            var scale = hi / _NVFP4_MAX
            if scale == 0.0:
                scale = 1.0
            var lo = x.load[1](IndexList[2](row, lo_k))[0] / scale
            var up = x.load[1](IndexList[2](row, lo_k + 1))[0] / scale
            var byte = _nvfp4_encode(lo) | (_nvfp4_encode(up) << 4)
            return SIMD[DType.uint8, width](byte)

        foreach[pack, target=target, simd_width=1](packed, ctx)


@extensibility.register("mxf353_nvfp4_matmul")
struct Mxf353NVFP4Matmul:
    """`a: [m, k/2] x b: [n, k/2] -> out: [m, n]`, dequantizing as it goes.

    Contracts over K with B laid out row-per-output-column, which is what
    the quantize op produces for a weight matrix.
    """

    @staticmethod
    def execute[
        target: StaticString
    ](
        out_tensor: OutputTensor[dtype=DType.float32, rank=2, ...],
        a_packed: InputTensor[dtype=DType.uint8, rank=2, ...],
        a_scales: InputTensor[dtype=DType.float32, rank=2, ...],
        b_packed: InputTensor[dtype=DType.uint8, rank=2, ...],
        b_scales: InputTensor[dtype=DType.float32, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        @always_inline
        def dot[width: Int](idx: Coord) capturing -> SIMD[DType.float32, width]:
            var at = coord_to_index_list(idx)
            var row = at[0]
            var col = at[1]
            var bytes = a_packed.dim_size(1)
            var acc: Float32 = 0.0
            for j in range(bytes):
                var block = (j * 2) // _NVFP4_BLOCK
                var a_scale = a_scales.load[1](IndexList[2](row, block))[0]
                var b_scale = b_scales.load[1](IndexList[2](col, block))[0]
                var a_byte = a_packed.load[1](IndexList[2](row, j))[0]
                var b_byte = b_packed.load[1](IndexList[2](col, j))[0]
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
            return SIMD[DType.float32, width](acc)

        foreach[dot, target=target, simd_width=1](out_tensor, ctx)


@extensibility.register("mxf353_relu")
struct Mxf353Relu:
    """Shape- and dtype-preserving ReLU, the pipeline's activation."""

    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        @always_inline
        def relu[width: Int](idx: Coord) capturing -> SIMD[x.dtype, width]:
            var v = x.load[width](idx)
            return max(v, 0)

        foreach[relu, target=target](output, ctx)

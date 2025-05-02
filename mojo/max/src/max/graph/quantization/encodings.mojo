# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Implementations of quantization encodings."""

# The Qn_K quantization implementations are based upon the corresponding
# implementations in llama.cpp, licensed under the MIT license:
#
# MIT License
#
# Copyright (c) 2023-2024 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import InlineArray, Optional
from math import sqrt
from sys import simdwidthof, sizeof

from algorithm import vectorize
from max.tensor import Tensor, TensorShape
from memory import UnsafePointer

from .quantization_encoding import QuantizationEncoding


struct BFloat16Encoding(QuantizationEncoding):
    """The bfloat16 quantization encoding.

    Like float32, the bfloat16 encoding uses 8 bits to store the exponent
    value, so it has the same numeric range as float32. However, it has just 7
    bits for the mantissa (compared to 23 bits available in float32), so it has
    less precision for the fractional part. This is often a better trade-off
    for ML applications, compared to traditional float16, which has less
    numeric range because it uses only 5 bits to store the exponent (though it
    has better precision with 10 bits for the mantissa).

    Because this holds the quantized data in a special packing format, it
    currently does not print float values at runtime—it's just a bag of
    bits in uint8 format.
    """

    @staticmethod
    def quantize(tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision input tensor to bfloat16.

        Only supports quantizing from float16 and float32, using a direct
        elementwise cast.

        Args:
            tensor: Full-precision tensor to quantize to bfloat16.

        Returns:
            Quantized bfloat16 tensor. The tensor datatype is `uint8`
            because this is simply a byte buffer. Each scalar is actually
            encoded into two bytes (16-bits).
        """
        if not tensor.num_elements():
            return Tensor[DType.uint8]()

        # Quantize to bfloat16 via elementwise cast.
        quantized = tensor.astype[DType.bfloat16]()

        # Compute bytes buffer shape as the tensor shape with 2 bytes per
        # bfloat16 element in the innermost dimension.
        # Note that this implies the storage is row major.
        tensor_shape = tensor.shape()
        buff_dims = List[Int, hint_trivial_type=True]()
        for i in range(tensor_shape.rank() - 1):
            buff_dims.append(tensor_shape[i])

        buff_dims.append(2 * tensor_shape[-1])

        return Tensor(
            TensorShape(buff_dims^),
            quantized._steal_ptr().bitcast[Scalar[DType.uint8]](),
        )

    @staticmethod
    fn id() -> String:
        """Identifier for the bfloat16 quantized encoding."""
        return "bfloat16"


struct Float32Encoding(QuantizationEncoding):
    """The float32 quantization encoding.

    This encoding is essentially an identity operation.
    It exists in order to be a default case for code that is generic over
    quantization encoding.
    """

    @staticmethod
    def quantize(_tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Unimplemented quantize method for float32.

        Since float32 is an identity encoding, it shouldn't define a quantize method.
        In particular, float32 values should be used with non-quantized ops,
        which expect dtype float32.
        This is in contrast to quantized ops, which expect dtype uint8 operands.
        So raise an exception here to avoid accidental bugs.
        """
        raise "float32 quantize intentionally not implemented"

    @staticmethod
    fn id() -> String:
        """Identifier for the float32 quantized encoding."""
        return "float32"


@value
struct _BlockQ40:
    """4-bit quantization.

    Constraints:
        The data layout must exactly match `block_q4_0` from ggml-quants.h.
    """

    alias QK4_0 = 32
    """Number of elements per Q4_0 block."""

    var d: Float16
    """Delta."""
    var qs: InlineArray[UInt8, Self.QK4_0 // 2]
    """Nibbles / quants."""

    def __init__(
        out self,
        d: Float16,
        qs: InlineArray[UInt8, Self.QK4_0 // 2],
    ):
        constrained[sizeof[Self]() == sizeof[Float16]() + (Self.QK4_0 // 2)]()

        self.d = d
        self.qs = qs

    @staticmethod
    fn elements_per_block() -> Int:
        """Returns the number of elements per Q4_0 block."""
        return Self.QK4_0


struct Q4_0Encoding(QuantizationEncoding):
    """The Q4_0 quantization encoding.

    Q4_0 is a block quantization scheme originally designed for
    [GGML](https://ggml.ai) in which each element (number) is reduced to an
    unsigned, fixed-point, 4-bit value. Multiple quantized elements are packed
    together in a block, all using the same float16 scale.

    The packing scheme requires that the innermost dimension is a factor of 32.
    When the tensor is quantized to Q4_0, each block of 32 scalar values is
    packed into 18 bytes. The first two bytes specify the float16 quantization
    scale, and the other 16 bytes hold the 32 values (one byte holds two 4-bit
    values).

    Because this holds the quantized data in a special packing format, it
    currently does not print float values at runtime—it's just a bag of
    bits in uint8 format.
    """

    @staticmethod
    def quantize(tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision tensor to Q4_0.

        Args:
            tensor: Full-precision tensor to quantize. The innermost dimension
                of the tensor must be a factor of 32.

        Returns:
            Quantized Q4_0 tensor. The tensor datatype is `uint8` because this
            is simply a bytes buffer. Each scalar is actually stored with 4 bits.

        Raises:
            If the last dimension size is not a factor of 32.
        """
        if not tensor.num_elements():
            return Tensor[DType.uint8]()

        alias elems_per_block = _BlockQ40.elements_per_block()
        tensor_shape = tensor.shape()
        cols = tensor_shape[-1]
        if cols % elems_per_block != 0:
            raise "num elements in row must be a multiple of Q4_0 block size"

        # Q4_0 quantizes row-wise, so compute the output shape as the same as
        # the input shape, except with the last dimension packed as _BlockQ40.
        buff_dims = List[Int, hint_trivial_type=True]()
        for i in range(tensor_shape.rank() - 1):
            buff_dims.append(tensor_shape[i])
        # Compute number of bytes in last block, which is packed.
        buff_dims.append((cols // elems_per_block) * sizeof[_BlockQ40]())

        # Allocate the output buffer and interpret it as an array of _BlockQ40.
        quantized = Tensor[DType.uint8](TensorShape(buff_dims^))
        quantized_ptr = rebind[UnsafePointer[_BlockQ40]](quantized.unsafe_ptr())

        tensor_ptr = tensor.unsafe_ptr()

        # Iterate over all blocks in the tensor.
        for block_idx in range(tensor.num_elements() // elems_per_block):
            # Track max and abs(max) over the block.
            block_abs_max = Float32(0.0)
            block_max = Float32(0.0)

            # Find and set the max over the block.
            for i in range(elems_per_block):
                val = tensor_ptr[block_idx * elems_per_block + i]
                if block_abs_max < abs(val):
                    block_abs_max = abs(val)
                    block_max = val

            # Compute float16 scale and its inverse to scale elems to [-8, 8].
            d = block_max / -8
            inv_d = 1.0 / d if d else Float32(0)

            # Write scale to output buffer.
            quantized_ptr[block_idx].d = d.cast[DType.float16]()

            @parameter
            for elem_idx in range(elems_per_block // 2):
                # x0: first half of block.
                x0 = inv_d * tensor[block_idx * elems_per_block + elem_idx]
                # x1: second half of block.
                x1 = (
                    inv_d
                    * tensor[
                        block_idx * elems_per_block
                        + elems_per_block // 2
                        + elem_idx
                    ]
                )

                # Offset by 8.5 and clamp to [0, 15].
                x0_int4 = min(15, (x0 + 8.5).cast[DType.int8]())
                x1_int4 = min(15, (x1 + 8.5).cast[DType.int8]())

                # Write first half elem to low bits, second half elem to high.
                low_bits = x0_int4.cast[DType.uint8]()
                high_bits = x1_int4.cast[DType.uint8]() << 4
                quantized_ptr[block_idx].qs[elem_idx] = low_bits | high_bits

        return quantized

    @staticmethod
    fn id() -> String:
        """Identifier for the Q4_0 quantized encoding."""
        return "q4_0"


# Note that there is a compile definition in ggml-quants.h that allows setting
# `QK_K=64`, which is useful for models with rows unaligned to 256 bits.
alias QK_K = 256
"""Size of superblock quantized elements, in bytes."""

alias K_SCALE_SIZE = 12
"""Size of superblock scales and mins, in bytes."""


def _sum_squares[count: Int](ptr: UnsafePointer[Float32]) -> Float32:
    sum_squares = Float32(0.0)

    @parameter
    fn agg[width: Int](i: Int):
        var xs = ptr.load[width=width](i)
        sum_squares += (xs * xs).reduce_add()

    vectorize[agg, simdwidthof[DType.float32](), size=count]()
    return sum_squares


def _pick_weights_q4_k_q5_k[
    count: Int
](ptr: UnsafePointer[Float32], rms: Float32) -> InlineArray[Float32, count]:
    weights = InlineArray[Float32, count](uninitialized=True)

    @parameter
    fn fill[width: Int](i: Int):
        var xs = ptr.load[width=width](i)
        weights.unsafe_ptr().store(i, rms + abs(xs))

    vectorize[fill, simdwidthof[DType.float32](), size=count]()
    return weights


def _find_extrema[
    count: Int
](ptr: UnsafePointer[Float32]) -> Tuple[Float32, Float32]:
    alias prefix_size = min(count, simdwidthof[Float32]())

    prefix = ptr.load[width=prefix_size]()
    min_value = prefix.reduce_min()
    max_value = prefix.reduce_max()

    @parameter
    fn agg_rest[width: Int](i: Int):
        var piece = ptr.load[width=width](i + prefix_size)
        min_value = min(min_value, piece.reduce_min())
        max_value = max(max_value, piece.reduce_max())

    vectorize[agg_rest, simdwidthof[Float32](), size = count - prefix_size]()
    return (min_value, max_value)


def _find_amax[
    count: Int
](ptr: UnsafePointer[Float32]) -> Tuple[Float32, Float32]:
    """Find absolute maximum and non-absolute value of the absolute maximum."""
    min, max = _find_extrema[count](ptr)
    abs_min = abs(min)
    abs_max = abs(max)
    if abs_max > abs_min:
        return (abs_max, max)
    else:
        return (abs_min, min)


fn _unsigned_symmetric_quantize[
    nmax: Int, size: Int
](x: SIMD[DType.float32, size], *, iscale: Float32) -> SIMD[DType.uint8, size]:
    # Clamping occurs in float32 instead of uint8 to avoid undefined behavior
    # if rounded result is out of bounds of uint8.
    return round(x * iscale).clamp(0, nmax).cast[DType.uint8]()


fn _biased_symmetric_quantize[
    nmax: Int, size: Int
](x: SIMD[DType.float32, size], *, iscale: Float32) -> SIMD[DType.uint8, size]:
    # Clamping occurs in float32 instead of uint8 to avoid undefined behavior
    # if rounded result is out of bounds of uint8.
    return (
        (round(x * iscale) + (nmax + 1) // 2).clamp(0, nmax).cast[DType.uint8]()
    )


def _biased_symmetric_quantize[
    count: Int, nmax: Int
](ptr: UnsafePointer[Float32], *, iscale: Float32) -> InlineArray[UInt8, count]:
    quants = InlineArray[UInt8, count](uninitialized=True)

    @parameter
    fn quantize_piece[width: Int](i: Int):
        var x = ptr.load[width=width](i)
        var quant = _biased_symmetric_quantize[nmax](x, iscale=iscale)
        quants.unsafe_ptr().store(i, quant)

    vectorize[quantize_piece, simdwidthof[DType.float32](), size=count]()
    return quants


fn _unbiased_symmetric_quantize[
    nmax: Int, size: Int
](x: SIMD[DType.float32, size], *, iscale: Float32) -> SIMD[DType.int8, size]:
    # Clamping occurs in float32 instead of uint8 to avoid undefined behavior
    # if rounded result is out of bounds of uint8.
    return (
        (round(x * iscale))
        .clamp(-(nmax + 1) // 2, nmax // 2)
        .cast[DType.int8]()
    )


def _unbiased_symmetric_quantize[
    count: Int, nmax: Int
](ptr: UnsafePointer[Float32], *, iscale: Float32) -> InlineArray[Int8, count]:
    quants = InlineArray[Int8, count](uninitialized=True)

    @parameter
    fn quantize_piece[width: Int](i: Int):
        var x = ptr.load[width=width](i)
        var quant = _unbiased_symmetric_quantize[nmax](x, iscale=iscale)
        quants.unsafe_ptr().store(i, quant)

    vectorize[quantize_piece, simdwidthof[DType.float32](), size=count]()
    return quants


def _unbiased_symmetric_qdq[
    count: Int, nmax: Int
](
    ptr: UnsafePointer[Float32], *, scale: Float32, iscale: Float32
) -> InlineArray[Int8, count]:
    quants = InlineArray[Int8, count](uninitialized=True)

    @parameter
    fn qdq_piece[width: Int](i: Int):
        var x = ptr.load[width=width](i)
        var quant = _unbiased_symmetric_quantize[nmax](x, iscale=iscale)
        quants.unsafe_ptr().store(i, quant)
        var dequant = quant.cast[DType.float32]() * scale
        ptr.store(i, dequant)

    vectorize[qdq_piece, simdwidthof[DType.float32](), size=count]()
    return quants


fn _asymmetric_quantize[
    nmax: Int, size: Int
](x: SIMD[DType.float32, size], *, iscale: Float32, min: Float32) -> SIMD[
    DType.uint8, size
]:
    # Clamping occurs in float32 instead of uint8 to avoid undefined behavior
    # if rounded result is out of bounds of uint8.
    return round((x - min) * iscale).clamp(0, Float32(nmax)).cast[DType.uint8]()


def _asymmetric_quantize[
    count: Int, nmax: Int
](ptr: UnsafePointer[Float32], *, iscale: Float32, min: Float32) -> InlineArray[
    UInt8, count
]:
    quants = InlineArray[UInt8, count](uninitialized=True)

    @parameter
    fn quantize_piece[width: Int](i: Int):
        var x = ptr.load[width=width](i)
        var quant = _asymmetric_quantize[nmax](x, iscale=iscale, min=min)
        quants.unsafe_ptr().store(i, quant)

    vectorize[quantize_piece, simdwidthof[DType.float32](), size=count]()
    return quants


def _measure_asymmetric_quant_error[
    count: Int
](
    ptr: UnsafePointer[Float32],
    *,
    quants: UnsafePointer[UInt8],
    weights: UnsafePointer[Float32],
    scale: Float32,
    min_x: Float32,
) -> Float32:
    """Compute the error a set of quants have against a ground truth.

    The quants must already have been computed.  Dequantization parameters must
    be provided.
    """
    error = Float32(0.0)

    @parameter
    fn agg[width: Int](i: Int):
        var x = ptr.load[width=width](i)
        var quant = quants.load[width=width](i)
        var weight = weights.load[width=width](i)
        var dequantized = quant.cast[DType.float32]() * scale + min_x
        var diff = dequantized - x
        error += (weight * (diff * diff)).reduce_add()

    vectorize[agg, simdwidthof[DType.float32](), size=count]()
    return error


def _measure_asymmetric_quant_error[
    count: Int, nmax: Int
](
    ptr: UnsafePointer[Float32],
    *,
    weights: UnsafePointer[Float32],
    scale: Float32,
    iscale: Float32,
    min_x: Float32,
) -> Float32:
    """Compute the error a set of quantization parameters would imply on data.

    Quants are computed internally and discarded after evaluation.  Both
    quantization and dequantization parameters must be provided.
    """
    error = Float32(0.0)

    @parameter
    fn agg[width: Int](i: Int):
        var x = ptr.load[width=width](i)
        var weight = weights.load[width=width](i)
        var quantized = _asymmetric_quantize[nmax](x, iscale=iscale, min=min_x)
        var dequantized = quantized.cast[DType.float32]() * scale + min_x
        var diff = dequantized - x
        error += (weight * (diff * diff)).reduce_add()

    vectorize[agg, simdwidthof[DType.float32](), size=count]()
    return error


@value
struct _AsymmetricDequantParameters:
    var scale: Float32
    var min: Float32


def _refit_asymmetric[
    count: Int
](
    ptr: UnsafePointer[Float32],
    quants: UnsafePointer[UInt8],
    weights: UnsafePointer[Float32],
) -> Optional[_AsymmetricDequantParameters]:
    sum_w = Float32(0.0)
    sum_x = Float32(0.0)
    sum_l = Float32(0.0)
    sum_l2 = Float32(0.0)
    sum_xl = Float32(0.0)

    @parameter
    fn agg[width: Int](i: Int):
        var x = ptr.load[width=width](i)
        var l = quants.load[width=width](i).cast[DType.float32]()
        var w = weights.load[width=width](i)
        sum_w += w.reduce_add()
        sum_x += (w * x).reduce_add()
        sum_l += (w * l).reduce_add()
        sum_l2 += (w * l * l).reduce_add()
        sum_xl += (w * l * x).reduce_add()

    vectorize[agg, simdwidthof[DType.float32](), size=count]()

    D = sum_w * sum_l2 - sum_l * sum_l
    if D <= 0:
        return None
    scale = (sum_w * sum_xl - sum_x * sum_l) / D
    min = (sum_l2 * sum_x - sum_l * sum_xl) / D
    return _AsymmetricDequantParameters(scale=scale, min=min)


def _pick_subblock_scale_min_q4_k_q5_k[
    count: Int, *, nmax: Int, rmin: Float32, rdelta: Float32, nstep: Int
](ptr: UnsafePointer[Float32]) -> _AsymmetricDequantParameters:
    rms = sqrt(_sum_squares[count](ptr) / Float32(count))
    weights = _pick_weights_q4_k_q5_k[count](ptr, rms)
    min_x, max_x = _find_extrema[count](ptr)
    if min_x > 0:
        min_x = 0
    if min_x == max_x:
        return _AsymmetricDequantParameters(scale=0.0, min=-min_x)
    iscale = Float32(nmax) / (max_x - min_x)
    scale = 1 / iscale
    best_error = _measure_asymmetric_quant_error[count, nmax](
        ptr,
        weights=weights.unsafe_ptr(),
        scale=scale,
        iscale=iscale,
        min_x=min_x,
    )
    for step in range(nstep + 1):
        trial_iscale = (rmin + step * rdelta + nmax) / (max_x - min_x)
        trial_quants = _asymmetric_quantize[count, nmax](
            ptr, min=min_x, iscale=trial_iscale
        )
        maybe_refitted = _refit_asymmetric[count](
            ptr, quants=trial_quants.unsafe_ptr(), weights=weights.unsafe_ptr()
        )
        if not maybe_refitted:
            continue
        refitted = maybe_refitted.value()
        # It's quite weird to be doing this error computation with the new
        # min/scale parameters but the old quants, but that's what
        # llama.cpp does, so we also do it in order to remain bug-for-bug
        # compatible.
        error = _measure_asymmetric_quant_error[count](
            ptr,
            quants=trial_quants.unsafe_ptr(),
            weights=weights.unsafe_ptr(),
            scale=refitted.scale,
            min_x=refitted.min,
        )

        if error < best_error:
            # min_x is, aside from the return value, also used for
            # trial_iscale computation in future trials, so it _probably_
            # should not be changed here, but that's what llama.cpp is
            # doing, and we're trying to be bug-for-bug compatible, so we
            # change it anyway.
            scale = refitted.scale
            min_x = refitted.min
            best_error = error

    return _AsymmetricDequantParameters(scale=scale, min=min_x)


@value
struct _BlockQ4K:
    """4-bit quantization.

    8 blocks of 32 elements each.
    Weights are represented as `x = a * q + b`.
    Effectively 4.5 bits per weight.

    Constraints:
        The data layout must exactly match `block_q4_K` from ggml-quants.h.
    """

    var d: Float16
    """Super-block scale for quantized scales."""

    var dmin: Float16
    """Super-block scale for quantized mins."""

    var scales: InlineArray[UInt8, K_SCALE_SIZE]
    """Scales and mins, quantized with 6 bits."""

    var qs: InlineArray[UInt8, QK_K // 2]
    """4-bit quants."""

    def __init__(
        out self,
        d: Float16,
        dmin: Float16,
        scales: InlineArray[UInt8, K_SCALE_SIZE],
        qs: InlineArray[UInt8, QK_K // 2],
    ):
        constrained[
            sizeof[Self]() == 2 * sizeof[Float16]() + K_SCALE_SIZE + QK_K // 2
        ]()

        self.d = d
        self.dmin = dmin
        self.scales = scales
        self.qs = qs

    @staticmethod
    fn elements_per_superblock() -> Int:
        """Returns the number of elements per Q4_K superblock."""
        return QK_K

    @staticmethod
    fn elements_per_subblock() -> Int:
        """Returns the number of elements per Q4_K subblock."""
        return 32

    @staticmethod
    fn num_subblocks() -> Int:
        """Returns the number of subblocks per Q4_K superblock."""
        return Self.elements_per_superblock() // Self.elements_per_subblock()


def _quantize_superblock_params[
    num_subblocks: Int
](
    *,
    mut scales: InlineArray[Float32, num_subblocks],
    mut mins: InlineArray[Float32, num_subblocks],
) -> Tuple[Float16, Float16, InlineArray[UInt8, K_SCALE_SIZE]]:
    constrained[
        num_subblocks == 8, "K scale packing only designed for 8 subblocks"
    ]()
    _, max_scale = _find_extrema[num_subblocks](scales.unsafe_ptr())
    # Mins are expected to usually be negative, so we actually take the minimum
    # here, invert the sign, and call _that_ the maximum.
    max_min, _ = _find_extrema[num_subblocks](mins.unsafe_ptr())
    max_min = -max_min
    if max_scale < 0:
        max_scale = 0
    if max_min < 0:
        max_min = 0
    alias nmax = 63
    d = (max_scale / nmax).cast[DType.float16]()
    dmin = (max_min / nmax).cast[DType.float16]()
    inv_scale = nmax / max_scale
    # inv_min's sign inverted here to invert the signs of 'min' below.
    inv_min = -nmax / max_min
    d_f32 = d.cast[DType.float32]()
    # neg_dmin_f32 inverting sign to undo the inversion from inv_min.
    neg_dmin_f32 = -dmin.cast[DType.float32]()
    # Layout of _BlockQ4K 'scales' array:
    # 76543210
    # S4[-S0-]
    # S5[-S1-]
    # S6[-S2-]
    # S7[-S3-]
    # M4[-M0-]
    # M5[-M1-]
    # M6[-M2-]
    # M7[-M3-]
    # [M4][S4]
    # [M5][S5]
    # [M6][S6]
    # [M7][S7]
    quant_scales = InlineArray[UInt8, K_SCALE_SIZE](uninitialized=True)
    for subblock_idx in range(num_subblocks):
        q_scale = _unsigned_symmetric_quantize[nmax](
            scales[subblock_idx], iscale=inv_scale
        )
        q_min = _unsigned_symmetric_quantize[nmax](
            mins[subblock_idx], iscale=inv_min
        )
        if subblock_idx < 4:
            quant_scales[subblock_idx] = q_scale
            quant_scales[subblock_idx + 4] = q_min
        else:
            quant_scales[subblock_idx + 4] = (q_scale & 0xF) | (
                q_min & 0xF
            ) << 4
            quant_scales[subblock_idx - 4] |= q_scale >> 4 << 6
            quant_scales[subblock_idx] |= q_min >> 4 << 6
        scales[subblock_idx] = q_scale.cast[DType.float32]() * d_f32
        mins[subblock_idx] = q_min.cast[DType.float32]() * neg_dmin_f32
    return (d, dmin, quant_scales)


def _qn_k_quantize[
    BlockType: Copyable & Movable,
    elems_per_superblock: Int,
    quantize_superblock: fn (UnsafePointer[Float32]) raises -> BlockType,
](tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
    if not tensor.num_elements():
        return Tensor[DType.uint8]()

    tensor_shape = tensor.shape()
    cols = tensor_shape[-1]
    if cols % elems_per_superblock != 0:
        raise "num elements in row must be a multiple of superblock size"

    # Qn_K quantizes row-wise, so compute the output shape as the same as
    # the input shape, except with the last dimension packed as BlockType.
    buff_dims = List[Int, hint_trivial_type=True]()
    for i in range(tensor_shape.rank() - 1):
        buff_dims.append(tensor_shape[i])
    # Compute number of bytes in last dimension, which is packed.
    buff_dims.append((cols // elems_per_superblock) * sizeof[BlockType]())

    # Allocate the output buffer and interpret it as an array of BlockType.
    quantized = Tensor[DType.uint8](TensorShape(buff_dims^))
    quantized_ptr = rebind[UnsafePointer[BlockType]](quantized.unsafe_ptr())

    tensor_ptr = tensor.unsafe_ptr()

    # Iterate over all blocks in the tensor.
    for superblock_idx in range(tensor.num_elements() // elems_per_superblock):
        quantized_ptr[superblock_idx] = quantize_superblock(
            tensor_ptr.offset(superblock_idx * elems_per_superblock)
        )

    return quantized


struct Q4_KEncoding(QuantizationEncoding):
    """The Q4_K quantization encoding.

    Because this holds the quantized data in a special packing format, it
    currently does not print float values at runtime—it's just a bag of
    bits in uint8 format.
    """

    @staticmethod
    def quantize(tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision tensor `tensor` to Q4_K.

        Args:
            tensor: Full-precision tensor to quantize. The innermost dimension
                of the tensor must be a factor of 256.

        Returns:
            Quantized Q4_K tensor. The tensor datatype is `uint8` because this
            is simply a bytes buffer. Each scalar is actually stored with 4 bits.

        Raises:
            If the last dimension size is not a factor of 256.
        """
        return _qn_k_quantize[
            _BlockQ4K,
            _BlockQ4K.elements_per_superblock(),
            Self._quantize_superblock,
        ](tensor)

    @staticmethod
    def _quantize_superblock(tensor_ptr: UnsafePointer[Float32]) -> _BlockQ4K:
        alias nmax = 15
        alias elems_per_subblock = _BlockQ4K.elements_per_subblock()
        alias num_subblocks = (
            _BlockQ4K.elements_per_superblock()
            // _BlockQ4K.elements_per_subblock()
        )

        # First compute subblock statistics.
        scales = InlineArray[Float32, num_subblocks](uninitialized=True)
        mins = InlineArray[Float32, num_subblocks](uninitialized=True)
        for subblock_idx in range(num_subblocks):
            subblock_params = _pick_subblock_scale_min_q4_k_q5_k[
                count=elems_per_subblock,
                nmax=nmax,
                rmin= -1.0,
                rdelta=0.1,
                nstep=20,
            ](tensor_ptr.offset(subblock_idx * elems_per_subblock))
            scales[subblock_idx] = subblock_params.scale
            mins[subblock_idx] = subblock_params.min

        # Quantize scales for block structure, and dequantize them again for
        # the upcoming quantization.
        (d, dmin, quant_scales) = _quantize_superblock_params(
            scales=scales, mins=mins
        )

        qs = InlineArray[UInt8, QK_K // 2](uninitialized=True)
        # Quantize two subblocks at a time.
        for subblock_idx in range(0, num_subblocks, 2):
            subblock_lsb_iscale = 1 / scales[subblock_idx]
            subblock_lsb_min = mins[subblock_idx]
            subblock_msb_iscale = 1 / scales[subblock_idx + 1]
            subblock_msb_min = mins[subblock_idx + 1]
            constrained[
                elems_per_subblock % simdwidthof[Float32]() == 0,
                "subblock cannot be divided into SIMD-width units.",
            ]()
            # Adjacent subblocks' elements are zipped together for packing in
            # memory.
            for elem_idx in range(
                0, elems_per_subblock, simdwidthof[Float32]()
            ):
                subblock_lsb_data = tensor_ptr.load[
                    width = simdwidthof[Float32]()
                ](subblock_idx * elems_per_subblock + elem_idx)
                subblock_msb_data = tensor_ptr.load[
                    width = simdwidthof[Float32]()
                ]((subblock_idx + 1) * elems_per_subblock + elem_idx)
                subblock_lsb_quants = _asymmetric_quantize[nmax](
                    subblock_lsb_data,
                    iscale=subblock_lsb_iscale,
                    min=subblock_lsb_min,
                )
                subblock_msb_quants = _asymmetric_quantize[nmax](
                    subblock_msb_data,
                    iscale=subblock_msb_iscale,
                    min=subblock_msb_min,
                )
                packed_quants = subblock_lsb_quants | (subblock_msb_quants << 4)
                qs.unsafe_ptr().store(
                    subblock_idx * (elems_per_subblock // 2) + elem_idx,
                    packed_quants,
                )

        return _BlockQ4K(d=d, dmin=dmin, scales=quant_scales^, qs=qs^)

    @staticmethod
    fn id() -> String:
        """Identifier for the Q4_K quantized encoding."""
        return "q4_k"


@value
struct _BlockQ5K:
    """5-bit quantization.

    8 blocks of 32 elements each.
    Weights are represented as `x = a * q + b`.
    Effectively 5.5 bits per weight.

    Constraints:
        The data layout must exactly match `block_q5_K` from ggml-quants.h.
    """

    var d: Float16
    """Super-block scale for quantized scales."""

    var dmin: Float16
    """Super-block scale for quantized mins."""

    var scales: InlineArray[UInt8, K_SCALE_SIZE]
    """Scales and mins, quantized with 6 bits."""

    var qh: InlineArray[UInt8, QK_K // 8]
    """High bit of quants."""

    var qs: InlineArray[UInt8, QK_K // 2]
    """Lower 4 bits of quants."""

    def __init__(
        out self,
        d: Float16,
        dmin: Float16,
        scales: InlineArray[UInt8, K_SCALE_SIZE],
        qh: InlineArray[UInt8, QK_K // 8],
        qs: InlineArray[UInt8, QK_K // 2],
    ):
        constrained[
            sizeof[Self]()
            == 2 * sizeof[Float16]() + K_SCALE_SIZE + QK_K // 2 + QK_K // 8
        ]()

        self.d = d
        self.dmin = dmin
        self.scales = scales
        self.qh = qh
        self.qs = qs

    @staticmethod
    fn elements_per_superblock() -> Int:
        """Returns the number of elements per Q5_K superblock."""
        return QK_K

    @staticmethod
    fn elements_per_subblock() -> Int:
        """Returns the number of elements per Q5_K subblock."""
        return 32

    @staticmethod
    fn num_subblocks() -> Int:
        """Returns the number of subblocks per Q5_K superblock."""
        return Self.elements_per_superblock() // Self.elements_per_subblock()


struct Q5_KEncoding(QuantizationEncoding):
    """The Q5_K quantization encoding.

    Because this holds the quantized data in a special packing format, it
    currently does not print float values at runtime—it's just a bag of
    bits in uint8 format.
    """

    @staticmethod
    def quantize(tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision tensor `tensor` to Q5_K.

        Args:
            tensor: Full-precision tensor to quantize. The innermost dimension
                of the tensor must be a factor of 256.

        Returns:
            Quantized Q5_K tensor. The tensor datatype is `uint8` because this
            is simply a bytes buffer. Each scalar is actually stored with 5 bits.

        Raises:
            If the last dimension size is not a factor of 256.
        """
        return _qn_k_quantize[
            _BlockQ5K,
            _BlockQ5K.elements_per_superblock(),
            Self._quantize_superblock,
        ](tensor)

    @staticmethod
    def _quantize_superblock(tensor_ptr: UnsafePointer[Float32]) -> _BlockQ5K:
        alias nmax = 31
        alias elems_per_subblock = _BlockQ5K.elements_per_subblock()
        alias num_subblocks = (
            _BlockQ5K.elements_per_superblock()
            // _BlockQ5K.elements_per_subblock()
        )

        # First compute subblock statistics.
        scales = InlineArray[Float32, num_subblocks](uninitialized=True)
        mins = InlineArray[Float32, num_subblocks](uninitialized=True)
        for subblock_idx in range(num_subblocks):
            subblock_params = _pick_subblock_scale_min_q4_k_q5_k[
                count=elems_per_subblock,
                nmax=nmax,
                rmin= -0.5,
                rdelta=0.1,
                nstep=15,
            ](tensor_ptr.offset(subblock_idx * elems_per_subblock))
            scales[subblock_idx] = subblock_params.scale
            mins[subblock_idx] = subblock_params.min

        # Quantize scales for block structure, and dequantize them again for
        # the upcoming quantization.
        (d, dmin, quant_scales) = _quantize_superblock_params(
            scales=scales, mins=mins
        )

        # qs is fully overwritten, but qh is bit-or'ed into, so it needs to be
        # initialized to zeros.
        qh = InlineArray[UInt8, QK_K // 8](0)
        qs = InlineArray[UInt8, QK_K // 2](uninitialized=True)
        # Quantize two subblocks at a time.
        for subblock_idx in range(0, num_subblocks, 2):
            subblock0_iscale = 1 / scales[subblock_idx]
            subblock0_min = mins[subblock_idx]
            subblock1_iscale = 1 / scales[subblock_idx + 1]
            subblock1_min = mins[subblock_idx + 1]
            qh_shift = subblock_idx // 2 * 2
            alias width = simdwidthof[Float32]()
            constrained[
                elems_per_subblock % width == 0,
                "subblock cannot be divided into SIMD-width units.",
            ]()
            # Adjacent subblocks' elements are zipped together for the
            # lower-bit outputs.  For the high-bit outputs, we only have two
            # bits, so can't fill a complete byte, so this is incrementally
            # filled out, but at least the lower-bit outputs can be filled out
            # a complete byte at a time.
            for elem_idx in range(0, elems_per_subblock, width):
                subblock0_data = tensor_ptr.load[width=width](
                    subblock_idx * elems_per_subblock + elem_idx
                )
                subblock1_data = tensor_ptr.load[width=width](
                    (subblock_idx + 1) * elems_per_subblock + elem_idx
                )
                subblock0_quants = _asymmetric_quantize[nmax](
                    subblock0_data, iscale=subblock0_iscale, min=subblock0_min
                )
                subblock1_quants = _asymmetric_quantize[nmax](
                    subblock1_data, iscale=subblock1_iscale, min=subblock1_min
                )
                packed_quant_lsbs = (subblock0_quants & 0xF) | (
                    subblock1_quants << 4
                )
                qs.unsafe_ptr().store(
                    subblock_idx * (elems_per_subblock // 2) + elem_idx,
                    packed_quant_lsbs,
                )
                packed_quant_msbs = qh.unsafe_ptr().load[width=width](elem_idx)
                packed_quant_msbs |= (subblock0_quants >> 4 << qh_shift) | (
                    subblock1_quants >> 4 << (qh_shift + 1)
                )
                qh.unsafe_ptr().store(elem_idx, packed_quant_msbs)

        return _BlockQ5K(d=d, dmin=dmin, scales=quant_scales, qh=qh^, qs=qs^)

    @staticmethod
    fn id() -> String:
        """Identifier for the Q5_K quantized encoding."""
        return "q5_k"


@value
struct _BlockQ6K:
    """6-bit quantization.

    16 blocks of 16 elements each.
    Weights are represented as `x = a * q`.
    Effectively 6.5625 bits per weight.

    Constraints:
        The data layout must exactly match `block_q6_K` from ggml-quants.h.
    """

    var ql: InlineArray[UInt8, QK_K // 2]
    """Quants: lower 4 bits."""

    var qh: InlineArray[UInt8, QK_K // 4]
    """Quants: upper 2 bits."""
    var scales: InlineArray[Int8, QK_K // 16]
    """Scales: quantized with 8 bits."""

    var d: Float16
    """Super-block scale."""

    def __init__(
        out self,
        ql: InlineArray[UInt8, QK_K // 2],
        qh: InlineArray[UInt8, QK_K // 4],
        scales: InlineArray[Int8, QK_K // 16],
        d: Float16,
    ):
        constrained[
            sizeof[Self]()
            == (3 * (QK_K // 4)) + (QK_K // 16) + sizeof[Float16]()
        ]()

        self.ql = ql
        self.qh = qh
        self.scales = scales
        self.d = d

    @staticmethod
    fn elements_per_superblock() -> Int:
        """Returns the number of elements per Q6_K superblock."""
        return QK_K

    @staticmethod
    fn elements_per_subblock() -> Int:
        """Returns the number of elements per Q6_K subblock."""
        return 16

    @staticmethod
    fn num_subblocks() -> Int:
        """Returns the number of subblocks per Q6_K superblock."""
        return Self.elements_per_superblock() // Self.elements_per_subblock()


def _measure_unbiased_symmetric_quant_stats[
    count: Int, *, nmax: Int
](ptr: UnsafePointer[Float32], quants: UnsafePointer[Int8]) -> Tuple[
    Float32, Float32
]:
    sum_lx = Float32(0.0)
    sum_l2 = Float32(0.0)

    @parameter
    fn agg[width: Int](i: Int):
        var x = ptr.load[width=width](i)
        var l = quants.load[width=width](i).cast[DType.float32]()
        var common = x * x * l
        sum_lx += (common * x).reduce_add()
        sum_l2 += (common * l).reduce_add()

    vectorize[agg, simdwidthof[Float32](), size=count]()
    return sum_lx, sum_l2


struct Q6_KEncoding(QuantizationEncoding):
    """The Q6_K quantization encoding.

    Because this holds the quantized data in a special packing format, it
    currently does not print float values at runtime—it's just a bag of
    bits in uint8 format.
    """

    @staticmethod
    def quantize(tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision tensor `tensor` to Q6_K.

        Args:
            tensor: Full-precision tensor to quantize. The innermost dimension
                of the tensor must be a factor of 256.

        Returns:
            Quantized Q6_K tensor. The tensor datatype is `uint8` because this
            is simply a bytes buffer. Each scalar is actually stored with 6 bits.

        Raises:
            If the last dimension size is not a factor of 256.
        """
        return _qn_k_quantize[
            _BlockQ6K,
            _BlockQ6K.elements_per_superblock(),
            Self._quantize_superblock,
        ](tensor)

    @staticmethod
    def _quantize_superblock(tensor_ptr: UnsafePointer[Float32]) -> _BlockQ6K:
        alias nmax = 63
        alias elems_per_subblock = _BlockQ6K.elements_per_subblock()
        alias num_subblocks = (
            _BlockQ6K.elements_per_superblock()
            // _BlockQ6K.elements_per_subblock()
        )
        # In a few places we'd like to use num_subblocks, but Mojo rejects it
        # due to a parameter mismatch (even though the _values_ are the same),
        # so constrain here and hard-code where we have to.
        constrained[
            num_subblocks == QK_K // 16,
            "hard-coded num_subblocks does not match computed value",
        ]()

        # First compute subblock statistics.
        scales = InlineArray[Float32, QK_K // 16](uninitialized=True)
        for subblock_idx in range(num_subblocks):
            scales[subblock_idx] = Self._pick_subblock_scale[
                count=elems_per_subblock,
                nmax=nmax,
            ](tensor_ptr.offset(subblock_idx * elems_per_subblock))

        # Quantize scales for block structure, and dequantize them again for
        # the upcoming quantization.
        (d, quant_scales) = Self._quantize_superblock_params(scales)

        qh = InlineArray[UInt8, QK_K // 4](uninitialized=True)
        ql = InlineArray[UInt8, QK_K // 2](uninitialized=True)
        # Quantize, 4 subblocks at a time, in such a way to match llama.cpp's
        # elaborate indexing structure.
        for outer_subblock_idx in range(0, num_subblocks, 8):
            for inner_subblock_idx in range(2):
                subblock_iscales = InlineArray[Float32, 4](
                    1 / scales[outer_subblock_idx + inner_subblock_idx],
                    1 / scales[outer_subblock_idx + inner_subblock_idx + 2],
                    1 / scales[outer_subblock_idx + inner_subblock_idx + 4],
                    1 / scales[outer_subblock_idx + inner_subblock_idx + 6],
                )
                alias width = simdwidthof[Float32]()
                constrained[
                    elems_per_subblock % width == 0,
                    "subblock cannot be divided into SIMD-width units.",
                ]()
                for elem_idx in range(0, elems_per_subblock, width):
                    subblock_quants = InlineArray[SIMD[DType.uint8, width], 4](
                        uninitialized=True
                    )
                    for i in range(4):
                        subblock_quants[i] = _biased_symmetric_quantize[nmax](
                            tensor_ptr.load[width=width](
                                (
                                    outer_subblock_idx
                                    + inner_subblock_idx
                                    + i * 2
                                )
                                * elems_per_subblock
                                + elem_idx
                            ),
                            iscale=subblock_iscales[i],
                        )
                    for i in range(2):
                        ql.unsafe_ptr().store(
                            inner_subblock_idx * elems_per_subblock
                            + (outer_subblock_idx + i * 4)
                            * (elems_per_subblock // 2)
                            + elem_idx,
                            (subblock_quants[i] & 0xF)
                            | ((subblock_quants[i + 2] & 0xF) << 4),
                        )
                    qh.unsafe_ptr().store(
                        outer_subblock_idx * (elems_per_subblock // 4)
                        + inner_subblock_idx * elems_per_subblock
                        + elem_idx,
                        (subblock_quants[0] >> 4)
                        | (subblock_quants[1] >> 4 << 2)
                        | (subblock_quants[2] >> 4 << 4)
                        | (subblock_quants[3] >> 4 << 6),
                    )

        return _BlockQ6K(ql=ql^, qh=qh^, scales=quant_scales^, d=d)

    @staticmethod
    def _pick_subblock_scale[
        *, count: Int, nmax: Int
    ](ptr: UnsafePointer[Float32]) -> Float32:
        alias nmax_signed_min = -(nmax + 1) // 2
        amax, amax_nonabs = _find_amax[count](ptr)
        # Make a first guess at a quantization scale.
        iscale = nmax_signed_min / amax_nonabs
        quants = _unbiased_symmetric_quantize[count, nmax=nmax](
            ptr, iscale=iscale
        )
        sum_lx, sum_l2 = _measure_unbiased_symmetric_quant_stats[
            count, nmax=nmax
        ](ptr, quants.unsafe_ptr())
        # Improve scale based on measured statistics.
        scale = sum_lx / sum_l2 if sum_l2 else 0
        best_corr = scale * sum_lx
        # Sweep, looking for a scale with higher correlation.
        for step in range(-9, 10):
            trial_iscale = (nmax_signed_min - step * Float32(0.1)) / amax_nonabs
            trial_quants = _unbiased_symmetric_quantize[count, nmax=nmax](
                ptr, iscale=trial_iscale
            )
            trial_sum_lx, trial_sum_l2 = (
                _measure_unbiased_symmetric_quant_stats[count, nmax=nmax](
                    ptr, trial_quants.unsafe_ptr()
                )
            )
            if (
                trial_sum_l2 > 0
                and trial_sum_lx * trial_sum_lx > best_corr * trial_sum_l2
            ):
                quants = trial_quants
                scale = trial_sum_lx / trial_sum_l2
                best_corr = scale * trial_sum_lx
        return scale

    @staticmethod
    def _quantize_superblock_params[
        count: Int
    ](mut scales: InlineArray[Float32, count]) -> Tuple[
        Float16, InlineArray[Int8, count]
    ]:
        scales_amax, scales_amax_nonabs = _find_amax[count](scales.unsafe_ptr())
        scale_iscale = -128 / scales_amax_nonabs
        d = (1 / scale_iscale).cast[DType.float16]()
        scale_quants = _unbiased_symmetric_qdq[count, nmax=255](
            scales.unsafe_ptr(),
            scale=d.cast[DType.float32](),
            iscale=scale_iscale,
        )
        return (d, scale_quants)

    @staticmethod
    fn id() -> String:
        """Identifier for the Q6_K quantized encoding."""
        return "q6_k"

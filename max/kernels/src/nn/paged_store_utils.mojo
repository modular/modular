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
"""Slot, RoPE and quantization helpers for paged quantized cache stores.

A cache page holds `page_size` records: all of their values, then all of their
scales. With one slot per page a record is a contiguous row of values followed
by its scales.
"""

from std.math import clamp, fma
from std.memory import bitcast
from std.sys import size_of
from layout import TileTensor
import max.gpu.primitives.warp as warp
from linalg.fp4_utils import cast_fp32_to_fp4e2m1


@inline(.always)
def raw_pointer[
    dtype: DType
](t: TileTensor[dtype, ...]) -> MutPointer[Scalar[dtype], MutUntrackedOrigin]:
    """Returns a generic pointer to a tensor's first element.

    Parameters:
        dtype: The element type.

    Args:
        t: The tensor.

    Returns:
        The tensor's base address, without its origin or address space.
    """
    return (
        t.ptr.unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_address_space_cast[.GENERIC]()
    )


@inline(.always)
def dense_pointer[
    dtype: DType
](t: TileTensor[dtype, ...]) -> MutPointer[Scalar[dtype], MutUntrackedOrigin]:
    """Returns a generic pointer to a row-major tensor's first element.

    Parameters:
        dtype: The element type.

    Args:
        t: The tensor, which a kernel indexes as dense row-major.

    Returns:
        The tensor's base address, without its origin or address space.
    """
    comptime assert t.is_row_major, "paged store tensors must be row-major"
    return raw_pointer(t)


def row_stride[
    dtype: DType, //, alignment: Int
](t: TileTensor[dtype, ...], name: StaticString) raises -> Int:
    """Returns the row stride of a matrix whose rows are contiguous.

    Parameters:
        dtype: The element type.
        alignment: The byte alignment every row must start at.

    Args:
        t: A rank-2 tensor with unit column stride.
        name: The tensor's name, for errors.

    Returns:
        The distance between rows, in elements.

    Raises:
        If a row is not contiguous or does not start at `alignment` bytes.
    """
    comptime assert t.rank == 2, "expected a [rows, columns] tensor"
    comptime size = size_of[dtype]()
    var stride = Int(t.dynamic_stride(0))
    if Int(t.dynamic_stride(1)) != 1:
        raise Error(String(name, " rows must be contiguous"))
    if (stride * size) % alignment or Int(t.ptr) % alignment:
        raise Error(
            String(name, " rows must start at ", alignment, "-byte boundaries")
        )
    return stride


@inline(.always)
def page_and_offset(slot: Int, page_size: Int) -> Tuple[Int, Int]:
    """Splits a slot into its page and its place in the page.

    Args:
        slot: A slot in `[0, 2^32)`.
        page_size: The slots in one page.

    Returns:
        The page and the slot's place in it.
    """
    # 32-bit division is a fraction of the 64-bit sequence, and 2^32 records
    # would take over a terabyte.
    var page, offset = divmod(UInt32(slot), UInt32(page_size))
    return Int(page), Int(offset)


@inline(.always)
def paged_record[
    value_bytes: Int, scale_bytes: Int
](
    cache: MutPointer[UInt8, MutAnyOrigin],
    slot: Int,
    page_size: Int,
    page_stride: Int,
) -> Tuple[MutPointer[UInt8, MutAnyOrigin], MutPointer[UInt8, MutAnyOrigin]]:
    """Locates a slot's values and scales in a paged cache.

    Parameters:
        value_bytes: The bytes of values in one record.
        scale_bytes: The bytes of scales in one record.

    Args:
        cache: The first page.
        slot: A slot in `[0, 2^32)`.
        page_size: The slots in one page.
        page_stride: The bytes from one page to the next.

    Returns:
        The slot's values and its scales.
    """
    var page, offset = page_and_offset(slot, page_size)
    var base = cache.unsafe_offset(page * page_stride)
    return (
        base.unsafe_offset(offset * value_bytes),
        base.unsafe_offset(page_size * value_bytes + offset * scale_bytes),
    )


@inline(.always)
def load_frequencies[
    pairs: Int
](
    frequencies: ImmPointer[Float32, ImmutAnyOrigin],
    row: Int,
    first_pair: Int,
) -> Tuple[SIMD[.float32, pairs], SIMD[.float32, pairs]]:
    """Loads the cosines and sines of consecutive RoPE pairs.

    A frequency row holds 32 cosines, then 32 sines.

    Parameters:
        pairs: The number of pairs.

    Args:
        frequencies: The frequency table, aligned to `4 * pairs` bytes.
        row: The offset of the position's row.
        first_pair: The first pair, a multiple of `pairs`.

    Returns:
        The cosines and the sines.
    """
    comptime alignment = 4 * pairs
    return (
        frequencies.unsafe_load[width=pairs, alignment=alignment](
            row + first_pair
        ),
        frequencies.unsafe_load[width=pairs, alignment=alignment](
            row + 32 + first_pair
        ),
    )


@inline(.always)
def load_frequencies[
    pairs: Int
](
    frequencies: TileTensor[.float32, ...], position: Int, first_pair: Int
) -> Tuple[SIMD[.float32, pairs], SIMD[.float32, pairs]]:
    """Loads the cosines and sines of consecutive RoPE pairs from a table.

    A frequency row holds 32 cosines, then 32 sines.

    Parameters:
        pairs: The number of pairs.

    Args:
        frequencies: The frequency table, one row per position, with rows
            aligned to `4 * pairs` bytes.
        position: The position's row.
        first_pair: The first pair, a multiple of `pairs`.

    Returns:
        The cosines and the sines.
    """
    return (
        frequencies.load[width=pairs]((position, first_pair)),
        frequencies.load[width=pairs]((position, 32 + first_pair)),
    )


@inline(.always)
def gptj_rotate[
    pairs: Int, *, odd_fuses_sine: Bool = False
](
    values: SIMD[.float32, 2 * pairs],
    cosine: SIMD[.float32, pairs],
    sine: SIMD[.float32, pairs],
) -> SIMD[.float32, 2 * pairs]:
    """Rotates interleaved (even, odd) channel pairs.

    Each output fuses one product into an FMA, as the reference stores'
    compilers contract `even * cos - odd * sin` and the odd output. Triton
    contracts `odd * cos + even * sin` around `odd * cos`, CUDA contracts
    `even * sin + odd * cos` around `even * sin`.

    Parameters:
        pairs: The number of pairs.
        odd_fuses_sine: Fuse `even * sin` rather than `odd * cos` into the
            odd output.

    Args:
        values: The pairs, even channels at even lanes.
        cosine: The cosine of each pair's angle.
        sine: The sine of each pair's angle.

    Returns:
        The rotated pairs.
    """
    var halves = values.deinterleave()
    var even = rebind[SIMD[.float32, pairs]](halves[0])
    var odd = rebind[SIMD[.float32, pairs]](halves[1])
    var rotated_odd: SIMD[.float32, pairs]
    comptime if odd_fuses_sine:
        rotated_odd = fma(even, sine, odd * cosine)
    else:
        rotated_odd = fma(odd, cosine, even * sine)
    return rebind[SIMD[.float32, 2 * pairs]](
        fma(even, cosine, -(odd * sine)).interleave(rotated_odd)
    )


@inline(.always)
def mxfp8_quantize[
    width: SIMDLength, //, lanes: Int
](values: SIMD[.float32, width]) -> Tuple[
    SIMD[.float8_e4m3fn, width], Scalar[.float8_e8m0fnu]
]:
    """Quantizes a group spread over adjacent lanes to E4M3 with a UE8M0 scale.

    The scale is the smallest power of two that fits the group's largest
    magnitude, floored at 1e-4, under 448.

    Parameters:
        width: The values each lane holds.
        lanes: The adjacent lanes that share one scale.

    Args:
        values: This lane's part of the group.

    Returns:
        The quantized values and the group's scale.
    """
    var peak = max(
        warp.lane_group_max[num_lanes=lanes](abs(values).reduce_max()),
        Float32(1e-4),
    )
    var scale = (peak * (1.0 / 448.0)).cast[.float8_e8m0fnu]()
    var quantized = clamp(
        values * (1.0 / scale.cast[.float32]()), -448.0, 448.0
    ).cast[.float8_e4m3fn]()
    return quantized, scale


@inline(.always)
def mxfp4_quantize[
    dtype: DType, width: SIMDLength, //, lanes: Int
](values: SIMD[dtype, width]) -> Tuple[
    SIMD[.uint32, Int(width) // 8], Scalar[.float8_e8m0fnu]
]:
    """Quantizes a group spread over adjacent lanes to E2M1 with a UE8M0 scale.

    The scale is the smallest power of two that brings the group's largest
    magnitude, floored at 6 * 2^-126, to at most 6.

    Parameters:
        dtype: The values' type, FP32 or BF16.
        width: The values each lane holds, a multiple of 8.
        lanes: The adjacent lanes that share one scale.

    Args:
        values: This lane's part of the group.

    Returns:
        The packed codes, even values in the low nibbles, and the group's
        scale.
    """
    comptime groups = Int(width) // 8
    comptime assert groups * 8 == Int(width), "width must be a multiple of 8"
    var peak = max(
        warp.lane_group_max[num_lanes=lanes](
            abs(values).reduce_max().cast[.float32]()
        ),
        # 6 * 2^-126, exact before the scale rounds up.
        Float32(7.052966104933725e-38),
    )
    var scale = (peak * (1.0 / 6.0)).cast[.float8_e8m0fnu]()
    # 2^(127 - e) is 1 / scale, exact without a division.
    var inverse = bitcast[.float32](
        (254 - UInt32(bitcast[.uint8](scale))) << 23
    )
    var scaled = values.cast[.float32]() * inverse
    var codes = SIMD[.uint32, groups]()
    comptime for i in range(groups):
        codes[i] = cast_fp32_to_fp4e2m1(scaled.slice[8, offset=8 * i]())
    return codes, scale

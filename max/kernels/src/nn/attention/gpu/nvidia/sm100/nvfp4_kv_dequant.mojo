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

"""NVFP4 KV-cache dequantization helpers for the SM100 sparse MLA decode kernel.

Target family: Blackwell sm_100.

An NVFP4 KV operand is a `uint8` cache that packs two e2m1 elements per byte
(element 0 in the low nibble) under one `float8_e4m3fn` scale per 16 elements.
The kernel dequantizes it to FP8 in shared memory, so Q@K and P@V stay on the
FP8 `tcgen05.mma.kind::f8f6f4` path.

An MLA row keeps its trailing `NVFP4_MLA_ROPE_DIM` rope dims as raw FP8 bytes
instead: the packed latent comes first, then the rope bytes, and only the
latent has scales. The rope bytes are exactly what an FP8 cache holds, so the
kernel copies them to shared memory without a conversion.

Each FP8 value is `e2m1 * sf` rounded once to e4m3. A 16-element group shares
one scale and e2m1 has only 8 magnitudes, so the eight products are converted
once and every element is then a byte lookup.

The `e4m3_magnitude_lut` / `e2m1x8_to_e4m3x8` pair mirrors the helpers in
`Kernels/lib/msa/nvfp4_kv.mojo`, which cannot be imported from `max/kernels`.
"""

from std.memory import bitcast
from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly

from max.gpu.intrinsics import byte_permute

from nn.attention.mha_operand import MHAOperand

# Elements covered by one E4M3 scale.
comptime NVFP4_SF_VECTOR_SIZE = 16

# Rope dims at the end of an MLA row, stored as unscaled FP8 bytes. Rope
# dominates the attention scores, so e2m1 there costs accuracy.
comptime NVFP4_MLA_ROPE_DIM = 64


def nvfp4_mla_row_bytes[input_q_depth: Int]() -> Int:
    """Bytes of one NVFP4 MLA cache row: the packed latent, then FP8 rope."""
    return (input_q_depth - NVFP4_MLA_ROPE_DIM) // 2 + NVFP4_MLA_ROPE_DIM


def nvfp4_mla_scale_row_bytes[input_q_depth: Int]() -> Int:
    """Scale bytes of one NVFP4 MLA row; only the latent has scales."""
    return (input_q_depth - NVFP4_MLA_ROPE_DIM) // NVFP4_SF_VECTOR_SIZE


def mla_kv_is_nvfp4[KVType: MHAOperand]() -> Bool:
    """Whether the KV operand stores packed NVFP4 (`uint8` + E4M3 scales).

    The operand's `quantization_granularity` counts packed bytes, so 16
    elements are 8 of them.
    """
    return (
        KVType.dtype == DType.uint8
        and KVType.quantization_enabled
        and KVType.scale_dtype == DType.float8_e4m3fn
        and KVType.quantization_granularity * 2 == NVFP4_SF_VECTOR_SIZE
    )


@always_inline
def e4m3_magnitude_lut(scale: Float32) -> SIMD[DType.uint32, 2]:
    """`e4m3(m * scale)` for the e2m1 magnitudes `{0, .5, 1, 1.5, 2, 3, 4, 6}`.

    Byte `i` is the entry for magnitude code `i`, laid out as the two words
    `byte_permute` indexes. `m * scale` is exact in f32 and is rounded once to
    e4m3 (nearest even, which is sign-symmetric), so
    `e4m3(-x) == e4m3(x) | 0x80`. The product needs up to 6 significant bits, so
    it is not always exact in e4m3; a scale from a real quantizer keeps `6 *
    scale` inside the e4m3 range.
    """
    var mags = SIMD[DType.float32, 8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    return bitcast[DType.uint32, 2]((mags * scale).cast[DType.float8_e4m3fn]())


@always_inline
def e2m1x8_to_e4m3x8(
    word: UInt32, lut: SIMD[DType.uint32, 2]
) -> SIMD[DType.uint32, 2]:
    """Converts eight e2m1 (element 0 in the low nibble) to eight e4m3 bytes.

    The magnitude comes from `lut` indexed by the code's low 3 bits. The sign
    comes from `prmt`'s sign-replicate mode: with `shifted = word << 4`, element
    `2i`'s sign is the msb of `shifted` byte `i` and element `2i + 1`'s is the
    msb of `word` byte `i`, so one `prmt` spreads four signs into 0x00/0xFF
    bytes that select the 0x80 bit.
    """
    var shifted = word << 4
    var codes = word & 0x77777777
    var lo = byte_permute(lut[0], lut[1], codes)
    var hi = byte_permute(lut[0], lut[1], codes >> 16)
    var lo_sign = byte_permute(shifted, word, 0xD9C8)
    var hi_sign = byte_permute(shifted, word, 0xFBEA)
    return SIMD[DType.uint32, 2](
        lo | (lo_sign & 0x80808080), hi | (hi_sign & 0x80808080)
    )


@always_inline
def nvfp4_dequant_group(
    packed: SIMD[DType.uint8, 8], sf: Float32
) -> SIMD[DType.uint32, 4]:
    """Expands one scale group (8 packed bytes, 16 e2m1) to 16 e4m3 bytes."""
    var words = bitcast[DType.uint32, 2](packed)
    var lut = e4m3_magnitude_lut(sf)
    var lo = e2m1x8_to_e4m3x8(words[0], lut)
    var hi = e2m1x8_to_e4m3x8(words[1], lut)
    return SIMD[DType.uint32, 4](lo[0], lo[1], hi[0], hi[1])


@always_inline
def e4m3x2_to_f16x2(pair: UInt32) -> UInt32:
    """Widens the two E4M3 bytes in the low half of `pair` to an f16 pair.

    Byte 0 lands in the low half. E4M3 widens exactly to f16.
    """
    return inlined_assembly[
        "cvt.rn.f16x2.e4m3x2 $0, $1;",
        UInt32,
        constraints="=r,h",
        has_side_effect=False,
    ](UInt16(pair & 0xFFFF))


@always_inline
def e4m3_magnitude_lut_f16(sf2: UInt32) -> SIMD[DType.uint32, 2]:
    """`e4m3_magnitude_lut` from a broadcast f16 scale pair.

    `m * scale` takes at most 6 significant bits and stays in f16's normal
    range for every E4M3 scale, so the f16 product is exact and its one
    rounding to e4m3 gives the f32 table's bytes.
    """
    var r = inlined_assembly[
        """{
.reg .b32 p0, p1, p2, p3;
.reg .b16 o0, o1, o2, o3;
mov.b32 p0, 0x38000000;
mov.b32 p1, 0x3E003C00;
mov.b32 p2, 0x42004000;
mov.b32 p3, 0x46004400;
mul.rn.f16x2 p0, $2, p0;
mul.rn.f16x2 p1, $2, p1;
mul.rn.f16x2 p2, $2, p2;
mul.rn.f16x2 p3, $2, p3;
cvt.rn.satfinite.e4m3x2.f16x2 o0, p0;
cvt.rn.satfinite.e4m3x2.f16x2 o1, p1;
cvt.rn.satfinite.e4m3x2.f16x2 o2, p2;
cvt.rn.satfinite.e4m3x2.f16x2 o3, p3;
mov.b32 $0, {o0, o1};
mov.b32 $1, {o2, o3};
}""",
        _RegisterPackType[UInt32, UInt32],
        constraints="=r,=r,r",
        has_side_effect=False,
    ](sf2)
    return SIMD[DType.uint32, 2](r[0], r[1])


@always_inline
def e2m1x8_scaled_to_e4m3x8(word: UInt32, sf2: UInt32) -> SIMD[DType.uint32, 2]:
    """Converts eight e2m1 (element 0 in the low nibble) times a broadcast f16
    scale pair to eight e4m3 bytes, through the hardware converts.

    Each byte widens to an f16 pair, `mul.f16x2` applies the scale exactly
    (see `e4m3_magnitude_lut_f16`), and one convert rounds the pair to e4m3,
    so the bytes match `e2m1x8_to_e4m3x8` for every positive scale.
    """
    var r = inlined_assembly[
        """{
.reg .b8 b0, b1, b2, b3;
.reg .b32 h0, h1, h2, h3;
.reg .b16 o0, o1, o2, o3;
mov.b32 {b0, b1, b2, b3}, $2;
cvt.rn.f16x2.e2m1x2 h0, b0;
cvt.rn.f16x2.e2m1x2 h1, b1;
cvt.rn.f16x2.e2m1x2 h2, b2;
cvt.rn.f16x2.e2m1x2 h3, b3;
mul.rn.f16x2 h0, h0, $3;
mul.rn.f16x2 h1, h1, $3;
mul.rn.f16x2 h2, h2, $3;
mul.rn.f16x2 h3, h3, $3;
cvt.rn.satfinite.e4m3x2.f16x2 o0, h0;
cvt.rn.satfinite.e4m3x2.f16x2 o1, h1;
cvt.rn.satfinite.e4m3x2.f16x2 o2, h2;
cvt.rn.satfinite.e4m3x2.f16x2 o3, h3;
mov.b32 $0, {o0, o1};
mov.b32 $1, {o2, o3};
}""",
        _RegisterPackType[UInt32, UInt32],
        constraints="=r,=r,r,r",
        has_side_effect=False,
    ](word, sf2)
    return SIMD[DType.uint32, 2](r[0], r[1])


@always_inline
def nvfp4_dequant_group_f16[
    hw_convert: Bool
](lo_word: UInt32, hi_word: UInt32, sf2: UInt32) -> SIMD[DType.uint32, 4]:
    """`nvfp4_dequant_group` from a broadcast f16 scale pair.

    `hw_convert` converts every element through the hardware converts;
    otherwise the f16 products build the byte table that `prmt` indexes.
    """
    var lo: SIMD[DType.uint32, 2]
    var hi: SIMD[DType.uint32, 2]
    comptime if hw_convert:
        lo = e2m1x8_scaled_to_e4m3x8(lo_word, sf2)
        hi = e2m1x8_scaled_to_e4m3x8(hi_word, sf2)
    else:
        var lut = e4m3_magnitude_lut_f16(sf2)
        lo = e2m1x8_to_e4m3x8(lo_word, lut)
        hi = e2m1x8_to_e4m3x8(hi_word, lut)
    return SIMD[DType.uint32, 4](lo[0], lo[1], hi[0], hi[1])

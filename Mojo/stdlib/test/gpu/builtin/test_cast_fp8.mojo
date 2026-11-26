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
"""The vector float32 -> float8 cast packs four bytes per 32-bit word.

On NVIDIA the two `cvt.rn.satfinite.*x2.f32` of a vector cast merge into the
word, while a width-1 cast takes the single-`cvt` path. This test pins the
packed words to the width-1 bytes, so it catches a word whose halves are
swapped, as well as any per-element conversion change.
"""

from std.memory import bitcast
from std.testing import assert_equal

from max.gpu import global_idx
from max.gpu.host import DeviceContext

comptime F32 = DType.float32
comptime NUM_GROUPS = 1 << 13


def packed_byte_words[
    dtype: DType,
](values: SIMD[F32, 4]) -> UInt32:
    """Encodes four values with the width-1 cast, `values[0]` in the low byte.
    """
    var word: UInt32 = 0
    for i in range(4):
        var encoded = bitcast[DType.uint8, 1](values[i].cast[dtype]())
        word |= UInt32(encoded) << UInt32(8 * i)
    return word


def cast_kernel[
    dtype: DType,
](
    src: MutPointer[Scalar[F32], MutAnyOrigin],
    packed4: MutPointer[UInt32, MutAnyOrigin],
    packed8: MutPointer[UInt32, MutAnyOrigin],
    per_element: MutPointer[UInt32, MutAnyOrigin],
):
    """Casts eight consecutive values with widths 4, 8 and 1, one word each."""
    var group = global_idx.x
    if group >= NUM_GROUPS:
        return
    var base = 8 * group

    var lo = SIMD[F32, 4](
        src[unsafe_offset=base],
        src[unsafe_offset=base + 1],
        src[unsafe_offset=base + 2],
        src[unsafe_offset=base + 3],
    )
    var hi = SIMD[F32, 4](
        src[unsafe_offset=base + 4],
        src[unsafe_offset=base + 5],
        src[unsafe_offset=base + 6],
        src[unsafe_offset=base + 7],
    )
    packed4[unsafe_offset=2 * group] = bitcast[DType.uint32, 1](
        lo.cast[dtype]()
    )
    packed4[unsafe_offset=2 * group + 1] = bitcast[DType.uint32, 1](
        hi.cast[dtype]()
    )

    var wide = SIMD[F32, 8](
        lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]
    )
    var wide_bytes = bitcast[DType.uint32, 2](wide.cast[dtype]())
    packed8[unsafe_offset=2 * group] = wide_bytes[0]
    packed8[unsafe_offset=2 * group + 1] = wide_bytes[1]

    per_element[unsafe_offset=2 * group] = packed_byte_words[dtype](lo)
    per_element[unsafe_offset=2 * group + 1] = packed_byte_words[dtype](hi)


def check_cast[
    dtype: DType,
]() raises:
    """Runs the vector casts and compares every word with width-1 casts."""
    var n = 8 * NUM_GROUPS
    var bad4 = 0
    var bad8 = 0
    with DeviceContext() as ctx:
        var host = ctx.enqueue_create_host_buffer[F32](n)
        ctx.synchronize()
        # An xorshift stream covers every exponent; the head of the buffer
        # holds the values that sit on the format's boundaries.
        var state = UInt32(0x9E3779B9)
        for i in range(n):
            state ^= state << 13
            state ^= state >> 17
            state ^= state << 5
            host[i] = bitcast[F32, 1](state)
        var edges: List[Float32] = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            448.0,
            464.0,
            465.0,
            480.0,
            1.0e30,
            -1.0e30,
            0.0009765625,
            0.001953125,
            0.0029296875,
            1.0e-9,
            3.0e38,
            -3.0e38,
        ]
        for i in range(len(edges)):
            host[i] = edges[i]
        host[len(edges)] = bitcast[F32, 1](UInt32(0x7F800000))
        host[len(edges) + 1] = bitcast[F32, 1](UInt32(0xFF800000))
        host[len(edges) + 2] = bitcast[F32, 1](UInt32(0x7FC00000))

        var src = ctx.enqueue_create_buffer[F32](n)
        var packed4 = ctx.enqueue_create_buffer[DType.uint32](2 * NUM_GROUPS)
        var packed8 = ctx.enqueue_create_buffer[DType.uint32](2 * NUM_GROUPS)
        var per_element = ctx.enqueue_create_buffer[DType.uint32](
            2 * NUM_GROUPS
        )
        ctx.enqueue_copy(src, host)
        ctx.enqueue_function[cast_kernel[dtype]](
            src,
            packed4,
            packed8,
            per_element,
            grid_dim=(NUM_GROUPS // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
        var packed4_host = ctx.enqueue_create_host_buffer[DType.uint32](
            2 * NUM_GROUPS
        )
        var packed8_host = ctx.enqueue_create_host_buffer[DType.uint32](
            2 * NUM_GROUPS
        )
        var per_element_host = ctx.enqueue_create_host_buffer[DType.uint32](
            2 * NUM_GROUPS
        )
        ctx.enqueue_copy(packed4_host, packed4)
        ctx.enqueue_copy(packed8_host, packed8)
        ctx.enqueue_copy(per_element_host, per_element)
        ctx.synchronize()

        for group in range(NUM_GROUPS):
            for word in range(2):
                var index = 2 * group + word
                if packed4_host[index] != per_element_host[index]:
                    bad4 += 1
                if packed8_host[index] != per_element_host[index]:
                    bad8 += 1
    assert_equal(bad4, 0, msg="width-4 words that differ from the scalar cast")
    assert_equal(bad8, 0, msg="width-8 words that differ from the scalar cast")


def main() raises:
    check_cast[DType.float8_e4m3fn]()
    check_cast[DType.float8_e5m2]()

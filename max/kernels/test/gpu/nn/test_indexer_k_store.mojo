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
"""Check index-key normalization, RoPE, packing, and paged write ownership."""

from std.math import ceil, ceildiv, exp2, fma, log2, rsqrt
from std.memory import bitcast
from std.testing import assert_equal
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.indexer_k_store import indexer_k_norm_rope_store


def _fp4_code(value: Float32) -> UInt8:
    # Independent nearest-even oracle, including negative zero.
    var magnitude = abs(value)
    var code: UInt8 = 0
    if magnitude > 5.0:
        code = 7
    elif magnitude >= 3.5:
        code = 6
    elif magnitude > 2.5:
        code = 5
    elif magnitude >= 1.75:
        code = 4
    elif magnitude > 1.25:
        code = 3
    elif magnitude >= 0.75:
        code = 2
    elif magnitude > 0.25:
        code = 1
    return code | UInt8((bitcast[.uint32](value) >> 28) & 8)


def _case[
    fp4: Bool, ratio: Int
](
    ctx: DeviceContext, count: Int, page_size: Int, seed: Int, epsilon: Float32
) raises:
    comptime key_stride = 136
    comptime freq_stride = 72
    comptime max_pos = 512
    comptime value_bytes = 64 if fp4 else 128
    var pages = ceildiv(2 * count + page_size, page_size)
    var page_stride = page_size * (value_bytes + 4) + 16
    var cache_size = pages * page_stride + 32
    var keys = ctx.enqueue_create_host_buffer[.bfloat16](
        max(count, 1) * key_stride
    )
    var weights = ctx.enqueue_create_host_buffer[.bfloat16](128)
    var positions = ctx.enqueue_create_host_buffer[.int64](max(count, 1))
    var slots = ctx.enqueue_create_host_buffer[.int64](max(count, 1))
    var frequencies = ctx.enqueue_create_host_buffer[.float32](
        max_pos * freq_stride
    )
    var expected = ctx.enqueue_create_host_buffer[.uint8](cache_size)
    var actual = ctx.enqueue_create_host_buffer[.uint8](cache_size)
    for i in range(cache_size):
        expected[i] = 0xFF
    for d in range(128):
        weights[d] = (Float32((d * 7 + seed) % 17 - 8) / 32 + 1).cast[
            .bfloat16
        ]()
    for p in range(max_pos):
        for d in range(32):
            # Exact quarter turns make group-position mistakes unambiguous.
            var phase = (p + d) % 4
            frequencies[p * freq_stride + d] = Float32(
                1 if phase == 0 else (-1 if phase == 2 else 0)
            )
            frequencies[p * freq_stride + 32 + d] = Float32(
                1 if phase == 1 else (-1 if phase == 3 else 0)
            )
    if seed == 19:
        for p in range(max_pos):
            for d in range(32):
                var cosine = frequencies[p * freq_stride + d]
                var sine = frequencies[p * freq_stride + 32 + d]
                frequencies[p * freq_stride + d] = 0.6 * cosine - 0.8 * sine
                frequencies[p * freq_stride + 32 + d] = (
                    0.8 * cosine + 0.6 * sine
                )
    for row in range(count):
        positions[row] = Int64((row * 3 + seed) % max_pos)
        slots[row] = Int64(
            -1 if row % 7 == 6 else 2 * count - row * 2 + page_size - 1
        )
        for d in range(128):
            keys[row * key_stride + d] = (
                Float32((row * 17 + d * 13 + seed) % 97 - 48) / 16
            ).cast[.bfloat16]()
        # Include a true zero row to exercise tiny FP4 scales and signed zero.
        if row % 11 == 1:
            for d in range(128):
                keys[row * key_stride + d] = 0
        if slots[row] < 0 or (Int(positions[row]) + 1) % ratio != 0:
            continue
        var values = List[Float32]()
        var square_sum = Float32(0)
        for d in range(128):
            var value = keys[row * key_stride + d].cast[.float32]()
            square_sum += value * value
        var norm = rsqrt(square_sum / 128 + epsilon)
        for d in range(128):
            values.append(
                (
                    keys[row * key_stride + d].cast[.float32]()
                    * norm
                    * weights[d].cast[.float32]()
                )
                .cast[.bfloat16]()
                .cast[.float32]()
            )
        var position = Int(positions[row]) // ratio * ratio
        for pair in range(32):
            var even = values[64 + 2 * pair]
            var odd = values[65 + 2 * pair]
            var cosine = frequencies[position * freq_stride + pair]
            var sine = frequencies[position * freq_stride + 32 + pair]
            values[64 + 2 * pair] = (
                (fma(even, cosine, -(odd * sine)))
                .cast[.bfloat16]()
                .cast[.float32]()
            )
            values[65 + 2 * pair] = (
                (fma(odd, cosine, even * sine))
                .cast[.bfloat16]()
                .cast[.float32]()
            )
        var slot = Int(slots[row])
        var base = slot // page_size * page_stride
        var offset = slot % page_size
        var scales = base + page_size * value_bytes + offset * 4
        comptime if fp4:
            for group in range(4):
                var peak = Float32(6.0 * 2.0**-126)
                for d in range(32):
                    peak = max(peak, abs(values[group * 32 + d]))
                var exponent = ceil(log2(peak * (1.0 / 6.0)))
                expected[scales + group] = UInt8(exponent + 127)
                for pair in range(16):
                    var lo = _fp4_code(
                        values[group * 32 + pair * 2] * exp2(-exponent)
                    )
                    var hi = _fp4_code(
                        values[group * 32 + pair * 2 + 1] * exp2(-exponent)
                    )
                    expected[base + offset * 64 + group * 16 + pair] = lo | (
                        hi << 4
                    )
        else:
            var peak = Float32(1e-4)
            for d in range(128):
                peak = max(peak, abs(values[d]))
            var exponent = ceil(log2(peak * (1.0 / 448.0)))
            var scale_bytes = bitcast[.uint8, 4](exp2(exponent))
            for d in range(4):
                expected[scales + d] = scale_bytes[d]
            for d in range(128):
                expected[base + offset * 128 + d] = bitcast[.uint8](
                    (values[d] * exp2(-exponent)).cast[.float8_e4m3fn]()
                )
    var d_keys = ctx.enqueue_create_buffer[.bfloat16](len(keys))
    var d_weights = ctx.enqueue_create_buffer[.bfloat16](len(weights))
    var d_positions = ctx.enqueue_create_buffer[.int64](len(positions))
    var d_slots = ctx.enqueue_create_buffer[.int64](len(slots))
    var d_frequencies = ctx.enqueue_create_buffer[.float32](len(frequencies))
    var d_cache = ctx.enqueue_create_buffer[.uint8](cache_size)
    ctx.enqueue_copy(dst_buf=d_keys, src_buf=keys)
    ctx.enqueue_copy(dst_buf=d_weights, src_buf=weights)
    ctx.enqueue_copy(dst_buf=d_positions, src_buf=positions)
    ctx.enqueue_copy(dst_buf=d_slots, src_buf=slots)
    ctx.enqueue_copy(dst_buf=d_frequencies, src_buf=frequencies)
    d_cache.enqueue_fill(0xFF)
    indexer_k_norm_rope_store[fp4, ratio](
        TileTensor(d_keys, row_major(count, key_stride)),
        TileTensor(d_positions, row_major(max(count, 1))),
        TileTensor(d_frequencies, row_major(max_pos, freq_stride)),
        TileTensor(d_weights, row_major(128)),
        TileTensor(d_slots, row_major(max(count, 1))),
        TileTensor(d_cache, row_major(pages, page_stride)),
        page_size,
        epsilon,
        ctx,
    )
    ctx.enqueue_copy(dst_buf=actual, src_buf=d_cache)
    ctx.synchronize()
    comptime if not fp4 and ratio == 1:
        if count == 63 and page_size == 1 and seed == 19:
            # Pinned vLLM byte at a BF16 tie: separately rounded products
            # produce 224 instead of the fused result 225.
            assert_equal(actual[13091], UInt8(225))
    for i in range(cache_size):
        if actual[i] != expected[i]:
            print(
                "mismatch",
                fp4,
                ratio,
                count,
                page_size,
                seed,
                i,
                actual[i],
                expected[i],
            )
        assert_equal(actual[i], expected[i])


def main() raises:
    var ctx = DeviceContext()
    comptime for mode in range(2):
        comptime fp4 = Bool(mode)
        comptime for ratio in range(1, 3):
            for count in [0, 1, 7, 8, 9, 63, 64, 65, 129]:
                for page_size in [1, 8, 64]:
                    for seed in [1, 19, 2026]:
                        for epsilon in [Float32(1e-6), Float32(1e-20)]:
                            _case[fp4, ratio](
                                ctx, count, page_size, seed, epsilon
                            )
    print("PASS 648 index-key cache cases")

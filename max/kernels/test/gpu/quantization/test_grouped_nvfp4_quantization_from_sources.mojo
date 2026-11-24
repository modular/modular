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
"""The per-token grouped NVFP4 quantize against the per-row one.

Each case routes tokens to `top_k` distinct experts, sorts the slots by
expert as `moe_create_indices` does, and runs both kernels on that order:
the per-row kernel reads each slot's token through the slot order, the
per-token kernel writes each token through the slot positions. From buffers
holding the same garbage, the payload and every scale byte must match,
padding rows included.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from max.gpu.host import DeviceContext
from std.testing import assert_equal
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from linalg.block_scaled_quantization import (
    grouped_quantize_dynamic_scaled_fp4_async,
    grouped_quantize_dynamic_scaled_fp4_from_sources,
)


def _next(mut state: UInt64) -> UInt64:
    state = state * 6364136223846793005 + 1442695040888963407
    return state >> 33


def test_from_sources[
    N: Int, num_groups: Int, top_k: Int
](
    ctx: DeviceContext,
    num_tokens: Int,
    routed_experts: Int,
    per_expert_sf: Bool,
    round_robin: Bool = False,
) raises:
    """Quantizing each token once matches quantizing each of its slots.

    Experts at or past `num_groups` belong to other ranks: their slots sort
    past the last range and neither kernel writes them. `round_robin` routes
    slot `k` of token `t` to expert `(t + k) % num_groups`, so every expert
    gets the same row count and whole scale tiles.
    """
    comptime dtype = DType.bfloat16
    comptime out_dtype = DType.uint8
    comptime scales_dtype = NVFP4_SF_DTYPE
    comptime K_tiles = ceildiv(N, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    comptime output_N = N // 2
    comptime scales_per_m_tile = K_tiles * SF_MN_GROUP_SIZE * SF_ATOM_K
    comptime SENTINEL = 0xEE
    comptime GARBAGE = 0x5A

    var num_slots = num_tokens * top_k
    var experts = unsafe_alloc[Int](max(num_slots, 1))
    var state = UInt64(0x9E3779B97F4A7C15) + UInt64(num_tokens)
    for t in range(num_tokens):
        for k in range(top_k):
            if round_robin:
                experts[t * top_k + k] = (t + k) % num_groups
                continue
            while True:
                var e = Int(_next(state) % UInt64(routed_experts))
                var fresh = True
                for j in range(k):
                    if experts[t * top_k + j] == e:
                        fresh = False
                if fresh:
                    experts[t * top_k + k] = e
                    break

    # A stable counting sort over the local groups, then one trailing range
    # for every other expert's slots.
    var counts = unsafe_alloc[Int](num_groups + 1)
    for g in range(num_groups + 1):
        counts[g] = 0
    for s in range(num_slots):
        counts[min(experts[s], num_groups)] += 1
    var row_offsets_host = unsafe_alloc[UInt32](num_groups + 1)
    var cursor = unsafe_alloc[Int](num_groups + 1)
    var at = 0
    for g in range(num_groups + 1):
        cursor[g] = at
        row_offsets_host[g] = UInt32(at)
        at += counts[g]
    var order_host = unsafe_alloc[UInt32](max(num_slots, 1))
    var restore_host = unsafe_alloc[UInt32](max(num_slots, 1))
    for s in range(num_slots):
        var key = min(experts[s], num_groups)
        order_host[cursor[key]] = UInt32(s)
        restore_host[s] = UInt32(cursor[key])
        cursor[key] += 1

    var scales_offsets_host = unsafe_alloc[UInt32](num_groups)
    var expert_ids_host = unsafe_alloc[Int32](num_groups)
    var sf_host = unsafe_alloc[Float32](num_groups)
    var tile_start = 0
    for g in range(num_groups):
        var rows = Int(row_offsets_host[g + 1]) - Int(row_offsets_host[g])
        scales_offsets_host[g] = UInt32(
            tile_start - Int(row_offsets_host[g]) // SF_MN_GROUP_SIZE
        )
        tile_start += ceildiv(rows, SF_MN_GROUP_SIZE)
        expert_ids_host[g] = Int32(g)
        sf_host[g] = (
            Float32(0.5) + Float32(0.03) * Float32(g) if per_expert_sf else 0.75
        )
    # As the graph op sizes it: a tile per 128 rows plus one per range that
    # can hold a row.
    var total_m_tiles = ceildiv(num_slots, SF_MN_GROUP_SIZE) + min(
        num_groups, num_slots
    )
    var total_scales = total_m_tiles * scales_per_m_tile

    var host_input = unsafe_alloc[Scalar[dtype]](max(num_tokens * N, 1))
    random(
        TileTensor(host_input, row_major(num_tokens, Idx[N])),
        min=-1.0,
        max=1.0,
    )
    # An all-zero scale group takes the zero-scale branch.
    if num_tokens > 0:
        for c in range(2 * NVFP4_SF_VECTOR_SIZE):
            host_input[c] = 0

    var dev_input = ctx.enqueue_create_buffer[dtype](max(num_tokens * N, 1))
    var dev_order = ctx.enqueue_create_buffer[.uint32](max(num_slots, 1))
    var dev_restore = ctx.enqueue_create_buffer[.uint32](max(num_slots, 1))
    var dev_row_offsets = ctx.enqueue_create_buffer[.uint32](num_groups + 1)
    var dev_scales_offsets = ctx.enqueue_create_buffer[.uint32](num_groups)
    var dev_expert_ids = ctx.enqueue_create_buffer[.int32](num_groups)
    var dev_sf = ctx.enqueue_create_buffer[.float32](num_groups)
    ctx.enqueue_copy(dev_input, host_input)
    ctx.enqueue_copy(dev_order, order_host)
    ctx.enqueue_copy(dev_restore, restore_host)
    ctx.enqueue_copy(dev_row_offsets, row_offsets_host)
    ctx.enqueue_copy(dev_scales_offsets, scales_offsets_host)
    ctx.enqueue_copy(dev_expert_ids, expert_ids_host)
    ctx.enqueue_copy(dev_sf, sf_host)

    # One slack row past the payload catches a store past the last slot.
    var payload_bytes = num_slots * output_N + output_N
    var scale_count = max(total_scales, 1)
    var sentinel_host = unsafe_alloc[Scalar[out_dtype]](payload_bytes)
    for i in range(payload_bytes):
        sentinel_host[i] = SENTINEL
    var garbage_host = unsafe_alloc[Scalar[scales_dtype]](scale_count)
    for i in range(scale_count):
        garbage_host[i] = bitcast[scales_dtype, 1](UInt8(GARBAGE))

    var input_tensor = TileTensor(
        dev_input, row_major(Coord(num_tokens, Idx[N]))
    )
    var row_offsets_tensor = TileTensor(
        dev_row_offsets, row_major(Idx[num_groups + 1])
    )
    var scales_offsets_tensor = TileTensor(
        dev_scales_offsets, row_major(Idx[num_groups])
    )
    var expert_ids_tensor = TileTensor(
        dev_expert_ids, row_major(Idx[num_groups])
    )
    var sf_tensor = TileTensor(dev_sf, row_major(Idx[num_groups]))
    var output_shape = Coord(num_slots, Idx[output_N])
    var scales_shape = Coord(
        total_m_tiles,
        Idx[K_tiles],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var ref_output = ctx.enqueue_create_buffer[out_dtype](payload_bytes)
    var ref_scales = ctx.enqueue_create_buffer[scales_dtype](scale_count)
    var new_output = ctx.enqueue_create_buffer[out_dtype](payload_bytes)
    var new_scales = ctx.enqueue_create_buffer[scales_dtype](scale_count)
    ctx.enqueue_copy(ref_output, sentinel_host)
    ctx.enqueue_copy(new_output, sentinel_host)
    ctx.enqueue_copy(ref_scales, garbage_host)
    ctx.enqueue_copy(new_scales, garbage_host)

    grouped_quantize_dynamic_scaled_fp4_async[
        row_indices_dtype=DType.uint32, row_index_divisor=top_k
    ](
        TileTensor(ref_output, row_major(output_shape)),
        TileTensor(ref_scales, row_major(scales_shape)),
        input_tensor,
        row_offsets_tensor,
        scales_offsets_tensor,
        expert_ids_tensor,
        sf_tensor,
        ctx,
        row_indices=TileTensor(dev_order, row_major(num_slots))
        .as_unsafe_any_origin()
        .as_imm(),
    )
    grouped_quantize_dynamic_scaled_fp4_from_sources[top_k=top_k](
        TileTensor(new_output, row_major(output_shape)),
        TileTensor(new_scales, row_major(scales_shape)),
        input_tensor,
        row_offsets_tensor,
        scales_offsets_tensor,
        expert_ids_tensor,
        sf_tensor,
        TileTensor(dev_restore, row_major(num_slots)),
        ctx,
    )

    var host_ref_output = unsafe_alloc[Scalar[out_dtype]](payload_bytes)
    var host_new_output = unsafe_alloc[Scalar[out_dtype]](payload_bytes)
    var host_ref_scales = unsafe_alloc[Scalar[scales_dtype]](scale_count)
    var host_new_scales = unsafe_alloc[Scalar[scales_dtype]](scale_count)
    ctx.enqueue_copy(host_ref_output, ref_output)
    ctx.enqueue_copy(host_new_output, new_output)
    ctx.enqueue_copy(host_ref_scales, ref_scales)
    ctx.enqueue_copy(host_new_scales, new_scales)
    ctx.synchronize()

    var payload_mismatches = 0
    for i in range(payload_bytes):
        payload_mismatches += Int(host_new_output[i] != host_ref_output[i])
    assert_equal(payload_mismatches, 0, msg="payload bytes differ")
    var scale_mismatches = 0
    for i in range(scale_count):
        scale_mismatches += Int(
            bitcast[DType.uint8, 1](host_new_scales[i])
            != bitcast[DType.uint8, 1](host_ref_scales[i])
        )
    assert_equal(scale_mismatches, 0, msg="scale bytes differ")

    host_ref_output.unsafe_free()
    host_new_output.unsafe_free()
    host_ref_scales.unsafe_free()
    host_new_scales.unsafe_free()
    sentinel_host.unsafe_free()
    garbage_host.unsafe_free()
    host_input.unsafe_free()
    experts.unsafe_free()
    counts.unsafe_free()
    cursor.unsafe_free()
    row_offsets_host.unsafe_free()
    order_host.unsafe_free()
    restore_host.unsafe_free()
    scales_offsets_host.unsafe_free()
    expert_ids_host.unsafe_free()
    sf_host.unsafe_free()


def main() raises:
    with DeviceContext() as ctx:
        # GLM-5.3's MoE: 6144 columns, 256 experts, top-8, at decode and at
        # a ragged batch with per-expert scale factors.
        test_from_sources[6144, 256, 8](ctx, 1, 256, per_expert_sf=False)
        test_from_sources[6144, 256, 8](ctx, 131, 256, per_expert_sf=True)
        # Ranges past one tile, whole tiles, and other ranks' slots.
        test_from_sources[512, 16, 4](ctx, 1000, 16, per_expert_sf=True)
        test_from_sources[512, 16, 4](
            ctx, 512, 16, per_expert_sf=False, round_robin=True
        )
        test_from_sources[512, 16, 4](ctx, 77, 64, per_expert_sf=True)
        test_from_sources[512, 16, 4](ctx, 0, 16, per_expert_sf=False)

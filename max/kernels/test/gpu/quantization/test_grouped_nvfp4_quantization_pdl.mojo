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
"""PDL ordering test for the grouped NVFP4 quantize kernel.

The grouped quantize launches with programmatic dependent launch, so it can
become resident while the kernel before it is still running. Its routing
operands (`row_offsets`, `scales_offsets`, `expert_ids`, `sf_tensor`) are
often written by that very kernel: a MoE layer derives the offsets from
`moe_create_indices` and, with per-expert activation scales, computes
`sf_tensor = 1 / input_scale` right before the quantize. The kernel must not
read any of them before `wait_on_dependent_grids()`.

Here a producer kernel releases its dependents on entry, sleeps, and only then
publishes the real operands over zeroed ones. A quantize that reads before its
wait sees every expert empty, exits every block early and leaves the output
at its sentinel; on the real model the stale table instead sent blocks past
the input rows (CUDA_ERROR_ILLEGAL_ADDRESS) or quantized against a stale
scale. The PDL output must match a run whose operands were published first.
"""

from std.math import ceildiv
from std.memory.alloc import unsafe_alloc
from std.time import sleep

from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.block_scaled_quantization import (
    grouped_quantize_dynamic_scaled_fp4_async,
)
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import launch_dependent_grids
from std.testing import assert_equal


def publish_after_delay[
    num_experts: Int
](
    row_offsets: MutPointer[UInt32, MutUntrackedOrigin],
    scales_offsets: MutPointer[UInt32, MutUntrackedOrigin],
    expert_ids: MutPointer[Int32, MutUntrackedOrigin],
    sf_tensor: MutPointer[Float32, MutUntrackedOrigin],
    src_row_offsets: MutPointer[UInt32, MutUntrackedOrigin],
    src_scales_offsets: MutPointer[UInt32, MutUntrackedOrigin],
    src_expert_ids: MutPointer[Int32, MutUntrackedOrigin],
    src_sf_tensor: MutPointer[Float32, MutUntrackedOrigin],
):
    """Releases dependents at once, then publishes the operands late."""
    launch_dependent_grids()
    sleep(0.05)
    for i in range(num_experts + 1):
        row_offsets[i] = src_row_offsets[i]
    for i in range(num_experts):
        scales_offsets[i] = src_scales_offsets[i]
        expert_ids[i] = src_expert_ids[i]
        sf_tensor[i] = src_sf_tensor[i]


def test_operands_from_predecessor_kernel(ctx: DeviceContext) raises:
    comptime dtype = DType.bfloat16
    comptime scales_dtype = NVFP4_SF_DTYPE
    comptime out_dtype = DType.uint8
    comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
    comptime N = 256
    comptime num_experts = 4
    comptime K_tiles = ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime output_N = N // 2
    comptime scales_per_m_tile = K_tiles * SF_MN_GROUP_SIZE * SF_ATOM_K
    comptime SENTINEL = 0xEE

    # An empty expert and a two-tile expert, as a decode/prefill mix routes.
    var counts: List[Int] = [3, 0, 5, 130]
    var row_offsets_host = unsafe_alloc[UInt32](num_experts + 1)
    var scales_offsets_host = unsafe_alloc[UInt32](num_experts)
    var expert_ids_host = unsafe_alloc[Int32](num_experts)
    var sf_tensor_host = unsafe_alloc[Float32](num_experts)
    var zeros_u32 = unsafe_alloc[UInt32](num_experts + 1)
    var zeros_i32 = unsafe_alloc[Int32](num_experts)
    var zeros_f32 = unsafe_alloc[Float32](num_experts)
    var tile_start = 0
    row_offsets_host[0] = 0
    zeros_u32[num_experts] = 0
    for i in range(num_experts):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(counts[i])
        scales_offsets_host[i] = UInt32(
            tile_start - Int(row_offsets_host[i]) // SF_MN_GROUP_SIZE
        )
        tile_start += ceildiv(counts[i], SF_MN_GROUP_SIZE)
        expert_ids_host[i] = Int32(i)
        sf_tensor_host[i] = Float32(1.5 + 0.5 * Float64(i))
        zeros_u32[i] = 0
        zeros_i32[i] = 0
        zeros_f32[i] = 0
    var total_tokens = Int(row_offsets_host[num_experts])
    var total_m_tiles = tile_start
    var total_scales = total_m_tiles * scales_per_m_tile

    var host_input = unsafe_alloc[Scalar[dtype]](total_tokens * N)
    random(
        TileTensor(host_input, row_major(total_tokens, Idx[N])),
        min=-1.0,
        max=1.0,
    )
    var dev_input = ctx.enqueue_create_buffer[dtype](total_tokens * N)
    ctx.enqueue_copy(dev_input, host_input)

    # The published operands: staging copies (the producer's source) and
    # the buffers the quantize reads, zeroed until the producer publishes.
    var src_row_offsets = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var src_scales_offsets = ctx.enqueue_create_buffer[.uint32](num_experts)
    var src_expert_ids = ctx.enqueue_create_buffer[.int32](num_experts)
    var src_sf_tensor = ctx.enqueue_create_buffer[.float32](num_experts)
    ctx.enqueue_copy(src_row_offsets, row_offsets_host)
    ctx.enqueue_copy(src_scales_offsets, scales_offsets_host)
    ctx.enqueue_copy(src_expert_ids, expert_ids_host)
    ctx.enqueue_copy(src_sf_tensor, sf_tensor_host)

    var sentinel_host = unsafe_alloc[Scalar[out_dtype]](total_tokens * output_N)
    for i in range(total_tokens * output_N):
        sentinel_host[i] = SENTINEL

    var input_tensor = TileTensor(dev_input, row_major(total_tokens, Idx[N]))
    var scales_shape = Coord(
        total_m_tiles,
        Idx[K_tiles],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    # Reference: the operands are already in place when the quantize runs.
    var ref_output = ctx.enqueue_create_buffer[out_dtype](
        total_tokens * output_N
    )
    var ref_scales = ctx.enqueue_create_buffer[scales_dtype](total_scales)
    ctx.enqueue_copy(ref_output, sentinel_host)
    grouped_quantize_dynamic_scaled_fp4_async(
        TileTensor(ref_output, row_major(total_tokens, Idx[output_N])),
        TileTensor(ref_scales, row_major(scales_shape)),
        input_tensor,
        TileTensor(src_row_offsets, row_major(Idx[num_experts + 1])),
        TileTensor(src_scales_offsets, row_major(Idx[num_experts])),
        TileTensor(src_expert_ids, row_major(Idx[num_experts])),
        TileTensor(src_sf_tensor, row_major(Idx[num_experts])),
        ctx,
    )
    ctx.synchronize()

    var row_offsets = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var scales_offsets = ctx.enqueue_create_buffer[.uint32](num_experts)
    var expert_ids = ctx.enqueue_create_buffer[.int32](num_experts)
    var sf_tensor = ctx.enqueue_create_buffer[.float32](num_experts)
    ctx.enqueue_copy(row_offsets, zeros_u32)
    ctx.enqueue_copy(scales_offsets, zeros_u32)
    ctx.enqueue_copy(expert_ids, zeros_i32)
    ctx.enqueue_copy(sf_tensor, zeros_f32)
    var output = ctx.enqueue_create_buffer[out_dtype](total_tokens * output_N)
    var scales = ctx.enqueue_create_buffer[scales_dtype](total_scales)
    ctx.enqueue_copy(output, sentinel_host)
    ctx.synchronize()

    ctx.enqueue_function[publish_after_delay[num_experts]](
        row_offsets,
        scales_offsets,
        expert_ids,
        sf_tensor,
        src_row_offsets,
        src_scales_offsets,
        src_expert_ids,
        src_sf_tensor,
        grid_dim=1,
        block_dim=1,
    )
    grouped_quantize_dynamic_scaled_fp4_async(
        TileTensor(output, row_major(total_tokens, Idx[output_N])),
        TileTensor(scales, row_major(scales_shape)),
        input_tensor,
        TileTensor(row_offsets, row_major(Idx[num_experts + 1])),
        TileTensor(scales_offsets, row_major(Idx[num_experts])),
        TileTensor(expert_ids, row_major(Idx[num_experts])),
        TileTensor(sf_tensor, row_major(Idx[num_experts])),
        ctx,
    )

    var host_ref_output = unsafe_alloc[Scalar[out_dtype]](
        total_tokens * output_N
    )
    var host_output = unsafe_alloc[Scalar[out_dtype]](total_tokens * output_N)
    var host_ref_scales = unsafe_alloc[Scalar[scales_dtype]](total_scales)
    var host_scales = unsafe_alloc[Scalar[scales_dtype]](total_scales)
    ctx.enqueue_copy(host_ref_output, ref_output)
    ctx.enqueue_copy(host_output, output)
    ctx.enqueue_copy(host_ref_scales, ref_scales)
    ctx.enqueue_copy(host_scales, scales)
    ctx.synchronize()

    var unwritten = 0
    for i in range(total_tokens * output_N):
        if Int(host_output[i]) == SENTINEL and Int(host_ref_output[i]) != (
            SENTINEL
        ):
            unwritten += 1
    assert_equal(
        unwritten,
        0,
        msg=(
            "quantize read its routing operands before the producer kernel"
            " published them"
        ),
    )
    for i in range(total_tokens * output_N):
        assert_equal(Int(host_output[i]), Int(host_ref_output[i]))
    for i in range(total_scales):
        assert_equal(
            host_scales[i].cast[DType.float32](),
            host_ref_scales[i].cast[DType.float32](),
        )

    row_offsets_host.unsafe_free()
    zeros_u32.unsafe_free()
    scales_offsets_host.unsafe_free()
    expert_ids_host.unsafe_free()
    sf_tensor_host.unsafe_free()
    zeros_i32.unsafe_free()
    zeros_f32.unsafe_free()
    host_input.unsafe_free()
    sentinel_host.unsafe_free()
    host_ref_output.unsafe_free()
    host_output.unsafe_free()
    host_ref_scales.unsafe_free()
    host_scales.unsafe_free()


def main() raises:
    with DeviceContext() as ctx:
        test_operands_from_predecessor_kernel(ctx)

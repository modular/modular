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

"""Edge-case and sanity tests for grouped_matmul_dynamic_scaled_fp8.

Most cases here are the zero edge cases where num_active_experts or
max_num_tokens_per_expert are zero: the function should return early without
error. One additional case uses a real (non-zero) per-expert a_offsets prefix
sum and checks the result against an independent host-side reference, so the
dispatcher (the SM100 persistent kernel, or the AMD/H100 naive fallback) is
exercised end-to-end rather than only compiled.

There are more comprehensive non-zero test cases for
grouped_matmul_sm100_blockwise_scaled_fp8 in
test_grouped_matmul_sm100_blockwise_fp8.mojo.
"""

from std.math import isnan
from std.random import rand

from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_dynamic_scaled_fp8,
)


def test_grouped_matmul_dynamic_scaled_fp8_zero_edge_case[
    num_experts: Int = 4,
    N: Int = 256,
    K: Int = 256,
](
    num_active_experts: Int,
    max_num_tokens_per_expert: Int,
    ctx: DeviceContext,
) raises:
    """Test grouped_matmul_dynamic_scaled_fp8 with zero edge cases.

    This test verifies that the function returns early without errors when
    either num_active_experts or max_num_tokens_per_expert (or both) are 0.

    Args:
        num_active_experts: Number of active experts (can be 0).
        max_num_tokens_per_expert: Maximum tokens per expert (can be 0).
        ctx: Device context for GPU operations.
    """
    comptime in_type = DType.float8_e4m3fn
    comptime out_type = DType.bfloat16
    comptime BLOCK_SCALE_K = 128

    print(
        "== test_grouped_matmul_dynamic_scaled_fp8_zero_edge_case",
        "num_experts:",
        num_experts,
        "N:",
        N,
        "K:",
        K,
        "num_active_experts:",
        num_active_experts,
        "max_num_tokens_per_expert:",
        max_num_tokens_per_expert,
    )

    # Use minimal buffer size for efficiency since nothing will execute
    var total_tokens = max(max_num_tokens_per_expert, 16)
    var num_offsets = max(num_active_experts + 1, 1)
    # Floor at 1: a device tile cannot wrap a zero-size buffer, and the
    # launcher early-returns before reading expert ids when there are none.
    var num_expert_ids = max(num_active_experts, 1)

    # Create host buffers
    var a_size = total_tokens * K

    var b_size = num_experts * N * K

    var c_size = total_tokens * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[in_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[in_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[out_type](c_size)

    var a_host = TileTensor(
        a_host_ptr,
        row_major(total_tokens, Idx[K]),
    )
    var b_host = TileTensor(
        b_host_ptr,
        row_major[num_experts, N, K](),
    )

    # Create offsets and expert_ids
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_offsets
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_expert_ids
    )

    # Set up offsets
    for i in range(num_offsets):
        a_offsets_host_ptr[i] = 0

    # Set up expert_ids
    for i in range(num_expert_ids):
        expert_ids_host_ptr[i] = Int32(i % num_experts)

    # Create scale buffers
    var a_scales_size = (K // BLOCK_SCALE_K) * total_tokens

    var b_scales_size = (
        num_experts * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)
    )

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        a_scales_size
    )
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        b_scales_size
    )

    var a_scales_host = TileTensor(
        a_scales_host_ptr,
        row_major(Idx[K // BLOCK_SCALE_K], total_tokens),
    )
    var b_scales_host = TileTensor(
        b_scales_host_ptr,
        row_major[num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K](),
    )

    # Initialize with random data
    random(a_host)
    random(b_host)
    random(a_scales_host)
    random(b_scales_host)

    # Create device buffers
    var a_device_buffer = ctx.enqueue_create_buffer[in_type](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[in_type](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[out_type](c_size)
    var a_offsets_device_buffer = ctx.enqueue_create_buffer[.uint32](
        num_offsets
    )
    var expert_ids_device_buffer = ctx.enqueue_create_buffer[.int32](
        num_expert_ids
    )
    var a_scales_device_buffer = ctx.enqueue_create_buffer[.float32](
        a_scales_size
    )
    var b_scales_device_buffer = ctx.enqueue_create_buffer[.float32](
        b_scales_size
    )

    var a_device = TileTensor(
        a_device_buffer,
        row_major(total_tokens, Idx[K]),
    )
    var b_device = TileTensor(
        b_device_buffer,
        row_major(Idx[num_experts], Idx[N], Idx[K]),
    )
    var c_device = TileTensor(
        c_device_buffer,
        row_major(total_tokens, Idx[N]),
    )
    var a_offsets_device = TileTensor(
        a_offsets_device_buffer,
        row_major(num_offsets),
    )
    var expert_ids_device = TileTensor(
        expert_ids_device_buffer,
        row_major(num_expert_ids),
    )
    var a_scales_device = TileTensor(
        a_scales_device_buffer,
        row_major(Idx[K // BLOCK_SCALE_K], total_tokens),
    )
    var b_scales_device = TileTensor(
        b_scales_device_buffer,
        row_major(
            Idx[num_experts], Idx[N // BLOCK_SCALE_K], Idx[K // BLOCK_SCALE_K]
        ),
    )

    # Copy to device
    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(a_offsets_device_buffer, a_offsets_host_ptr)
    if num_expert_ids > 0:
        ctx.enqueue_copy(expert_ids_device_buffer, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device_buffer, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device_buffer, b_scales_host_ptr)

    # Call with the specified zero edge case parameters
    # This should return early without error
    grouped_matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=1,
        n_scale_granularity=BLOCK_SCALE_K,
        k_scale_granularity=BLOCK_SCALE_K,
        transpose_b=True,
    ](
        c_device,
        a_device,
        b_device,
        a_scales_device,
        b_scales_device,
        a_offsets_device,
        expert_ids_device,
        max_num_tokens_per_expert=max_num_tokens_per_expert,
        num_active_experts=num_active_experts,
        ctx=ctx,
    )

    ctx.synchronize()
    print("  ✓ Successfully handled edge case")

    # Cleanup
    _ = a_device_buffer^
    _ = b_device_buffer^
    _ = c_device_buffer^
    _ = a_offsets_device_buffer^
    _ = expert_ids_device_buffer^
    _ = a_scales_device_buffer^
    _ = b_scales_device_buffer^


def test_grouped_matmul_dynamic_scaled_fp8_nonzero_sanity[
    num_experts: Int = 8,
    N: Int = 256,
    K: Int = 256,
](ctx: DeviceContext) raises:
    """Non-zero correctness case for grouped_matmul_dynamic_scaled_fp8.

    Every expert gets a real (non-zero) row count via a genuine prefix-sum
    a_offsets, unlike the all-zero-offsets cases above where every thread
    early-returns before touching memory. Checked against an independent
    host-side reference, mirroring
    test_batched_matmul_dynamic_scaled_fp8_naive.mojo.
    """
    comptime in_type = DType.float8_e4m3fn
    comptime out_type = DType.bfloat16
    comptime BLOCK_SCALE_K = 128
    comptime tokens_per_expert = 2
    comptime total_tokens = num_experts * tokens_per_expert

    print(
        "== test_grouped_matmul_dynamic_scaled_fp8_nonzero_sanity",
        "num_experts:",
        num_experts,
        "N:",
        N,
        "K:",
        K,
        "tokens_per_expert:",
        tokens_per_expert,
    )

    var a_size = total_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_tokens * N
    var a_scales_size = (K // BLOCK_SCALE_K) * total_tokens
    var b_scales_size = (
        num_experts * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)
    )

    var a_host_ptr = ctx.enqueue_create_host_buffer[in_type](a_size)
    var a_host = TileTensor(a_host_ptr, row_major[total_tokens, K]())
    var b_host_ptr = ctx.enqueue_create_host_buffer[in_type](b_size)
    var b_host = TileTensor(b_host_ptr, row_major[num_experts, N, K]())
    var c_host_ptr = ctx.enqueue_create_host_buffer[out_type](c_size)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[out_type](c_size)

    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_experts + 1
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_experts
    )

    # Real prefix sum, e.g. [0, 2, 4, ..., 16]: every expert is active with
    # `tokens_per_expert` real rows, unlike the all-zero offsets above.
    for i in range(num_experts + 1):
        a_offsets_host_ptr[i] = UInt32(i * tokens_per_expert)
    for i in range(num_experts):
        expert_ids_host_ptr[i] = Int32(i)

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        a_scales_size
    )
    var a_scales_host = TileTensor(
        a_scales_host_ptr, row_major[K // BLOCK_SCALE_K, total_tokens]()
    )
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        b_scales_size
    )
    var b_scales_host = TileTensor(
        b_scales_host_ptr,
        row_major[num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K](),
    )

    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    rand(a_scales_host._storage, a_scales_host.num_elements())
    rand(b_scales_host._storage, b_scales_host.num_elements())

    var a_device_buffer = ctx.enqueue_create_buffer[in_type](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[in_type](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[out_type](c_size)
    var a_offsets_device_buffer = ctx.enqueue_create_buffer[.uint32](
        num_experts + 1
    )
    var expert_ids_device_buffer = ctx.enqueue_create_buffer[.int32](
        num_experts
    )
    var a_scales_device_buffer = ctx.enqueue_create_buffer[.float32](
        a_scales_size
    )
    var b_scales_device_buffer = ctx.enqueue_create_buffer[.float32](
        b_scales_size
    )

    var a_device = TileTensor(a_device_buffer, row_major[total_tokens, K]())
    var b_device = TileTensor(b_device_buffer, row_major[num_experts, N, K]())
    var c_device = TileTensor(c_device_buffer, row_major[total_tokens, N]())
    var a_offsets_device = TileTensor(
        a_offsets_device_buffer, row_major[num_experts + 1]()
    )
    var expert_ids_device = TileTensor(
        expert_ids_device_buffer, row_major[num_experts]()
    )
    var a_scales_device = TileTensor(
        a_scales_device_buffer,
        row_major[K // BLOCK_SCALE_K, total_tokens](),
    )
    var b_scales_device = TileTensor(
        b_scales_device_buffer,
        row_major[num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K](),
    )

    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(a_offsets_device_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_device_buffer, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device_buffer, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device_buffer, b_scales_host_ptr)

    grouped_matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=1,
        n_scale_granularity=BLOCK_SCALE_K,
        k_scale_granularity=BLOCK_SCALE_K,
        transpose_b=True,
    ](
        c_device,
        a_device,
        b_device,
        a_scales_device,
        b_scales_device,
        a_offsets_device,
        expert_ids_device,
        max_num_tokens_per_expert=tokens_per_expert,
        num_active_experts=num_experts,
        ctx=ctx,
    )

    ctx.enqueue_copy(c_host_ptr, c_device_buffer)
    ctx.synchronize()

    # Independent host-side reference. Index-for-index match with
    # naive_blockwise_scaled_fp8_grouped_matmul_kernel
    # (linalg/fp8_quantization.mojo): a_scales is keyed by the *global*
    # token row (m_scale_granularity=1), b_scales by (expert, n block, k
    # block).
    for expert_slot in range(num_experts):
        var a_start_row = Int(a_offsets_host_ptr[expert_slot])
        var m_local_count = (
            Int(a_offsets_host_ptr[expert_slot + 1]) - a_start_row
        )
        var expert = Int(expert_ids_host_ptr[expert_slot])
        for m_local in range(m_local_count):
            var m_global = a_start_row + m_local
            for n in range(N):
                var accum = Scalar[DType.float32](0)
                for k in range(K):
                    var a_val = a_host_ptr[m_global * K + k].cast[
                        DType.float32
                    ]()
                    var b_val = b_host_ptr[expert * N * K + n * K + k].cast[
                        DType.float32
                    ]()
                    var a_scale = a_scales_host_ptr[
                        (k // BLOCK_SCALE_K) * total_tokens + m_global
                    ]
                    var b_scale = b_scales_host_ptr[
                        expert * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)
                        + (n // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)
                        + (k // BLOCK_SCALE_K)
                    ]
                    accum += (
                        a_val
                        * b_val
                        * a_scale.cast[DType.float32]()
                        * b_scale.cast[DType.float32]()
                    )
                c_ref_host_ptr[m_global * N + n] = accum.cast[out_type]()

    comptime rtol = 1e-2
    comptime atol = 1e-2
    var ndiff = 0
    for i in range(c_size):
        var got = c_host_ptr[i]
        var ref_val = c_ref_host_ptr[i]
        var got_f32 = got.cast[DType.float32]()
        var ref_f32 = ref_val.cast[DType.float32]()
        var diff = got_f32 - ref_f32
        var abs_diff = diff if diff >= 0 else -diff
        var abs_ref = ref_f32 if ref_f32 >= 0 else -ref_f32
        if isnan(got_f32) or abs_diff > atol + rtol * abs_ref:
            ndiff += 1
            if ndiff <= 5:
                print("  diff @", i, "got=", got, "ref=", ref_val)
    if ndiff > 0:
        raise Error(
            "grouped blockwise-fp8 matmul diverged from host reference at "
            + String(ndiff)
            + " of "
            + String(c_size)
            + " positions"
        )
    print("  PASSED")

    # Cleanup
    _ = a_device_buffer^
    _ = b_device_buffer^
    _ = c_device_buffer^
    _ = a_offsets_device_buffer^
    _ = expert_ids_device_buffer^
    _ = a_scales_device_buffer^
    _ = b_scales_device_buffer^


def main() raises:
    """Run all edge case tests for grouped_matmul_dynamic_scaled_fp8."""
    with DeviceContext() as ctx:
        # Test zero num_active_experts (with non-zero max_num_tokens_per_expert)
        test_grouped_matmul_dynamic_scaled_fp8_zero_edge_case[
            num_experts=4,
            N=256,
            K=256,
        ](num_active_experts=0, max_num_tokens_per_expert=64, ctx=ctx)

        # Test zero max_num_tokens_per_expert (with non-zero num_active_experts)
        test_grouped_matmul_dynamic_scaled_fp8_zero_edge_case[
            num_experts=4,
            N=256,
            K=256,
        ](num_active_experts=2, max_num_tokens_per_expert=0, ctx=ctx)

        # Test both zero
        test_grouped_matmul_dynamic_scaled_fp8_zero_edge_case[
            num_experts=4,
            N=256,
            K=256,
        ](num_active_experts=0, max_num_tokens_per_expert=0, ctx=ctx)

        # Non-zero sanity case: real per-expert offsets, checked against an
        # independent host reference (the cases above only exercise the
        # all-offsets-zero early return).
        test_grouped_matmul_dynamic_scaled_fp8_nonzero_sanity(ctx=ctx)

    print("\n✓ All edge case tests passed!")

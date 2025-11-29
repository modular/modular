# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Test edge cases for grouped_matmul_dynamic_scaled_fp8.

This test focuses on the zero edge cases where num_active_experts or
max_num_tokens_per_expert are zero. The function should return early without
error in these cases.

There are more comprehensive non-zero test cases for
grouped_matmul_sm100_blockwise_scaled_fp8 in
test_grouped_matmul_sm100_blockwise_fp8.mojo.
"""

from buffer.dimlist import Dim, DimList
from gpu.host import DeviceContext

from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_dynamic_scaled_fp8,
)
from utils.index import Index


def test_grouped_matmul_dynamic_scaled_fp8_zero_edge_case[
    num_experts: Int = 4,
    N: Int = 256,
    K: Int = 256,
](num_active_experts: Int, max_num_tokens_per_expert: Int, ctx: DeviceContext,):
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
    var total_tokens = max(16, max_num_tokens_per_expert)
    var num_offsets = max(1, num_active_experts + 1)
    var num_expert_ids = max(0, num_active_experts)

    # Create host buffers
    comptime static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = DimList(total_tokens, K)
    var a_host = HostNDBuffer[in_type, 2, static_a_shape](dynamic_a_shape)

    comptime static_b_shape = DimList(num_experts, N, K)
    var b_host = HostNDBuffer[in_type, 3, static_b_shape](static_b_shape)

    comptime static_c_shape = DimList(Dim(), N)
    var dynamic_c_shape = DimList(total_tokens, N)
    var c_host = HostNDBuffer[out_type, 2, static_c_shape](dynamic_c_shape)

    # Create offsets and expert_ids
    var a_offsets_host = HostNDBuffer[DType.uint32, 1, DimList(Dim())](
        num_offsets
    )
    var expert_ids_host = HostNDBuffer[DType.int32, 1](num_expert_ids)

    # Set up offsets
    for i in range(num_offsets):
        a_offsets_host.tensor[i] = 0

    # Set up expert_ids
    for i in range(num_expert_ids):
        expert_ids_host.tensor[i] = i % num_experts

    # Create scale buffers
    comptime static_a_scales_shape = DimList(K // BLOCK_SCALE_K, Dim())
    var dynamic_a_scales_shape = DimList(K // BLOCK_SCALE_K, total_tokens)
    var a_scales_host = HostNDBuffer[DType.float32, 2, static_a_scales_shape](
        dynamic_a_scales_shape
    )

    comptime static_b_scales_shape = DimList(
        num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K
    )
    var b_scales_host = HostNDBuffer[DType.float32, 3, static_b_scales_shape](
        static_b_scales_shape
    )

    # Initialize with random data
    random(a_host.tensor)
    random(b_host.tensor)
    random(a_scales_host.tensor)
    random(b_scales_host.tensor)

    # Create device buffers
    var a_device = DeviceNDBuffer[in_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[in_type, 3, static_b_shape](
        static_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[out_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var a_offsets_device = DeviceNDBuffer[DType.uint32, 1](num_offsets, ctx=ctx)
    var expert_ids_device = DeviceNDBuffer[DType.int32, 1](
        num_expert_ids, ctx=ctx
    )
    var a_scales_device = DeviceNDBuffer[
        DType.float32, 2, static_a_scales_shape
    ](dynamic_a_scales_shape, ctx=ctx)
    var b_scales_device = DeviceNDBuffer[
        DType.float32, 3, static_b_scales_shape
    ](static_b_scales_shape, ctx=ctx)

    # Copy to device
    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_offsets_device.buffer, a_offsets_host.tensor.data)
    if num_expert_ids > 0:
        ctx.enqueue_copy(expert_ids_device.buffer, expert_ids_host.tensor.data)
    ctx.enqueue_copy(a_scales_device.buffer, a_scales_host.tensor.data)
    ctx.enqueue_copy(b_scales_device.buffer, b_scales_host.tensor.data)

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
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        a_scales_device.tensor,
        b_scales_device.tensor,
        a_offsets_device.tensor,
        expert_ids_device.tensor,
        max_num_tokens_per_expert=max_num_tokens_per_expert,
        num_active_experts=num_active_experts,
        ctx=ctx,
    )

    ctx.synchronize()
    print("  ✓ Successfully handled edge case")

    # Cleanup
    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_offsets_host
    _ = expert_ids_host
    _ = a_scales_host
    _ = b_scales_host
    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_offsets_device
    _ = expert_ids_device
    _ = a_scales_device
    _ = b_scales_device


def main():
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

    print("\n✓ All edge case tests passed!")

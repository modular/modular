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

import itertools

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from gpu.host.info import B200
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from linalg.lora import shrink_qkv_permute_3mn_sm100 as shrink_qkv_permute_3mn
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from testing import assert_almost_equal

from utils import IndexList
from utils.index import Index


fn test[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    print(
        num_active_experts,
        "active of",
        num_experts,
        "experts of shape",
        expert_shape,
    )
    print("tokens:", end="")
    for i in range(len(num_tokens_by_expert)):
        print(num_tokens_by_expert[i], end=" ")
    print("expert ids:", end="")
    for i in range(len(expert_ids)):
        print(expert_ids[i], end=" ")
    print()

    comptime a_type = in_type
    comptime b_type = in_type
    comptime c_type = out_type

    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    # Total and max number of tokens
    total_num_tokens = 0
    max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]
        max_num_tokens_by_expert = max(
            max_num_tokens_by_expert, num_tokens_by_expert[i]
        )

    # Define shapes
    comptime static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = IndexList[2](total_num_tokens, K)
    var a_size = total_num_tokens * K

    comptime actual_N = 3 * N
    comptime static_c_ref_shape = DimList(Dim(), actual_N)
    var dynamic_c_ref_shape = IndexList[2](total_num_tokens, actual_N)
    var c_ref_size = total_num_tokens * actual_N

    comptime static_lora_c_shape = DimList(3, Dim(), N)
    var dynamic_lora_c_shape = IndexList[3](3, total_num_tokens, N)
    var lora_c_size = 3 * total_num_tokens * N

    comptime static_b_shape = DimList(num_experts, 3 * N, K)
    var b_size = num_experts * 3 * N * K

    comptime a_layout = Layout.row_major(UNKNOWN_VALUE, K)
    comptime b_layout = Layout.row_major(num_experts, 3 * N, K)
    comptime c_layout = Layout.row_major(3, UNKNOWN_VALUE, N)
    comptime c_ref_layout = Layout.row_major(UNKNOWN_VALUE, actual_N)

    # Host allocations
    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(lora_c_size)
    var c_ref_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_ref_size)
    var a_offsets_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        num_experts + 1
    )
    var expert_ids_host_ptr = UnsafePointer[Scalar[DType.int32]].alloc(
        num_experts
    )

    var a_host = LayoutTensor[a_type, a_layout](
        a_host_ptr,
        RuntimeLayout[a_layout].row_major(dynamic_a_shape),
    )
    var b_host = LayoutTensor[b_type, b_layout](
        b_host_ptr,
        RuntimeLayout[b_layout].row_major(IndexList[3](num_experts, 3 * N, K)),
    )
    var c_host = LayoutTensor[c_type, c_layout](
        c_host_ptr,
        RuntimeLayout[c_layout].row_major(dynamic_lora_c_shape),
    )
    var c_ref_host = LayoutTensor[c_type, c_ref_layout](
        c_ref_host_ptr,
        RuntimeLayout[c_ref_layout].row_major(dynamic_c_ref_shape),
    )

    # Setup offsets and expert ids
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = Int32(expert_ids[i])

    # Initialize matmul inputs
    random(a_host)
    random(b_host)

    # Device allocations
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev_buffer = ctx.enqueue_create_buffer[c_type](lora_c_size)
    var c_ref_dev_buffer = ctx.enqueue_create_buffer[c_type](c_ref_size)
    var a_offsets_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        num_experts + 1
    )
    var expert_ids_dev_buffer = ctx.enqueue_create_buffer[DType.int32](
        num_experts
    )

    var a_dev = NDBuffer[a_type, 2, _, static_a_shape](
        a_dev_buffer.unsafe_ptr(),
        DimList(total_num_tokens, K),
    )
    var b_dev = NDBuffer[b_type, 3, _, static_b_shape](
        b_dev_buffer.unsafe_ptr(),
        static_b_shape,
    )
    var c_dev = NDBuffer[c_type, 3, _, static_lora_c_shape](
        c_dev_buffer.unsafe_ptr(),
        DimList(3, total_num_tokens, N),
    )
    var c_ref_dev = NDBuffer[c_type, 2, _, static_c_ref_shape](
        c_ref_dev_buffer.unsafe_ptr(),
        DimList(total_num_tokens, actual_N),
    )
    var a_offsets_dev = NDBuffer[DType.uint32, 1](
        a_offsets_dev_buffer.unsafe_ptr(),
        num_experts + 1,
    )
    var expert_ids_dev = NDBuffer[DType.int32, 1](
        expert_ids_dev_buffer.unsafe_ptr(),
        num_experts,
    )

    # Move inputs to device
    ctx.enqueue_copy(a_dev_buffer, a_host_ptr)
    ctx.enqueue_copy(b_dev_buffer, b_host_ptr)
    ctx.enqueue_copy(a_offsets_dev_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_dev_buffer, expert_ids_host_ptr)
    ctx.synchronize()

    naive_grouped_matmul(
        c_ref_dev,
        a_dev,
        b_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    shrink_qkv_permute_3mn(
        c_dev,
        a_dev,
        b_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev_buffer)
    ctx.enqueue_copy(c_host_ptr, c_dev_buffer)
    ctx.synchronize()

    rtol = 1e-2

    for qkv_idx, m, n in itertools.product(
        range(3), range(total_num_tokens), range(N)
    ):
        var expect = c_ref_host[m, qkv_idx * N + n][0]

        var actual = c_host[qkv_idx, m, n][0]
        assert_almost_equal(
            actual,
            expect,
            msg=String(
                "qkv_idx: ",
                qkv_idx,
                " m: ",
                m,
                " n: ",
                n,
                " ref: ",
                expect,
                " actual: ",
                actual,
            ),
            rtol=rtol,
        )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_ref_host_ptr.free()
    a_offsets_host_ptr.free()
    expert_ids_host_ptr.free()
    _ = a_dev_buffer^
    _ = b_dev_buffer^
    _ = c_dev_buffer^
    _ = c_ref_dev_buffer^
    _ = a_offsets_dev_buffer^
    _ = expert_ids_dev_buffer^


def main():
    with DeviceContext() as ctx:
        # QKV perm dim test

        comptime is_sm100_kernel_applicable = ctx.default_device_info == B200

        comptime if not is_sm100_kernel_applicable:
            return

        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(192, 1024),
        ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(256, 256),
        ](1, [128], [0], ctx)

        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(16, 256),
        ](1, [128], [0], ctx)

        # unaligned matmul
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(1024, 256),
        ](1, [200], [0], ctx)

        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(512, 1024),
        ](1, [256], [0], ctx)

        # simple expert routing
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(256, 64),
        ](1, [128], [2], ctx)

        # simple aligned group routing
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(256, 64),
        ](3, [32, 32 * 3, 32 * 7], [2, 0, 1], ctx)

        # simple unaligned group routing
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(256, 64),
        ](2, [10, 60], [2, 0], ctx)

        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(2880, 512),
        ](2, [10, 60], [2, 0], ctx)

        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(5760, 512),
        ](2, [10, 60], [2, 0], ctx)

        # Multiple matmuls selecting part of experts
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(768, 1024),
        ](2, [128, 256], [0, 2], ctx)

        # Multiple matmuls selecting part of experts
        # num_tokens not multiple of tile size
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(1280, 1024),
        ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        # Multiple matmuls selecting part of experts
        # num_tokens not multiple of tile size
        # expert N dimension not multiple of 256
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(192, 1024),
        ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(1280, 16),
        ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(16, 1024),
        ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        # Multiple matmuls selecting part of experts with epilogue
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(768, 1024),
        ](2, [128, 256], [0, 2], ctx)

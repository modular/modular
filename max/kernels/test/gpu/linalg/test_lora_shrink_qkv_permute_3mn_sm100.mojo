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

from collections import OptionalReg

from buffer.dimlist import Dim, DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig
from testing import assert_almost_equal
from gpu.host.info import B200
from linalg.lora import shrink_qkv_permute_3mn_sm100 as shrink_qkv_permute_3mn

from utils import IndexList
from utils.index import Index
import itertools


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

    # Create host A C buffers
    comptime static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = DimList(total_num_tokens, K)
    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    comptime actual_N = 3 * N
    comptime static_c_shape = DimList(Dim(), actual_N)
    var dynamic_c_shape = DimList(total_num_tokens, actual_N)

    comptime static_lora_c_shape = DimList(3, Dim(), N)
    var dynamic_lora_c_shape = DimList(3, total_num_tokens, N)

    var c_host = HostNDBuffer[c_type, 3, static_lora_c_shape](
        dynamic_lora_c_shape
    )
    var c_ref_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var a_offsets_host = HostNDBuffer[DType.uint32, 1, DimList(Dim())](
        num_experts + 1
    )

    # Create host B buffers
    comptime static_b_shape = DimList(num_experts, 3 * N, K)
    var b_host = HostNDBuffer[b_type, 3, static_b_shape](static_b_shape)
    var expert_ids_host = HostNDBuffer[DType.int32, 1](num_experts)

    # Setup  offsets and expert ids
    a_offsets_host.tensor[0] = 0
    for i in range(num_active_experts):
        a_offsets_host.tensor[i + 1] = (
            a_offsets_host.tensor[i] + num_tokens_by_expert[i]
        )
        expert_ids_host.tensor[i] = expert_ids[i]

    # Initialize matmul inputs
    random(a_host.tensor)
    random(b_host.tensor)

    # Create device buffers
    var a_dev = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var c_dev = DeviceNDBuffer[c_type, 3, static_lora_c_shape](
        dynamic_lora_c_shape, ctx=ctx
    )
    var c_ref_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var b_dev = DeviceNDBuffer[b_type, 3, static_b_shape](
        static_b_shape, ctx=ctx
    )
    var a_offsets_dev = DeviceNDBuffer[DType.uint32, 1](
        num_experts + 1, ctx=ctx
    )
    var expert_ids_dev = DeviceNDBuffer[DType.int32, 1](num_experts, ctx=ctx)

    # Move inputs to device
    ctx.enqueue_copy(a_dev.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_dev.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_offsets_dev.buffer, a_offsets_host.tensor.data)
    ctx.enqueue_copy(expert_ids_dev.buffer, expert_ids_host.tensor.data)
    ctx.synchronize()

    naive_grouped_matmul(
        c_ref_dev.tensor,
        a_dev.tensor,
        b_dev.tensor,
        a_offsets_dev.tensor,
        expert_ids_dev.tensor,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    var c_dev_ndbuffer = c_dev.tensor

    shrink_qkv_permute_3mn(
        c_dev.tensor,
        a_dev.tensor,
        b_dev.tensor,
        a_offsets_dev.tensor,
        expert_ids_dev.tensor,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(c_ref_host.tensor.data, c_ref_dev.buffer)
    ctx.enqueue_copy(c_host.tensor.data, c_dev.buffer)
    ctx.synchronize()

    rtol = 1e-2
    c_ref_host_buffer = c_ref_host.tensor
    c_host_buffer = c_host.tensor

    for qkv_idx, m, n in itertools.product(
        range(3), range(total_num_tokens), range(N)
    ):
        var expect = c_ref_host_buffer[m, qkv_idx * N + n]

        var actual = c_host_buffer[qkv_idx, m, n]
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


def main():
    with DeviceContext() as ctx:
        # QKV perm dim test

        comptime is_sm100_kernel_applicable = ctx.default_device_info is B200

        @parameter
        if not is_sm100_kernel_applicable:
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

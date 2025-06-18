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

from collections.optional import OptionalReg
from math import ceildiv
from sys import alignof, has_nvidia_gpu_accelerator, simdwidthof

from algorithm.functional import elementwise
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import barrier, block_dim, block_idx, thread_idx
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.host.info import DEFAULT_GPU_ARCH
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    arange,
    fill,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static
from linalg import vendor_blas
from linalg.grouped_matmul import grouped_matmul_sm90, naive_grouped_matmul
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig, MatmulKernels
from memory import memset_zero, stack_allocation
from memory.pointer import _GPUAddressSpace as GPUAddressSpace
from testing import assert_almost_equal

from utils import IndexList
from utils.index import Index
from utils.numerics import FPUtils


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

    alias a_type = in_type
    alias b_type = in_type
    alias c_type = out_type

    alias N = expert_shape[0]
    alias K = expert_shape[1]

    # Total and max number of tokens
    total_num_tokens = 0
    max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]
        max_num_tokens_by_expert = max(
            max_num_tokens_by_expert, num_tokens_by_expert[i]
        )

    # Create host A C buffers
    alias static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = DimList(total_num_tokens, K)
    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    alias static_c_shape = DimList(Dim(), N)
    var dynamic_c_shape = DimList(total_num_tokens, N)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_ref_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var a_offsets_host = HostNDBuffer[DType.uint32, 1, DimList(Dim())](
        num_active_experts + 1
    )

    # Create host B buffers
    alias static_b_shape = DimList(num_experts, N, K)
    var b_host = HostNDBuffer[b_type, 3, static_b_shape](static_b_shape)
    var expert_ids_host = HostNDBuffer[DType.uint32, 1](num_active_experts)

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
    var c_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_ref_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var b_dev = DeviceNDBuffer[b_type, 3, static_b_shape](
        static_b_shape, ctx=ctx
    )
    var a_offsets_dev = DeviceNDBuffer[DType.uint32, 1](
        num_active_experts + 1, ctx=ctx
    )
    var expert_ids_dev = DeviceNDBuffer[DType.uint32, 1](
        num_active_experts, ctx=ctx
    )

    # Move inputs to device
    ctx.enqueue_copy(a_dev.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_dev.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_offsets_dev.buffer, a_offsets_host.tensor.data)
    ctx.enqueue_copy(expert_ids_dev.buffer, expert_ids_host.tensor.data)

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

    alias config = MatmulConfig[
        a_type,
        b_type,
        c_type,
        transpose_b=True,
        mma_shape = Index(64, 256, 16),
    ](
        block_tile_shape=Index(128, 256, 64),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=False,
    )

    grouped_matmul_sm90[
        transpose_b=True,
        wgmma_shape = Index(64, 256, 16),
        config=config,
    ](
        c_dev.tensor,
        a_dev.tensor,
        a_offsets_dev.tensor,
        max_num_tokens_by_expert,
        b_dev.tensor,
        expert_ids_dev.tensor,
        num_active_experts,
        ctx,
    )

    ctx.enqueue_copy(c_ref_host.tensor.data, c_ref_dev.buffer)
    ctx.enqueue_copy(c_host.tensor.data, c_dev.buffer)
    ctx.synchronize()

    rtol = 1e-2
    c_ref_host_buffer = c_ref_host.tensor
    c_host_buffer = c_host.tensor
    for m in range(total_num_tokens):
        for n in range(N):
            var expect = c_ref_host_buffer[m, n]
            var actual = c_host_buffer[m, n]
            assert_almost_equal(
                actual, expect, msg=String("m: ", m, " n: ", n), rtol=rtol
            )

    _ = c_dev
    _ = c_ref_dev
    _ = a_dev
    _ = b_dev
    _ = a_offsets_dev
    _ = expert_ids_dev
    _ = c_host
    _ = c_ref_host
    _ = a_host
    _ = b_host
    _ = a_offsets_host
    _ = expert_ids_host


fn main() raises:
    with DeviceContext() as ctx:
        # Single matmul
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(256, 256),
        ](1, List[Int](128), List[Int](0), ctx)

        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(512, 1024),
        ](1, List[Int](256), List[Int](0), ctx)

        # Multiple matmuls selecting part of experts
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(768, 1024),
        ](2, List[Int](128, 256), List[Int](0, 2), ctx)

        # Multiple matmuls selecting part of experts
        # num_tokesn not multiple of tile size
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(1280, 1024),
        ](4, List[Int](27, 1500, 300, 150), List[Int](0, 3, 2, 4), ctx)

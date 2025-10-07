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

"""
Expert Parallelism (EP) Communication Kernel.
"""

import compiler_internal as compiler
from gpu.host import DeviceBuffer, get_gpu_target
from gpu.host.info import is_gpu
from runtime.asyncrt import DeviceContextPtr
from sys.info import align_of, simd_width_of, size_of
from tensor_internal import InputTensor, OutputTensor
from tensor_internal.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)

from shmem import shmem_init, shmem_malloc
from shmem.ep_comm import EPMsgConfig


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Initialization Kernel
# ===-----------------------------------------------------------------------===#


@compiler.register("ep.init")
struct Struct_ep_init:
    @always_inline
    @staticmethod
    fn execute[
        dispatch_dtype: DType,
        combine_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int, //,
        target: StaticString,
    ](
        dev_ptrs: OutputTensor[dtype = DType.uint64, rank=2],
        atomic_counters_0: MutableInputTensor[dtype = DType.int32],
        atomic_counters_1: MutableInputTensor[dtype = DType.int32],
        context: DeviceContextPtr,
    ) raises:
        """This kernel initializes the vendor library for Expert Parallelism
        on the current GPU device. It also allocates symmetric memory buffers.

        Parameters:
            dispatch_dtype: DType used during token dispatch to experts.
            combine_dtype: DType used when combining expert outputs.
            hidden_size: Size of the model's hidden dimension.
            top_k: Number of experts each token is routed to.
            n_experts: Total number of experts across all GPUs.
            max_token_per_rank: Maximum number of tokens per GPU.
            n_gpus_per_node: Number of GPUs per node.
            target: Target for this kernel.

        Arguments:
            dev_ptrs: Output tensor to store device pointers. Shape [2, 3] where:
                     - First dimension: buffer groups (0=dispatch, 1=combine)
                     - Second dimension: buffer types (0=send, 1=recv, 2=recv_count)
            atomic_counters_0: Atomic counters for buffer group 0.
            atomic_counters_1: Atomic counters for buffer group 1.
            context: GPU device context
        """
        # Ensure this kernel only runs on GPU targets
        constrained[is_gpu[target](), "EP is only supported on GPU."]()
        var gpu_ctx = context.get_device_context()

        alias gpu_target = get_gpu_target()
        alias gpu_simd_width = simd_width_of[DType.uint8, target=gpu_target]()
        alias gpu_alignment = align_of[
            SIMD[DType.uint8, gpu_simd_width], target=gpu_target
        ]()

        # Calculate message sizes for dispatch and combine phases
        alias dispatch_msg_size = EPMsgConfig(
            dispatch_dtype, hidden_size, top_k, gpu_alignment
        ).msg_size()
        # Combine messages only contain the processed token
        alias combine_msg_size = hidden_size * size_of[combine_dtype]()

        # Calculate buffer sizes for dispatch phase
        alias dispatch_send_size = max_token_per_rank * dispatch_msg_size
        alias dispatch_recv_size = n_experts * max_token_per_rank * dispatch_msg_size

        # Calculate buffer sizes for combine phase
        alias combine_send_size = n_experts * max_token_per_rank * combine_msg_size
        alias combine_recv_size = top_k * max_token_per_rank * combine_msg_size

        # Initialize atomic counters to zero for synchronization
        # These counters coordinate work between different thread blocks.
        var atomic_counters_0_buf = DeviceBuffer(
            gpu_ctx,
            atomic_counters_0._ptr,
            atomic_counters_0.size(),
            owning=False,
        )
        gpu_ctx.enqueue_memset(atomic_counters_0_buf, Int32(0))
        var atomic_counters_1_buf = DeviceBuffer(
            gpu_ctx,
            atomic_counters_1._ptr,
            atomic_counters_1.size(),
            owning=False,
        )
        gpu_ctx.enqueue_memset(atomic_counters_1_buf, Int32(0))

        # Initialize the SHMEM library for this GPU
        _ = shmem_init(Int(gpu_ctx.id()), n_gpus_per_node)

        # Allocate SHMEM buffers for dispatch phase
        var dispatch_send_p = shmem_malloc[DType.uint8](dispatch_send_size)
        var dispatch_recv_p = shmem_malloc[DType.uint8](dispatch_recv_size)
        var dispatch_recv_count_p = shmem_malloc[DType.uint64](n_experts)

        # Allocate SHMEM buffers for combine phase
        var combine_send_p = shmem_malloc[DType.uint8](combine_send_size)
        var combine_recv_p = shmem_malloc[DType.uint8](combine_recv_size)
        var combine_recv_count_p = shmem_malloc[DType.uint64](n_experts)

        # Initialize receive count buffers to MAX_FINITE
        # This sentinel value indicates that no data has been received yet
        var dispatch_recv_count_buf = DeviceBuffer(
            gpu_ctx, dispatch_recv_count_p, n_experts, owning=False
        )
        gpu_ctx.enqueue_memset(dispatch_recv_count_buf, UInt64.MAX_FINITE)

        var combine_recv_count_buf = DeviceBuffer(
            gpu_ctx, combine_recv_count_p, n_experts, owning=False
        )
        gpu_ctx.enqueue_memset(combine_recv_count_buf, UInt64.MAX_FINITE)

        # Group 0: Dispatch phase buffer pointers
        dev_ptrs[0, 0] = UInt64(Int(dispatch_send_p))
        dev_ptrs[0, 1] = UInt64(Int(dispatch_recv_p))
        dev_ptrs[0, 2] = UInt64(Int(dispatch_recv_count_p))

        # Group 1: Combine phase buffer pointers
        dev_ptrs[1, 0] = UInt64(Int(combine_send_p))
        dev_ptrs[1, 1] = UInt64(Int(combine_recv_p))
        dev_ptrs[1, 2] = UInt64(Int(combine_recv_count_p))

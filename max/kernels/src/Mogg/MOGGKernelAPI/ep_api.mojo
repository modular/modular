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
from layout import Layout
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, get_safe_task_id
from sys.info import align_of, simd_width_of, size_of
from sys.intrinsics import _unsafe_aliasing_address_to_pointer
from tensor_internal import InputTensor, OutputTensor
from tensor_internal.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)

from shmem import shmem_init_thread, shmem_module_init, shmem_malloc
from shmem.ep_comm import (
    BF16TokenFormat,
    dispatch_kernel,
    dispatch_cb_kernel,
    combine_kernel,
    combine_cb_kernel,
)


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
        alias token_fmt_type = BF16TokenFormat[
            output_layout = Layout(), hidden_size, top_k, gpu_alignment
        ]
        alias dispatch_msg_size = token_fmt_type.msg_size()
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
        shmem_init_thread(gpu_ctx, n_gpus_per_node)

        # Allocate SHMEM buffers for dispatch phase
        var dispatch_send_p = shmem_malloc[DType.uint8](
            UInt(dispatch_send_size)
        )
        var dispatch_recv_p = shmem_malloc[DType.uint8](
            UInt(dispatch_recv_size)
        )
        var dispatch_recv_count_p = shmem_malloc[DType.uint64](UInt(n_experts))

        # Allocate SHMEM buffers for combine phase
        var combine_send_p = shmem_malloc[DType.uint8](UInt(combine_send_size))
        var combine_recv_p = shmem_malloc[DType.uint8](UInt(combine_recv_size))
        var combine_recv_count_p = shmem_malloc[DType.uint64](UInt(n_experts))

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


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Dispatch Kernel
# ===-----------------------------------------------------------------------===#


@compiler.register("ep.dispatch")
struct Struct_ep_dispatch:
    @always_inline
    @staticmethod
    fn execute[
        dispatch_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int, //,
        target: StaticString,
    ](
        atomic_counters_0: MutableInputTensor[dtype = DType.int32, rank=1],
        input_tokens: InputTensor[dtype=dispatch_dtype, rank=2],
        topk_ids: InputTensor[dtype = DType.int32, rank=2],
        send_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_count_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism dispatch kernel.

        This function launches the dispatch_kernel from ep_comm.mojo to
        initiate token distribution across expert devices. The kernel uses
        SHMEM for efficient GPU-to-GPU communication without CPU involvement.

        Parameters:
            dispatch_dtype: Data type for tokens during dispatch phase.
            hidden_size: Model hidden dimension size.
            top_k: Number of experts each token is routed to.
            n_experts: Total experts across all devices.
            max_token_per_rank: Maximum tokens any device can send.
            n_gpus_per_node: GPUs per physical node.
            n_nodes: Number of physical nodes.
            target: Target.

        Arguments:
            atomic_counters_0: Synchronization counters for buffer group 0.
                Used to coordinate between different thread blocks.
            input_tokens: Tokens to dispatch to experts.
            topk_ids: Expert assignments from router.
            send_ptrs: SHMEM send buffer pointers for each local GPU.
            recv_ptrs: SHMEM receive buffer pointers for each local GPU.
            recv_count_ptrs: SHMEM receive count buffer pointers for each local
                GPU.
            context: Device context pointer
        """
        # Ensure this kernel only runs on GPU targets
        constrained[is_gpu[target](), "EP is only supported on GPU."]()

        var input_tokens_tensor = input_tokens.to_layout_tensor()
        var topk_ids_tensor = topk_ids.to_layout_tensor()

        # Ensure the shape for the input tensors are correct
        constrained[
            input_tokens_tensor.shape[1]() == hidden_size,
            "EP dispatch: input tokens shape doesn't match hidden size.",
        ]()
        constrained[
            topk_ids_tensor.shape[1]() == top_k,
            "EP dispatch: topk ids shape doesn't match top k.",
        ]()

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        alias hw_info = gpu_ctx.default_device_info
        alias gpu_target = get_gpu_target()
        alias gpu_simd_width = simd_width_of[DType.uint8, target=gpu_target]()
        alias gpu_alignment = align_of[
            SIMD[DType.uint8, gpu_simd_width], target=gpu_target
        ]()
        alias token_fmt_type = BF16TokenFormat[
            output_layout = Layout(), hidden_size, top_k, gpu_alignment
        ]

        alias n_ranks = n_gpus_per_node * n_nodes

        alias dispatch = dispatch_kernel[
            dispatch_dtype,
            hw_info.max_thread_block_size,
            input_tokens_tensor.layout,
            topk_ids_tensor.layout,
            hw_info.sm_count,
            n_experts // (hw_info.max_thread_block_size // hw_info.warp_size),
            n_experts,
            n_ranks,
            max_token_per_rank,
            token_fmt_type,
        ]

        @always_inline
        @parameter
        fn description_fn() -> String:
            # fmt: off
            return String(
                "dispatch_dtype=", dispatch_dtype,
                ";hidden_size=", hidden_size,
                ";top_k=", top_k,
                ";n_experts=", n_experts,
                ";max_token_per_rank=", max_token_per_rank,
                ";n_gpus_per_node=", n_gpus_per_node,
                ";n_nodes=", n_nodes,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.dispatch",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var func = gpu_ctx.compile_function[dispatch]()
            shmem_module_init(func)

            var send_buf_p = _unsafe_aliasing_address_to_pointer[UInt8](
                Int(send_ptrs[gpu_id])
            )
            var recv_buf_p = _unsafe_aliasing_address_to_pointer[UInt8](
                Int(recv_ptrs[gpu_id])
            )
            var recv_count_p = _unsafe_aliasing_address_to_pointer[UInt64](
                Int(recv_count_ptrs[gpu_id])
            )

            gpu_ctx.enqueue_function(
                func,
                input_tokens_tensor,
                topk_ids_tensor,
                send_buf_p,
                recv_buf_p,
                recv_count_p,
                atomic_counters_0._ptr,
                Int32(gpu_id),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )


@compiler.register("ep.dispatch_cb")
struct Struct_ep_dispatch_cb:
    @always_inline
    @staticmethod
    fn execute[
        dispatch_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int, //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2],
        row_offsets: OutputTensor[dtype = DType.uint32, rank=1],
        expert_ids: OutputTensor[dtype = DType.int32, rank=1],
        expert_usage_stats_host: OutputTensor[dtype = DType.uint32, rank=1],
        src_info: OutputTensor[dtype = DType.int32, rank=2],
        atomic_counters_0: MutableInputTensor[dtype = DType.int32, rank=1],
        recv_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_count_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel.

        This function launches the dispatch_cb_kernel from ep_comm.mojo to
        complete the token dispatch phase. It waits for all local SHMEM
        transfers to finish, then organizes the received tokens for grouped
        matmul computation.

        Parameters:
            dispatch_dtype: Data type for tokens during dispatch phase.
            hidden_size: Model hidden dimension size.
            top_k: Number of experts each token is routed to.
            n_experts: Total experts across all devices.
            max_token_per_rank: Maximum tokens any device can send.
            n_gpus_per_node: GPUs per physical node.
            n_nodes: Number of physical nodes.
            target: Target.

        Arguments:
            output_tokens: Aggregated tokens ready for grouped matmul
                computation.
            row_offsets: Cumulative token counts for grouped matmul.
            expert_ids: Local expert IDs for grouped matmul.
            expert_usage_stats_host: Statistics for grouped matmul kernel.
            src_info: Source routing information for combine phase.
            atomic_counters_0: Synchronization counters from dispatch phase.
            recv_ptrs: SHMEM receive buffer pointers for each local GPU.
            recv_count_ptrs: SHMEM receive count buffer pointers for each local
                GPU.
            context: Device context pointer
        """
        # Ensure this kernel only runs on GPU targets
        constrained[is_gpu[target](), "EP is only supported on GPU."]()

        var output_tokens_tensor = output_tokens.to_layout_tensor()
        var row_offsets_tensor = row_offsets.to_layout_tensor()
        var expert_ids_tensor = expert_ids.to_layout_tensor()
        var src_info_tensor = src_info.to_layout_tensor()

        # Ensure the shape for the input tensors are correct
        constrained[
            output_tokens_tensor.shape[1]() == hidden_size,
            "EP dispatch_cb: output tokens shape doesn't match hidden size.",
        ]()

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        alias hw_info = gpu_ctx.default_device_info
        alias gpu_target = get_gpu_target()
        alias gpu_simd_width = simd_width_of[DType.uint8, target=gpu_target]()
        alias gpu_alignment = align_of[
            SIMD[DType.uint8, gpu_simd_width], target=gpu_target
        ]()

        alias n_ranks = n_gpus_per_node * n_nodes

        constrained[dispatch_dtype == DType.bfloat16]()
        var format_handler = BF16TokenFormat[hidden_size, top_k, gpu_alignment](
            output_tokens_tensor.bitcast[DType.bfloat16]()
        )

        alias dispatch_cb = dispatch_cb_kernel[
            hw_info.max_thread_block_size,
            output_tokens_tensor.layout,
            row_offsets_tensor.layout,
            expert_ids_tensor.layout,
            src_info_tensor.layout,
            hw_info.sm_count,
            1,
            n_experts,
            n_ranks,
            max_token_per_rank,
            __type_of(format_handler),
        ]

        @always_inline
        @parameter
        fn description_fn() -> String:
            # fmt: off
            return String(
                "dispatch_dtype=", dispatch_dtype,
                ";hidden_size=", hidden_size,
                ";top_k=", top_k,
                ";n_experts=", n_experts,
                ";max_token_per_rank=", max_token_per_rank,
                ";n_gpus_per_node=", n_gpus_per_node,
                ";n_nodes=", n_nodes,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.dispatch_cb",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var recv_buf_p = _unsafe_aliasing_address_to_pointer[UInt8](
                Int(recv_ptrs[gpu_id])
            )
            var recv_count_p = _unsafe_aliasing_address_to_pointer[UInt64](
                Int(recv_count_ptrs[gpu_id])
            )

            gpu_ctx.enqueue_function[dispatch_cb](
                format_handler,
                row_offsets_tensor,
                expert_ids_tensor,
                src_info_tensor,
                recv_buf_p,
                recv_count_p,
                atomic_counters_0._ptr,
                Int32(gpu_id),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )

            # The grouped matmul kernel needs this tensor to be filled
            expert_usage_stats_host[0] = (
                n_ranks * max_token_per_rank
            )  # max number of tokens per expert
            expert_usage_stats_host[1] = (
                n_experts // n_ranks
            )  # number of active experts


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Combine Kernel
# ===-----------------------------------------------------------------------===#


@compiler.register("ep.combine")
struct Struct_ep_combine:
    @always_inline
    @staticmethod
    fn execute[
        combine_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int, //,
        target: StaticString,
    ](
        atomic_counters_1: MutableInputTensor[dtype = DType.int32, rank=1],
        input_tokens: InputTensor[dtype=combine_dtype, rank=2],
        src_info: InputTensor[dtype = DType.int32, rank=2],
        send_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_count_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism combine kernel.

        This function launches the combine_kernel from ep_comm.mojo to initiate
        sending expert outputs back to their original devices. The kernel uses
        source routing information to determine destinations.

        Parameters:
            combine_dtype: Data type for tokens during combine phase.
            hidden_size: Model hidden dimension size.
            top_k: Number of experts each token was routed to.
            n_experts: Total experts across all devices.
            max_token_per_rank: Maximum tokens any device can send.
            n_gpus_per_node: GPUs per physical node.
            n_nodes: Number of physical nodes.
            target: Target.

        Arguments:
            atomic_counters_1: Synchronization counters for buffer group 1.
                Used to coordinate between different thread blocks.
            input_tokens: Expert output tokens to send back to original devices.
            src_info: Source routing information from dispatch phase.
            send_ptrs: SHMEM send buffer pointers for each local GPU.
            recv_ptrs: SHMEM receive buffer pointers for each local GPU.
            recv_count_ptrs: SHMEM receive count buffer pointers for each local
                GPU.
            context: Device context pointer.
        """
        # Ensure this kernel only runs on GPU targets
        constrained[is_gpu[target](), "EP is only supported on GPU."]()

        var input_tokens_tensor = input_tokens.to_layout_tensor()
        var src_info_tensor = src_info.to_layout_tensor()

        # Ensure the shape for the input tensors are correct
        constrained[
            input_tokens_tensor.shape[1]() == hidden_size,
            "EP combine: input tokens shape doesn't match hidden size.",
        ]()

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        alias hw_info = gpu_ctx.default_device_info
        alias combine_msg_size = hidden_size * size_of[combine_dtype]()

        alias n_ranks = n_gpus_per_node * n_nodes

        alias combine = combine_kernel[
            combine_dtype,
            hw_info.max_thread_block_size,
            input_tokens_tensor.layout,
            src_info_tensor.layout,
            hw_info.sm_count,
            top_k,
            n_experts,
            n_ranks,
            combine_msg_size,
            max_token_per_rank,
        ]

        @always_inline
        @parameter
        fn description_fn() -> String:
            # fmt: off
            return String(
                "combine_dtype=", combine_dtype,
                ";hidden_size=", hidden_size,
                ";top_k=", top_k,
                ";n_experts=", n_experts,
                ";max_token_per_rank=", max_token_per_rank,
                ";n_gpus_per_node=", n_gpus_per_node,
                ";n_nodes=", n_nodes,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.combine",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var func = gpu_ctx.compile_function[combine]()
            shmem_module_init(func)

            var send_buf_p = _unsafe_aliasing_address_to_pointer[UInt8](
                Int(send_ptrs[gpu_id])
            )
            var recv_buf_p = _unsafe_aliasing_address_to_pointer[UInt8](
                Int(recv_ptrs[gpu_id])
            )
            var recv_count_p = _unsafe_aliasing_address_to_pointer[UInt64](
                Int(recv_count_ptrs[gpu_id])
            )

            gpu_ctx.enqueue_function(
                func,
                input_tokens_tensor,
                src_info_tensor,
                send_buf_p,
                recv_buf_p,
                recv_count_p,
                atomic_counters_1._ptr,
                Int32(gpu_id),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )


@compiler.register("ep.combine_cb")
struct Struct_ep_combine_cb:
    @always_inline
    @staticmethod
    fn execute[
        combine_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int, //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=combine_dtype, rank=3],
        atomic_counters_1: MutableInputTensor[dtype = DType.int32, rank=1],
        recv_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_count_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism combine completion kernel.

        This function launches the combine_cb_kernel from ep_comm.mojo to complete
        the expert output gathering phase. It waits for all SHMEM transfers to
        finish, then organizes tokens back to their original format.

        Parameters:
            combine_dtype: Data type for tokens during combine phase.
            hidden_size: Model hidden dimension size.
            top_k: Number of experts each token was routed to.
            n_experts: Total experts across all devices.
            max_token_per_rank: Maximum tokens any device can receive.
            n_gpus_per_node: GPUs per physical node.
            n_nodes: Number of physical nodes.
            target: Target.

        Arguments:
            output_tokens: Final output tensor with expert results.
            atomic_counters_1: Synchronization counters from combine phase.
            recv_ptrs: SHMEM receive buffer pointers for each local GPU.
            recv_count_ptrs: SHMEM receive count buffer pointers for each local
                GPU.
            context: Device context pointer.
        """
        # Ensure this kernel only runs on GPU targets
        constrained[is_gpu[target](), "EP is only supported on GPU."]()

        var output_tokens_tensor = output_tokens.to_layout_tensor()

        # Ensure the shape for the output tensor is correct
        constrained[
            output_tokens_tensor.shape[2]() == hidden_size,
            "EP combine: output tokens shape doesn't match hidden size.",
        ]()

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        alias hw_info = gpu_ctx.default_device_info
        alias combine_msg_size = hidden_size * size_of[combine_dtype]()

        alias n_ranks = n_gpus_per_node * n_nodes

        alias combine_cb = combine_cb_kernel[
            combine_dtype,
            hw_info.max_thread_block_size,
            output_tokens_tensor.layout,
            hw_info.sm_count,
            1,
            top_k,
            n_experts,
            n_ranks,
            combine_msg_size,
            max_token_per_rank,
        ]

        @always_inline
        @parameter
        fn description_fn() -> String:
            # fmt: off
            return String(
                "combine_dtype=", combine_dtype,
                ";hidden_size=", hidden_size,
                ";top_k=", top_k,
                ";n_experts=", n_experts,
                ";max_token_per_rank=", max_token_per_rank,
                ";n_gpus_per_node=", n_gpus_per_node,
                ";n_nodes=", n_nodes,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.combine_cb",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var recv_buf_p = _unsafe_aliasing_address_to_pointer[UInt8](
                Int(recv_ptrs[gpu_id])
            )
            var recv_count_p = _unsafe_aliasing_address_to_pointer[UInt64](
                Int(recv_count_ptrs[gpu_id])
            )

            gpu_ctx.enqueue_function[combine_cb](
                output_tokens_tensor,
                recv_buf_p,
                recv_count_p,
                atomic_counters_1._ptr,
                Int32(gpu_id),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )

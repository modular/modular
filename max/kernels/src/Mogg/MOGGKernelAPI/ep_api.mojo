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

from collections import OptionalReg

import compiler_internal as compiler
from gpu.grid_controls import pdl_launch_attributes
from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from gpu.host.info import is_gpu
from layout import Layout, LayoutTensor, RuntimeLayout
from memory import LegacyUnsafePointer
from utils.index import IndexList

comptime OpaquePointer = LegacyUnsafePointer[
    mut=True, NoneType, origin=MutAnyOrigin
]
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, get_safe_task_id
from sys.info import align_of, simd_width_of, size_of
from sys.ffi import external_call
from tensor import InputTensor, OutputTensor
from tensor.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from tensor.managed_tensor_slice import (
    _FusedOutputTensor as FusedOutputTensor,
)

from shmem import (
    shmem_init_thread,
    shmem_malloc,
    shmem_module_init,
    shmem_my_pe,
)
from shmem.ep_comm import (
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    dispatch_kernel,
    dispatch_cb_kernel,
    combine_kernel,
    combine_cb_kernel,
    elementwise_epilogue_type,
    fused_silu_kernel,
    fused_silu_fp8_kernel,
)


# This should eventually be moved to ffi.mojo with a more general global cache method
# cache key is a string and cache value is a pointer.
@always_inline
fn global_cache_lookup(key: String) -> OpaquePointer:
    return external_call["KGEN_CompilerRT_GetGlobalOrNull", OpaquePointer](
        key.unsafe_ptr(), key.byte_length()
    )


@always_inline
fn global_cache_insert(key: String, value: OpaquePointer):
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(key),
        value,
    )


@always_inline
fn unsafe_aliasing_address_to_device_buffer[
    dtype: DType,
](var addr: Int, size: Int, ctx: DeviceContext) -> DeviceBuffer[dtype]:
    return DeviceBuffer[dtype](
        ctx,
        UnsafePointer[Scalar[dtype], MutExternalOrigin](
            unsafe_from_address=addr
        ),
        size,
        owning=False,
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
        n_gpus_per_node: Int,
        dispatch_scale_granularity: StaticString,
        dispatch_scale_dtype: DType,
        //,
        target: StaticString,
    ](
        dev_ptrs: OutputTensor[dtype = DType.uint64, rank=2],
        my_rank_tensor: OutputTensor[dtype = DType.int32, rank=1],
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
            dispatch_scale_granularity: FP8 quant granularity of the dispatch tokens.
            dispatch_scale_dtype: DType of the dispatch scale.
            target: Target for this kernel.

        Arguments:
            dev_ptrs: Output tensor to store device pointers. Shape [2, 3] where:
                     - First dimension: buffer groups (0=dispatch, 1=combine)
                     - Second dimension: buffer types (0=send, 1=recv, 2=recv_count)
            my_rank_tensor: Output tensor to store current device's rank.
            atomic_counters_0: Atomic counters for buffer group 0.
            atomic_counters_1: Atomic counters for buffer group 1.
            context: GPU device context
        """
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."
        var gpu_ctx = context.get_device_context()

        comptime gpu_target = get_gpu_target()
        comptime gpu_simd_width = simd_width_of[
            DType.uint8, target=gpu_target
        ]()
        comptime gpu_alignment = align_of[
            SIMD[DType.uint8, gpu_simd_width], target=gpu_target
        ]()

        # Calculate buffer sizes for dispatch phase
        var dispatch_msg_size: Int

        # Infer message sizes for dispatch phases
        @parameter
        if dispatch_dtype.is_float8():
            comptime token_fmt_type = BlockwiseFP8TokenFormat[
                fp8_dtype=dispatch_dtype,
                scales_dtype=dispatch_scale_dtype,
                output_layout = Layout(),
                scales_layout = Layout(),
                hidden_size,
                top_k,
                gpu_alignment,
            ]
            dispatch_msg_size = token_fmt_type.msg_size()

        else:
            comptime token_fmt_type = BF16TokenFormat[
                output_layout = Layout(), hidden_size, top_k, gpu_alignment
            ]
            dispatch_msg_size = token_fmt_type.msg_size()

        var dispatch_send_size = max_token_per_rank * dispatch_msg_size
        var dispatch_recv_size = (
            n_experts * max_token_per_rank * dispatch_msg_size
        )

        # Calculate buffer sizes for combine phase
        # Combine messages only contain the processed token
        comptime combine_msg_size = hidden_size * size_of[combine_dtype]()
        comptime combine_send_size = n_experts * max_token_per_rank * combine_msg_size
        comptime combine_recv_size = top_k * max_token_per_rank * combine_msg_size

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

        # Store current device's rank
        var my_rank = Int32(shmem_my_pe())
        my_rank_tensor[0] = my_rank


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Dispatch Kernel
# ===-----------------------------------------------------------------------===#


@compiler.register("ep.dispatch")
struct Struct_ep_dispatch:
    @always_inline
    @staticmethod
    fn execute[
        input_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
    ](
        atomic_counters_0: MutableInputTensor[dtype = DType.int32, rank=1],
        input_tokens: InputTensor[dtype=input_dtype, rank=2],
        topk_ids: InputTensor[dtype = DType.int32, rank=2],
        send_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_count_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism dispatch kernel.

        This function launches the dispatch_kernel from ep_comm.mojo to
        initiate token distribution across expert devices. In multi-node
        scenarios, all the communication buffers need to be allocated using
        `shmem_malloc`.

        Parameters:
            input_dtype: Data type of the input tokens.
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
            send_ptrs: Send buffer pointers for each local GPU.
            recv_ptrs: Receive buffer pointers for each local GPU.
            recv_count_ptrs: Receive count buffer pointers for each local GPU.
            context: Device context pointer
        """
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."

        var input_tokens_tensor = (
            input_tokens.to_layout_tensor().get_immutable()
        )
        var topk_ids_tensor = topk_ids.to_layout_tensor().get_immutable()

        # Ensure the shape for the input tensors are correct
        __comptime_assert (
            input_tokens_tensor.shape[1]() == hidden_size
        ), "EP dispatch: input tokens shape doesn't match hidden size."
        __comptime_assert (
            topk_ids_tensor.shape[1]() == top_k
        ), "EP dispatch: topk ids shape doesn't match top k."

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        var my_rank = Int32(shmem_my_pe())
        comptime hw_info = gpu_ctx.default_device_info
        comptime gpu_target = get_gpu_target()
        comptime gpu_simd_width = simd_width_of[
            DType.uint8, target=gpu_target
        ]()
        comptime gpu_alignment = align_of[
            SIMD[DType.uint8, gpu_simd_width], target=gpu_target
        ]()
        comptime token_fmt_type = BF16TokenFormat[
            output_layout = Layout(), hidden_size, top_k, gpu_alignment
        ]

        comptime n_ranks = n_gpus_per_node * n_nodes

        comptime dispatch = dispatch_kernel[
            input_dtype,
            hw_info.max_thread_block_size,
            input_tokens_tensor.layout,
            topk_ids_tensor.layout,
            hw_info.sm_count,
            n_experts // (hw_info.max_thread_block_size // hw_info.warp_size),
            n_experts,
            n_ranks,
            max_token_per_rank,
            n_gpus_per_node,  # p2p world size
            token_fmt_type,
        ]

        @always_inline
        @parameter
        fn description_fn() -> String:
            # fmt: off
            return String(
                "input_dtype=", input_dtype,
                ";hidden_size=", hidden_size,
                ";top_k=", top_k,
                ";n_experts=", n_experts,
                ";max_token_per_rank=", max_token_per_rank,
                ";n_gpus_per_node=", n_gpus_per_node,
                ";n_nodes=", n_nodes,
                ";my_rank=", my_rank,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.dispatch",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var func = gpu_ctx.compile_function[dispatch, dispatch]()
            var cached_module_key = String("EP_DISPATCH_INITED_DEV_", gpu_id)

            # Don't initialize the module repeatedly
            if not Int(global_cache_lookup(cached_module_key)):
                shmem_module_init(func)
                global_cache_insert(
                    cached_module_key,
                    UnsafePointer[NoneType, MutExternalOrigin](
                        unsafe_from_address=1
                    ),
                )

            var send_ptr = UnsafePointer[UInt8, MutExternalOrigin](
                unsafe_from_address=Int(send_ptrs[gpu_id])
            )

            # Create inline arrays to store all the p2p accessible pointers
            var recv_ptrs_arr = InlineArray[
                UnsafePointer[UInt8, MutExternalOrigin], n_gpus_per_node
            ](fill={})
            var recv_count_ptrs_arr = InlineArray[
                UnsafePointer[UInt64, MutExternalOrigin], n_gpus_per_node
            ](fill={})

            var atomic_counters_ptr = UnsafePointer[Int32, MutExternalOrigin](
                atomic_counters_0._ptr
            )

            @parameter
            for i in range(n_gpus_per_node):
                recv_ptrs_arr[i] = UnsafePointer[UInt8, MutExternalOrigin](
                    unsafe_from_address=Int(recv_ptrs[i])
                )
                recv_count_ptrs_arr[i] = UnsafePointer[
                    UInt64, MutExternalOrigin
                ](unsafe_from_address=Int(recv_count_ptrs[i]))

            gpu_ctx.enqueue_function(
                func,
                input_tokens_tensor,
                topk_ids_tensor,
                send_ptr,
                recv_ptrs_arr,
                recv_count_ptrs_arr,
                atomic_counters_ptr,
                my_rank,
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
        n_nodes: Int,
        //,
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
        complete the token dispatch phase. It waits for all inter-device
        communication to complete, then organizes the received tokens into a
        format suitable for grouped matmul computation.

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
            recv_ptrs: Receive buffer pointers for each local GPU.
            recv_count_ptrs: Receive count buffer pointers for each local GPU.
            context: Device context pointer
        """
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."

        var output_tokens_tensor = output_tokens.to_layout_tensor()
        var row_offsets_tensor = row_offsets.to_layout_tensor()
        var expert_ids_tensor = expert_ids.to_layout_tensor()
        var src_info_tensor = src_info.to_layout_tensor()

        # Ensure the shape for the input tensors are correct
        __comptime_assert (
            output_tokens_tensor.shape[1]() == hidden_size
        ), "EP dispatch_cb: output tokens shape doesn't match hidden size."

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        var my_rank = Int32(shmem_my_pe())
        comptime hw_info = gpu_ctx.default_device_info
        comptime gpu_target = get_gpu_target()
        comptime gpu_simd_width = simd_width_of[
            DType.uint8, target=gpu_target
        ]()
        comptime gpu_alignment = align_of[
            SIMD[DType.uint8, gpu_simd_width], target=gpu_target
        ]()

        comptime n_ranks = n_gpus_per_node * n_nodes

        __comptime_assert dispatch_dtype == DType.bfloat16
        var format_handler = BF16TokenFormat[hidden_size, top_k, gpu_alignment](
            output_tokens_tensor.bitcast[DType.bfloat16]()
        )

        comptime dispatch_cb = dispatch_cb_kernel[
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
            type_of(format_handler),
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
                ";my_rank=", my_rank,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.dispatch_cb",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var recv_buf_ptr = UnsafePointer[UInt8, MutExternalOrigin](
                unsafe_from_address=Int(recv_ptrs[gpu_id])
            )
            var recv_count_ptr = UnsafePointer[UInt64, MutExternalOrigin](
                unsafe_from_address=Int(recv_count_ptrs[gpu_id])
            )
            var atomic_counters_ptr = UnsafePointer[Int32, MutExternalOrigin](
                atomic_counters_0._ptr
            )

            gpu_ctx.enqueue_function[dispatch_cb, dispatch_cb](
                format_handler,
                row_offsets_tensor,
                expert_ids_tensor,
                src_info_tensor,
                recv_buf_ptr,
                recv_count_ptr,
                atomic_counters_ptr,
                my_rank,
                OptionalReg[
                    LayoutTensor[
                        dispatch_dtype, Layout.row_major[2](), ImmutAnyOrigin
                    ]
                ](),
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


@compiler.register("ep.dispatch_cb.fused_shared_expert")
struct Struct_ep_dispatch_cb_fused_shared_expert:
    @always_inline
    @staticmethod
    fn execute[
        dispatch_dtype: DType,
        shared_expert_input_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
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
        input_tokens: InputTensor[dtype=shared_expert_input_dtype, rank=2],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel.

        This function launches the dispatch_cb_kernel from ep_comm.mojo to
        complete the token dispatch phase. It waits for all inter-device
        communication to complete, then organizes the received tokens into a
        format suitable for grouped matmul computation. This kernel also packs
        the shared expert's inputs with the routed experts' inputs.

        Parameters:
            dispatch_dtype: Data type for tokens during dispatch phase.
            shared_expert_input_dtype: Data type for the shared expert inputs.
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
            recv_ptrs: Receive buffer pointers for each local GPU.
            recv_count_ptrs: Receive count buffer pointers for each local GPU.
            input_tokens: Input tokens for the shared experts.
            context: Device context pointer"""
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."

        var output_tokens_tensor = output_tokens.to_layout_tensor()
        var row_offsets_tensor = row_offsets.to_layout_tensor()
        var expert_ids_tensor = expert_ids.to_layout_tensor()
        var src_info_tensor = src_info.to_layout_tensor()
        var input_tokens_tensor = input_tokens.to_layout_tensor()

        var _input_tokens = LayoutTensor[
            shared_expert_input_dtype, Layout.row_major[2](), ImmutAnyOrigin
        ](
            input_tokens_tensor.ptr,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                input_tokens_tensor.runtime_layout.shape.value.canonicalize()
            ),
        )

        var maybe_input_tokens = OptionalReg[type_of(_input_tokens)](
            _input_tokens
        )

        # Ensure the shape for the input tensors are correct
        __comptime_assert (
            output_tokens_tensor.shape[1]() == hidden_size
        ), "EP dispatch_cb: output tokens shape doesn't match hidden size."

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        var my_rank = Int32(shmem_my_pe())
        comptime hw_info = gpu_ctx.default_device_info
        comptime gpu_target = get_gpu_target()
        comptime gpu_simd_width = simd_width_of[
            DType.uint8, target=gpu_target
        ]()
        comptime gpu_alignment = align_of[
            SIMD[DType.uint8, gpu_simd_width], target=gpu_target
        ]()

        comptime n_ranks = n_gpus_per_node * n_nodes

        __comptime_assert dispatch_dtype == DType.bfloat16
        var format_handler = BF16TokenFormat[hidden_size, top_k, gpu_alignment](
            output_tokens_tensor.bitcast[DType.bfloat16]()
        )

        comptime dispatch_cb = dispatch_cb_kernel[
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
            type_of(format_handler),
            fused_shared_expert=True,
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
                ";my_rank=", my_rank,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.dispatch_cb",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var recv_buf_ptr = UnsafePointer[UInt8, MutExternalOrigin](
                unsafe_from_address=Int(recv_ptrs[gpu_id])
            )
            var recv_count_ptr = UnsafePointer[UInt64, MutExternalOrigin](
                unsafe_from_address=Int(recv_count_ptrs[gpu_id])
            )
            var atomic_counters_ptr = UnsafePointer[Int32, MutExternalOrigin](
                atomic_counters_0._ptr
            )

            gpu_ctx.enqueue_function[dispatch_cb, dispatch_cb](
                format_handler,
                row_offsets_tensor,
                expert_ids_tensor,
                src_info_tensor,
                recv_buf_ptr,
                recv_count_ptr,
                atomic_counters_ptr,
                my_rank,
                maybe_input_tokens,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )

            # The grouped matmul kernel needs this tensor to be filled
            expert_usage_stats_host[0] = (
                n_ranks * max_token_per_rank
            )  # max number of tokens per expert
            expert_usage_stats_host[1] = (
                n_experts // n_ranks + 1
            )  # number of active experts


@compiler.register("ep.dispatch.fp8")
struct Struct_ep_dispatch_fp8:
    @always_inline
    @staticmethod
    fn execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        dispatch_scale_dtype: DType,
        //,
        target: StaticString,
    ](
        atomic_counters_0: MutableInputTensor[dtype = DType.int32, rank=1],
        input_tokens: InputTensor[dtype=input_dtype, rank=2],
        topk_ids: InputTensor[dtype = DType.int32, rank=2],
        send_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_count_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism dispatch kernel.

        This function launches the dispatch_kernel from ep_comm.mojo to
        initiate token distribution across expert devices. In multi-node
        scenarios, all the communication buffers need to be allocated using
        `shmem_malloc`.

        Parameters:
            input_dtype: Data type of the input tokens.
            dispatch_dtype: Data type to dispatch tokens to experts.
            hidden_size: Model hidden dimension size.
            top_k: Number of experts each token is routed to.
            n_experts: Total experts across all devices.
            max_token_per_rank: Maximum tokens any device can send.
            n_gpus_per_node: GPUs per physical node.
            n_nodes: Number of physical nodes.
            dispatch_scale_granularity: FP8 quant granularity of the dispatch tokens.
            dispatch_scale_dtype: DType of the dispatch scale.
            target: Target.

        Arguments:
            atomic_counters_0: Synchronization counters for buffer group 0.
                Used to coordinate between different thread blocks.
            input_tokens: Tokens to dispatch to experts.
            topk_ids: Expert assignments from router.
            send_ptrs: Send buffer pointers for each local GPU.
            recv_ptrs: Receive buffer pointers for each local GPU.
            recv_count_ptrs: Receive count buffer pointers for each local GPU.
            context: Device context pointer
        """
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."

        var input_tokens_tensor = (
            input_tokens.to_layout_tensor().get_immutable()
        )
        var topk_ids_tensor = topk_ids.to_layout_tensor().get_immutable()

        # Ensure the shape for the input tensors are correct
        __comptime_assert (
            input_tokens_tensor.shape[1]() == hidden_size
        ), "EP dispatch: input tokens shape doesn't match hidden size."
        __comptime_assert (
            topk_ids_tensor.shape[1]() == top_k
        ), "EP dispatch: topk ids shape doesn't match top k."
        __comptime_assert (
            dispatch_scale_granularity == "block"
        ), "EP dispatch.fp8: dispatch scale granularity must be block."

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        var my_rank = Int32(shmem_my_pe())
        comptime hw_info = gpu_ctx.default_device_info
        comptime gpu_target = get_gpu_target()
        comptime gpu_simd_width = simd_width_of[
            DType.uint8, target=gpu_target
        ]()
        comptime gpu_alignment = align_of[
            SIMD[DType.uint8, gpu_simd_width], target=gpu_target
        ]()
        comptime token_fmt_type = BlockwiseFP8TokenFormat[
            fp8_dtype=dispatch_dtype,
            scales_dtype=dispatch_scale_dtype,
            output_layout = Layout(),
            scales_layout = Layout(),
            hidden_size,
            top_k,
            gpu_alignment,
        ]

        comptime n_ranks = n_gpus_per_node * n_nodes

        comptime dispatch = dispatch_kernel[
            input_dtype,
            hw_info.max_thread_block_size,
            input_tokens_tensor.layout,
            topk_ids_tensor.layout,
            hw_info.sm_count,
            n_experts // (hw_info.max_thread_block_size // hw_info.warp_size),
            n_experts,
            n_ranks,
            max_token_per_rank,
            n_gpus_per_node,  # p2p world size
            token_fmt_type,
        ]

        @always_inline
        @parameter
        fn description_fn() -> String:
            # fmt: off
            return String(
                "input_dtype=", input_dtype,
                ";dispatch_dtype=", dispatch_dtype,
                ";dispatch_scale_granularity=", dispatch_scale_granularity,
                ";dispatch_scale_dtype=", dispatch_scale_dtype,
                ";hidden_size=", hidden_size,
                ";top_k=", top_k,
                ";n_experts=", n_experts,
                ";max_token_per_rank=", max_token_per_rank,
                ";n_gpus_per_node=", n_gpus_per_node,
                ";n_nodes=", n_nodes,
                ";my_rank=", my_rank,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.dispatch.fp8",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var func = gpu_ctx.compile_function[dispatch, dispatch]()
            var cached_module_key = String("EP_DISPATCH_INITED_DEV_", gpu_id)

            # Don't initialize the module repeatedly
            if not Int(global_cache_lookup(cached_module_key)):
                shmem_module_init(func)
                global_cache_insert(
                    cached_module_key,
                    UnsafePointer[NoneType, MutExternalOrigin](
                        unsafe_from_address=1
                    ),
                )

            var send_buf = unsafe_aliasing_address_to_device_buffer[
                DType.uint8
            ](Int(send_ptrs[gpu_id]), 1, gpu_ctx)

            # Create inline arrays to store all the p2p accessible pointers
            var recv_ptrs_arr = InlineArray[
                UnsafePointer[UInt8, MutExternalOrigin], n_gpus_per_node
            ](fill={})
            var recv_count_ptrs_arr = InlineArray[
                UnsafePointer[UInt64, MutExternalOrigin], n_gpus_per_node
            ](fill={})

            var atomic_counters_ptr = UnsafePointer[Int32, MutExternalOrigin](
                atomic_counters_0._ptr
            )

            @parameter
            for i in range(n_gpus_per_node):
                recv_ptrs_arr[i] = UnsafePointer[UInt8, MutExternalOrigin](
                    unsafe_from_address=Int(recv_ptrs[i])
                )
                recv_count_ptrs_arr[i] = UnsafePointer[
                    UInt64, MutExternalOrigin
                ](unsafe_from_address=Int(recv_count_ptrs[i]))

            gpu_ctx.enqueue_function(
                func,
                input_tokens_tensor,
                topk_ids_tensor,
                send_buf,
                recv_ptrs_arr,
                recv_count_ptrs_arr,
                atomic_counters_ptr,
                my_rank,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )


@compiler.register("ep.dispatch_cb.fp8")
struct Struct_ep_dispatch_cb_fp8:
    @always_inline
    @staticmethod
    fn execute[
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=2],
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
        complete the token dispatch phase. It waits for all inter-device
        communication to complete, then organizes the received tokens into a
        format suitable for grouped matmul computation.

        Parameters:
            dispatch_dtype: Data type for tokens during dispatch phase.
            dispatch_scale_dtype: DType of the scales.
            hidden_size: Model hidden dimension size.
            top_k: Number of experts each token is routed to.
            n_experts: Total experts across all devices.
            max_token_per_rank: Maximum tokens any device can send.
            n_gpus_per_node: GPUs per physical node.
            n_nodes: Number of physical nodes.
            dispatch_scale_granularity: FP8 quant granularity of the dispatch tokens.
            target: Target.

        Arguments:
            output_tokens: Aggregated tokens ready for grouped matmul
                computation.
            output_scales: Scales of the aggregated tokens.
            row_offsets: Cumulative token counts for grouped matmul.
            expert_ids: Local expert IDs for grouped matmul.
            expert_usage_stats_host: Statistics for grouped matmul kernel.
            src_info: Source routing information for combine phase.
            atomic_counters_0: Synchronization counters from dispatch phase.
            recv_ptrs: Receive buffer pointers for each local GPU.
            recv_count_ptrs: Receive count buffer pointers for each local GPU.
            context: Device context pointer
        """
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."
        __comptime_assert (
            dispatch_scale_granularity == "block"
        ), "dispatch scale granularity must be block."

        var output_tokens_tensor = output_tokens.to_layout_tensor()
        var output_scales_tensor = output_scales.to_layout_tensor()
        var row_offsets_tensor = row_offsets.to_layout_tensor()
        var expert_ids_tensor = expert_ids.to_layout_tensor()
        var src_info_tensor = src_info.to_layout_tensor()

        # Ensure the shape for the input tensors are correct
        __comptime_assert (
            output_tokens_tensor.shape[1]() == hidden_size
        ), "EP dispatch_cb: output tokens shape doesn't match hidden size."

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        var my_rank = Int32(shmem_my_pe())
        comptime hw_info = gpu_ctx.default_device_info
        comptime gpu_target = get_gpu_target()
        comptime gpu_simd_width = simd_width_of[
            DType.uint8, target=gpu_target
        ]()
        comptime gpu_alignment = align_of[
            SIMD[DType.uint8, gpu_simd_width], target=gpu_target
        ]()

        comptime n_ranks = n_gpus_per_node * n_nodes

        var format_handler = BlockwiseFP8TokenFormat[
            hidden_size, top_k, gpu_alignment
        ](output_tokens_tensor, output_scales_tensor)

        # In order to use TMA, the scales of tokens for each expert must be
        # alligned to 16 bytes.
        comptime expert_m_padding = 16 // size_of[dispatch_scale_dtype]()

        comptime dispatch_cb = dispatch_cb_kernel[
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
            type_of(format_handler),
            expert_m_padding=expert_m_padding,
        ]

        @always_inline
        @parameter
        fn description_fn() -> String:
            # fmt: off
            return String(
                "dispatch_dtype=", dispatch_dtype,
                ";dispatch_scale_dtype=", dispatch_scale_dtype,
                ";dispatch_scale_granularity=", dispatch_scale_granularity,
                ";hidden_size=", hidden_size,
                ";top_k=", top_k,
                ";n_experts=", n_experts,
                ";max_token_per_rank=", max_token_per_rank,
                ";n_gpus_per_node=", n_gpus_per_node,
                ";n_nodes=", n_nodes,
                ";my_rank=", my_rank,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.dispatch_cb.fp8",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var recv_buf_ptr = UnsafePointer[UInt8, MutExternalOrigin](
                unsafe_from_address=Int(recv_ptrs[gpu_id])
            )
            var recv_count_ptr = UnsafePointer[UInt64, MutExternalOrigin](
                unsafe_from_address=Int(recv_count_ptrs[gpu_id])
            )
            var atomic_counters_ptr = UnsafePointer[Int32, MutExternalOrigin](
                atomic_counters_0._ptr
            )

            gpu_ctx.enqueue_function[dispatch_cb, dispatch_cb](
                format_handler,
                row_offsets_tensor,
                expert_ids_tensor,
                src_info_tensor,
                recv_buf_ptr,
                recv_count_ptr,
                atomic_counters_ptr,
                my_rank,
                OptionalReg[
                    LayoutTensor[
                        DType.bfloat16, Layout.row_major[2](), ImmutAnyOrigin
                    ]
                ](),
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


@compiler.register("ep.dispatch_cb.fp8.fused_shared_expert")
struct Struct_ep_dispatch_cb_fp8_fused_shared_expert:
    @always_inline
    @staticmethod
    fn execute[
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        shared_expert_input_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=2],
        row_offsets: OutputTensor[dtype = DType.uint32, rank=1],
        expert_ids: OutputTensor[dtype = DType.int32, rank=1],
        expert_usage_stats_host: OutputTensor[dtype = DType.uint32, rank=1],
        src_info: OutputTensor[dtype = DType.int32, rank=2],
        atomic_counters_0: MutableInputTensor[dtype = DType.int32, rank=1],
        recv_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_count_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        input_tokens: InputTensor[dtype=shared_expert_input_dtype, rank=2],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel.

        This function launches the dispatch_cb_kernel from ep_comm.mojo to
        complete the token dispatch phase. It waits for all inter-device
        communication to complete, then organizes the received tokens into a
        format suitable for grouped matmul computation. This kernel also packs
        the shared expert's inputs with the routed experts' inputs.

        Parameters:
            dispatch_dtype: Data type for tokens during dispatch phase.
            dispatch_scale_dtype: DType of the scales.
            shared_expert_input_dtype: Data type for the shared expert inputs.
            hidden_size: Model hidden dimension size.
            top_k: Number of experts each token is routed to.
            n_experts: Total experts across all devices.
            max_token_per_rank: Maximum tokens any device can send.
            n_gpus_per_node: GPUs per physical node.
            n_nodes: Number of physical nodes.
            dispatch_scale_granularity: FP8 quant granularity of the dispatch tokens.
            target: Target.

        Arguments:
            output_tokens: Aggregated tokens ready for grouped matmul
                computation.
            output_scales: Scales of the aggregated tokens.
            row_offsets: Cumulative token counts for grouped matmul.
            expert_ids: Local expert IDs for grouped matmul.
            expert_usage_stats_host: Statistics for grouped matmul kernel.
            src_info: Source routing information for combine phase.
            atomic_counters_0: Synchronization counters from dispatch phase.
            recv_ptrs: Receive buffer pointers for each local GPU.
            recv_count_ptrs: Receive count buffer pointers for each local GPU.
            input_tokens: Input tokens for the shared experts.
            context: Device context pointer
        """
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."
        __comptime_assert (
            dispatch_scale_granularity == "block"
        ), "dispatch scale granularity must be block."

        var output_tokens_tensor = output_tokens.to_layout_tensor()
        var output_scales_tensor = output_scales.to_layout_tensor()
        var row_offsets_tensor = row_offsets.to_layout_tensor()
        var expert_ids_tensor = expert_ids.to_layout_tensor()
        var src_info_tensor = src_info.to_layout_tensor()
        var input_tokens_tensor = input_tokens.to_layout_tensor()

        var _input_tokens = LayoutTensor[
            shared_expert_input_dtype, Layout.row_major[2](), ImmutAnyOrigin
        ](
            input_tokens_tensor.ptr,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                input_tokens_tensor.runtime_layout.shape.value.canonicalize()
            ),
        )

        var maybe_input_tokens = OptionalReg[type_of(_input_tokens)](
            _input_tokens
        )

        # Ensure the shape for the input tensors are correct
        __comptime_assert (
            output_tokens_tensor.shape[1]() == hidden_size
        ), "EP dispatch_cb: output tokens shape doesn't match hidden size."

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        var my_rank = Int32(shmem_my_pe())
        comptime hw_info = gpu_ctx.default_device_info
        comptime gpu_target = get_gpu_target()
        comptime gpu_simd_width = simd_width_of[
            DType.uint8, target=gpu_target
        ]()
        comptime gpu_alignment = align_of[
            SIMD[DType.uint8, gpu_simd_width], target=gpu_target
        ]()

        comptime n_ranks = n_gpus_per_node * n_nodes

        var format_handler = BlockwiseFP8TokenFormat[
            hidden_size, top_k, gpu_alignment
        ](output_tokens_tensor, output_scales_tensor)

        # In order to use TMA, the scales of tokens for each expert must be
        # alligned to 16 bytes.
        comptime expert_m_padding = 16 // size_of[dispatch_scale_dtype]()

        comptime dispatch_cb = dispatch_cb_kernel[
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
            type_of(format_handler),
            expert_m_padding=expert_m_padding,
            fused_shared_expert=True,
        ]

        @always_inline
        @parameter
        fn description_fn() -> String:
            # fmt: off
            return String(
                "dispatch_dtype=", dispatch_dtype,
                ";dispatch_scale_dtype=", dispatch_scale_dtype,
                ";dispatch_scale_granularity=", dispatch_scale_granularity,
                ";hidden_size=", hidden_size,
                ";top_k=", top_k,
                ";n_experts=", n_experts,
                ";max_token_per_rank=", max_token_per_rank,
                ";n_gpus_per_node=", n_gpus_per_node,
                ";n_nodes=", n_nodes,
                ";my_rank=", my_rank,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.dispatch_cb.fp8",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var recv_buf_ptr = UnsafePointer[UInt8, MutExternalOrigin](
                unsafe_from_address=Int(recv_ptrs[gpu_id])
            )
            var recv_count_ptr = UnsafePointer[UInt64, MutExternalOrigin](
                unsafe_from_address=Int(recv_count_ptrs[gpu_id])
            )
            var atomic_counters_ptr = UnsafePointer[Int32, MutExternalOrigin](
                atomic_counters_0._ptr
            )

            gpu_ctx.enqueue_function[dispatch_cb, dispatch_cb](
                format_handler,
                row_offsets_tensor,
                expert_ids_tensor,
                src_info_tensor,
                recv_buf_ptr,
                recv_count_ptr,
                atomic_counters_ptr,
                my_rank,
                maybe_input_tokens,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )

            # The grouped matmul kernel needs this tensor to be filled
            expert_usage_stats_host[0] = (
                n_ranks * max_token_per_rank
            )  # max number of tokens per expert
            expert_usage_stats_host[1] = (
                n_experts // n_ranks + 1
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
        n_nodes: Int,
        //,
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
        source routing information to determine destinations. In multi-node
        scenarios, all the communication buffers need to be allocated using
        `shmem_malloc`.

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
            send_ptrs: Send buffer pointers for each local GPU.
            recv_ptrs: Receive buffer pointers for each local GPU.
            recv_count_ptrs: Receive count buffer pointers for each local GPU.
            context: Device context pointer.
        """
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."

        var input_tokens_tensor = input_tokens.to_layout_tensor()
        var src_info_tensor = src_info.to_layout_tensor()

        # Ensure the shape for the input tensors are correct
        __comptime_assert (
            input_tokens_tensor.shape[1]() == hidden_size
        ), "EP combine: input tokens shape doesn't match hidden size."

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        var my_rank = Int32(shmem_my_pe())
        comptime hw_info = gpu_ctx.default_device_info
        comptime combine_msg_size = hidden_size * size_of[combine_dtype]()

        comptime n_ranks = n_gpus_per_node * n_nodes

        comptime combine = combine_kernel[
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
            n_gpus_per_node,
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
                ";my_rank=", my_rank,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.combine",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var func = gpu_ctx.compile_function[combine, combine]()
            var cached_module_key = String("EP_COMBINE_INITED_DEV_", gpu_id)

            # Don't initialize the module repeatedly
            if not Int(global_cache_lookup(cached_module_key)):
                shmem_module_init(func)
                global_cache_insert(
                    cached_module_key,
                    UnsafePointer[NoneType, MutExternalOrigin](
                        unsafe_from_address=1
                    ),
                )

            var send_ptr = UnsafePointer[UInt8, MutExternalOrigin](
                unsafe_from_address=Int(send_ptrs[gpu_id])
            )

            # Create inline arrays to store all the p2p accessible pointers
            var recv_ptrs_arr = InlineArray[
                UnsafePointer[UInt8, MutExternalOrigin], n_gpus_per_node
            ](fill={})
            var recv_count_ptrs_arr = InlineArray[
                UnsafePointer[UInt64, MutExternalOrigin], n_gpus_per_node
            ](fill={})

            var atomic_counters_1_ptr = UnsafePointer[Int32, MutExternalOrigin](
                atomic_counters_1._ptr
            )

            @parameter
            for i in range(n_gpus_per_node):
                recv_ptrs_arr[i] = UnsafePointer[UInt8, MutExternalOrigin](
                    unsafe_from_address=Int(recv_ptrs[i])
                )
                recv_count_ptrs_arr[i] = UnsafePointer[
                    UInt64, MutExternalOrigin
                ](unsafe_from_address=Int(recv_count_ptrs[i]))

            gpu_ctx.enqueue_function(
                func,
                input_tokens_tensor,
                src_info_tensor,
                send_ptr,
                recv_ptrs_arr,
                recv_count_ptrs_arr,
                atomic_counters_1_ptr,
                my_rank,
                OptionalReg[
                    LayoutTensor[
                        combine_dtype, Layout.row_major[2](), MutAnyOrigin
                    ]
                ](),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )


@compiler.register("ep.combine.fused_shared_expert")
struct Struct_ep_combine_fused_shared_expert:
    @always_inline
    @staticmethod
    fn execute[
        combine_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=combine_dtype, rank=2],
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
        source routing information to determine destinations. This kernel will
        also filter out the shared expert's outputs and store them in a separate
        tensor. In multi-node scenarios, all the communication buffers need to
        be allocated using `shmem_malloc`.

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
            output_tokens: Output tokens for the shared experts.
            atomic_counters_1: Synchronization counters for buffer group 1.
                Used to coordinate between different thread blocks.
            input_tokens: Expert output tokens to send back to original devices.
            src_info: Source routing information from dispatch phase.
            send_ptrs: Send buffer pointers for each local GPU.
            recv_ptrs: Receive buffer pointers for each local GPU.
            recv_count_ptrs: Receive count buffer pointers for each local GPU.
            context: Device context pointer.
        """
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."

        var input_tokens_tensor = input_tokens.to_layout_tensor()
        var src_info_tensor = src_info.to_layout_tensor()
        var output_tokens_tensor = output_tokens.to_layout_tensor()

        var _output_tokens = LayoutTensor[
            combine_dtype, Layout.row_major[2](), MutAnyOrigin
        ](
            output_tokens_tensor.ptr,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                output_tokens_tensor.runtime_layout.shape.value.canonicalize()
            ),
        )

        var maybe_output_tokens = OptionalReg[type_of(_output_tokens)](
            _output_tokens
        )

        # Ensure the shape for the input tensors are correct
        __comptime_assert (
            input_tokens_tensor.shape[1]() == hidden_size
        ), "EP combine: input tokens shape doesn't match hidden size."

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        var my_rank = Int32(shmem_my_pe())
        comptime hw_info = gpu_ctx.default_device_info
        comptime combine_msg_size = hidden_size * size_of[combine_dtype]()

        comptime n_ranks = n_gpus_per_node * n_nodes

        comptime combine = combine_kernel[
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
            n_gpus_per_node,
            fused_shared_expert=True,
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
                ";my_rank=", my_rank,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.combine",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var func = gpu_ctx.compile_function[combine, combine]()
            var cached_module_key = String("EP_COMBINE_INITED_DEV_", gpu_id)

            # Don't initialize the module repeatedly
            if not Int(global_cache_lookup(cached_module_key)):
                shmem_module_init(func)
                global_cache_insert(
                    cached_module_key,
                    UnsafePointer[NoneType, MutExternalOrigin](
                        unsafe_from_address=1
                    ),
                )

            var send_ptr = UnsafePointer[UInt8, MutExternalOrigin](
                unsafe_from_address=Int(send_ptrs[gpu_id])
            )

            # Create inline arrays to store all the p2p accessible pointers
            var recv_ptrs_arr = InlineArray[
                UnsafePointer[UInt8, MutExternalOrigin], n_gpus_per_node
            ](fill={})
            var recv_count_ptrs_arr = InlineArray[
                UnsafePointer[UInt64, MutExternalOrigin], n_gpus_per_node
            ](fill={})

            var atomic_counters_1_ptr = UnsafePointer[Int32, MutExternalOrigin](
                atomic_counters_1._ptr
            )

            @parameter
            for i in range(n_gpus_per_node):
                recv_ptrs_arr[i] = UnsafePointer[UInt8, MutExternalOrigin](
                    unsafe_from_address=Int(recv_ptrs[i])
                )
                recv_count_ptrs_arr[i] = UnsafePointer[
                    UInt64, MutExternalOrigin
                ](unsafe_from_address=Int(recv_count_ptrs[i]))

            gpu_ctx.enqueue_function(
                func,
                input_tokens_tensor,
                src_info_tensor,
                send_ptr,
                recv_ptrs_arr,
                recv_count_ptrs_arr,
                atomic_counters_1_ptr,
                my_rank,
                maybe_output_tokens,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )


@compiler.register("ep.combine_cb")
struct Struct_ep_combine_cb:
    @parameter
    @always_inline
    @staticmethod
    fn execute[
        combine_dtype: DType,
        router_weights_dtype: DType,
        //,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        lambdas_have_fusion: Bool,
        target: StaticString,
    ](
        output_tokens: FusedOutputTensor[dtype=combine_dtype, rank=2],
        atomic_counters_1: MutableInputTensor[dtype = DType.int32, rank=1],
        recv_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        recv_count_ptrs: InputTensor[dtype = DType.uint64, rank=1],
        router_weights: InputTensor[dtype=router_weights_dtype, rank=2],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism combine completion kernel.

        This function launches the combine_cb_kernel from ep_comm.mojo to
        complete the token combine phase. It waits for all inter-device
        communication to complete, then computes the weighted sum of routed
        expert outputs for each token.

        Parameters:
            combine_dtype: Data type for tokens during combine phase.
            router_weights_dtype: Data type for router weights.
            hidden_size: Model hidden dimension size.
            top_k: Number of experts each token was routed to.
            n_experts: Total experts across all devices.
            max_token_per_rank: Maximum tokens any device can receive.
            n_gpus_per_node: GPUs per physical node.
            n_nodes: Number of physical nodes.
            lambdas_have_fusion: Whether we need to use fused output lambda.
            target: Target.

        Arguments:
            output_tokens: Final output tensor with expert results.
            atomic_counters_1: Synchronization counters from combine phase.
            recv_ptrs: Receive buffer pointers for each local GPU.
            recv_count_ptrs: Receive count buffer pointers for each local GPU.
            router_weights: Router weights for the current device.
            context: Device context pointer.
        """
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."

        var output_tokens_tensor = output_tokens.to_layout_tensor()
        var router_weights_tensor = router_weights.to_layout_tensor()

        @parameter
        @always_inline
        @__copy_capture(router_weights_tensor)
        fn router_weights_fn(token_idx: Int, topk_id: Int) -> Float32:
            return router_weights_tensor.load[width=1](token_idx, topk_id).cast[
                DType.float32
            ]()

        # Ensure the shape for the output tensor is correct
        __comptime_assert (
            output_tokens_tensor.shape[1]() == hidden_size
        ), "EP combine: output tokens shape doesn't match hidden size."

        var gpu_ctx = context.get_device_context()
        var gpu_id = Int(gpu_ctx.id())
        var my_rank = Int32(shmem_my_pe())
        comptime hw_info = gpu_ctx.default_device_info
        comptime combine_msg_size = hidden_size * size_of[combine_dtype]()
        comptime n_ranks = n_gpus_per_node * n_nodes

        @parameter
        @always_inline
        fn output_fn[
            dtype: DType, width: Int, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[dtype, width]):
            output_tokens._lambda_store[
                width=width, element_alignment=alignment
            ](
                coords,
                rebind[SIMD[combine_dtype, width]](val),
            )

        comptime combine_cb = combine_cb_kernel[
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
            router_weights_wrapper = OptionalReg[
                fn (Int, Int) capturing -> Float32
            ](router_weights_fn),
            elementwise_lambda_fn = OptionalReg[elementwise_epilogue_type](
                output_fn
            ) if lambdas_have_fusion else None,
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
                ";my_rank=", my_rank,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.combine_cb",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            var recv_buf_ptr = UnsafePointer[UInt8, MutExternalOrigin](
                unsafe_from_address=Int(recv_ptrs[gpu_id])
            )
            var recv_count_ptr = UnsafePointer[UInt64, MutExternalOrigin](
                unsafe_from_address=Int(recv_count_ptrs[gpu_id])
            )
            var atomic_counters_1_ptr = UnsafePointer[Int32, MutExternalOrigin](
                atomic_counters_1._ptr
            )

            gpu_ctx.enqueue_function[combine_cb, combine_cb](
                output_tokens_tensor,
                recv_buf_ptr,
                recv_count_ptr,
                atomic_counters_1_ptr,
                my_rank,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Utils
# ===-----------------------------------------------------------------------===#


@compiler.register("ep.fused_silu")
struct Struct_ep_fused_silu:
    @always_inline
    @staticmethod
    fn execute[
        output_dtype: DType,
        input_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_dtype, rank=2],
        input: InputTensor[dtype=input_dtype, rank=2],
        row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism fused SILU kernel.

        This function launches the fused_silu kernel to perform the SILU
        operation for all the MLPs in the EP MoE module. We need to manually
        implement the custom operation here is because after the EP dispatch
        phase, the actual number of received tokens is not known to the host.

        This kernel will read the row offsets to determine the actual number of
        received tokens in the input tensor, and then only perform the SILU
        operation on the received tokens.
        """
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."

        var output_tensor = output.to_layout_tensor()
        var input_tensor = input.to_layout_tensor().get_immutable()
        var row_offsets_tensor = row_offsets.to_layout_tensor().get_immutable()

        var gpu_ctx = context.get_device_context()
        comptime hw_info = gpu_ctx.default_device_info

        comptime fused_silu = fused_silu_kernel[
            output_dtype,
            input_dtype,
            output_tensor.layout,
            input_tensor.layout,
            row_offsets_tensor.layout,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
        ]

        @always_inline
        @parameter
        fn description_fn() -> String:
            # fmt: off
            return String(
                "output_dtype=", output_dtype,
                ";input_dtype=", input_dtype,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.fused_silu",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu, fused_silu](
                output_tensor,
                input_tensor,
                row_offsets_tensor,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(),
            )


@compiler.register("ep.fused_silu_fp8")
struct Struct_ep_fused_silu_fp8:
    @always_inline
    @staticmethod
    fn execute[
        fp8_dtype: DType,
        scales_dtype: DType,
        input_dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=fp8_dtype, rank=2],
        scales: OutputTensor[dtype=scales_dtype, rank=2],
        input: InputTensor[dtype=input_dtype, rank=2],
        row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism fused SILU kernel with FP8
        quantization.

        This function launches the fused_silu_fp8 kernel to perform the SILU
        operation for all the MLPs in the EP MoE module.

        This kernel will read the row offsets to determine the actual number of
        received tokens in the input tensor, and then only perform the SILU
        operation on the received tokens. Once the SILU operation is performed,
        the output will be quantized to the FP8 format. The scales tensor
        will be stored in a transposed way.
        """
        # Ensure this kernel only runs on GPU targets
        __comptime_assert is_gpu[target](), "EP is only supported on GPU."

        comptime group_size = 128

        var output_tensor = output.to_layout_tensor()
        var scales_tensor = scales.to_layout_tensor()
        var input_tensor = input.to_layout_tensor().get_immutable()
        var row_offsets_tensor = row_offsets.to_layout_tensor().get_immutable()

        var gpu_ctx = context.get_device_context()
        comptime hw_info = gpu_ctx.default_device_info

        comptime fused_silu_fp8 = fused_silu_fp8_kernel[
            fp8_dtype,
            scales_dtype,
            input_dtype,
            output_tensor.layout,
            scales_tensor.layout,
            input_tensor.layout,
            row_offsets_tensor.layout,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
            group_size,
        ]

        @always_inline
        @parameter
        fn description_fn() -> String:
            # fmt: off
            return String(
                "fp8_dtype=", fp8_dtype,
                ";scales_dtype=", scales_dtype,
                ";input_dtype=", input_dtype,
                ";group_size=", group_size,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.fused_silu_fp8",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu_fp8, fused_silu_fp8](
                output_tensor,
                scales_tensor,
                input_tensor,
                row_offsets_tensor,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(),
            )

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

"""
Expert Parallelism (EP) Communication Kernel.
"""


import compiler_internal as compiler
from comm.sync import is_p2p_enabled
from std.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from std.gpu.host import DeviceBuffer
from std.gpu.host.info import is_gpu
from std.memory.unsafe_pointer import pointer_to_int
from layout import Coord, TileTensor, Idx, coord_to_index_list
from layout.tile_tensor import row_major
from std.utils import StaticTuple
from std.utils.index import IndexList

from std.runtime.asyncrt import DeviceContextPtr, DeviceContextPtrList
from std.runtime.tracing import Trace, TraceLevel, get_safe_task_id
from std.sys.info import size_of, has_amd_gpu_accelerator
from tensor import (
    InputTensor,
    InputVariadicTensors,
    OutputTensor,
    OutputVariadicTensors,
)
from tensor.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from tensor.managed_tensor_slice import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)
from tensor.managed_tensor_slice import (
    _FusedOutputTensor as FusedOutputTensor,
)
from tensor.managed_tensor_slice import (
    _FusedOutputVariadicTensors as FusedOutputVariadicTensors,
)

from .MOGGKernelAPI import _launch_device_collective

from shmem import (
    shmem_init_thread_mpi,
    shmem_init_thread_tcp,
    shmem_malloc,
    shmem_my_pe,
)
from shmem.ep import (
    ep_combine_async_kernel_api,
    ep_combine_wait_kernel_api,
    ep_dispatch_async_kernel_api,
    ep_dispatch_wait_kernel_api,
    ep_fused_combine_kernel_api,
    ep_fused_dispatch_kernel_api,
)
from shmem.ep_comm import (
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    EPLocalSyncCounters,
    MXFP4TokenFormat,
    NVFP4TokenFormat,
    elementwise_epilogue_type,
    fused_silu_kernel,
    fused_silu_fp8_kernel,
    fused_silu_mxfp4_kernel,
    fused_silu_nvfp4_kernel,
)

comptime RT_LAYOUT_2D = type_of(row_major(Idx(Int64(1)), Idx(Int64(1))))

# ===-----------------------------------------------------------------------===#
# Expert Parallelism Initialization Kernel
# ===-----------------------------------------------------------------------===#


@compiler.register("ep.init")
struct Struct_ep_init:
    @always_inline
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        combine_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_dtype: DType,
        dispatch_fmt_str: StaticString,
        //,
        target: StaticString,
    ](
        dev_ptrs: OutputTensor[dtype=DType.uint64, rank=2, ...],
        my_rank_tensor: OutputTensor[dtype=DType.int32, rank=1, ...],
        atomic_counters_0: MutableInputTensor[dtype=DType.int32, ...],
        atomic_counters_1: MutableInputTensor[dtype=DType.int32, ...],
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
            n_nodes: Number of physical nodes.
            dispatch_scale_dtype: DType of the dispatch scale.
            dispatch_fmt_str: String indicating the dispatch format.
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
        comptime assert is_gpu[target](), "EP is only supported on GPU."
        var gpu_ctx = context.get_device_context()

        # Calculate buffer sizes for dispatch phase
        var dispatch_msg_size: Int

        # Infer message sizes for dispatch phases
        comptime if dispatch_fmt_str == "BlockwiseFP8":
            comptime token_fmt_type = BlockwiseFP8TokenFormat[
                fp8_dtype=dispatch_dtype,
                scales_dtype=dispatch_scale_dtype,
                output_layout=RT_LAYOUT_2D,
                scales_layout=RT_LAYOUT_2D,
                hidden_size,
                top_k,
            ]
            dispatch_msg_size = token_fmt_type.msg_size()

        elif dispatch_fmt_str == "NVFP4":
            comptime token_fmt_type = NVFP4TokenFormat[
                fp4_dtype=dispatch_dtype,
                scales_dtype=dispatch_scale_dtype,
                output_layout=RT_LAYOUT_2D,
                scales_offset_layout=RT_LAYOUT_2D,
                hidden_size,
                top_k,
            ]
            dispatch_msg_size = token_fmt_type.msg_size()

        elif dispatch_fmt_str == "MXFP4":
            comptime token_fmt_type = MXFP4TokenFormat[
                fp4_dtype=dispatch_dtype,
                scales_dtype=dispatch_scale_dtype,
                output_layout=RT_LAYOUT_2D,
                scales_layout=RT_LAYOUT_2D,
                hidden_size,
                top_k,
            ]
            dispatch_msg_size = token_fmt_type.msg_size()

        elif dispatch_fmt_str == "BF16":
            comptime token_fmt_type = BF16TokenFormat[
                output_layout=RT_LAYOUT_2D, hidden_size, top_k
            ]
            dispatch_msg_size = token_fmt_type.msg_size()

        else:
            raise Error("Invalid dispatch format string: ", dispatch_fmt_str)

        var dispatch_send_size = max_token_per_rank * dispatch_msg_size
        var dispatch_recv_size = (
            n_experts * max_token_per_rank * dispatch_msg_size
        )

        # Calculate buffer sizes for combine phase
        # Combine messages only contain the processed token
        comptime combine_msg_size = hidden_size * size_of[combine_dtype]()
        comptime combine_send_size = max_token_per_rank * combine_msg_size * min(
            n_experts, n_gpus_per_node * n_nodes * top_k
        )
        comptime combine_recv_size = top_k * max_token_per_rank * combine_msg_size

        # Initialize atomic counters to zero for synchronization
        # These counters coordinate work between different thread blocks.
        comptime assert (
            atomic_counters_0.static_spec.static_size
            == EPLocalSyncCounters[n_experts].total_size()
        ), "Atomic counters 0 size doesn't match expected size."
        var atomic_counters_0_buf = atomic_counters_0.to_device_buffer(gpu_ctx)
        gpu_ctx.enqueue_memset(atomic_counters_0_buf, Int32(0))

        comptime assert (
            atomic_counters_1.static_spec.static_size
            == EPLocalSyncCounters[n_experts].total_size()
        ), "Atomic counters 1 size doesn't match expected size."
        var atomic_counters_1_buf = atomic_counters_1.to_device_buffer(gpu_ctx)
        gpu_ctx.enqueue_memset(atomic_counters_1_buf, Int32(0))

        var dispatch_send_p: UnsafePointer[UInt8, MutAnyOrigin]
        var dispatch_recv_p: UnsafePointer[UInt8, MutAnyOrigin]
        var dispatch_recv_count_p: UnsafePointer[UInt64, MutAnyOrigin]

        var combine_send_p: Optional[UnsafePointer[UInt8, MutAnyOrigin]]
        var combine_recv_p: UnsafePointer[UInt8, MutAnyOrigin]
        var combine_recv_count_p: UnsafePointer[UInt64, MutAnyOrigin]

        comptime if n_nodes > 1:
            # Initialize the SHMEM library for this GPU
            comptime if has_amd_gpu_accelerator():
                shmem_init_thread_tcp(gpu_ctx, gpus_per_node=n_gpus_per_node)
            else:
                shmem_init_thread_mpi(gpu_ctx, gpus_per_node=n_gpus_per_node)

            # Allocate SHMEM buffers for dispatch phase
            dispatch_send_p = shmem_malloc[DType.uint8](dispatch_send_size)
            dispatch_recv_p = shmem_malloc[DType.uint8](dispatch_recv_size)
            dispatch_recv_count_p = shmem_malloc[DType.uint64](n_experts)

            # Allocate SHMEM buffers for combine phase
            combine_send_p = shmem_malloc[DType.uint8](combine_send_size)
            combine_recv_p = shmem_malloc[DType.uint8](combine_recv_size)
            combine_recv_count_p = shmem_malloc[DType.uint64](n_experts)

        else:
            if not is_p2p_enabled():
                raise Error("P2P is not supported on this system.")
            dispatch_send_p = gpu_ctx.enqueue_create_buffer[DType.uint8](
                dispatch_send_size
            ).take_ptr()
            dispatch_recv_p = gpu_ctx.enqueue_create_buffer[DType.uint8](
                dispatch_recv_size
            ).take_ptr()
            dispatch_recv_count_p = gpu_ctx.enqueue_create_buffer[DType.uint64](
                n_experts
            ).take_ptr()

            # When all the devices are on the same node, we skip the combine
            # send buffer and directly send tokens to each device's recv buffer.
            # Hence, we don't need to allocate the combine send buffer.
            combine_send_p = None
            combine_recv_p = gpu_ctx.enqueue_create_buffer[DType.uint8](
                combine_recv_size
            ).take_ptr()
            combine_recv_count_p = gpu_ctx.enqueue_create_buffer[DType.uint64](
                n_experts
            ).take_ptr()

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
        dev_ptrs[1, 0] = UInt64(pointer_to_int(combine_send_p))
        dev_ptrs[1, 1] = UInt64(Int(combine_recv_p))
        dev_ptrs[1, 2] = UInt64(Int(combine_recv_count_p))

        # Store current device's rank
        var my_rank: Int32

        comptime if n_nodes > 1:
            my_rank = Int32(shmem_my_pe())
        else:
            my_rank = Int32(gpu_ctx.id())
        my_rank_tensor[0] = my_rank


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Async Dispatch Kernel
# ===-----------------------------------------------------------------------===#


@compiler.register("ep.dispatch_async")
struct Struct_ep_dispatch_async:
    @always_inline
    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_fmt_str: StaticString,
        //,
        target: StaticString,
    ](
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=input_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism async dispatch kernel. Tokens are
        transferred in either Blockwise FP8 or BF16 format.
        """

        comptime if dispatch_fmt_str == "BlockwiseFP8":
            comptime token_fmt_type = BlockwiseFP8TokenFormat[
                fp8_dtype=dispatch_dtype,
                scales_dtype=DType.float32,
                output_layout=RT_LAYOUT_2D,
                scales_layout=RT_LAYOUT_2D,
                hidden_size,
                top_k,
            ]
            ep_dispatch_async_kernel_api[
                token_fmt_type,
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                target,
            ](
                atomic_counters.to_tile_tensor[DType.int64](),
                input_tokens.to_tile_tensor[DType.int64]().as_immut(),
                topk_ids.to_tile_tensor[DType.int64]().as_immut(),
                send_ptrs.to_tile_tensor[DType.int64](),
                recv_ptrs.to_tile_tensor[DType.int64](),
                recv_count_ptrs.to_tile_tensor[DType.int64](),
                context,
            )

        elif dispatch_fmt_str == "BF16":
            comptime token_fmt_type = BF16TokenFormat[
                output_layout=RT_LAYOUT_2D, hidden_size, top_k
            ]

            ep_dispatch_async_kernel_api[
                token_fmt_type,
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                target,
            ](
                atomic_counters.to_tile_tensor[DType.int64](),
                input_tokens.to_tile_tensor[DType.int64]().as_immut(),
                topk_ids.to_tile_tensor[DType.int64]().as_immut(),
                send_ptrs.to_tile_tensor[DType.int64](),
                recv_ptrs.to_tile_tensor[DType.int64](),
                recv_count_ptrs.to_tile_tensor[DType.int64](),
                context,
            )

        else:
            raise Error("Invalid dispatch format string: ", dispatch_fmt_str)


@compiler.register("ep.dispatch_async.nvfp4")
struct Struct_ep_dispatch_async_nvfp4:
    @always_inline
    @staticmethod
    @parameter
    def execute[
        input_dtype: DType,
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
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=input_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        input_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism async dispatch kernel. Tokens are
        transferred in NVFP4 format.
        """
        var input_scales_tensor = input_scales.to_tile_tensor[DType.int64]()
        comptime assert input_scales_tensor.flat_rank == 1

        @parameter
        @always_inline
        @__copy_capture(input_scales_tensor)
        def input_scales_fn[dtype: DType](expert_id: Int) -> Scalar[dtype]:
            # Currently only use one global input scale for all experts
            return rebind[Scalar[dtype]](input_scales_tensor[0].cast[dtype]())

        comptime token_fmt_type = NVFP4TokenFormat[
            fp4_dtype=dispatch_dtype,
            scales_dtype=DType.float8_e4m3fn,
            output_layout=RT_LAYOUT_2D,
            scales_offset_layout=RT_LAYOUT_2D,
            hidden_size,
            top_k,
        ]
        ep_dispatch_async_kernel_api[
            token_fmt_type,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
            input_scales_wrapper=input_scales_fn,
        ](
            atomic_counters.to_tile_tensor[DType.int64](),
            input_tokens.to_tile_tensor[DType.int64]().as_immut(),
            topk_ids.to_tile_tensor[DType.int64]().as_immut(),
            send_ptrs.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("ep.dispatch_async.mxfp4")
struct Struct_ep_dispatch_async_mxfp4:
    @always_inline
    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
    ](
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=input_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism async dispatch kernel. Tokens are
        transferred in MXFP4 format with per-token even-mode scales packed
        alongside the FP4 quants in the send buffer.
        """
        comptime token_fmt_type = MXFP4TokenFormat[
            fp4_dtype=dispatch_dtype,
            scales_dtype=dispatch_scale_dtype,
            output_layout=RT_LAYOUT_2D,
            scales_layout=RT_LAYOUT_2D,
            hidden_size,
            top_k,
        ]
        ep_dispatch_async_kernel_api[
            token_fmt_type,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            atomic_counters.to_tile_tensor[DType.int64](),
            input_tokens.to_tile_tensor[DType.int64]().as_immut(),
            topk_ids.to_tile_tensor[DType.int64]().as_immut(),
            send_ptrs.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            context,
        )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Dispatch Wait Kernel
# ===-----------------------------------------------------------------------===#


@compiler.register("ep.dispatch_wait")
struct Struct_ep_dispatch_wait:
    @always_inline
    @staticmethod
    def execute[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=DType.bfloat16, rank=2, ...],
        row_offsets: OutputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=DType.int32, rank=1, ...],
        src_info: OutputTensor[dtype=DType.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in BF16 format.
        """

        # Ensure the shape for the input tensors are correct
        comptime assert (
            Int(output_tokens.static_spec.shape_tuple[1]) == hidden_size
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = BF16TokenFormat[hidden_size, top_k](
            output_tokens.to_tile_tensor[DType.int64]()
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            format_handler,
            row_offsets.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            src_info.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            atomic_counters.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("ep.dispatch_wait.fp8")
struct Struct_ep_dispatch_wait_fp8:
    @always_inline
    @staticmethod
    def execute[
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
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=2, ...],
        row_offsets: OutputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=DType.int32, rank=1, ...],
        src_info: OutputTensor[dtype=DType.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in Blockwise FP8 format.
        """

        var output_tokens_tensor = output_tokens.to_tile_tensor[DType.int64]()
        var output_scales_tensor = output_scales.to_tile_tensor[DType.int64]()
        # Ensure the shape for the input tensors are correct
        comptime assert (
            output_tokens_tensor.static_shape[1] == hidden_size
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = BlockwiseFP8TokenFormat[hidden_size, top_k](
            output_tokens_tensor,
            output_scales_tensor,
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            format_handler,
            row_offsets.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            src_info.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            atomic_counters.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("ep.dispatch_wait.nvfp4")
struct Struct_ep_dispatch_wait_nvfp4:
    @always_inline
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=5, ...],
        row_offsets: OutputTensor[dtype=DType.uint32, rank=1, ...],
        scales_offsets: OutputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=DType.int32, rank=1, ...],
        src_info: OutputTensor[dtype=DType.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in NVFP4 format.
        """
        var output_tokens_tensor = output_tokens.to_tile_tensor[DType.int64]()
        var output_scales_tensor = output_scales.to_tile_tensor[DType.int64]()
        var scales_offsets_tensor = scales_offsets.to_tile_tensor[DType.int64]()

        comptime assert (
            output_tokens_tensor.static_shape[1] * 2 == hidden_size
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = NVFP4TokenFormat[hidden_size, top_k](
            output_tokens_tensor,
            output_scales_tensor,
            scales_offsets_tensor,
            context[],
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            format_handler,
            row_offsets.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            src_info.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            atomic_counters.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("ep.dispatch_wait.mxfp4")
struct Struct_ep_dispatch_wait_mxfp4:
    @always_inline
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=2, ...],
        row_offsets: OutputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=DType.int32, rank=1, ...],
        src_info: OutputTensor[dtype=DType.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in MXFP4 format: two FP4 elements packed per ``uint8`` in
        ``output_tokens`` with per-token even-mode scales in ``output_scales``.
        """
        var output_tokens_tensor = output_tokens.to_tile_tensor[DType.int64]()
        var output_scales_tensor = output_scales.to_tile_tensor[DType.int64]()

        comptime assert (
            output_tokens_tensor.static_shape[1] * 2 == hidden_size
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = MXFP4TokenFormat[hidden_size, top_k](
            output_tokens_tensor,
            output_scales_tensor,
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            format_handler,
            row_offsets.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            src_info.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            atomic_counters.to_tile_tensor[DType.int64](),
            context,
        )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Fused Dispatch Kernel
# ===-----------------------------------------------------------------------===#


@compiler.register("ep.dispatch")
struct Struct_ep_dispatch:
    @always_inline
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=DType.bfloat16, rank=2, ...],
        row_offsets: OutputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=DType.int32, rank=1, ...],
        src_info: OutputTensor[dtype=DType.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=dispatch_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the fused Expert Parallelism dispatch kernel."""

        comptime assert dispatch_dtype == DType.bfloat16

        var output_tokens_tensor = output_tokens.to_tile_tensor[DType.int64]()
        var format_handler = BF16TokenFormat[hidden_size, top_k](
            output_tokens_tensor
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            target,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            src_info.to_tile_tensor[DType.int64](),
            atomic_counters.to_tile_tensor[DType.int64](),
            input_tokens.to_tile_tensor[DType.int64](),
            topk_ids.to_tile_tensor[DType.int64](),
            send_ptrs.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("ep.dispatch.fp8")
struct Struct_ep_dispatch_fp8:
    @always_inline
    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=2, ...],
        row_offsets: OutputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=DType.int32, rank=1, ...],
        src_info: OutputTensor[dtype=DType.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=input_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the fused Expert Parallelism FP8 dispatch kernel. Tokens are
        dispatched in Blockwise FP8 format.
        """
        var output_tokens_tensor = output_tokens.to_tile_tensor[DType.int64]()
        var output_scales_tensor = output_scales.to_tile_tensor[DType.int64]()

        var format_handler = BlockwiseFP8TokenFormat[hidden_size, top_k](
            output_tokens_tensor, output_scales_tensor
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            target,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            src_info.to_tile_tensor[DType.int64](),
            atomic_counters.to_tile_tensor[DType.int64](),
            input_tokens.to_tile_tensor[DType.int64](),
            topk_ids.to_tile_tensor[DType.int64](),
            send_ptrs.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("ep.dispatch.nvfp4")
struct Struct_ep_dispatch_nvfp4:
    @always_inline
    @staticmethod
    @parameter
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=5, ...],
        row_offsets: OutputTensor[dtype=DType.uint32, rank=1, ...],
        scales_offsets: OutputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=DType.int32, rank=1, ...],
        src_info: OutputTensor[dtype=DType.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=input_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        input_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the fused Expert Parallelism NVFP4 dispatch kernel. Tokens
        are dispatched in NVFP4 format.
        """
        var output_tokens_tensor = output_tokens.to_tile_tensor[DType.int64]()
        var output_scales_tensor = output_scales.to_tile_tensor[DType.int64]()
        var scales_offsets_tensor = scales_offsets.to_tile_tensor[DType.int64]()
        var input_scales_tensor = input_scales.to_tile_tensor[DType.int64]()
        comptime assert input_scales_tensor.flat_rank == 1

        @parameter
        @always_inline
        @__copy_capture(input_scales_tensor)
        def input_scales_fn[dtype: DType](expert_id: Int) -> Scalar[dtype]:
            # Currently only use one global input scale for all experts
            return rebind[Scalar[dtype]](input_scales_tensor[0].cast[dtype]())

        var format_handler = NVFP4TokenFormat[hidden_size, top_k](
            output_tokens_tensor,
            output_scales_tensor,
            scales_offsets_tensor,
            context[],
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            target,
            input_scales_wrapper=input_scales_fn,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            src_info.to_tile_tensor[DType.int64](),
            atomic_counters.to_tile_tensor[DType.int64](),
            input_tokens.to_tile_tensor[DType.int64](),
            topk_ids.to_tile_tensor[DType.int64](),
            send_ptrs.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.distributed.ep.dispatch.nvfp4")
struct DistributedEPDispatchNVFP4:
    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tokens: OutputVariadicTensors[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputVariadicTensors[
            dtype=dispatch_scale_dtype, rank=5, ...
        ],
        row_offsets: OutputVariadicTensors[dtype=DType.uint32, rank=1, ...],
        scales_offsets: OutputVariadicTensors[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputVariadicTensors[dtype=DType.int32, rank=1, ...],
        src_info: OutputVariadicTensors[dtype=DType.int32, rank=2, ...],
        input_tokens: InputVariadicTensors[dtype=input_dtype, rank=2, ...],
        topk_ids: InputVariadicTensors[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        input_scales: InputVariadicTensors[dtype=DType.float32, rank=1, ...],
        atomic_counters: MutableInputVariadicTensors[
            dtype=DType.int32, rank=1, ...
        ],
        dev_ctxs: DeviceContextPtrList,
    ) capturing raises:
        """Multi-device fused Expert Parallelism NVFP4 dispatch.

        Launches the EP dispatch kernel on all devices simultaneously via
        _ep_launch_device_collective. Each device routes its tokens to experts
        based on top-k IDs, quantizes them to NVFP4 format, and sends them
        to the appropriate peer devices.
        """
        comptime num_devices = input_tokens.size

        # Filter the dev_ctxs list to only GPU devices. The op also takes
        # CPU host-pointer tensors (send_ptrs, recv_ptrs, recv_count_ptrs),
        # so the DeviceContextPtrList may contain CPU contexts.
        var gpu_ctxs_tuple = StaticTuple[DeviceContextPtr, num_devices]()
        var dev_idx = 0
        for i in range(dev_ctxs.size):
            if dev_idx < num_devices and dev_ctxs[i].api() != "cpu":
                gpu_ctxs_tuple[dev_idx] = dev_ctxs.ptrs[i]
                dev_idx += 1
        if dev_idx != num_devices:
            raise Error("Invalid number of GPU device contexts")
        var gpu_ctxs = DeviceContextPtrList[num_devices](gpu_ctxs_tuple)

        @always_inline
        def launch_dispatch[
            index: Int
        ]() raises {
            read output_tokens,
            read output_scales,
            read row_offsets,
            read scales_offsets,
            read expert_ids,
            read src_info,
            read atomic_counters,
            read input_tokens,
            read topk_ids,
            read send_ptrs,
            read recv_ptrs,
            read recv_count_ptrs,
            read input_scales,
            read gpu_ctxs,
        }:
            var out_tokens = output_tokens[index].to_tile_tensor[DType.int64]()
            var out_scales = output_scales[index].to_tile_tensor[DType.int64]()
            var sc_offsets = scales_offsets[index].to_tile_tensor[DType.int64]()
            var in_scales = input_scales[index].to_tile_tensor[DType.int64]()
            comptime assert in_scales.flat_rank == 1

            @parameter
            @always_inline
            @__copy_capture(in_scales)
            def input_scales_fn[dtype: DType](expert_id: Int) -> Scalar[dtype]:
                return rebind[Scalar[dtype]](in_scales[0].cast[dtype]())

            var format_handler = NVFP4TokenFormat[hidden_size, top_k](
                out_tokens,
                out_scales,
                sc_offsets,
                gpu_ctxs[index],
            )

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
                target,
                input_scales_wrapper=input_scales_fn,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[DType.int64](),
                expert_ids[index].to_tile_tensor[DType.int64](),
                src_info[index].to_tile_tensor[DType.int64](),
                atomic_counters[index].to_tile_tensor[DType.int64](),
                input_tokens[index].to_tile_tensor[DType.int64](),
                topk_ids[index].to_tile_tensor[DType.int64](),
                send_ptrs[index].to_tile_tensor[DType.int64](),
                recv_ptrs[index].to_tile_tensor[DType.int64](),
                recv_count_ptrs[index].to_tile_tensor[DType.int64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs)


@compiler.register("mo.distributed.ep.dispatch.mxfp4")
struct DistributedEPDispatchMXFP4:
    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tokens: OutputVariadicTensors[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputVariadicTensors[
            dtype=dispatch_scale_dtype, rank=2, ...
        ],
        row_offsets: OutputVariadicTensors[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputVariadicTensors[dtype=DType.int32, rank=1, ...],
        src_info: OutputVariadicTensors[dtype=DType.int32, rank=2, ...],
        input_tokens: InputVariadicTensors[dtype=input_dtype, rank=2, ...],
        topk_ids: InputVariadicTensors[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        atomic_counters: MutableInputVariadicTensors[
            dtype=DType.int32, rank=1, ...
        ],
        dev_ctxs: DeviceContextPtrList,
    ) capturing raises:
        """Multi-device fused Expert Parallelism NVFP4 dispatch.

        Launches the EP dispatch kernel on all devices simultaneously via
        _ep_launch_device_collective. Each device routes its tokens to experts
        based on top-k IDs, quantizes them to MXFP4 format, and sends them
        to the appropriate peer devices.
        """
        comptime num_devices = input_tokens.size

        # Filter the dev_ctxs list to only GPU devices. The op also takes
        # CPU host-pointer tensors (send_ptrs, recv_ptrs, recv_count_ptrs),
        # so the DeviceContextPtrList may contain CPU contexts.
        var gpu_ctxs_tuple = StaticTuple[DeviceContextPtr, num_devices]()
        var dev_idx = 0
        for i in range(dev_ctxs.size):
            if dev_idx < num_devices and dev_ctxs[i].api() != "cpu":
                gpu_ctxs_tuple[dev_idx] = dev_ctxs.ptrs[i]
                dev_idx += 1
        if dev_idx != num_devices:
            raise Error("Invalid number of GPU device contexts")
        var gpu_ctxs = DeviceContextPtrList[num_devices](gpu_ctxs_tuple)

        @always_inline
        def launch_dispatch[
            index: Int
        ]() raises {
            read output_tokens,
            read output_scales,
            read row_offsets,
            read expert_ids,
            read src_info,
            read atomic_counters,
            read input_tokens,
            read topk_ids,
            read send_ptrs,
            read recv_ptrs,
            read recv_count_ptrs,
            read gpu_ctxs,
        }:
            var out_tokens = output_tokens[index].to_tile_tensor[DType.int64]()
            var out_scales = output_scales[index].to_tile_tensor[DType.int64]()

            var format_handler = MXFP4TokenFormat[hidden_size, top_k](
                out_tokens,
                out_scales,
            )

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
                target,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[DType.int64](),
                expert_ids[index].to_tile_tensor[DType.int64](),
                src_info[index].to_tile_tensor[DType.int64](),
                atomic_counters[index].to_tile_tensor[DType.int64](),
                input_tokens[index].to_tile_tensor[DType.int64](),
                topk_ids[index].to_tile_tensor[DType.int64](),
                send_ptrs[index].to_tile_tensor[DType.int64](),
                recv_ptrs[index].to_tile_tensor[DType.int64](),
                recv_count_ptrs[index].to_tile_tensor[DType.int64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs)


@compiler.register("mo.distributed.ep.dispatch")
struct DistributedEPDispatch:
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tokens: OutputVariadicTensors[dtype=DType.bfloat16, rank=2, ...],
        row_offsets: OutputVariadicTensors[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputVariadicTensors[dtype=DType.int32, rank=1, ...],
        src_info: OutputVariadicTensors[dtype=DType.int32, rank=2, ...],
        input_tokens: InputVariadicTensors[dtype=dispatch_dtype, rank=2, ...],
        topk_ids: InputVariadicTensors[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        atomic_counters: MutableInputVariadicTensors[
            dtype=DType.int32, rank=1, ...
        ],
        dev_ctxs: DeviceContextPtrList,
    ) capturing raises:
        """Multi-device fused Expert Parallelism BF16 dispatch."""
        comptime num_devices = input_tokens.size
        comptime assert dispatch_dtype == DType.bfloat16

        var gpu_ctxs_tuple = StaticTuple[DeviceContextPtr, num_devices]()
        var dev_idx = 0
        for i in range(dev_ctxs.size):
            if dev_idx < num_devices and dev_ctxs[i].api() != "cpu":
                gpu_ctxs_tuple[dev_idx] = dev_ctxs.ptrs[i]
                dev_idx += 1
        if dev_idx != num_devices:
            raise Error("Invalid number of GPU device contexts")
        var gpu_ctxs = DeviceContextPtrList[num_devices](gpu_ctxs_tuple)

        @always_inline
        def launch_dispatch[
            index: Int
        ]() raises {
            read output_tokens,
            read row_offsets,
            read expert_ids,
            read src_info,
            read atomic_counters,
            read input_tokens,
            read topk_ids,
            read send_ptrs,
            read recv_ptrs,
            read recv_count_ptrs,
            read gpu_ctxs,
        }:
            var out_tokens = output_tokens[index].to_tile_tensor[DType.int64]()
            var format_handler = BF16TokenFormat[hidden_size, top_k](out_tokens)

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
                target,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[DType.int64](),
                expert_ids[index].to_tile_tensor[DType.int64](),
                src_info[index].to_tile_tensor[DType.int64](),
                atomic_counters[index].to_tile_tensor[DType.int64](),
                input_tokens[index].to_tile_tensor[DType.int64](),
                topk_ids[index].to_tile_tensor[DType.int64](),
                send_ptrs[index].to_tile_tensor[DType.int64](),
                recv_ptrs[index].to_tile_tensor[DType.int64](),
                recv_count_ptrs[index].to_tile_tensor[DType.int64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs)


@compiler.register("mo.distributed.ep.dispatch.fp8")
struct DistributedEPDispatchFP8:
    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        fused_shared_expert: Bool,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tokens: OutputVariadicTensors[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputVariadicTensors[
            dtype=dispatch_scale_dtype, rank=2, ...
        ],
        row_offsets: OutputVariadicTensors[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputVariadicTensors[dtype=DType.int32, rank=1, ...],
        src_info: OutputVariadicTensors[dtype=DType.int32, rank=2, ...],
        input_tokens: InputVariadicTensors[dtype=input_dtype, rank=2, ...],
        topk_ids: InputVariadicTensors[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        atomic_counters: MutableInputVariadicTensors[
            dtype=DType.int32, rank=1, ...
        ],
        dev_ctxs: DeviceContextPtrList,
    ) capturing raises:
        """Multi-device fused Expert Parallelism FP8 dispatch."""
        comptime num_devices = input_tokens.size

        var gpu_ctxs_tuple = StaticTuple[DeviceContextPtr, num_devices]()
        var dev_idx = 0
        for i in range(dev_ctxs.size):
            if dev_idx < num_devices and dev_ctxs[i].api() != "cpu":
                gpu_ctxs_tuple[dev_idx] = dev_ctxs.ptrs[i]
                dev_idx += 1
        if dev_idx != num_devices:
            raise Error("Invalid number of GPU device contexts")
        var gpu_ctxs = DeviceContextPtrList[num_devices](gpu_ctxs_tuple)

        @always_inline
        def launch_dispatch[
            index: Int
        ]() raises {
            read output_tokens,
            read output_scales,
            read row_offsets,
            read expert_ids,
            read src_info,
            read atomic_counters,
            read input_tokens,
            read topk_ids,
            read send_ptrs,
            read recv_ptrs,
            read recv_count_ptrs,
            read gpu_ctxs,
        }:
            var out_tokens = output_tokens[index].to_tile_tensor[DType.int64]()
            var out_scales = output_scales[index].to_tile_tensor[DType.int64]()
            var format_handler = BlockwiseFP8TokenFormat[hidden_size, top_k](
                out_tokens, out_scales
            )

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
                target,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[DType.int64](),
                expert_ids[index].to_tile_tensor[DType.int64](),
                src_info[index].to_tile_tensor[DType.int64](),
                atomic_counters[index].to_tile_tensor[DType.int64](),
                input_tokens[index].to_tile_tensor[DType.int64](),
                topk_ids[index].to_tile_tensor[DType.int64](),
                send_ptrs[index].to_tile_tensor[DType.int64](),
                recv_ptrs[index].to_tile_tensor[DType.int64](),
                recv_count_ptrs[index].to_tile_tensor[DType.int64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs)


@compiler.register("mo.distributed.ep.combine")
struct DistributedEPCombine:
    @staticmethod
    def execute[
        combine_dtype: DType,
        router_weights_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        lambdas_have_fusion: Bool,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tokens: FusedOutputVariadicTensors[
            dtype=combine_dtype, rank=2, ...
        ],
        input_tokens: InputVariadicTensors[dtype=combine_dtype, rank=2, ...],
        src_info: InputVariadicTensors[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        router_weights: InputVariadicTensors[
            dtype=router_weights_dtype, rank=2, ...
        ],
        atomic_counters: MutableInputVariadicTensors[
            dtype=DType.int32, rank=1, ...
        ],
        dev_ctxs: DeviceContextPtrList,
    ) capturing raises:
        """Multi-device fused Expert Parallelism combine with output fusion."""
        comptime num_devices = input_tokens.size

        var gpu_ctxs_tuple = StaticTuple[DeviceContextPtr, num_devices]()
        var dev_idx = 0
        for i in range(dev_ctxs.size):
            if dev_idx < num_devices and dev_ctxs[i].api() != "cpu":
                gpu_ctxs_tuple[dev_idx] = dev_ctxs.ptrs[i]
                dev_idx += 1
        if dev_idx != num_devices:
            raise Error("Invalid number of GPU device contexts")
        var gpu_ctxs = DeviceContextPtrList[num_devices](gpu_ctxs_tuple)

        @always_inline
        def launch_combine[
            index: Int
        ]() raises {
            read output_tokens,
            read input_tokens,
            read src_info,
            read send_ptrs,
            read recv_ptrs,
            read recv_count_ptrs,
            read router_weights,
            read atomic_counters,
            read gpu_ctxs,
        }:
            var rw_tensor = router_weights[index].to_tile_tensor[DType.int64]()

            @parameter
            @always_inline
            @__copy_capture(rw_tensor)
            def router_weights_fn[
                width: Int
            ](token_idx: Int, topk_id: Int) -> SIMD[DType.float32, width]:
                return rw_tensor.load[width=width](
                    (Idx(token_idx), Idx(topk_id))
                ).cast[DType.float32]()

            @parameter
            @always_inline
            def output_fn[
                dtype: DType, width: Int, *, alignment: Int = 1
            ](coords: IndexList[2], val: SIMD[dtype, width]):
                output_tokens[index]._lambda_store[
                    width=width, element_alignment=alignment
                ](
                    coords,
                    rebind[SIMD[combine_dtype, width]](val),
                )

            ep_fused_combine_kernel_api[
                hidden_size,
                top_k,
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                target,
                router_weights_wrapper=router_weights_fn,
                epilogue_fn=Optional[elementwise_epilogue_type](
                    output_fn
                ) if lambdas_have_fusion else None,
                fused_shared_expert=fused_shared_expert,
            ](
                output_tokens[index].to_tile_tensor[DType.int64](),
                atomic_counters[index].to_tile_tensor[DType.int64](),
                input_tokens[index].to_tile_tensor[DType.int64](),
                src_info[index].to_tile_tensor[DType.int64](),
                send_ptrs[index].to_tile_tensor[DType.int64](),
                recv_ptrs[index].to_tile_tensor[DType.int64](),
                recv_count_ptrs[index].to_tile_tensor[DType.int64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_combine, gpu_ctxs)


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Combine Kernel
# ===-----------------------------------------------------------------------===#


@compiler.register("ep.combine_async")
struct Struct_ep_combine_async:
    @always_inline
    @staticmethod
    def execute[
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
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=combine_dtype, rank=2, ...],
        src_info: InputTensor[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism combine kernel."""

        ep_combine_async_kernel_api[
            combine_dtype,
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            atomic_counters.to_tile_tensor[DType.int64](),
            input_tokens.to_tile_tensor[DType.int64](),
            src_info.to_tile_tensor[DType.int64](),
            send_ptrs.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("ep.combine_wait")
struct Struct_ep_combine_wait:
    @parameter
    @always_inline
    @staticmethod
    def execute[
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
        output_tokens: FusedOutputTensor[dtype=combine_dtype, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        router_weights: InputTensor[dtype=router_weights_dtype, rank=2, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism combine completion kernel."""
        var router_weights_tensor = router_weights.to_tile_tensor[DType.int64]()
        comptime assert router_weights_tensor.flat_rank == 2
        comptime assert router_weights_tensor.flat_rank >= 2

        @parameter
        @always_inline
        @__copy_capture(router_weights_tensor)
        def router_weights_fn[
            width: Int
        ](token_idx: Int, topk_id: Int) -> SIMD[DType.float32, width]:
            return router_weights_tensor.load[width=width](
                (Idx(token_idx), Idx(topk_id))
            ).cast[DType.float32]()

        @parameter
        @always_inline
        def output_fn[
            dtype: DType, width: SIMDSize, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[dtype, width]):
            output_tokens._lambda_store[
                width=width, element_alignment=alignment
            ](
                coords,
                rebind[SIMD[combine_dtype, width]](val),
            )

        ep_combine_wait_kernel_api[
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
            router_weights_wrapper=router_weights_fn,
            epilogue_fn=Optional[elementwise_epilogue_type](
                output_fn
            ) if lambdas_have_fusion else None,
        ](
            output_tokens.to_tile_tensor[DType.int64](),
            atomic_counters.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("ep.combine")
struct Struct_ep_combine:
    @always_inline
    @staticmethod
    @parameter
    def execute[
        combine_dtype: DType,
        router_weights_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        lambdas_have_fusion: Bool,
        skip_a2a: Bool,
        //,
        target: StaticString,
    ](
        output_tokens: FusedOutputTensor[dtype=combine_dtype, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=combine_dtype, rank=2, ...],
        src_info: InputTensor[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        router_weights: InputTensor[dtype=router_weights_dtype, rank=2, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the fused Expert Parallelism combine kernel."""
        var router_weights_tensor = router_weights.to_tile_tensor[DType.int64]()
        comptime assert router_weights_tensor.flat_rank == 2
        comptime assert router_weights_tensor.flat_rank >= 2

        @parameter
        @always_inline
        @__copy_capture(router_weights_tensor)
        def router_weights_fn[
            width: Int
        ](token_idx: Int, topk_id: Int) -> SIMD[DType.float32, width]:
            return router_weights_tensor.load[width=width](
                (Idx(token_idx), Idx(topk_id))
            ).cast[DType.float32]()

        @parameter
        @always_inline
        def output_fn[
            dtype: DType, width: SIMDSize, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[dtype, width]):
            output_tokens._lambda_store[
                width=width, element_alignment=alignment
            ](
                coords,
                rebind[SIMD[combine_dtype, width]](val),
            )

        ep_fused_combine_kernel_api[
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
            router_weights_wrapper=router_weights_fn,
            epilogue_fn=Optional[elementwise_epilogue_type](
                output_fn
            ) if lambdas_have_fusion else None,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
        ](
            output_tokens.to_tile_tensor[DType.int64](),
            atomic_counters.to_tile_tensor[DType.int64](),
            input_tokens.to_tile_tensor[DType.int64](),
            src_info.to_tile_tensor[DType.int64](),
            send_ptrs.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("ep.combine.skip_a2a")
struct Struct_ep_combine_skip_a2a:
    @always_inline
    @staticmethod
    @parameter
    def execute[
        combine_dtype: DType,
        router_weights_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        lambdas_have_fusion: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        //,
        target: StaticString,
    ](
        output_tokens: FusedOutputTensor[dtype=combine_dtype, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=combine_dtype, rank=2, ...],
        src_info: InputTensor[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        router_weights: InputTensor[dtype=router_weights_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=DType.int32, rank=2, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the fused Expert Parallelism combine kernel."""
        var router_weights_tensor = router_weights.to_tile_tensor[DType.int64]()
        comptime assert router_weights_tensor.flat_rank == 2
        comptime assert router_weights_tensor.flat_rank >= 2

        @parameter
        @always_inline
        @__copy_capture(router_weights_tensor)
        def router_weights_fn[
            width: Int
        ](token_idx: Int, topk_id: Int) -> SIMD[DType.float32, width]:
            return router_weights_tensor.load[width=width](
                (Idx(token_idx), Idx(topk_id))
            ).cast[DType.float32]()

        @parameter
        @always_inline
        def output_fn[
            dtype: DType, width: SIMDSize, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[dtype, width]):
            output_tokens._lambda_store[
                width=width, element_alignment=alignment
            ](
                coords,
                rebind[SIMD[combine_dtype, width]](val),
            )

        ep_fused_combine_kernel_api[
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
            router_weights_wrapper=router_weights_fn,
            epilogue_fn=Optional[elementwise_epilogue_type](
                output_fn
            ) if lambdas_have_fusion else None,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            output_tokens.to_tile_tensor[DType.int64](),
            atomic_counters.to_tile_tensor[DType.int64](),
            input_tokens.to_tile_tensor[DType.int64](),
            src_info.to_tile_tensor[DType.int64](),
            send_ptrs.to_tile_tensor[DType.int64](),
            recv_ptrs.to_tile_tensor[DType.int64](),
            recv_count_ptrs.to_tile_tensor[DType.int64](),
            context,
            topk_ids._ptr.as_immutable().unsafe_origin_cast[
                ImmutExternalOrigin
            ](),
        )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Utils
# ===-----------------------------------------------------------------------===#


@compiler.register("ep.fused_silu")
struct Struct_ep_fused_silu:
    @always_inline
    @staticmethod
    def execute[
        output_dtype: DType,
        input_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_dtype, rank=2, ...],
        input: InputTensor[dtype=input_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
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
        comptime assert is_gpu[target](), "EP is only supported on GPU."

        var output_tensor = output.to_tile_tensor[DType.int64]()
        var input_tensor = input.to_tile_tensor[DType.int64]().as_immut()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            DType.int64
        ]().as_immut()

        var gpu_ctx = context.get_device_context()
        comptime hw_info = gpu_ctx.default_device_info

        comptime fused_silu = fused_silu_kernel[
            output_dtype,
            input_dtype,
            output_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
        ]

        @always_inline
        @parameter
        def description_fn() -> String:
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
            gpu_ctx.enqueue_function[fused_silu](
                output_tensor,
                input_tensor,
                row_offsets_tensor,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )


@compiler.register("ep.fused_silu.fp8")
struct Struct_ep_fused_silu_fp8:
    @always_inline
    @staticmethod
    def execute[
        fp8_dtype: DType,
        scales_dtype: DType,
        input_dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=fp8_dtype, rank=2, ...],
        scales: OutputTensor[dtype=scales_dtype, rank=2, ...],
        input: InputTensor[dtype=input_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
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
        comptime assert is_gpu[target](), "EP is only supported on GPU."

        comptime group_size = 128

        var output_tensor = output.to_tile_tensor[DType.int64]()
        var scales_tensor = scales.to_tile_tensor[DType.int64]()
        var input_tensor = input.to_tile_tensor[DType.int64]().as_immut()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            DType.int64
        ]().as_immut()

        var gpu_ctx = context.get_device_context()
        comptime hw_info = gpu_ctx.default_device_info

        comptime fused_silu_fp8 = fused_silu_fp8_kernel[
            fp8_dtype,
            scales_dtype,
            input_dtype,
            output_tensor.LayoutType,
            scales_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
            group_size,
        ]

        @always_inline
        @parameter
        def description_fn() -> String:
            # fmt: off
            return String(
                "fp8_dtype=", fp8_dtype,
                ";scales_dtype=", scales_dtype,
                ";input_dtype=", input_dtype,
                ";group_size=", group_size,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.fused_silu.fp8",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu_fp8](
                output_tensor,
                scales_tensor,
                input_tensor,
                row_offsets_tensor,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )


@compiler.register("ep.fused_silu.mxfp4")
struct Struct_ep_fused_silu_mxfp4:
    @always_inline
    @staticmethod
    def execute[
        fp4_dtype: DType,
        scales_dtype: DType,
        input_dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=fp4_dtype, rank=2, ...],
        scales: OutputTensor[dtype=scales_dtype, rank=2, ...],
        input: InputTensor[dtype=input_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism fused SILU kernel with MXFP4
        quantization.

        This function launches the fused_silu_mxfp4 kernel to perform the SILU
        operation for all the MLPs in the EP MoE module.

        This kernel will read the row offsets to determine the actual number of
        received tokens in the input tensor, and then only perform the SILU
        operation on the received tokens. Once the SILU operation is performed,
        the output will be quantized to the MXFP4 format.
        """
        # Ensure this kernel only runs on GPU targets
        comptime assert is_gpu[target](), "EP is only supported on GPU."

        var output_tensor = output.to_tile_tensor[DType.int64]()
        var scales_tensor = scales.to_tile_tensor[DType.int64]()
        var input_tensor = input.to_tile_tensor[DType.int64]().as_immut()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            DType.int64
        ]().as_immut()

        var gpu_ctx = context.get_device_context()
        comptime hw_info = gpu_ctx.default_device_info

        comptime fused_silu_mxfp4 = fused_silu_mxfp4_kernel[
            fp4_dtype,
            scales_dtype,
            input_dtype,
            output_tensor.LayoutType,
            scales_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
        ]

        @always_inline
        @parameter
        def description_fn() -> String:
            # fmt: off
            return String(
                "fp4_dtype=", fp4_dtype,
                ";scales_dtype=", scales_dtype,
                ";input_dtype=", input_dtype,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.fused_silu.mxfp4",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu_mxfp4](
                output_tensor,
                scales_tensor,
                input_tensor,
                row_offsets_tensor,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )


@compiler.register("ep.fused_silu.nvfp4")
struct Struct_ep_fused_silu_nvfp4:
    @always_inline
    @staticmethod
    def execute[
        fp4_dtype: DType,
        scales_dtype: DType,
        input_dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=fp4_dtype, rank=2, ...],
        scales: OutputTensor[dtype=scales_dtype, rank=5, ...],
        input: InputTensor[dtype=input_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        scales_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        input_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        context: DeviceContextPtr,
    ) raises:
        """Execute the Expert Parallelism fused SILU kernel with NVFP4
        quantization.

        This function launches the fused_silu_nvfp4 kernel to perform the SILU
        operation for all the MLPs in the EP MoE module.

        This kernel will read the row offsets to determine the actual number of
        received tokens in the input tensor, and then only perform the SILU
        operation on the received tokens. Once the SILU operation is performed,
        the output will be quantized to the NVFP4 format. The scales tensor
        will be padded and zero-filled.
        """
        # Ensure this kernel only runs on GPU targets
        comptime assert is_gpu[target](), "EP is only supported on GPU."

        var output_tensor = output.to_tile_tensor[DType.int64]()
        var scales_tensor = scales.to_tile_tensor[DType.int64]()
        var input_tensor = input.to_tile_tensor[DType.int64]().as_immut()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            DType.int64
        ]().as_immut()
        var scales_offsets_tensor = scales_offsets.to_tile_tensor[
            DType.int64
        ]().as_immut()
        var input_scales_tensor = input_scales.to_tile_tensor[
            DType.int64
        ]().as_immut()

        var gpu_ctx = context.get_device_context()
        comptime hw_info = gpu_ctx.default_device_info

        comptime fused_silu_nvfp4 = fused_silu_nvfp4_kernel[
            fp4_dtype,
            scales_dtype,
            input_dtype,
            output_tensor.LayoutType,
            scales_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            scales_offsets_tensor.LayoutType,
            input_scales_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
        ]

        @always_inline
        @parameter
        def description_fn() -> String:
            # fmt: off
            return String(
                "fp4_dtype=", fp4_dtype,
                ";scales_dtype=", scales_dtype,
                ";input_dtype=", input_dtype,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.fused_silu.nvfp4",
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu_nvfp4](
                output_tensor,
                scales_tensor,
                input_tensor,
                row_offsets_tensor,
                scales_offsets_tensor,
                input_scales_tensor,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

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


import extensibility
from comm.sync import is_p2p_enabled
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceContextArray
from max.gpu.host.info import is_gpu
from std.memory.pointer import pointer_to_int
from layout.tile_tensor import row_major
from layout import Coord, TileTensor, coord_to_index_list

from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id
from std.math import align_up
from std.sys.info import size_of
from extensibility import (
    Fused,
    Input,
    MutableInput,
    Output,
    Storable,
    StoreAccess,
    Tensor,
    TensorTuple,
    TileTensorable,
    UnsafeFusedView,
    unsafe_fused_view,
)

from comm.device_collective import _launch_device_collective
from .kernels import _check_tensor_tuple

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
    ep_token_recv,
)
from shmem.ep_comm import (
    EP_WARP_TOKENS_MIN_TOKENS,
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    EPLocalSyncCounters,
    ExpertLoadSink,
    ExpertLoadTally,
    MXTokenFormat,
    NVBlockScaledTokenFormat,
    ep_dispatch_recv_size,
    NullExpertLoad,
    fused_silu_kernel,
    fused_silu_mx_kernel,
    fused_silu_mxfp6_kernel,
    fused_silu_nvfp4_kernel,
)
from linalg.mx_format import MXFormat
from linalg.fp6_utils import FP6Format
from linalg.swiglu_activation import get_swiglu_activation

comptime RT_LAYOUT_2D = type_of(row_major(Int64(1), Int64(1)))


@extensibility.register("ep.init")
struct Struct_ep_init:
    """Registers the `ep.init` graph op with the graph compiler."""

    @inline(.always)
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
        *,
        nvfp4_dyn_global_scales: Bool = False,
    ](
        dev_ptrs: Some[Output & Tensor],
        my_rank_tensor: Some[Output & Tensor],
        atomic_counters_0: Some[MutableInput & Tensor],
        atomic_counters_1: Some[MutableInput & Tensor],
        context: DeviceContext,
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
            nvfp4_dyn_global_scales: Whether the `BLOCK_SCALED_NV` messages
                carry a per-token global scale, which makes them larger.

        Arguments:
            dev_ptrs: Output tensor to store device pointers. Shape [2, 3] where:
                     - First dimension: buffer groups (0=dispatch, 1=combine)
                     - Second dimension: buffer types (0=send, 1=recv, 2=recv_count)
            my_rank_tensor: Output tensor to store current device's rank.
            atomic_counters_0: Atomic counters for buffer group 0.
            atomic_counters_1: Atomic counters for buffer group 1.
            context: GPU device context
        """
        comptime assert conforms_to(type_of(dev_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(my_rank_tensor), TileTensorable)
        comptime assert conforms_to(type_of(atomic_counters_0), TileTensorable)
        comptime assert conforms_to(type_of(atomic_counters_1), TileTensorable)
        comptime assert type_of(dev_ptrs).rank == 2
        comptime assert type_of(my_rank_tensor).rank == 1
        comptime assert type_of(atomic_counters_0).rank == 1
        comptime assert type_of(atomic_counters_1).rank == 1
        # Ensure this kernel only runs on GPU targets
        comptime assert context.target.is_gpu(), "EP is only supported on GPU."
        var gpu_ctx = context

        # Calculate buffer sizes for dispatch phase
        var dispatch_msg_size: Int

        # Infer message sizes for dispatch phases
        comptime match dispatch_fmt_str:
            case "BlockwiseFP8":
                comptime token_fmt_type = BlockwiseFP8TokenFormat[
                    fp8_dtype=dispatch_dtype,
                    scales_dtype=dispatch_scale_dtype,
                    output_layout=RT_LAYOUT_2D,
                    scales_layout=RT_LAYOUT_2D,
                    hidden_size,
                    top_k,
                ]
                dispatch_msg_size = token_fmt_type.msg_size()

            case "BLOCK_SCALED_NV":
                comptime token_fmt_type = NVBlockScaledTokenFormat[
                    quant_dtype=dispatch_dtype,
                    scales_dtype=dispatch_scale_dtype,
                    output_layout=RT_LAYOUT_2D,
                    scales_offset_layout=RT_LAYOUT_2D,
                    hidden_size,
                    top_k,
                    nvfp4_dyn_global_scales=nvfp4_dyn_global_scales,
                ]
                dispatch_msg_size = token_fmt_type.msg_size()

            case "MXFP4":
                comptime token_fmt_type = MXTokenFormat[
                    quant_dtype=dispatch_dtype,
                    scales_dtype=dispatch_scale_dtype,
                    output_layout=RT_LAYOUT_2D,
                    scales_layout=RT_LAYOUT_2D,
                    hidden_size,
                    top_k,
                ]
                dispatch_msg_size = token_fmt_type.msg_size()

            case "MXFP6":
                comptime token_fmt_type = MXTokenFormat[
                    quant_dtype=dispatch_dtype,
                    scales_dtype=dispatch_scale_dtype,
                    output_layout=RT_LAYOUT_2D,
                    scales_layout=RT_LAYOUT_2D,
                    hidden_size,
                    top_k,
                    mx_format=MXFormat.FP6_E2M3,
                ]
                dispatch_msg_size = token_fmt_type.msg_size()

            case "BF16":
                comptime token_fmt_type = BF16TokenFormat[
                    output_layout=RT_LAYOUT_2D, hidden_size, top_k
                ]
                dispatch_msg_size = token_fmt_type.msg_size()

            case _:
                raise Error(
                    "Invalid dispatch format string: ", dispatch_fmt_str
                )

        var dispatch_send_size = max_token_per_rank * dispatch_msg_size
        var dispatch_recv_size = ep_dispatch_recv_size(
            n_experts=n_experts,
            n_ranks=n_gpus_per_node * n_nodes,
            max_tokens_per_rank=max_token_per_rank,
            msg_bytes=dispatch_msg_size,
            token_recv=ep_token_recv[n_nodes](),
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
        # A config that can host a fused EP kernel allocates the reserved
        # words too (`allocation_size()`); the memset zeroes them as well.
        comptime counters_0_size = type_of(atomic_counters_0)._shape_types[
            0
        ].static_value
        comptime counters_1_size = type_of(atomic_counters_1)._shape_types[
            0
        ].static_value
        comptime assert (
            counters_0_size == EPLocalSyncCounters[n_experts].total_size()
            or counters_0_size
            == EPLocalSyncCounters[n_experts].allocation_size()
        ), "Atomic counters 0 size doesn't match expected size."
        var atomic_counters_0_buf = atomic_counters_0.to_tile_tensor[
            .int32
        ]().to_device_buffer(gpu_ctx)
        gpu_ctx.enqueue_memset(atomic_counters_0_buf, Int32(0))

        comptime assert (
            counters_1_size == counters_0_size
        ), "Atomic counters 1 size doesn't match expected size."
        var atomic_counters_1_buf = atomic_counters_1.to_tile_tensor[
            .int32
        ]().to_device_buffer(gpu_ctx)
        gpu_ctx.enqueue_memset(atomic_counters_1_buf, Int32(0))

        var dispatch_send_p: MutPointer[UInt8, MutUntrackedOrigin]
        var dispatch_recv_p: MutPointer[UInt8, MutUntrackedOrigin]
        var dispatch_recv_count_p: MutPointer[UInt64, MutUntrackedOrigin]

        var combine_send_p: Optional[MutPointer[UInt8, MutUntrackedOrigin]]
        var combine_recv_p: MutPointer[UInt8, MutUntrackedOrigin]
        var combine_recv_count_p: MutPointer[UInt64, MutUntrackedOrigin]

        comptime if n_nodes > 1:
            # Initialize the SHMEM library for this GPU
            comptime if gpu_ctx.target.is_amd_gpu():
                shmem_init_thread_tcp(gpu_ctx, gpus_per_node=n_gpus_per_node)
            else:
                shmem_init_thread_mpi(gpu_ctx, gpus_per_node=n_gpus_per_node)

            # Allocate SHMEM buffers for dispatch phase
            dispatch_send_p = shmem_malloc[.uint8](dispatch_send_size)
            dispatch_recv_p = shmem_malloc[.uint8](dispatch_recv_size)
            dispatch_recv_count_p = shmem_malloc[.uint64](n_experts)

            # Allocate SHMEM buffers for combine phase
            combine_send_p = shmem_malloc[.uint8](combine_send_size)
            combine_recv_p = shmem_malloc[.uint8](combine_recv_size)
            combine_recv_count_p = shmem_malloc[.uint64](n_experts)

        else:
            comptime if n_gpus_per_node > 1:
                if not is_p2p_enabled():
                    raise Error("P2P is not supported on this system.")
            dispatch_send_p = gpu_ctx.enqueue_create_buffer[.uint8](
                dispatch_send_size
            ).take_ptr()
            dispatch_recv_p = gpu_ctx.enqueue_create_buffer[.uint8](
                dispatch_recv_size
            ).take_ptr()
            dispatch_recv_count_p = gpu_ctx.enqueue_create_buffer[.uint64](
                n_experts
            ).take_ptr()

            # When all the devices are on the same node, we skip the combine
            # send buffer and directly send tokens to each device's recv buffer.
            # Hence, we don't need to allocate the combine send buffer.
            combine_send_p = None
            combine_recv_p = gpu_ctx.enqueue_create_buffer[.uint8](
                combine_recv_size
            ).take_ptr()
            combine_recv_count_p = gpu_ctx.enqueue_create_buffer[.uint64](
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

        var dev_ptrs_tensor = dev_ptrs.to_tile_tensor[.uint64]()

        # Group 0: Dispatch phase buffer pointers
        dev_ptrs_tensor[0, 0] = UInt64(Int(dispatch_send_p))
        dev_ptrs_tensor[0, 1] = UInt64(Int(dispatch_recv_p))
        dev_ptrs_tensor[0, 2] = UInt64(Int(dispatch_recv_count_p))

        # Group 1: Combine phase buffer pointers
        dev_ptrs_tensor[1, 0] = UInt64(pointer_to_int(combine_send_p))
        dev_ptrs_tensor[1, 1] = UInt64(Int(combine_recv_p))
        dev_ptrs_tensor[1, 2] = UInt64(Int(combine_recv_count_p))

        # Store current device's rank
        var my_rank: Int32

        comptime if n_nodes > 1:
            my_rank = Int32(shmem_my_pe())
        else:
            my_rank = Int32(gpu_ctx.device().id)
        my_rank_tensor.to_tile_tensor[.int32]()[0] = my_rank


@extensibility.register("ep.dispatch_async")
struct Struct_ep_dispatch_async:
    """Registers the `ep.dispatch_async` graph op with the graph compiler."""

    @inline(.always)
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
        dispatch_fmt_str: StaticString,
        //,
    ](
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism async dispatch kernel. Tokens are
        transferred in either Blockwise FP8 or BF16 format.

        Parameters:
            dispatch_dtype: `DType` used for the quantized token
                payload during dispatch (inferred).
            dispatch_scale_dtype: `DType` of the block scales
                accompanying the dispatched tokens (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            dispatch_fmt_str: String selecting the dispatch token
                format, either `"BlockwiseFP8"` or `"BF16"`
                (inferred).

        Args:
            atomic_counters: Atomic counters coordinating work across
                thread blocks during the dispatch phase.
            input_tokens: Input tokens to dispatch to experts. Shape
                `[num_tokens, hidden_size]`.
            topk_ids: Top-k expert IDs per token. Shape
                `[num_tokens, top_k]`.
            send_ptrs: Send buffer pointers for the dispatch phase.
            recv_ptrs: Receive buffer pointers for the dispatch
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                tokens received per expert.
            context: GPU device context for the current device.
        """
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(topk_ids).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1

        comptime match dispatch_fmt_str:
            case "BlockwiseFP8":
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
                ](
                    atomic_counters.to_tile_tensor[.int32](),
                    input_tokens.to_tile_tensor().as_imm(),
                    topk_ids.to_tile_tensor[.int32]().as_imm(),
                    send_ptrs.to_tile_tensor[.uint64](),
                    recv_ptrs.to_tile_tensor[.uint64](),
                    recv_count_ptrs.to_tile_tensor[.uint64](),
                    context,
                )

            case "BF16":
                comptime token_fmt_type = BF16TokenFormat[
                    output_layout=RT_LAYOUT_2D, hidden_size, top_k
                ]

                ep_dispatch_async_kernel_api[
                    token_fmt_type,
                    n_experts,
                    max_token_per_rank,
                    n_gpus_per_node,
                    n_nodes,
                ](
                    atomic_counters.to_tile_tensor[.int32](),
                    input_tokens.to_tile_tensor().as_imm(),
                    topk_ids.to_tile_tensor[.int32]().as_imm(),
                    send_ptrs.to_tile_tensor[.uint64](),
                    recv_ptrs.to_tile_tensor[.uint64](),
                    recv_count_ptrs.to_tile_tensor[.uint64](),
                    context,
                )

            case _:
                raise Error(
                    "Invalid dispatch format string: ", dispatch_fmt_str
                )


@extensibility.register("ep.dispatch_async.block.scaled.nv")
struct Struct_ep_dispatch_async_block_scaled_nv:
    """Registers the `ep.dispatch_async.block.scaled.nv` graph op with the graph compiler.
    """

    @inline(.always)
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
    ](
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        input_scales: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism async dispatch kernel. Tokens are
        transferred in NVFP4 format.

        Parameters:
            dispatch_dtype: `DType` used for the quantized token
                payload during dispatch (inferred).
            dispatch_scale_dtype: `DType` of the block scales
                accompanying the dispatched tokens (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).

        Args:
            atomic_counters: Atomic counters coordinating work across
                thread blocks during the dispatch phase.
            input_tokens: Input tokens to dispatch to experts. Shape
                `[num_tokens, hidden_size]`.
            topk_ids: Top-k expert IDs per token. Shape
                `[num_tokens, top_k]`.
            send_ptrs: Send buffer pointers for the dispatch phase.
            recv_ptrs: Receive buffer pointers for the dispatch
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                tokens received per expert.
            input_scales: Global input scales for NVFP4 quantization.
            context: GPU device context for the current device.
        """
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(input_scales), TileTensorable)
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(topk_ids).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        comptime assert type_of(input_scales).rank == 1
        var input_scales_tensor = input_scales.to_tile_tensor[.float32]()
        comptime assert input_scales_tensor.flat_rank == 1

        @inline(.always)
        def input_scales_fn[
            dtype: DType
        ](expert_id: Int) {var input_scales_tensor} -> Scalar[dtype]:
            # Currently only use one global input scale for all experts
            return input_scales_tensor.load[width=1](Coord(0)).cast[dtype]()

        comptime token_fmt_type = NVBlockScaledTokenFormat[
            quant_dtype=dispatch_dtype,
            scales_dtype=dispatch_scale_dtype,
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
        ](
            atomic_counters.to_tile_tensor[.int32](),
            input_tokens.to_tile_tensor().as_imm(),
            topk_ids.to_tile_tensor[.int32]().as_imm(),
            send_ptrs.to_tile_tensor[.uint64](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            context,
            input_scales_fn,
        )


@extensibility.register("ep.dispatch_async.mxfp4")
struct Struct_ep_dispatch_async_mxfp4:
    """Registers the `ep.dispatch_async.mxfp4` graph op with the graph compiler.
    """

    @inline(.always)
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
        *,
        MX_FORMAT: StaticString = "auto",
    ](
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism async dispatch kernel. Tokens are
        transferred in MXFP4 format with per-token even-mode scales packed
        alongside the FP4 quants in the send buffer.

        Parameters:
            dispatch_dtype: `DType` used for the quantized token
                payload during dispatch (inferred).
            dispatch_scale_dtype: `DType` of the per-token even-mode
                scales accompanying the dispatched tokens (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            MX_FORMAT: Name of the MX element format, or `auto` to take it
                from the dtype. See `MXFormat.from_name`. Names the MX
                encoding of the wire payload. It cannot be inferred from
                `dispatch_dtype`: MXFP4 and MXFP6 both travel as
                `DType.uint8` and differ only in bits per element. The
                default `auto` falls back to FP4 for a `uint8` payload and
                FP8 E4M3 otherwise.

        Args:
            atomic_counters: Atomic counters coordinating work across
                thread blocks during the dispatch phase.
            input_tokens: Input tokens to dispatch to experts. Shape
                `[num_tokens, hidden_size]`.
            topk_ids: Top-k expert IDs per token. Shape
                `[num_tokens, top_k]`.
            send_ptrs: Send buffer pointers for the dispatch phase.
            recv_ptrs: Receive buffer pointers for the dispatch
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                tokens received per expert.
            context: GPU device context for the current device.
        """
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(topk_ids).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        comptime mx_fmt = MXFormat.from_dtype[
            dispatch_dtype
        ]() if MX_FORMAT == "auto" else (MXFormat.from_name[MX_FORMAT]())
        comptime token_fmt_type = MXTokenFormat[
            quant_dtype=dispatch_dtype,
            scales_dtype=dispatch_scale_dtype,
            output_layout=RT_LAYOUT_2D,
            scales_layout=RT_LAYOUT_2D,
            hidden_size,
            top_k,
            mx_format=mx_fmt,
        ]
        ep_dispatch_async_kernel_api[
            token_fmt_type,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
        ](
            atomic_counters.to_tile_tensor[.int32](),
            input_tokens.to_tile_tensor().as_imm(),
            topk_ids.to_tile_tensor[.int32]().as_imm(),
            send_ptrs.to_tile_tensor[.uint64](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            context,
        )


@inline(.always)
def _mut_pointer_tile(
    tile: TileTensor[mut=True, address_space=.GENERIC, ...],
) -> TileTensor[tile.dtype, tile.LayoutType, MutUntrackedOrigin]:
    """Rebuilds `tile` as a mutable view over its pointer.

    The EP token formats hold pointer-backed tiles, so a projected argument
    with another storage policy aborts here at run time.
    """
    return TileTensor[tile.dtype, tile.LayoutType, MutUntrackedOrigin](
        ptr=tile.ptr.unsafe_origin_cast[MutUntrackedOrigin](),
        layout=tile.layout,
    )


@inline(.always)
def _expert_load_tally(
    expert_load: Some[MutableInput & Tensor], layer_idx: Some[Input & Tensor]
) -> ExpertLoadTally:
    """Wraps the `.expert_load` ops' tally inputs: a `[layers,
    n_local_experts]` uint64 running total and the int32 `[1]` MoE layer index
    the EPLB router also reads."""
    comptime assert conforms_to(type_of(expert_load), TileTensorable)
    comptime assert conforms_to(type_of(layer_idx), TileTensorable)
    comptime assert type_of(expert_load).rank == 2
    comptime assert type_of(layer_idx).rank == 1
    return ExpertLoadTally(
        MutPointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=Int(expert_load.to_tile_tensor[.uint64]().ptr)
        ),
        ImmPointer[Int32, ImmUntrackedOrigin](
            unsafe_from_address=Int(layer_idx.to_tile_tensor[.int32]().ptr)
        ),
    )


@extensibility.register("ep.dispatch_wait")
struct Struct_ep_dispatch_wait:
    """Registers the `ep.dispatch_wait` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        num_input_tokens: Int = -1,
    ](
        output_tokens: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in BF16 format.
        """
        Struct_ep_dispatch_wait.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            num_input_tokens=num_input_tokens,
        ](
            output_tokens,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            recv_ptrs,
            recv_count_ptrs,
            context,
            NullExpertLoad(),
        )

    @inline(.always)
    @staticmethod
    def run[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        num_input_tokens: Int = -1,
        ExpertLoadT: ExpertLoadSink = NullExpertLoad,
    ](
        output_tokens: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
        expert_load: ExpertLoadT,
    ) raises:
        """Runs `ep.dispatch_wait`, reporting each local expert's routed-token
        count to `expert_load`."""
        comptime assert conforms_to(type_of(output_tokens), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1

        # Ensure the shape for the input tensors are correct
        comptime assert (
            type_of(output_tokens)._shape_types[1].static_value == hidden_size
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = BF16TokenFormat[hidden_size, top_k](
            _mut_pointer_tile(output_tokens.to_tile_tensor[.bfloat16]())
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.uint32](),
            expert_ids.to_tile_tensor[.int32](),
            src_info.to_tile_tensor[.int32](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            atomic_counters.to_tile_tensor[.int32](),
            context,
            num_input_tokens,
            expert_load=expert_load,
        )


@extensibility.register("ep.dispatch_wait.expert_load")
struct Struct_ep_dispatch_wait_expert_load:
    """Registers `ep.dispatch_wait.expert_load`: `ep.dispatch_wait` that also adds each local
    expert's routed-token count to an EPLB load tally.
    """

    @inline(.always)
    @staticmethod
    def execute[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        num_input_tokens: Int = -1,
    ](
        output_tokens: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        expert_load: Some[MutableInput & Tensor],
        layer_idx: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Runs `ep.dispatch_wait` and adds each local expert's routed-token count to
        row `layer_idx[0]` of the `[layers, n_local_experts]` uint64
        `expert_load` tally."""
        Struct_ep_dispatch_wait.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            num_input_tokens=num_input_tokens,
        ](
            output_tokens,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            recv_ptrs,
            recv_count_ptrs,
            context,
            _expert_load_tally(expert_load, layer_idx),
        )


@extensibility.register("ep.dispatch_wait.fp8")
struct Struct_ep_dispatch_wait_fp8:
    """Registers the `ep.dispatch_wait.fp8` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        //,
        num_input_tokens: Int = -1,
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in Blockwise FP8 format.
        """
        Struct_ep_dispatch_wait_fp8.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            dispatch_scale_granularity=dispatch_scale_granularity,
            num_input_tokens=num_input_tokens,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            recv_ptrs,
            recv_count_ptrs,
            context,
            NullExpertLoad(),
        )

    @inline(.always)
    @staticmethod
    def run[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        num_input_tokens: Int = -1,
        ExpertLoadT: ExpertLoadSink = NullExpertLoad,
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
        expert_load: ExpertLoadT,
    ) raises:
        """Runs `ep.dispatch_wait.fp8`, reporting each local expert's routed-token
        count to `expert_load`."""
        comptime assert conforms_to(type_of(output_tokens), TileTensorable)
        comptime assert conforms_to(type_of(output_scales), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(output_scales).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1

        var output_tokens_tensor = output_tokens.to_tile_tensor()
        var output_scales_tensor = output_scales.to_tile_tensor()
        # Ensure the shape for the input tensors are correct
        comptime assert (
            output_tokens_tensor.static_shape[1] == hidden_size
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = BlockwiseFP8TokenFormat[hidden_size, top_k](
            _mut_pointer_tile(output_tokens_tensor),
            _mut_pointer_tile(output_scales_tensor),
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.uint32](),
            expert_ids.to_tile_tensor[.int32](),
            src_info.to_tile_tensor[.int32](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            atomic_counters.to_tile_tensor[.int32](),
            context,
            num_input_tokens,
            expert_load=expert_load,
        )


@extensibility.register("ep.dispatch_wait.fp8.expert_load")
struct Struct_ep_dispatch_wait_fp8_expert_load:
    """Registers `ep.dispatch_wait.fp8.expert_load`: `ep.dispatch_wait.fp8` that also adds each local
    expert's routed-token count to an EPLB load tally.
    """

    @inline(.always)
    @staticmethod
    def execute[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        //,
        num_input_tokens: Int = -1,
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        expert_load: Some[MutableInput & Tensor],
        layer_idx: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Runs `ep.dispatch_wait.fp8` and adds each local expert's routed-token count to
        row `layer_idx[0]` of the `[layers, n_local_experts]` uint64
        `expert_load` tally."""
        Struct_ep_dispatch_wait_fp8.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            dispatch_scale_granularity=dispatch_scale_granularity,
            num_input_tokens=num_input_tokens,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            recv_ptrs,
            recv_count_ptrs,
            context,
            _expert_load_tally(expert_load, layer_idx),
        )


@extensibility.register("ep.dispatch_wait.block.scaled.nv")
struct Struct_ep_dispatch_wait_block_scaled_nv:
    """Registers the `ep.dispatch_wait.block.scaled.nv` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        num_input_tokens: Int = -1,
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        scales_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in NVFP4 format.
        """
        Struct_ep_dispatch_wait_block_scaled_nv.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            num_input_tokens=num_input_tokens,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            scales_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            recv_ptrs,
            recv_count_ptrs,
            context,
            NullExpertLoad(),
        )

    @inline(.always)
    @staticmethod
    def run[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        num_input_tokens: Int = -1,
        ExpertLoadT: ExpertLoadSink = NullExpertLoad,
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        scales_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
        expert_load: ExpertLoadT,
    ) raises:
        """Runs `ep.dispatch_wait.block.scaled.nv`, reporting each local expert's routed-token
        count to `expert_load`."""
        comptime assert conforms_to(type_of(output_tokens), TileTensorable)
        comptime assert conforms_to(type_of(output_scales), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(scales_offsets), TileTensorable)
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(output_scales).rank == 5
        comptime assert type_of(row_offsets).rank == 1
        comptime assert type_of(scales_offsets).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        var output_tokens_tensor = output_tokens.to_tile_tensor()
        var output_scales_tensor = output_scales.to_tile_tensor()
        var scales_offsets_tensor = scales_offsets.to_tile_tensor[.uint32]()

        comptime assert (
            output_tokens_tensor.static_shape[1] * 2 == hidden_size
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = NVBlockScaledTokenFormat[hidden_size, top_k](
            _mut_pointer_tile(output_tokens_tensor),
            _mut_pointer_tile(output_scales_tensor),
            _mut_pointer_tile(scales_offsets_tensor),
            context,
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.uint32](),
            expert_ids.to_tile_tensor[.int32](),
            src_info.to_tile_tensor[.int32](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            atomic_counters.to_tile_tensor[.int32](),
            context,
            num_input_tokens,
            expert_load=expert_load,
        )


@extensibility.register("ep.dispatch_wait.block.scaled.nv.expert_load")
struct Struct_ep_dispatch_wait_block_scaled_nv_expert_load:
    """Registers `ep.dispatch_wait.block.scaled.nv.expert_load`: `ep.dispatch_wait.block.scaled.nv` that also adds each local
    expert's routed-token count to an EPLB load tally.
    """

    @inline(.always)
    @staticmethod
    def execute[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        num_input_tokens: Int = -1,
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        scales_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        expert_load: Some[MutableInput & Tensor],
        layer_idx: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Runs `ep.dispatch_wait.block.scaled.nv` and adds each local expert's routed-token count to
        row `layer_idx[0]` of the `[layers, n_local_experts]` uint64
        `expert_load` tally."""
        Struct_ep_dispatch_wait_block_scaled_nv.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            num_input_tokens=num_input_tokens,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            scales_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            recv_ptrs,
            recv_count_ptrs,
            context,
            _expert_load_tally(expert_load, layer_idx),
        )


@extensibility.register("ep.dispatch_wait.mxfp4")
struct Struct_ep_dispatch_wait_mxfp4:
    """Registers the `ep.dispatch_wait.mxfp4` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        num_input_tokens: Int = -1,
        *,
        MX_FORMAT: StaticString = "auto",
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in MXFP4 format: two FP4 elements packed per ``uint8`` in
        ``output_tokens`` with per-token even-mode scales in ``output_scales``.

        The kernel writes the E8M0 activation scale directly into the up-proj
        grouped matmul's per-expert fixed-stride ``scale_4d`` slot layout (KS224
        up-proj fusion). A slot holds ``max_padded_M`` rows, the most tokens one
        expert can receive rounded up to 32, so the scales output tensor has
        shape ``[n_local_experts * max_padded_M, K_SCALES]``.
        """
        Struct_ep_dispatch_wait_mxfp4.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            num_input_tokens=num_input_tokens,
            MX_FORMAT=MX_FORMAT,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            recv_ptrs,
            recv_count_ptrs,
            context,
            NullExpertLoad(),
        )

    @inline(.always)
    @staticmethod
    def run[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        num_input_tokens: Int = -1,
        *,
        MX_FORMAT: StaticString = "auto",
        ExpertLoadT: ExpertLoadSink = NullExpertLoad,
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
        expert_load: ExpertLoadT,
    ) raises:
        """Runs `ep.dispatch_wait.mxfp4`, reporting each local expert's routed-token
        count to `expert_load`."""
        comptime assert conforms_to(type_of(output_tokens), TileTensorable)
        comptime assert conforms_to(type_of(output_scales), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime dispatch_dtype = type_of(output_tokens).dtype
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(output_scales).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        comptime max_padded_M = align_up(
            max_token_per_rank * n_gpus_per_node * n_nodes, 32
        )
        var output_tokens_tensor = output_tokens.to_tile_tensor()
        var output_scales_tensor = output_scales.to_tile_tensor()

        comptime mx_fmt = MXFormat.from_dtype[
            dispatch_dtype
        ]() if MX_FORMAT == "auto" else (MXFormat.from_name[MX_FORMAT]())
        comptime assert (
            output_tokens_tensor.static_shape[1] * 8
            == hidden_size * mx_fmt.bits_per_element()
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = MXTokenFormat[
            hidden_size,
            top_k,
            mx_format=mx_fmt,
        ](
            _mut_pointer_tile(output_tokens_tensor),
            _mut_pointer_tile(output_scales_tensor),
            max_padded_M,
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.uint32](),
            expert_ids.to_tile_tensor[.int32](),
            src_info.to_tile_tensor[.int32](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            atomic_counters.to_tile_tensor[.int32](),
            context,
            num_input_tokens,
            expert_load=expert_load,
        )


@extensibility.register("ep.dispatch_wait.mxfp4.expert_load")
struct Struct_ep_dispatch_wait_mxfp4_expert_load:
    """Registers `ep.dispatch_wait.mxfp4.expert_load`: `ep.dispatch_wait.mxfp4` that also adds each local
    expert's routed-token count to an EPLB load tally.
    """

    @inline(.always)
    @staticmethod
    def execute[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        num_input_tokens: Int = -1,
        *,
        MX_FORMAT: StaticString = "auto",
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        expert_load: Some[MutableInput & Tensor],
        layer_idx: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Runs `ep.dispatch_wait.mxfp4` and adds each local expert's routed-token count to
        row `layer_idx[0]` of the `[layers, n_local_experts]` uint64
        `expert_load` tally."""
        Struct_ep_dispatch_wait_mxfp4.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            num_input_tokens=num_input_tokens,
            MX_FORMAT=MX_FORMAT,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            recv_ptrs,
            recv_count_ptrs,
            context,
            _expert_load_tally(expert_load, layer_idx),
        )


@extensibility.register("ep.dispatch")
struct Struct_ep_dispatch:
    """Registers the `ep.dispatch` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
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
    ](
        output_tokens: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism dispatch kernel.

        Routes tokens to experts based on top-k IDs, sends them to
        peer devices in BF16 format, waits for incoming tokens, and
        writes them to the output buffer along with their routing
        metadata.

        Parameters:
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens any GPU can
                receive (inferred).
            n_gpus_per_node: Number of GPUs per physical node
                (inferred).
            n_nodes: Number of physical nodes in the deployment
                (inferred).
            fused_shared_expert: Whether a shared expert is fused
                into the dispatch kernel (inferred).
            skip_a2a: Whether to skip the all-to-all communication
                and send tokens only within the current device
                (inferred).
            allreduce_world_size: Number of ranks participating in
                the allreduce following dispatch (inferred).

        Args:
            output_tokens: Output tensor storing the received tokens
                in BF16 format. Shape `[num_tokens, hidden_size]`.
            row_offsets: Output tensor storing the row offsets for
                the received tokens.
            expert_ids: Output tensor storing the expert ID for
                each received token.
            src_info: Output tensor recording the originating rank
                and token index for each received token. Shape
                `[num_tokens, 2]`.
            atomic_counters: Atomic counters coordinating work
                across thread blocks during the dispatch phase.
            input_tokens: Input tokens to dispatch to experts. Shape
                `[num_tokens, hidden_size]`.
            topk_ids: Input tensor of top-k expert IDs per token.
                Shape `[num_tokens, top_k]`.
            send_ptrs: Send buffer pointers for the dispatch phase.
            recv_ptrs: Receive buffer pointers for the dispatch
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                tokens received per expert.
            context: GPU device context for the current device.
        """
        Struct_ep_dispatch.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            output_tokens,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            context,
            NullExpertLoad(),
        )

    @inline(.always)
    @staticmethod
    def run[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        ExpertLoadT: ExpertLoadSink = NullExpertLoad,
    ](
        output_tokens: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
        expert_load: ExpertLoadT,
    ) raises:
        """Runs `ep.dispatch`, reporting each local expert's routed-token
        count to `expert_load`."""
        comptime assert conforms_to(type_of(output_tokens), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime dispatch_dtype = type_of(input_tokens).dtype
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(topk_ids).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1

        comptime assert dispatch_dtype == .bfloat16

        var output_tokens_tensor = output_tokens.to_tile_tensor[.bfloat16]()
        var format_handler = BF16TokenFormat[hidden_size, top_k](
            output_tokens_tensor
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.uint32](),
            expert_ids.to_tile_tensor[.int32](),
            src_info.to_tile_tensor[.int32](),
            atomic_counters.to_tile_tensor[.int32](),
            input_tokens.to_tile_tensor(),
            topk_ids.to_tile_tensor[.int32](),
            send_ptrs.to_tile_tensor[.uint64](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            context,
            expert_load=expert_load,
        )


@extensibility.register("ep.dispatch.expert_load")
struct Struct_ep_dispatch_expert_load:
    """Registers `ep.dispatch.expert_load`: `ep.dispatch` that also adds each local
    expert's routed-token count to an EPLB load tally.
    """

    @inline(.always)
    @staticmethod
    def execute[
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
    ](
        output_tokens: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        expert_load: Some[MutableInput & Tensor],
        layer_idx: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Runs `ep.dispatch` and adds each local expert's routed-token count to
        row `layer_idx[0]` of the `[layers, n_local_experts]` uint64
        `expert_load` tally."""
        Struct_ep_dispatch.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            output_tokens,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            context,
            _expert_load_tally(expert_load, layer_idx),
        )


@extensibility.register("ep.dispatch.fp8")
struct Struct_ep_dispatch_fp8:
    """Registers the `ep.dispatch.fp8` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
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
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism FP8 dispatch kernel. Tokens are
        dispatched in Blockwise FP8 format.
        """
        Struct_ep_dispatch_fp8.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            dispatch_scale_granularity=dispatch_scale_granularity,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            context,
            NullExpertLoad(),
        )

    @inline(.always)
    @staticmethod
    def run[
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
        ExpertLoadT: ExpertLoadSink = NullExpertLoad,
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
        expert_load: ExpertLoadT,
    ) raises:
        """Runs `ep.dispatch.fp8`, reporting each local expert's routed-token
        count to `expert_load`."""
        comptime assert conforms_to(type_of(output_tokens), TileTensorable)
        comptime assert conforms_to(type_of(output_scales), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(output_scales).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(topk_ids).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        var output_tokens_tensor = output_tokens.to_tile_tensor()
        var output_scales_tensor = output_scales.to_tile_tensor()

        var format_handler = BlockwiseFP8TokenFormat[hidden_size, top_k](
            _mut_pointer_tile(output_tokens_tensor),
            _mut_pointer_tile(output_scales_tensor),
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.uint32](),
            expert_ids.to_tile_tensor[.int32](),
            src_info.to_tile_tensor[.int32](),
            atomic_counters.to_tile_tensor[.int32](),
            input_tokens.to_tile_tensor(),
            topk_ids.to_tile_tensor[.int32](),
            send_ptrs.to_tile_tensor[.uint64](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            context,
            expert_load=expert_load,
        )


@extensibility.register("ep.dispatch.fp8.expert_load")
struct Struct_ep_dispatch_fp8_expert_load:
    """Registers `ep.dispatch.fp8.expert_load`: `ep.dispatch.fp8` that also adds each local
    expert's routed-token count to an EPLB load tally.
    """

    @inline(.always)
    @staticmethod
    def execute[
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
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        expert_load: Some[MutableInput & Tensor],
        layer_idx: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Runs `ep.dispatch.fp8` and adds each local expert's routed-token count to
        row `layer_idx[0]` of the `[layers, n_local_experts]` uint64
        `expert_load` tally."""
        Struct_ep_dispatch_fp8.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            dispatch_scale_granularity=dispatch_scale_granularity,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            context,
            _expert_load_tally(expert_load, layer_idx),
        )


@extensibility.register("ep.dispatch.block.scaled.nv")
struct Struct_ep_dispatch_block_scaled_nv:
    """Registers the `ep.dispatch.block.scaled.nv` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
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
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        scales_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        input_scales: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism NVFP4 dispatch kernel. Tokens
        are dispatched in NVFP4 format.

        Parameters:
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            fused_shared_expert: Whether a shared expert is fused
                into the dispatch kernel (inferred).
            skip_a2a: Whether to skip the all-to-all communication
                and send tokens only within the current device
                (inferred).
            allreduce_world_size: Number of ranks participating in
                the allreduce following dispatch (inferred).

        Args:
            output_tokens: Output tensor storing the received tokens
                in NVFP4 format.
            output_scales: Output tensor storing the NVFP4 block
                scales for the received tokens.
            row_offsets: Output tensor storing the row offsets for
                the received tokens.
            scales_offsets: Output tensor storing the offsets into
                the scales buffer for the received tokens.
            expert_ids: Output tensor storing the expert ID for
                each received token.
            src_info: Output tensor recording the originating rank
                and token index for each received token. Shape
                `[num_tokens, 2]`.
            atomic_counters: Atomic counters coordinating work
                across thread blocks during the dispatch phase.
            input_tokens: Input tokens to dispatch to experts. Shape
                `[num_tokens, hidden_size]`.
            topk_ids: Input tensor of top-k expert IDs per token.
                Shape `[num_tokens, top_k]`.
            send_ptrs: Send buffer pointers for the dispatch phase.
            recv_ptrs: Receive buffer pointers for the dispatch
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                tokens received per expert.
            input_scales: Global input scales for NVFP4 quantization.
            context: GPU device context for the current device.
        """
        Struct_ep_dispatch_block_scaled_nv.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            scales_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            input_scales,
            context,
            NullExpertLoad(),
        )

    @inline(.always)
    @staticmethod
    def run[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        ExpertLoadT: ExpertLoadSink = NullExpertLoad,
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        scales_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        input_scales: Some[Input & Tensor],
        context: DeviceContext,
        expert_load: ExpertLoadT,
    ) raises:
        """Runs `ep.dispatch.block.scaled.nv`, reporting each local expert's routed-token
        count to `expert_load`."""
        comptime assert conforms_to(type_of(output_tokens), TileTensorable)
        comptime assert conforms_to(type_of(output_scales), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(scales_offsets), TileTensorable)
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(input_scales), TileTensorable)
        comptime input_dtype = type_of(input_tokens).dtype
        comptime dispatch_dtype = type_of(output_tokens).dtype
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(output_scales).rank == 5
        comptime assert type_of(row_offsets).rank == 1
        comptime assert type_of(scales_offsets).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(topk_ids).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        comptime assert type_of(input_scales).rank == 1
        var output_tokens_tensor = output_tokens.to_tile_tensor()
        var output_scales_tensor = output_scales.to_tile_tensor()
        var scales_offsets_tensor = scales_offsets.to_tile_tensor[.uint32]()
        var input_scales_tensor = input_scales.to_tile_tensor[.float32]()
        comptime assert input_scales_tensor.flat_rank == 1

        @inline(.always)
        def input_scales_fn[
            dtype: DType
        ](expert_id: Int) {var input_scales_tensor} -> Scalar[dtype]:
            # Currently only use one global input scale for all experts
            return input_scales_tensor.load[width=1](Coord(0)).cast[dtype]()

        var format_handler = NVBlockScaledTokenFormat[hidden_size, top_k](
            _mut_pointer_tile(output_tokens_tensor),
            _mut_pointer_tile(output_scales_tensor),
            _mut_pointer_tile(scales_offsets_tensor),
            context,
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.uint32](),
            expert_ids.to_tile_tensor[.int32](),
            src_info.to_tile_tensor[.int32](),
            atomic_counters.to_tile_tensor[.int32](),
            input_tokens.to_tile_tensor(),
            topk_ids.to_tile_tensor[.int32](),
            send_ptrs.to_tile_tensor[.uint64](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            context,
            input_scales_fn,
            expert_load=expert_load,
        )


@extensibility.register("ep.dispatch.block.scaled.nv.expert_load")
struct Struct_ep_dispatch_block_scaled_nv_expert_load:
    """Registers `ep.dispatch.block.scaled.nv.expert_load`: `ep.dispatch.block.scaled.nv` that also adds each local
    expert's routed-token count to an EPLB load tally.
    """

    @inline(.always)
    @staticmethod
    def execute[
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
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        scales_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        input_scales: Some[Input & Tensor],
        expert_load: Some[MutableInput & Tensor],
        layer_idx: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Runs `ep.dispatch.block.scaled.nv` and adds each local expert's routed-token count to
        row `layer_idx[0]` of the `[layers, n_local_experts]` uint64
        `expert_load` tally."""
        Struct_ep_dispatch_block_scaled_nv.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            scales_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            input_scales,
            context,
            _expert_load_tally(expert_load, layer_idx),
        )


def _check_rowwise_scales_rows(
    output_rowwise_scales: TileTensor[DType.bfloat16, ...],
    output_tokens: TileTensor[...],
) raises:
    """Rejects a per-row scales output that cannot hold one scale per
    received row, since the token format sizes it from `output_tokens`."""
    comptime assert output_rowwise_scales.rank == 1
    comptime assert output_tokens.rank == 2
    var scale_rows = Int(output_rowwise_scales.dim[0]())
    var token_rows = Int(output_tokens.dim[0]())
    if scale_rows != token_rows:
        raise Error(
            "EP dispatch: output_rowwise_scales has ",
            scale_rows,
            " rows, but output_tokens has ",
            token_rows,
        )


@extensibility.register("ep.dispatch.block.scaled.nv.dyn_global_scales")
struct Struct_ep_dispatch_block_scaled_nv_dyn_global_scales:
    """Registers the `ep.dispatch.block.scaled.nv.dyn_global_scales` graph op
    with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
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
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        output_rowwise_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        scales_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism NVFP4 dispatch kernel with
        dynamic global scales.

        Each token is quantized against its own global scale,
        `2688 / rowmax`, rather than a caller-supplied input scale. The
        scale's BF16 inverse travels with the token and lands in
        `output_rowwise_scales`, so a received row dequantizes as
        `fp4 * block_scale * rowwise_scale`.

        Parameters:
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            fused_shared_expert: Whether a shared expert is fused
                into the dispatch kernel (inferred).
            skip_a2a: Whether to skip the all-to-all communication
                and send tokens only within the current device
                (inferred).
            allreduce_world_size: Number of ranks participating in
                the allreduce following dispatch (inferred).

        Args:
            output_tokens: Output tensor storing the received tokens
                in NVFP4 format.
            output_scales: Output tensor storing the NVFP4 block
                scales for the received tokens.
            output_rowwise_scales: Output tensor storing each received
                token's BF16 inverse global scale, `rowmax / 2688`. Shape
                `[max_recv_tokens]`, one entry per row of `output_tokens`.
            row_offsets: Output tensor storing the row offsets for
                the received tokens.
            scales_offsets: Output tensor storing the offsets into
                the scales buffer for the received tokens.
            expert_ids: Output tensor storing the expert ID for
                each received token.
            src_info: Output tensor recording the originating rank
                and token index for each received token. Shape
                `[num_tokens, 2]`.
            atomic_counters: Atomic counters coordinating work
                across thread blocks during the dispatch phase.
            input_tokens: Input tokens to dispatch to experts. Shape
                `[num_tokens, hidden_size]`.
            topk_ids: Input tensor of top-k expert IDs per token.
                Shape `[num_tokens, top_k]`.
            send_ptrs: Send buffer pointers for the dispatch phase.
            recv_ptrs: Receive buffer pointers for the dispatch
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                tokens received per expert.
            context: GPU device context for the current device.
        """
        Struct_ep_dispatch_block_scaled_nv_dyn_global_scales.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            output_tokens,
            output_scales,
            output_rowwise_scales,
            row_offsets,
            scales_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            context,
            NullExpertLoad(),
        )

    @inline(.always)
    @staticmethod
    def run[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        ExpertLoadT: ExpertLoadSink = NullExpertLoad,
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        output_rowwise_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        scales_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
        expert_load: ExpertLoadT,
    ) raises:
        """Runs `ep.dispatch.block.scaled.nv.dyn_global_scales`, reporting each local expert's routed-token
        count to `expert_load`."""
        comptime assert conforms_to(type_of(output_tokens), TileTensorable)
        comptime assert conforms_to(type_of(output_scales), TileTensorable)
        comptime assert conforms_to(
            type_of(output_rowwise_scales), TileTensorable
        )
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(scales_offsets), TileTensorable)
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime input_dtype = type_of(input_tokens).dtype
        comptime dispatch_dtype = type_of(output_tokens).dtype
        comptime dispatch_scale_dtype = type_of(output_scales).dtype
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(output_scales).rank == 5
        comptime assert type_of(output_rowwise_scales).rank == 1
        comptime assert type_of(row_offsets).rank == 1
        comptime assert type_of(scales_offsets).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(topk_ids).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        _check_rowwise_scales_rows(
            output_rowwise_scales.to_tile_tensor[.bfloat16](),
            output_tokens.to_tile_tensor(),
        )

        var format_handler = NVBlockScaledTokenFormat[
            hidden_size, top_k, nvfp4_dyn_global_scales=True
        ](
            _mut_pointer_tile(output_tokens.to_tile_tensor()),
            _mut_pointer_tile(output_scales.to_tile_tensor()),
            _mut_pointer_tile(scales_offsets.to_tile_tensor[.uint32]()),
            context,
            # The format reads the row scales through their flat pointer; a
            # non-pointer storage policy aborts here at run time.
            output_rowwise_scales.to_tile_tensor[.bfloat16]()
            .as_unsafe_any_origin()
            .ptr,
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.uint32](),
            expert_ids.to_tile_tensor[.int32](),
            src_info.to_tile_tensor[.int32](),
            atomic_counters.to_tile_tensor[.int32](),
            input_tokens.to_tile_tensor(),
            topk_ids.to_tile_tensor[.int32](),
            send_ptrs.to_tile_tensor[.uint64](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            context,
            expert_load=expert_load,
        )


@extensibility.register(
    "ep.dispatch.block.scaled.nv.dyn_global_scales.expert_load"
)
struct Struct_ep_dispatch_block_scaled_nv_dyn_global_scales_expert_load:
    """Registers `ep.dispatch.block.scaled.nv.dyn_global_scales.expert_load`: `ep.dispatch.block.scaled.nv.dyn_global_scales` that also adds each local
    expert's routed-token count to an EPLB load tally.
    """

    @inline(.always)
    @staticmethod
    def execute[
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
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        output_rowwise_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        scales_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        expert_load: Some[MutableInput & Tensor],
        layer_idx: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Runs `ep.dispatch.block.scaled.nv.dyn_global_scales` and adds each local expert's routed-token count to
        row `layer_idx[0]` of the `[layers, n_local_experts]` uint64
        `expert_load` tally."""
        Struct_ep_dispatch_block_scaled_nv_dyn_global_scales.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            output_tokens,
            output_scales,
            output_rowwise_scales,
            row_offsets,
            scales_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            context,
            _expert_load_tally(expert_load, layer_idx),
        )


@extensibility.register("ep.dispatch.mxfp4")
struct Struct_ep_dispatch_mxfp4:
    """Registers the `ep.dispatch.mxfp4` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
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
        *,
        MX_FORMAT: StaticString = "auto",
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism MXFP4 dispatch kernel. Tokens
        are dispatched in MXFP4 format.

        The wait-side copy writes the E8M0 activation scale directly into the
        up-proj grouped matmul's per-expert fixed-stride ``scale_4d`` slot
        layout (KS224 up-proj fusion). A slot holds ``max_padded_M`` rows, the
        most tokens one expert can receive rounded up to 32, so the scales
        output tensor has shape ``[n_local_experts * max_padded_M, K_SCALES]``.
        """
        Struct_ep_dispatch_mxfp4.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
            MX_FORMAT=MX_FORMAT,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            context,
            NullExpertLoad(),
        )

    @inline(.always)
    @staticmethod
    def run[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        *,
        MX_FORMAT: StaticString = "auto",
        ExpertLoadT: ExpertLoadSink = NullExpertLoad,
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
        expert_load: ExpertLoadT,
    ) raises:
        """Runs `ep.dispatch.mxfp4`, reporting each local expert's routed-token
        count to `expert_load`."""
        comptime assert conforms_to(type_of(output_tokens), TileTensorable)
        comptime assert conforms_to(type_of(output_scales), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime dispatch_dtype = type_of(output_tokens).dtype
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(output_scales).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(topk_ids).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        comptime max_padded_M = align_up(
            max_token_per_rank * n_gpus_per_node * n_nodes, 32
        )
        var output_tokens_tensor = output_tokens.to_tile_tensor()
        var output_scales_tensor = output_scales.to_tile_tensor()

        comptime mx_fmt = MXFormat.from_dtype[
            dispatch_dtype
        ]() if MX_FORMAT == "auto" else (MXFormat.from_name[MX_FORMAT]())
        var format_handler = MXTokenFormat[
            hidden_size,
            top_k,
            mx_format=mx_fmt,
        ](
            _mut_pointer_tile(output_tokens_tensor),
            _mut_pointer_tile(output_scales_tensor),
            max_padded_M,
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.uint32](),
            expert_ids.to_tile_tensor[.int32](),
            src_info.to_tile_tensor[.int32](),
            atomic_counters.to_tile_tensor[.int32](),
            input_tokens.to_tile_tensor(),
            topk_ids.to_tile_tensor[.int32](),
            send_ptrs.to_tile_tensor[.uint64](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            context,
            expert_load=expert_load,
        )


@extensibility.register("ep.dispatch.mxfp4.expert_load")
struct Struct_ep_dispatch_mxfp4_expert_load:
    """Registers `ep.dispatch.mxfp4.expert_load`: `ep.dispatch.mxfp4` that also adds each local
    expert's routed-token count to an EPLB load tally.
    """

    @inline(.always)
    @staticmethod
    def execute[
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
        *,
        MX_FORMAT: StaticString = "auto",
    ](
        output_tokens: Some[Output & Tensor],
        output_scales: Some[Output & Tensor],
        row_offsets: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        src_info: Some[Output & Tensor],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        expert_load: Some[MutableInput & Tensor],
        layer_idx: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Runs `ep.dispatch.mxfp4` and adds each local expert's routed-token count to
        row `layer_idx[0]` of the `[layers, n_local_experts]` uint64
        `expert_load` tally."""
        Struct_ep_dispatch_mxfp4.run[
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_token_per_rank=max_token_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=n_nodes,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
            MX_FORMAT=MX_FORMAT,
        ](
            output_tokens,
            output_scales,
            row_offsets,
            expert_ids,
            src_info,
            atomic_counters,
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            context,
            _expert_load_tally(expert_load, layer_idx),
        )


@extensibility.register("mo.distributed.ep.dispatch.block.scaled.nv")
struct DistributedEPDispatchBlockScaledNV:
    """Registers the `mo.distributed.ep.dispatch.block.scaled.nv` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        OutTokens: TypeList[Trait=Output & Tensor, ...],
        OutScales: TypeList[Trait=Output & Tensor, ...],
        RowOffsets: TypeList[Trait=Output & Tensor, ...],
        ScalesOffsets: TypeList[Trait=Output & Tensor, ...],
        ExpertIds: TypeList[Trait=Output & Tensor, ...],
        SrcInfos: TypeList[Trait=Output & Tensor, ...],
        InTokens: TypeList[Trait=Input & Tensor, ...],
        TopkIds: TypeList[Trait=Input & Tensor, ...],
        SendPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvCountPtrs: TypeList[Trait=Input & Tensor, ...],
        InScales: TypeList[Trait=Input & Tensor, ...],
        Counters: TypeList[Trait=MutableInput & Tensor, ...],
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        //,
        _trace_name: StaticString,
    ](
        output_tokens: TensorTuple[*OutTokens],
        output_scales: TensorTuple[*OutScales],
        row_offsets: TensorTuple[*RowOffsets],
        scales_offsets: TensorTuple[*ScalesOffsets],
        expert_ids: TensorTuple[*ExpertIds],
        src_info: TensorTuple[*SrcInfos],
        input_tokens: TensorTuple[*InTokens],
        topk_ids: TensorTuple[*TopkIds],
        send_ptrs: TensorTuple[*SendPtrs],
        recv_ptrs: TensorTuple[*RecvPtrs],
        recv_count_ptrs: TensorTuple[*RecvCountPtrs],
        input_scales: TensorTuple[*InScales],
        atomic_counters: TensorTuple[*Counters],
        dev_ctxs: DeviceContextArray[_, _],
    ) raises:
        """Multi-device fused Expert Parallelism NVFP4 dispatch.

        Launches the EP dispatch kernel on all devices simultaneously via
        _ep_launch_device_collective. Each device routes its tokens to experts
        based on top-k IDs, quantizes them to NVFP4 format, and sends them
        to the appropriate peer devices.

        Parameters:
            OutTokens: The element types of `output_tokens` (inferred).
            OutScales: The element types of `output_scales` (inferred).
            RowOffsets: The element types of `row_offsets` (inferred).
            ScalesOffsets: The element types of `scales_offsets` (inferred).
            ExpertIds: The element types of `expert_ids` (inferred).
            SrcInfos: The element types of `src_info` (inferred).
            InTokens: The element types of `input_tokens` (inferred).
            TopkIds: The element types of `topk_ids` (inferred).
            SendPtrs: The element types of `send_ptrs` (inferred).
            RecvPtrs: The element types of `recv_ptrs` (inferred).
            RecvCountPtrs: The element types of `recv_count_ptrs` (inferred).
            InScales: The element types of `input_scales` (inferred).
            Counters: The element types of `atomic_counters` (inferred).
            hidden_size: Size of the model's hidden dimension (inferred).
            top_k: Number of experts each token is routed to (inferred).
            n_experts: Total number of experts across all GPUs (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            fused_shared_expert: Whether a shared expert is fused into the
                dispatch kernel (inferred).
            _trace_name: Trace label for this op.

        Args:
            output_tokens: Output variadic tensors storing the dispatched
                NVFP4-quantized tokens, one per device.
            output_scales: Output variadic tensors storing the NVFP4 block
                scales, one per device.
            row_offsets: Output variadic tensors storing the row offsets for
                the received tokens, one per device.
            scales_offsets: Output variadic tensors storing the offsets into
                the scales buffer, one per device.
            expert_ids: Output variadic tensors storing the expert ID for
                each received token, one per device.
            src_info: Output variadic tensors recording the originating
                rank and token index for each received token, one per
                device.
            input_tokens: Input variadic tensors of tokens to dispatch,
                one per device.
            topk_ids: Input variadic tensors of top-k expert IDs per token,
                one per device.
            send_ptrs: Send buffer pointers for the dispatch phase, one
                per device.
            recv_ptrs: Receive buffer pointers for the dispatch phase,
                one per device.
            recv_count_ptrs: Receive count buffer pointers tracking tokens
                received per expert, one per device.
            input_scales: Input variadic tensors of global input scales
                for NVFP4 quantization, one per device.
            atomic_counters: Atomic counters coordinating work across
                thread blocks, one per device.
            dev_ctxs: List of GPU device contexts, one per device.
        """
        comptime num_devices = len(InTokens)
        _check_tensor_tuple[num_devices, OutTokens[0].dtype, 2, *OutTokens]()
        _check_tensor_tuple[num_devices, OutScales[0].dtype, 5, *OutScales]()
        _check_tensor_tuple[num_devices, .uint32, 1, *RowOffsets]()
        _check_tensor_tuple[num_devices, .uint32, 1, *ScalesOffsets]()
        _check_tensor_tuple[num_devices, .int32, 1, *ExpertIds]()
        _check_tensor_tuple[num_devices, .int32, 2, *SrcInfos]()
        _check_tensor_tuple[num_devices, InTokens[0].dtype, 2, *InTokens]()
        _check_tensor_tuple[num_devices, .int32, 2, *TopkIds]()
        _check_tensor_tuple[num_devices, .uint64, 1, *SendPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvCountPtrs]()
        _check_tensor_tuple[num_devices, .float32, 1, *InScales]()
        _check_tensor_tuple[num_devices, .int32, 1, *Counters]()

        var gpu_ctxs = dev_ctxs.filter_gpu_contexts[num_devices]()

        @inline(.always)
        def launch_dispatch[
            index: Int
        ]() raises {
            imm output_tokens,
            imm output_scales,
            imm row_offsets,
            imm scales_offsets,
            imm expert_ids,
            imm src_info,
            imm atomic_counters,
            imm input_tokens,
            imm topk_ids,
            imm send_ptrs,
            imm recv_ptrs,
            imm recv_count_ptrs,
            imm input_scales,
            imm gpu_ctxs,
        }:
            comptime assert conforms_to(
                type_of(output_tokens[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(output_scales[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(row_offsets[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(scales_offsets[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(expert_ids[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(src_info[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(atomic_counters[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(input_tokens[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(topk_ids[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(send_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_count_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(input_scales[index]), TileTensorable
            )
            var out_tokens = output_tokens[index].to_tile_tensor()
            var out_scales = output_scales[index].to_tile_tensor()
            var sc_offsets = scales_offsets[index].to_tile_tensor[.uint32]()
            var in_scales = input_scales[index].to_tile_tensor[.float32]()
            comptime assert in_scales.flat_rank == 1

            @inline(.always)
            def input_scales_fn[
                dtype: DType
            ](expert_id: Int) {var in_scales} -> Scalar[dtype]:
                return in_scales.load[width=1](Coord(0)).cast[dtype]()

            var format_handler = NVBlockScaledTokenFormat[hidden_size, top_k](
                _mut_pointer_tile(out_tokens),
                _mut_pointer_tile(out_scales),
                _mut_pointer_tile(sc_offsets),
                gpu_ctxs[index],
            )

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
                # A prefill chunk's dispatch waited on the monitor's serial
                # per-expert scan and sent one token per CTA at a time; the
                # histogram count and the warp-per-token send keep the same
                # bytes and receive layout (KERN-3829). Only single-node NVFP4
                # without a fused shared expert takes them.
                dispatch_warp_min_tokens=EP_WARP_TOKENS_MIN_TOKENS,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[.uint32](),
                expert_ids[index].to_tile_tensor[.int32](),
                src_info[index].to_tile_tensor[.int32](),
                atomic_counters[index].to_tile_tensor[.int32](),
                input_tokens[index].to_tile_tensor(),
                topk_ids[index].to_tile_tensor[.int32](),
                send_ptrs[index].to_tile_tensor[.uint64](),
                recv_ptrs[index].to_tile_tensor[.uint64](),
                recv_count_ptrs[index].to_tile_tensor[.uint64](),
                gpu_ctxs[index],
                input_scales_fn,
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs.copy())


@extensibility.register(
    "mo.distributed.ep.dispatch.block.scaled.nv.dyn_global_scales"
)
struct DistributedEPDispatchBlockScaledNVDynGlobalScales:
    """Registers the
    `mo.distributed.ep.dispatch.block.scaled.nv.dyn_global_scales` graph op
    with the graph compiler.
    """

    @staticmethod
    def execute[
        OutTokens: TypeList[Trait=Output & Tensor, ...],
        OutScales: TypeList[Trait=Output & Tensor, ...],
        OutRowwiseScales: TypeList[Trait=Output & Tensor, ...],
        RowOffsets: TypeList[Trait=Output & Tensor, ...],
        ScalesOffsets: TypeList[Trait=Output & Tensor, ...],
        ExpertIds: TypeList[Trait=Output & Tensor, ...],
        SrcInfos: TypeList[Trait=Output & Tensor, ...],
        InTokens: TypeList[Trait=Input & Tensor, ...],
        TopkIds: TypeList[Trait=Input & Tensor, ...],
        SendPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvCountPtrs: TypeList[Trait=Input & Tensor, ...],
        Counters: TypeList[Trait=MutableInput & Tensor, ...],
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        //,
        _trace_name: StaticString,
    ](
        output_tokens: TensorTuple[*OutTokens],
        output_scales: TensorTuple[*OutScales],
        output_rowwise_scales: TensorTuple[*OutRowwiseScales],
        row_offsets: TensorTuple[*RowOffsets],
        scales_offsets: TensorTuple[*ScalesOffsets],
        expert_ids: TensorTuple[*ExpertIds],
        src_info: TensorTuple[*SrcInfos],
        input_tokens: TensorTuple[*InTokens],
        topk_ids: TensorTuple[*TopkIds],
        send_ptrs: TensorTuple[*SendPtrs],
        recv_ptrs: TensorTuple[*RecvPtrs],
        recv_count_ptrs: TensorTuple[*RecvCountPtrs],
        atomic_counters: TensorTuple[*Counters],
        dev_ctxs: DeviceContextArray[_, _],
    ) raises:
        """Multi-device fused Expert Parallelism NVFP4 dispatch with dynamic
        global scales.

        Launches the EP dispatch kernel on all devices simultaneously via
        _ep_launch_device_collective. Each device routes its tokens to experts
        based on top-k IDs, quantizes each token to NVFP4 against its own
        global scale, `2688 / rowmax`, and sends it to the appropriate peer
        devices. The receivers store the scales' BF16 inverses in
        `output_rowwise_scales`.

        Parameters:
            OutTokens: The element types of `output_tokens` (inferred).
            OutScales: The element types of `output_scales` (inferred).
            OutRowwiseScales: The element types of `output_rowwise_scales`
                (inferred).
            RowOffsets: The element types of `row_offsets` (inferred).
            ScalesOffsets: The element types of `scales_offsets` (inferred).
            ExpertIds: The element types of `expert_ids` (inferred).
            SrcInfos: The element types of `src_info` (inferred).
            InTokens: The element types of `input_tokens` (inferred).
            TopkIds: The element types of `topk_ids` (inferred).
            SendPtrs: The element types of `send_ptrs` (inferred).
            RecvPtrs: The element types of `recv_ptrs` (inferred).
            RecvCountPtrs: The element types of `recv_count_ptrs` (inferred).
            Counters: The element types of `atomic_counters` (inferred).
            hidden_size: Size of the model's hidden dimension (inferred).
            top_k: Number of experts each token is routed to (inferred).
            n_experts: Total number of experts across all GPUs (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            fused_shared_expert: Whether a shared expert is fused into the
                dispatch kernel (inferred).
            _trace_name: Trace label for this op.

        Args:
            output_tokens: Output variadic tensors storing the dispatched
                NVFP4-quantized tokens, one per device.
            output_scales: Output variadic tensors storing the NVFP4 block
                scales, one per device.
            output_rowwise_scales: Output variadic tensors storing each
                received token's BF16 inverse global scale, one per device.
            row_offsets: Output variadic tensors storing the row offsets for
                the received tokens, one per device.
            scales_offsets: Output variadic tensors storing the offsets into
                the scales buffer, one per device.
            expert_ids: Output variadic tensors storing the expert ID for
                each received token, one per device.
            src_info: Output variadic tensors recording the originating
                rank and token index for each received token, one per
                device.
            input_tokens: Input variadic tensors of tokens to dispatch,
                one per device.
            topk_ids: Input variadic tensors of top-k expert IDs per token,
                one per device.
            send_ptrs: Send buffer pointers for the dispatch phase, one
                per device.
            recv_ptrs: Receive buffer pointers for the dispatch phase,
                one per device.
            recv_count_ptrs: Receive count buffer pointers tracking tokens
                received per expert, one per device.
            atomic_counters: Atomic counters coordinating work across
                thread blocks, one per device.
            dev_ctxs: List of GPU device contexts, one per device.
        """
        comptime num_devices = len(InTokens)
        _check_tensor_tuple[num_devices, OutTokens[0].dtype, 2, *OutTokens]()
        _check_tensor_tuple[num_devices, OutScales[0].dtype, 5, *OutScales]()
        _check_tensor_tuple[num_devices, .bfloat16, 1, *OutRowwiseScales]()
        _check_tensor_tuple[num_devices, .uint32, 1, *RowOffsets]()
        _check_tensor_tuple[num_devices, .uint32, 1, *ScalesOffsets]()
        _check_tensor_tuple[num_devices, .int32, 1, *ExpertIds]()
        _check_tensor_tuple[num_devices, .int32, 2, *SrcInfos]()
        _check_tensor_tuple[num_devices, InTokens[0].dtype, 2, *InTokens]()
        _check_tensor_tuple[num_devices, .int32, 2, *TopkIds]()
        _check_tensor_tuple[num_devices, .uint64, 1, *SendPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvCountPtrs]()
        _check_tensor_tuple[num_devices, .int32, 1, *Counters]()

        var gpu_ctxs = dev_ctxs.filter_gpu_contexts[num_devices]()
        comptime for i in range(num_devices):
            comptime assert conforms_to(
                type_of(output_rowwise_scales[i]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(output_tokens[i]), TileTensorable
            )
            _check_rowwise_scales_rows(
                output_rowwise_scales[i].to_tile_tensor[.bfloat16](),
                output_tokens[i].to_tile_tensor(),
            )

        @inline(.always)
        def launch_dispatch[
            index: Int
        ]() raises {
            imm output_tokens,
            imm output_scales,
            imm output_rowwise_scales,
            imm row_offsets,
            imm scales_offsets,
            imm expert_ids,
            imm src_info,
            imm atomic_counters,
            imm input_tokens,
            imm topk_ids,
            imm send_ptrs,
            imm recv_ptrs,
            imm recv_count_ptrs,
            imm gpu_ctxs,
        }:
            comptime assert conforms_to(
                type_of(output_tokens[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(output_scales[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(output_rowwise_scales[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(row_offsets[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(scales_offsets[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(expert_ids[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(src_info[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(atomic_counters[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(input_tokens[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(topk_ids[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(send_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_count_ptrs[index]), TileTensorable
            )
            var format_handler = NVBlockScaledTokenFormat[
                hidden_size, top_k, nvfp4_dyn_global_scales=True
            ](
                _mut_pointer_tile(output_tokens[index].to_tile_tensor()),
                _mut_pointer_tile(output_scales[index].to_tile_tensor()),
                _mut_pointer_tile(
                    scales_offsets[index].to_tile_tensor[.uint32]()
                ),
                gpu_ctxs[index],
                # The format reads the row scales through their flat pointer; a
                # non-pointer storage policy aborts here at run time.
                output_rowwise_scales[index]
                .to_tile_tensor[.bfloat16]()
                .as_unsafe_any_origin()
                .ptr,
            )

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[.uint32](),
                expert_ids[index].to_tile_tensor[.int32](),
                src_info[index].to_tile_tensor[.int32](),
                atomic_counters[index].to_tile_tensor[.int32](),
                input_tokens[index].to_tile_tensor(),
                topk_ids[index].to_tile_tensor[.int32](),
                send_ptrs[index].to_tile_tensor[.uint64](),
                recv_ptrs[index].to_tile_tensor[.uint64](),
                recv_count_ptrs[index].to_tile_tensor[.uint64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs.copy())


@extensibility.register("mo.distributed.ep.dispatch.mxfp4")
struct DistributedEPDispatchMXFP4:
    """Registers the `mo.distributed.ep.dispatch.mxfp4` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        OutTokens: TypeList[Trait=Output & Tensor, ...],
        OutScales: TypeList[Trait=Output & Tensor, ...],
        RowOffsets: TypeList[Trait=Output & Tensor, ...],
        ExpertIds: TypeList[Trait=Output & Tensor, ...],
        SrcInfos: TypeList[Trait=Output & Tensor, ...],
        InTokens: TypeList[Trait=Input & Tensor, ...],
        TopkIds: TypeList[Trait=Input & Tensor, ...],
        SendPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvCountPtrs: TypeList[Trait=Input & Tensor, ...],
        Counters: TypeList[Trait=MutableInput & Tensor, ...],
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        //,
        _trace_name: StaticString,
        *,
        mx_format: StaticString = "auto",
    ](
        output_tokens: TensorTuple[*OutTokens],
        output_scales: TensorTuple[*OutScales],
        row_offsets: TensorTuple[*RowOffsets],
        expert_ids: TensorTuple[*ExpertIds],
        src_info: TensorTuple[*SrcInfos],
        input_tokens: TensorTuple[*InTokens],
        topk_ids: TensorTuple[*TopkIds],
        send_ptrs: TensorTuple[*SendPtrs],
        recv_ptrs: TensorTuple[*RecvPtrs],
        recv_count_ptrs: TensorTuple[*RecvCountPtrs],
        atomic_counters: TensorTuple[*Counters],
        dev_ctxs: DeviceContextArray[_, _],
    ) raises:
        """Multi-device fused Expert Parallelism MXFP4 dispatch.

        Launches the EP dispatch kernel on all devices simultaneously via
        _ep_launch_device_collective. Each device routes its tokens to experts
        based on top-k IDs, quantizes them to MXFP4 format, and sends them
        to the appropriate peer devices.

        Parameters:
            OutTokens: The element types of `output_tokens` (inferred).
            OutScales: The element types of `output_scales` (inferred).
            RowOffsets: The element types of `row_offsets` (inferred).
            ExpertIds: The element types of `expert_ids` (inferred).
            SrcInfos: The element types of `src_info` (inferred).
            InTokens: The element types of `input_tokens` (inferred).
            TopkIds: The element types of `topk_ids` (inferred).
            SendPtrs: The element types of `send_ptrs` (inferred).
            RecvPtrs: The element types of `recv_ptrs` (inferred).
            RecvCountPtrs: The element types of `recv_count_ptrs` (inferred).
            Counters: The element types of `atomic_counters` (inferred).
            hidden_size: Size of the model's hidden dimension (inferred).
            top_k: Number of experts each token is routed to (inferred).
            n_experts: Total number of experts across all GPUs (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            fused_shared_expert: Whether a shared expert is fused into the
                dispatch kernel (inferred).
            _trace_name: Trace label for this op.
            mx_format: Name of the MX element format, or `auto` to take it
                from the dtype. See `MXFormat.from_name`. Names the MX
                encoding of the wire payload. It cannot be inferred from
                `dispatch_dtype`: MXFP4 and MXFP6 both travel as
                `DType.uint8` and differ only in bits per element.

        Args:
            output_tokens: Output variadic tensors storing the dispatched
                MXFP4-quantized tokens, one per device.
            output_scales: Output variadic tensors storing the dispatched
                tokens' E8M0 scales in the grouped matmul's per-expert slot
                layout, one per device.
            row_offsets: Output variadic tensors storing the row offsets for
                the received tokens, one per device.
            expert_ids: Output variadic tensors storing the expert ID for
                each received token, one per device.
            src_info: Output variadic tensors recording the originating
                rank and token index for each received token, one per
                device.
            input_tokens: Input variadic tensors of tokens to dispatch,
                one per device.
            topk_ids: Input variadic tensors of top-k expert IDs per token,
                one per device.
            send_ptrs: Send buffer pointers for the dispatch phase, one
                per device.
            recv_ptrs: Receive buffer pointers for the dispatch phase,
                one per device.
            recv_count_ptrs: Receive count buffer pointers tracking tokens
                received per expert, one per device.
            atomic_counters: Atomic counters coordinating work across
                thread blocks, one per device.
            dev_ctxs: List of GPU device contexts, one per device.
        """
        comptime max_padded_m = align_up(
            max_token_per_rank * n_gpus_per_node * n_nodes, 32
        )
        comptime num_devices = len(InTokens)
        comptime dispatch_dtype = OutTokens[0].dtype
        _check_tensor_tuple[num_devices, OutTokens[0].dtype, 2, *OutTokens]()
        _check_tensor_tuple[num_devices, OutScales[0].dtype, 2, *OutScales]()
        _check_tensor_tuple[num_devices, .uint32, 1, *RowOffsets]()
        _check_tensor_tuple[num_devices, .int32, 1, *ExpertIds]()
        _check_tensor_tuple[num_devices, .int32, 2, *SrcInfos]()
        _check_tensor_tuple[num_devices, InTokens[0].dtype, 2, *InTokens]()
        _check_tensor_tuple[num_devices, .int32, 2, *TopkIds]()
        _check_tensor_tuple[num_devices, .uint64, 1, *SendPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvCountPtrs]()
        _check_tensor_tuple[num_devices, .int32, 1, *Counters]()

        var gpu_ctxs = dev_ctxs.filter_gpu_contexts[num_devices]()

        @inline(.always)
        def launch_dispatch[
            index: Int
        ]() raises {
            imm output_tokens,
            imm output_scales,
            imm row_offsets,
            imm expert_ids,
            imm src_info,
            imm atomic_counters,
            imm input_tokens,
            imm topk_ids,
            imm send_ptrs,
            imm recv_ptrs,
            imm recv_count_ptrs,
            imm gpu_ctxs,
        }:
            comptime assert conforms_to(
                type_of(output_tokens[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(output_scales[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(row_offsets[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(expert_ids[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(src_info[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(atomic_counters[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(input_tokens[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(topk_ids[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(send_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_count_ptrs[index]), TileTensorable
            )
            var out_tokens = output_tokens[index].to_tile_tensor()
            var out_scales = output_scales[index].to_tile_tensor()

            comptime mx_fmt = MXFormat.from_dtype[
                dispatch_dtype
            ]() if mx_format == "auto" else (MXFormat.from_name[mx_format]())
            var format_handler = MXTokenFormat[
                hidden_size,
                top_k,
                mx_format=mx_fmt,
            ](
                _mut_pointer_tile(out_tokens),
                _mut_pointer_tile(out_scales),
                max_padded_m,
            )

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[.uint32](),
                expert_ids[index].to_tile_tensor[.int32](),
                src_info[index].to_tile_tensor[.int32](),
                atomic_counters[index].to_tile_tensor[.int32](),
                input_tokens[index].to_tile_tensor(),
                topk_ids[index].to_tile_tensor[.int32](),
                send_ptrs[index].to_tile_tensor[.uint64](),
                recv_ptrs[index].to_tile_tensor[.uint64](),
                recv_count_ptrs[index].to_tile_tensor[.uint64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs.copy())


@extensibility.register("mo.distributed.ep.dispatch")
struct DistributedEPDispatch:
    """Registers the `mo.distributed.ep.dispatch` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        OutTokens: TypeList[Trait=Output & Tensor, ...],
        RowOffsets: TypeList[Trait=Output & Tensor, ...],
        ExpertIds: TypeList[Trait=Output & Tensor, ...],
        SrcInfos: TypeList[Trait=Output & Tensor, ...],
        InTokens: TypeList[Trait=Input & Tensor, ...],
        TopkIds: TypeList[Trait=Input & Tensor, ...],
        SendPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvCountPtrs: TypeList[Trait=Input & Tensor, ...],
        Counters: TypeList[Trait=MutableInput & Tensor, ...],
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        //,
        _trace_name: StaticString,
    ](
        output_tokens: TensorTuple[*OutTokens],
        row_offsets: TensorTuple[*RowOffsets],
        expert_ids: TensorTuple[*ExpertIds],
        src_info: TensorTuple[*SrcInfos],
        input_tokens: TensorTuple[*InTokens],
        topk_ids: TensorTuple[*TopkIds],
        send_ptrs: TensorTuple[*SendPtrs],
        recv_ptrs: TensorTuple[*RecvPtrs],
        recv_count_ptrs: TensorTuple[*RecvCountPtrs],
        atomic_counters: TensorTuple[*Counters],
        dev_ctxs: DeviceContextArray[_, _],
    ) raises:
        """Multi-device fused Expert Parallelism BF16 dispatch.

        Parameters:
            OutTokens: The element types of `output_tokens` (inferred).
            RowOffsets: The element types of `row_offsets` (inferred).
            ExpertIds: The element types of `expert_ids` (inferred).
            SrcInfos: The element types of `src_info` (inferred).
            InTokens: The element types of `input_tokens` (inferred).
            TopkIds: The element types of `topk_ids` (inferred).
            SendPtrs: The element types of `send_ptrs` (inferred).
            RecvPtrs: The element types of `recv_ptrs` (inferred).
            RecvCountPtrs: The element types of `recv_count_ptrs` (inferred).
            Counters: The element types of `atomic_counters` (inferred).
            hidden_size: Size of the model's hidden dimension (inferred).
            top_k: Number of experts each token is routed to (inferred).
            n_experts: Total number of experts across all GPUs (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            fused_shared_expert: Whether a shared expert is fused into the
                dispatch kernel (inferred).
            _trace_name: Trace label for this op.

        Args:
            output_tokens: Output variadic tensors storing the dispatched
                tokens, one per device.
            row_offsets: Output variadic tensors storing the row offsets for
                the received tokens, one per device.
            expert_ids: Output variadic tensors storing the expert ID for
                each received token, one per device.
            src_info: Output variadic tensors recording the originating
                rank and token index for each received token, one per
                device.
            input_tokens: Input variadic tensors of tokens to dispatch,
                one per device.
            topk_ids: Input variadic tensors of top-k expert IDs per token,
                one per device.
            send_ptrs: Send buffer pointers for the dispatch phase, one
                per device.
            recv_ptrs: Receive buffer pointers for the dispatch phase,
                one per device.
            recv_count_ptrs: Receive count buffer pointers tracking tokens
                received per expert, one per device.
            atomic_counters: Atomic counters coordinating work across
                thread blocks, one per device.
            dev_ctxs: List of GPU device contexts, one per device.
        """
        comptime num_devices = len(InTokens)
        comptime dispatch_dtype = InTokens[0].dtype
        comptime assert dispatch_dtype == .bfloat16
        _check_tensor_tuple[num_devices, .bfloat16, 2, *OutTokens]()
        _check_tensor_tuple[num_devices, .uint32, 1, *RowOffsets]()
        _check_tensor_tuple[num_devices, .int32, 1, *ExpertIds]()
        _check_tensor_tuple[num_devices, .int32, 2, *SrcInfos]()
        _check_tensor_tuple[num_devices, dispatch_dtype, 2, *InTokens]()
        _check_tensor_tuple[num_devices, .int32, 2, *TopkIds]()
        _check_tensor_tuple[num_devices, .uint64, 1, *SendPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvCountPtrs]()
        _check_tensor_tuple[num_devices, .int32, 1, *Counters]()

        var gpu_ctxs = dev_ctxs.filter_gpu_contexts[num_devices]()

        @inline(.always)
        def launch_dispatch[
            index: Int
        ]() raises {
            imm output_tokens,
            imm row_offsets,
            imm expert_ids,
            imm src_info,
            imm atomic_counters,
            imm input_tokens,
            imm topk_ids,
            imm send_ptrs,
            imm recv_ptrs,
            imm recv_count_ptrs,
            imm gpu_ctxs,
        }:
            comptime assert conforms_to(
                type_of(output_tokens[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(row_offsets[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(expert_ids[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(src_info[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(atomic_counters[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(input_tokens[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(topk_ids[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(send_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_count_ptrs[index]), TileTensorable
            )
            var out_tokens = output_tokens[index].to_tile_tensor[.bfloat16]()
            var format_handler = BF16TokenFormat[hidden_size, top_k](out_tokens)

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[.uint32](),
                expert_ids[index].to_tile_tensor[.int32](),
                src_info[index].to_tile_tensor[.int32](),
                atomic_counters[index].to_tile_tensor[.int32](),
                input_tokens[index].to_tile_tensor(),
                topk_ids[index].to_tile_tensor[.int32](),
                send_ptrs[index].to_tile_tensor[.uint64](),
                recv_ptrs[index].to_tile_tensor[.uint64](),
                recv_count_ptrs[index].to_tile_tensor[.uint64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs.copy())


@extensibility.register("mo.distributed.ep.dispatch.fp8")
struct DistributedEPDispatchFP8:
    """Registers the `mo.distributed.ep.dispatch.fp8` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        OutTokens: TypeList[Trait=Output & Tensor, ...],
        OutScales: TypeList[Trait=Output & Tensor, ...],
        RowOffsets: TypeList[Trait=Output & Tensor, ...],
        ExpertIds: TypeList[Trait=Output & Tensor, ...],
        SrcInfos: TypeList[Trait=Output & Tensor, ...],
        InTokens: TypeList[Trait=Input & Tensor, ...],
        TopkIds: TypeList[Trait=Input & Tensor, ...],
        SendPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvCountPtrs: TypeList[Trait=Input & Tensor, ...],
        Counters: TypeList[Trait=MutableInput & Tensor, ...],
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        fused_shared_expert: Bool,
        //,
        _trace_name: StaticString,
    ](
        output_tokens: TensorTuple[*OutTokens],
        output_scales: TensorTuple[*OutScales],
        row_offsets: TensorTuple[*RowOffsets],
        expert_ids: TensorTuple[*ExpertIds],
        src_info: TensorTuple[*SrcInfos],
        input_tokens: TensorTuple[*InTokens],
        topk_ids: TensorTuple[*TopkIds],
        send_ptrs: TensorTuple[*SendPtrs],
        recv_ptrs: TensorTuple[*RecvPtrs],
        recv_count_ptrs: TensorTuple[*RecvCountPtrs],
        atomic_counters: TensorTuple[*Counters],
        dev_ctxs: DeviceContextArray[_, _],
    ) raises:
        """Multi-device fused Expert Parallelism FP8 dispatch.

        Launches the EP dispatch kernel on all devices simultaneously via
        _launch_device_collective. Each device routes its tokens to experts
        based on top-k IDs, quantizes them to Blockwise FP8 format, and sends
        them to peer devices.

        Parameters:
            OutTokens: The element types of `output_tokens` (inferred).
            OutScales: The element types of `output_scales` (inferred).
            RowOffsets: The element types of `row_offsets` (inferred).
            ExpertIds: The element types of `expert_ids` (inferred).
            SrcInfos: The element types of `src_info` (inferred).
            InTokens: The element types of `input_tokens` (inferred).
            TopkIds: The element types of `topk_ids` (inferred).
            SendPtrs: The element types of `send_ptrs` (inferred).
            RecvPtrs: The element types of `recv_ptrs` (inferred).
            RecvCountPtrs: The element types of `recv_count_ptrs` (inferred).
            Counters: The element types of `atomic_counters` (inferred).
            hidden_size: Size of the model's hidden dimension (inferred).
            top_k: Number of experts each token is routed to (inferred).
            n_experts: Total number of experts across all GPUs (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            dispatch_scale_granularity: Block size used for the blockwise
                FP8 quantization scales (inferred).
            fused_shared_expert: Whether a shared expert is fused into the
                dispatch kernel (inferred).
            _trace_name: Trace label for this op.

        Args:
            output_tokens: Output variadic tensors storing the dispatched
                Blockwise FP8-quantized tokens, one per device.
            output_scales: Output variadic tensors storing the block scales
                for the dispatched tokens, one per device.
            row_offsets: Output variadic tensors storing the row offsets for
                the received tokens, one per device.
            expert_ids: Output variadic tensors storing the expert ID for
                each received token, one per device.
            src_info: Output variadic tensors recording the originating
                rank and token index for each received token, one per
                device.
            input_tokens: Input variadic tensors of tokens to dispatch,
                one per device.
            topk_ids: Input variadic tensors of top-k expert IDs per token,
                one per device.
            send_ptrs: Send buffer pointers for the dispatch phase, one
                per device.
            recv_ptrs: Receive buffer pointers for the dispatch phase,
                one per device.
            recv_count_ptrs: Receive count buffer pointers tracking tokens
                received per expert, one per device.
            atomic_counters: Atomic counters coordinating work across
                thread blocks, one per device.
            dev_ctxs: List of GPU device contexts, one per device.
        """
        comptime num_devices = len(InTokens)
        _check_tensor_tuple[num_devices, OutTokens[0].dtype, 2, *OutTokens]()
        _check_tensor_tuple[num_devices, OutScales[0].dtype, 2, *OutScales]()
        _check_tensor_tuple[num_devices, .uint32, 1, *RowOffsets]()
        _check_tensor_tuple[num_devices, .int32, 1, *ExpertIds]()
        _check_tensor_tuple[num_devices, .int32, 2, *SrcInfos]()
        _check_tensor_tuple[num_devices, InTokens[0].dtype, 2, *InTokens]()
        _check_tensor_tuple[num_devices, .int32, 2, *TopkIds]()
        _check_tensor_tuple[num_devices, .uint64, 1, *SendPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvCountPtrs]()
        _check_tensor_tuple[num_devices, .int32, 1, *Counters]()

        var gpu_ctxs = dev_ctxs.filter_gpu_contexts[num_devices]()

        @inline(.always)
        def launch_dispatch[
            index: Int
        ]() raises {
            imm output_tokens,
            imm output_scales,
            imm row_offsets,
            imm expert_ids,
            imm src_info,
            imm atomic_counters,
            imm input_tokens,
            imm topk_ids,
            imm send_ptrs,
            imm recv_ptrs,
            imm recv_count_ptrs,
            imm gpu_ctxs,
        }:
            comptime assert conforms_to(
                type_of(output_tokens[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(output_scales[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(row_offsets[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(expert_ids[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(src_info[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(atomic_counters[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(input_tokens[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(topk_ids[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(send_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_count_ptrs[index]), TileTensorable
            )
            var out_tokens = output_tokens[index].to_tile_tensor()
            var out_scales = output_scales[index].to_tile_tensor()
            var format_handler = BlockwiseFP8TokenFormat[hidden_size, top_k](
                _mut_pointer_tile(out_tokens), _mut_pointer_tile(out_scales)
            )

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[.uint32](),
                expert_ids[index].to_tile_tensor[.int32](),
                src_info[index].to_tile_tensor[.int32](),
                atomic_counters[index].to_tile_tensor[.int32](),
                input_tokens[index].to_tile_tensor(),
                topk_ids[index].to_tile_tensor[.int32](),
                send_ptrs[index].to_tile_tensor[.uint64](),
                recv_ptrs[index].to_tile_tensor[.uint64](),
                recv_count_ptrs[index].to_tile_tensor[.uint64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs.copy())


@extensibility.register("mo.distributed.ep.combine")
struct DistributedEPCombine:
    """Registers the `mo.distributed.ep.combine` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Fused & StoreAccess, ...],
        Tokens: TypeList[Trait=Input & Tensor, ...],
        SrcInfos: TypeList[Trait=Input & Tensor, ...],
        SendPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvPtrs: TypeList[Trait=Input & Tensor, ...],
        RecvCountPtrs: TypeList[Trait=Input & Tensor, ...],
        RouterWeights: TypeList[Trait=Input & Tensor, ...],
        Counters: TypeList[Trait=MutableInput & Tensor, ...],
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        //,
        _trace_name: StaticString,
    ](
        output_tokens: TensorTuple[*Outs],
        input_tokens: TensorTuple[*Tokens],
        src_info: TensorTuple[*SrcInfos],
        send_ptrs: TensorTuple[*SendPtrs],
        recv_ptrs: TensorTuple[*RecvPtrs],
        recv_count_ptrs: TensorTuple[*RecvCountPtrs],
        router_weights: TensorTuple[*RouterWeights],
        atomic_counters: TensorTuple[*Counters],
        dev_ctxs: DeviceContextArray[_, _],
    ) raises:
        """Multi-device fused Expert Parallelism combine with output fusion.

        Parameters:
            Outs: The element types of `output_tokens` (inferred).
            Tokens: The element types of `input_tokens` (inferred).
            SrcInfos: The element types of `src_info` (inferred).
            SendPtrs: The element types of `send_ptrs` (inferred).
            RecvPtrs: The element types of `recv_ptrs` (inferred).
            RecvCountPtrs: The element types of `recv_count_ptrs` (inferred).
            RouterWeights: The element types of `router_weights` (inferred).
            Counters: The element types of `atomic_counters` (inferred).
            hidden_size: Size of the model's hidden dimension.
            top_k: Number of experts each token is routed to.
            n_experts: Total number of experts across all GPUs.
            max_token_per_rank: Maximum number of tokens per GPU.
            n_gpus_per_node: Number of GPUs per node.
            n_nodes: Number of physical nodes.
            fused_shared_expert: Whether a shared expert is fused into the
                combine kernel.
            _trace_name: Trace label for this op.

        Args:
            output_tokens: Fused output variadic tensors storing the combined
                expert outputs, one per device.
            input_tokens: Input variadic tensors of expert-processed tokens to
                combine, one per device.
            src_info: Source info tensors recording the originating rank and
                token index for each received token, one per device.
            send_ptrs: Send buffer pointers for the combine phase, one per
                device.
            recv_ptrs: Receive buffer pointers for the combine phase, one per
                device.
            recv_count_ptrs: Receive count buffer pointers tracking tokens
                received per expert, one per device.
            router_weights: Router weights used to scale each expert's
                contribution, one per device.
            atomic_counters: Atomic counters coordinating work across thread
                blocks, one per device.
            dev_ctxs: List of GPU device contexts, one per device.
        """
        comptime num_devices = len(Tokens)
        comptime combine_dtype = Tokens[0].dtype
        comptime router_weights_dtype = RouterWeights[0].dtype
        _check_tensor_tuple[num_devices, combine_dtype, 2, *Outs]()
        _check_tensor_tuple[num_devices, combine_dtype, 2, *Tokens]()
        _check_tensor_tuple[num_devices, .int32, 2, *SrcInfos]()
        _check_tensor_tuple[num_devices, .uint64, 1, *SendPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvPtrs]()
        _check_tensor_tuple[num_devices, .uint64, 1, *RecvCountPtrs]()
        _check_tensor_tuple[
            num_devices, router_weights_dtype, 2, *RouterWeights
        ]()
        _check_tensor_tuple[num_devices, .int32, 1, *Counters]()

        var gpu_ctxs = dev_ctxs.filter_gpu_contexts[num_devices]()
        var output_views = TensorTuple._fused_view(output_tokens)

        @inline(.always)
        def launch_combine[
            index: Int
        ]() raises {
            imm output_tokens,
            imm output_views,
            imm input_tokens,
            imm src_info,
            imm send_ptrs,
            imm recv_ptrs,
            imm recv_count_ptrs,
            imm router_weights,
            imm atomic_counters,
            imm gpu_ctxs,
        }:
            comptime assert (
                Outs[index]._shape_types[1].static_value == hidden_size
            ), "EP combine: output tokens shape doesn't match hidden size."
            comptime assert conforms_to(
                type_of(router_weights[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(atomic_counters[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(input_tokens[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(src_info[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(send_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_ptrs[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(recv_count_ptrs[index]), TileTensorable
            )
            var rw_tensor = router_weights[index].to_tile_tensor()

            @inline(.always)
            def router_weights_fn[
                width: Int
            ](token_idx: Int, topk_id: Int) {var rw_tensor} -> SIMD[
                .float32, width
            ]:
                return rw_tensor.load[width=width]((token_idx, topk_id)).cast[
                    DType.float32
                ]()

            # Stores into this device's fused output. It runs on the device, so
            # it copies the views.
            @inline(.always)
            def output_fn[
                dtype: DType,
                width: SIMDLength,
                *,
                alignment: Int = 1,
            ](coords: Coord, val: SIMD[dtype, width]) {var output_views}:
                comptime assert conforms_to(
                    type_of(output_views[index]), Storable
                )
                output_views[index].store[
                    width=width, element_alignment=alignment
                ](
                    coords,
                    rebind[SIMD[type_of(output_views[index]).dtype, width]](
                        val
                    ),
                )

            ep_fused_combine_kernel_api[
                hidden_size,
                top_k,
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert=fused_shared_expert,
            ](
                coord_to_index_list(output_tokens[index].shape())[0],
                atomic_counters[index].to_tile_tensor[.int32](),
                input_tokens[index].to_tile_tensor[combine_dtype](),
                src_info[index].to_tile_tensor[.int32](),
                send_ptrs[index].to_tile_tensor[.uint64](),
                recv_ptrs[index].to_tile_tensor[.uint64](),
                recv_count_ptrs[index].to_tile_tensor[.uint64](),
                gpu_ctxs[index],
                output_fn,
                router_weights_fn,
            )

        _launch_device_collective[num_devices](launch_combine, gpu_ctxs.copy())


@extensibility.register("ep.combine_async")
struct Struct_ep_combine_async:
    """Registers the `ep.combine_async` graph op with the graph compiler."""

    @inline(.always)
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
    ](
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        src_info: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism combine kernel.

        Sends expert-processed output tokens back to their original
        devices asynchronously, without waiting for the transfers to
        complete.

        Parameters:
            combine_dtype: `DType` used for the token payload during
                the combine phase (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).

        Args:
            atomic_counters: Atomic counters coordinating work across
                thread blocks during the combine phase.
            input_tokens: Expert output tokens to send back to their
                original devices. Shape `[num_tokens, hidden_size]`.
            src_info: Source routing information from the dispatch
                phase recording the originating rank and token index
                for each token. Shape `[num_tokens, 2]`.
            send_ptrs: Send buffer pointers for the combine phase.
            recv_ptrs: Receive buffer pointers for the combine phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                the number of tokens received per expert.
            context: GPU device context for the current device.
        """
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1

        ep_combine_async_kernel_api[
            combine_dtype,
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
        ](
            atomic_counters.to_tile_tensor[.int32](),
            input_tokens.to_tile_tensor[combine_dtype](),
            src_info.to_tile_tensor[.int32](),
            send_ptrs.to_tile_tensor[.uint64](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            context,
        )


@extensibility.register("ep.combine_wait")
struct Struct_ep_combine_wait:
    """Registers the `ep.combine_wait` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        combine_dtype: DType,
        //,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
    ](
        output_tokens: Some[Output & Fused & StoreAccess],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        router_weights: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism combine completion kernel.

        Waits for incoming expert outputs and computes the weighted
        sum of routed expert outputs for each token using the router
        weights.

        Parameters:
            combine_dtype: `DType` used for the token payload during
                the combine phase (inferred).
            hidden_size: Size of the model's hidden dimension.
            top_k: Number of experts each token is routed to.
            n_experts: Total number of experts across all GPUs.
            max_token_per_rank: Maximum number of tokens any GPU can
                receive.
            n_gpus_per_node: Number of GPUs per physical node.
            n_nodes: Number of physical nodes in the deployment.

        Args:
            output_tokens: Fused output tensor storing the weighted
                sum of routed expert outputs for each token. Shape
                `[num_tokens, hidden_size]`.
            atomic_counters: Atomic counters coordinating work
                across thread blocks during the combine phase.
            recv_ptrs: Receive buffer pointers for the combine
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                the number of tokens received per expert.
            router_weights: Router weights for the current device
                used to compute the weighted sum of expert outputs.
                Shape `[num_tokens, top_k]`.
            context: GPU device context for the current device.
        """
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(router_weights), TileTensorable)
        comptime assert type_of(output_tokens).dtype == combine_dtype
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        comptime assert type_of(router_weights).rank == 2
        comptime assert (
            type_of(output_tokens)._shape_types[1].static_value == hidden_size
        ), "EP combine: output tokens shape doesn't match hidden size."
        var router_weights_tensor = router_weights.to_tile_tensor()
        comptime assert router_weights_tensor.flat_rank == 2
        comptime assert router_weights_tensor.flat_rank >= 2

        comptime assert conforms_to(type_of(output_tokens), UnsafeFusedView)
        var out_view = unsafe_fused_view(output_tokens)
        comptime assert conforms_to(type_of(out_view), Storable)

        @inline(.always)
        def router_weights_fn[
            width: Int
        ](token_idx: Int, topk_id: Int) {var router_weights_tensor} -> SIMD[
            .float32, width
        ]:
            return router_weights_tensor.load[width=width](
                (token_idx, topk_id)
            ).cast[.float32]()

        @inline(.always)
        def output_fn[
            dtype: DType,
            width: SIMDLength,
            *,
            alignment: Int = 1,
        ](coords: Coord, val: SIMD[dtype, width]) {var out_view}:
            out_view.store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[type_of(out_view).dtype, width]](val),
            )

        ep_combine_wait_kernel_api[
            combine_dtype,
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
        ](
            coord_to_index_list(output_tokens.shape())[0],
            atomic_counters.to_tile_tensor[.int32](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            context,
            output_fn,
            router_weights_fn,
        )


@extensibility.register("ep.combine_async.mx")
struct Struct_ep_combine_async_mx:
    """Registers the `ep.combine_async.mx` graph op with the graph compiler."""

    @inline(.always)
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
        *,
        mx_min_tokens: Int,
    ](
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        src_info: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        total_tokens: UInt32,
        context: DeviceContext,
    ) raises:
        """`ep.combine_async` that sends MXFP8 messages for large steps.

        When `total_tokens` reaches `mx_min_tokens`, each expert output goes
        back to its rank as MXFP8 (E4M3 values with an E8M0 scale per 32
        elements, about half the bytes), which is lossy; smaller steps send it
        exactly. `total_tokens` must be the same on every rank, and the
        matching `ep.combine_wait.mx` must get the same value and threshold.

        Parameters:
            combine_dtype: Expert output dtype (inferred).
            hidden_size: Model hidden size (inferred).
            top_k: Experts each token is routed to (inferred).
            n_experts: Experts across all GPUs (inferred).
            max_token_per_rank: Most tokens per GPU (inferred).
            n_gpus_per_node: GPUs per node (inferred).
            n_nodes: Physical nodes (inferred).
            mx_min_tokens: Smallest `total_tokens` sent as MXFP8.

        Args:
            atomic_counters: Atomic counters coordinating the combine.
            input_tokens: Expert outputs. Shape `[num_tokens, hidden_size]`.
            src_info: Each row's originating rank and token from the dispatch.
            send_ptrs: Send buffer pointers for the combine phase.
            recv_ptrs: Receive buffer pointers for the combine phase.
            recv_count_ptrs: Receive count buffer pointers.
            total_tokens: Tokens of the step across all ranks.
            context: GPU device context for the current device.
        """
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        var atomic_counters_tile = atomic_counters.to_tile_tensor[.int32]()
        var input_tokens_tile = input_tokens.to_tile_tensor[combine_dtype]()
        var src_info_tile = src_info.to_tile_tensor[.int32]()
        var send_ptrs_tile = send_ptrs.to_tile_tensor[.uint64]()
        var recv_ptrs_tile = recv_ptrs.to_tile_tensor[.uint64]()
        var recv_count_ptrs_tile = recv_count_ptrs.to_tile_tensor[.uint64]()

        @inline(.always)
        def launch[mx_msg: Bool]() raises {imm}:
            ep_combine_async_kernel_api[
                combine_dtype,
                hidden_size,
                top_k,
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                mx_msg=mx_msg,
            ](
                atomic_counters_tile,
                input_tokens_tile,
                src_info_tile,
                send_ptrs_tile,
                recv_ptrs_tile,
                recv_count_ptrs_tile,
                context,
            )

        if Int(total_tokens) >= mx_min_tokens:
            launch[True]()
        else:
            launch[False]()


@extensibility.register("ep.combine_wait.mx")
struct Struct_ep_combine_wait_mx:
    """Registers the `ep.combine_wait.mx` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        combine_dtype: DType,
        //,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        *,
        mx_min_tokens: Int,
    ](
        output_tokens: Some[Output & Fused & StoreAccess],
        atomic_counters: Some[MutableInput & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        router_weights: Some[Input & Tensor],
        total_tokens: UInt32,
        context: DeviceContext,
    ) raises:
        """`ep.combine_wait` for the messages `ep.combine_async.mx` sent.

        Reads MXFP8 messages when `total_tokens` reaches `mx_min_tokens` and
        exact ones otherwise, then computes each token's weighted sum of its
        routed expert outputs.

        Parameters:
            combine_dtype: Output dtype (inferred).
            hidden_size: Model hidden size.
            top_k: Experts each token is routed to.
            n_experts: Experts across all GPUs.
            max_token_per_rank: Most tokens any GPU can receive.
            n_gpus_per_node: GPUs per physical node.
            n_nodes: Physical nodes.
            mx_min_tokens: Smallest `total_tokens` sent as MXFP8.

        Args:
            output_tokens: `[num_tokens, hidden_size]` weighted sums.
            atomic_counters: Atomic counters coordinating the combine.
            recv_ptrs: Receive buffer pointers for the combine phase.
            recv_count_ptrs: Receive count buffer pointers.
            router_weights: `[num_tokens, top_k]` router weights.
            total_tokens: Tokens of the step across all ranks.
            context: GPU device context for the current device.
        """
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(router_weights), TileTensorable)
        comptime assert type_of(output_tokens).dtype == combine_dtype
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        comptime assert type_of(router_weights).rank == 2
        comptime assert (
            type_of(output_tokens)._shape_types[1].static_value == hidden_size
        ), "EP combine: output tokens shape doesn't match hidden size."
        var router_weights_tensor = router_weights.to_tile_tensor()
        comptime assert router_weights_tensor.flat_rank == 2

        comptime assert conforms_to(type_of(output_tokens), UnsafeFusedView)
        var out_view = unsafe_fused_view(output_tokens)
        comptime assert conforms_to(type_of(out_view), Storable)

        @inline(.always)
        def router_weights_fn[
            width: Int
        ](token_idx: Int, topk_id: Int) {var router_weights_tensor} -> SIMD[
            .float32, width
        ]:
            return router_weights_tensor.load[width=width](
                (token_idx, topk_id)
            ).cast[.float32]()

        @inline(.always)
        def output_fn[
            dtype: DType,
            width: SIMDLength,
            *,
            alignment: Int = 1,
        ](coords: Coord, val: SIMD[dtype, width]) {var out_view}:
            out_view.store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[type_of(out_view).dtype, width]](val),
            )

        var num_tokens = coord_to_index_list(output_tokens.shape())[0]
        var atomic_counters_tile = atomic_counters.to_tile_tensor[.int32]()
        var recv_ptrs_tile = recv_ptrs.to_tile_tensor[.uint64]()
        var recv_count_ptrs_tile = recv_count_ptrs.to_tile_tensor[.uint64]()

        @inline(.always)
        def launch[mx_msg: Bool]() raises {imm}:
            ep_combine_wait_kernel_api[
                combine_dtype,
                hidden_size,
                top_k,
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                mx_msg=mx_msg,
            ](
                num_tokens,
                atomic_counters_tile,
                recv_ptrs_tile,
                recv_count_ptrs_tile,
                context,
                output_fn,
                router_weights_fn,
            )

        if Int(total_tokens) >= mx_min_tokens:
            launch[True]()
        else:
            launch[False]()


@extensibility.register("ep.combine")
struct Struct_ep_combine:
    """Registers the `ep.combine` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        //,
    ](
        output_tokens: Some[Output & Fused & StoreAccess],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        src_info: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        router_weights: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism combine kernel.

        Sends expert outputs back to their original devices, waits for
        all transfers to complete, and computes the weighted sum of
        routed expert outputs for each token.

        Parameters:
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens any GPU can
                receive (inferred).
            n_gpus_per_node: Number of GPUs per physical node
                (inferred).
            n_nodes: Number of physical nodes in the deployment
                (inferred).
            fused_shared_expert: Whether a shared expert is fused into
                the combine kernel, adding its output to the routed
                expert outputs (inferred).
            skip_a2a: Whether to skip the all-to-all communication and
                send tokens only within the current device (inferred).

        Args:
            output_tokens: Fused output tensor storing the weighted
                sum of routed expert outputs for each token. Shape
                `[num_tokens, hidden_size]`.
            atomic_counters: Atomic counters coordinating work across
                thread blocks during the combine phase.
            input_tokens: Expert output tokens to send back to their
                original devices. Shape `[num_tokens, hidden_size]`.
            src_info: Source routing information from the dispatch
                phase recording the originating rank and token index
                for each token. Shape `[num_tokens, 2]`.
            send_ptrs: Send buffer pointers for the combine phase,
                one per local GPU.
            recv_ptrs: Receive buffer pointers for the combine phase,
                one per local GPU.
            recv_count_ptrs: Receive count buffer pointers tracking the
                number of tokens received per expert.
            router_weights: Router weights for the current device used
                to compute the weighted sum of expert outputs. Shape
                `[num_tokens, top_k]`.
            context: GPU device context for the current device.
        """
        comptime assert (
            not skip_a2a
        ), "ep.combine: skip_a2a is lowered to ep.combine.skip_a2a."
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(router_weights), TileTensorable)
        comptime combine_dtype = type_of(output_tokens).dtype
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        comptime assert type_of(router_weights).rank == 2
        comptime assert (
            type_of(output_tokens)._shape_types[1].static_value == hidden_size
        ), "EP combine: output tokens shape doesn't match hidden size."
        var router_weights_tensor = router_weights.to_tile_tensor()
        comptime assert router_weights_tensor.flat_rank == 2
        comptime assert router_weights_tensor.flat_rank >= 2

        comptime assert conforms_to(type_of(output_tokens), UnsafeFusedView)
        var out_view = unsafe_fused_view(output_tokens)
        comptime assert conforms_to(type_of(out_view), Storable)

        @inline(.always)
        def router_weights_fn[
            width: Int
        ](token_idx: Int, topk_id: Int) {var router_weights_tensor} -> SIMD[
            .float32, width
        ]:
            return router_weights_tensor.load[width=width](
                (token_idx, topk_id)
            ).cast[.float32]()

        @inline(.always)
        def output_fn[
            dtype: DType,
            width: SIMDLength,
            *,
            alignment: Int = 1,
        ](coords: Coord, val: SIMD[dtype, width]) {var out_view}:
            out_view.store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[type_of(out_view).dtype, width]](val),
            )

        ep_fused_combine_kernel_api[
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
        ](
            coord_to_index_list(output_tokens.shape())[0],
            atomic_counters.to_tile_tensor[.int32](),
            input_tokens.to_tile_tensor[combine_dtype](),
            src_info.to_tile_tensor[.int32](),
            send_ptrs.to_tile_tensor[.uint64](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            context,
            output_fn,
            router_weights_fn,
        )


@extensibility.register("ep.combine.skip_a2a")
struct Struct_ep_combine_skip_a2a:
    """Registers the `ep.combine.skip_a2a` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
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
    ](
        output_tokens: Some[Output & Fused & StoreAccess],
        atomic_counters: Some[MutableInput & Tensor],
        input_tokens: Some[Input & Tensor],
        src_info: Some[Input & Tensor],
        send_ptrs: Some[Input & Tensor],
        recv_ptrs: Some[Input & Tensor],
        recv_count_ptrs: Some[Input & Tensor],
        router_weights: Some[Input & Tensor],
        topk_ids: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism combine kernel.

        Sends expert outputs back to their original devices, waits for
        all transfers to complete, and computes the weighted sum of
        routed expert outputs for each token. When `skip_a2a` is set,
        skips the all-to-all communication and reduces expert outputs
        locally using an allreduce of size `allreduce_world_size`.

        Parameters:
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens any GPU can
                receive (inferred).
            n_gpus_per_node: Number of GPUs per physical node
                (inferred).
            n_nodes: Number of physical nodes in the deployment
                (inferred).
            fused_shared_expert: Whether a shared expert is fused
                into the combine kernel, adding its output to the
                routed expert outputs (inferred).
            skip_a2a: Whether to skip the all-to-all communication
                and reduce expert outputs locally instead
                (inferred).
            allreduce_world_size: World size for the local allreduce
                used when `skip_a2a` is set (inferred).

        Args:
            output_tokens: Fused output tensor storing the weighted
                sum of routed expert outputs for each token. Shape
                `[num_tokens, hidden_size]`.
            atomic_counters: Atomic counters coordinating work
                across thread blocks during the combine phase.
            input_tokens: Expert output tokens to send back to
                their original devices. Shape
                `[num_tokens, hidden_size]`.
            src_info: Source routing information from the dispatch
                phase recording the originating rank and token
                index for each token. Shape `[num_tokens, 2]`.
            send_ptrs: Send buffer pointers for the combine phase,
                one per local GPU.
            recv_ptrs: Receive buffer pointers for the combine
                phase, one per local GPU.
            recv_count_ptrs: Receive count buffer pointers tracking
                the number of tokens received per expert.
            router_weights: Router weights for the current device
                used to compute the weighted sum of expert outputs.
                Shape `[num_tokens, top_k]`.
            topk_ids: Top-k expert IDs selected for each token.
                Shape `[num_tokens, top_k]`.
            context: GPU device context for the current device.
        """
        comptime assert conforms_to(type_of(atomic_counters), TileTensorable)
        comptime assert conforms_to(type_of(input_tokens), TileTensorable)
        comptime assert conforms_to(type_of(src_info), TileTensorable)
        comptime assert conforms_to(type_of(send_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(recv_count_ptrs), TileTensorable)
        comptime assert conforms_to(type_of(router_weights), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime combine_dtype = type_of(output_tokens).dtype
        comptime assert type_of(output_tokens).rank == 2
        comptime assert type_of(atomic_counters).rank == 1
        comptime assert type_of(input_tokens).rank == 2
        comptime assert type_of(src_info).rank == 2
        comptime assert type_of(send_ptrs).rank == 1
        comptime assert type_of(recv_ptrs).rank == 1
        comptime assert type_of(recv_count_ptrs).rank == 1
        comptime assert type_of(router_weights).rank == 2
        comptime assert type_of(topk_ids).rank == 2
        comptime assert (
            type_of(output_tokens)._shape_types[1].static_value == hidden_size
        ), "EP combine: output tokens shape doesn't match hidden size."
        var router_weights_tensor = router_weights.to_tile_tensor()
        comptime assert router_weights_tensor.flat_rank == 2
        comptime assert router_weights_tensor.flat_rank >= 2

        comptime assert conforms_to(type_of(output_tokens), UnsafeFusedView)
        var out_view = unsafe_fused_view(output_tokens)
        comptime assert conforms_to(type_of(out_view), Storable)

        @inline(.always)
        def router_weights_fn[
            width: Int
        ](token_idx: Int, topk_id: Int) {var router_weights_tensor} -> SIMD[
            .float32, width
        ]:
            return router_weights_tensor.load[width=width](
                (token_idx, topk_id)
            ).cast[.float32]()

        @inline(.always)
        def output_fn[
            dtype: DType,
            width: SIMDLength,
            *,
            alignment: Int = 1,
        ](coords: Coord, val: SIMD[dtype, width]) {var out_view}:
            out_view.store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[type_of(out_view).dtype, width]](val),
            )

        ep_fused_combine_kernel_api[
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            coord_to_index_list(output_tokens.shape())[0],
            atomic_counters.to_tile_tensor[.int32](),
            input_tokens.to_tile_tensor[combine_dtype](),
            src_info.to_tile_tensor[.int32](),
            send_ptrs.to_tile_tensor[.uint64](),
            recv_ptrs.to_tile_tensor[.uint64](),
            recv_count_ptrs.to_tile_tensor[.uint64](),
            context,
            output_fn,
            router_weights_fn,
            topk_ids.to_tile_tensor[.int32]().as_imm().ptr,
        )


@extensibility.register("ep.fused_silu")
struct Struct_ep_fused_silu:
    """Registers the `ep.fused_silu` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        activation: StaticString = "silu",
        swiglu_alpha: Float32 = 0.0,
        swiglu_limit: Float32 = 0.0,
        interleaved: Bool = False,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        row_offsets: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism fused SILU kernel.

        This function launches the fused_silu kernel to perform the SILU
        operation for all the MLPs in the EP MoE module. We need to manually
        implement the custom operation here is because after the EP dispatch
        phase, the actual number of received tokens is not known to the host.

        This kernel will read the row offsets to determine the actual number of
        received tokens in the input tensor, and then only perform the SILU
        operation on the received tokens.

        The gated activation is built at compile time from the `activation`,
        `swiglu_alpha` and `swiglu_limit` parameters.
        `interleaved` reads each input row as `[gate_0, up_0, ...]` instead
        of `[gate | up]`.
        """
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime output_dtype = type_of(output).dtype
        comptime input_dtype = type_of(input).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(input).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        # Ensure this kernel only runs on GPU targets
        comptime assert context.target.is_gpu(), "EP is only supported on GPU."

        var output_tensor = output.to_tile_tensor()
        var input_tensor = input.to_tile_tensor().as_imm()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            .uint32, mut=False
        ]()

        var gpu_ctx = context
        comptime hw_info = gpu_ctx.default_device_info

        comptime act = get_swiglu_activation[
            activation,
            swiglu_alpha,
            swiglu_limit,
        ]

        comptime fused_silu = fused_silu_kernel[
            output_dtype,
            input_dtype,
            output_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
            activation=act,
            interleaved=interleaved,
        ]

        @inline(.always)
        def description_fn() {imm} -> String:
            # fmt: off
            return String(
                "output_dtype=", output_dtype,
                ";input_dtype=", input_dtype,
                ";activation=", activation,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "ep.fused_silu",
            Trace[TraceLevel.OP]._get_detail_str(description_fn),
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


@extensibility.register("ep.fused_silu.mxfp4")
struct Struct_ep_fused_silu_mxfp4:
    """Registers the `ep.fused_silu.mxfp4` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        *,
        max_padded_M: Int,
        activation: StaticString = "silu",
        swiglu_alpha: Float32 = 0.0,
        swiglu_limit: Float32 = 0.0,
        # `ep.fused_silu.mxfp8` shares this body and overrides the label.
        trace_name: StaticString = "ep.fused_silu.mxfp4",
    ](
        output: Some[Output & Tensor],
        scales: Some[Output & Tensor],
        input: Some[Input & Tensor],
        row_offsets: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism fused SILU kernel with MX
        quantization.

        Shared body for `ep.fused_silu.mxfp4` and `ep.fused_silu.mxfp8`:
        `fused_silu_mx_kernel` takes its element packing from `quant_dtype`.
        `trace_name` only labels the op in traces; it is defaulted and no
        caller sets it through MOGG's `parameters` dict, so the mxfp8
        registration overrides it directly at the Mojo call site.

        This function launches the shared `fused_silu_mx_kernel` to perform the SILU
        operation for all the MLPs in the EP MoE module.

        This kernel will read the row offsets to determine the actual number of
        received tokens in the input tensor, and then only perform the SILU
        operation on the received tokens. Once the SILU operation is performed,
        the output will be quantized to the MXFP4 or MXFP8 format.

        The kernel writes the E8M0 scale directly into the grouped matmul's
        per-expert fixed-stride `scale_4d` slot layout (slot stride
        `max_padded_M * K_SCALES`; KS64 fusion), so the scales output tensor
        has shape `[n_local_experts * max_padded_M, K_SCALES]`.

        The gated activation is built at compile time from the `activation`,
        `swiglu_alpha` and `swiglu_limit` parameters.
        """
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(scales), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime quant_dtype = type_of(output).dtype
        comptime scales_dtype = type_of(scales).dtype
        comptime input_dtype = type_of(input).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(scales).rank == 2
        comptime assert type_of(input).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        # Ensure this kernel only runs on GPU targets
        comptime assert context.target.is_gpu(), "EP is only supported on GPU."
        comptime assert (
            max_padded_M > 0 and max_padded_M % 32 == 0
        ), "max_padded_M must be a positive multiple of 32"

        var output_tensor = output.to_tile_tensor()
        var scales_tensor = scales.to_tile_tensor()
        var input_tensor = input.to_tile_tensor().as_imm()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            .uint32, mut=False
        ]()

        var gpu_ctx = context
        comptime hw_info = gpu_ctx.default_device_info

        comptime act = get_swiglu_activation[
            activation,
            swiglu_alpha,
            swiglu_limit,
        ]

        comptime fused_silu_mx = fused_silu_mx_kernel[
            quant_dtype,
            scales_dtype,
            input_dtype,
            output_tensor.LayoutType,
            scales_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
            activation=act,
        ]

        @inline(.always)
        def description_fn() {imm} -> String:
            # fmt: off
            return String(
                "quant_dtype=", quant_dtype,
                ";scales_dtype=", scales_dtype,
                ";input_dtype=", input_dtype,
                ";activation=", activation,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            trace_name,
            Trace[TraceLevel.OP]._get_detail_str(description_fn),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu_mx](
                output_tensor,
                scales_tensor,
                input_tensor,
                row_offsets_tensor,
                Int32(max_padded_M),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )


@extensibility.register("ep.fused_silu.mxfp8")
struct Struct_ep_fused_silu_mxfp8:
    """Registers the `ep.fused_silu.mxfp8` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        *,
        max_padded_M: Int,
        activation: StaticString = "silu",
        swiglu_alpha: Float32 = 0.0,
        swiglu_limit: Float32 = 0.0,
    ](
        output: Some[Output & Tensor],
        scales: Some[Output & Tensor],
        input: Some[Input & Tensor],
        row_offsets: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the EP fused SILU kernel with MXFP8 quantization.

        Same body as `ep.fused_silu.mxfp4`: `fused_silu_mx_kernel` takes its
        element packing from the output dtype, so one `fp8_e4m3fn` byte per
        element here rather than two FP4 nibbles, leaving `output` at the full
        hidden size along axis 1. `scales` is shaped
        `[n_local_experts * max_padded_M, K_SCALES]`.
        """
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(scales), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(scales).rank == 2
        comptime assert type_of(input).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        Struct_ep_fused_silu_mxfp4.execute[
            max_padded_M=max_padded_M,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            trace_name="ep.fused_silu.mxfp8",
        ](output, scales, input, row_offsets, context)


@extensibility.register("ep.fused_silu.mxfp6")
struct Struct_ep_fused_silu_mxfp6:
    """Registers the `ep.fused_silu.mxfp6` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        *,
        max_padded_M: Int,
        FP6_FORMAT: Int = 0,
        activation: StaticString = "silu",
        swiglu_alpha: Float32 = 0.0,
        swiglu_limit: Float32 = 0.0,
    ](
        output: Some[Output & Tensor],
        scales: Some[Output & Tensor],
        input: Some[Input & Tensor],
        row_offsets: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        """Execute the EP fused SILU kernel with MXFP6 quantization.

        Unlike `ep.fused_silu.mxfp4` and `.mxfp8`, which share one body that
        switches on elements-per-byte, FP6 needs its own kernel: four codes per
        three bytes is a ratio of 4/3, which that integer cannot represent.
        `output` is packed uint8 at three quarters of the hidden size.

        `FP6_FORMAT` selects the element encoding (0 = E2M3, 1 = E3M2). Both
        occupy six bits and pack identically, so nothing downstream can recover
        it from the bytes -- it must match what the checkpoint declares.

        The gated activation is built at compile time from the `activation`,
        `swiglu_alpha` and `swiglu_limit` parameters.
        """
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(scales), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime scales_dtype = type_of(scales).dtype
        comptime input_dtype = type_of(input).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(scales).rank == 2
        comptime assert type_of(input).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        comptime assert context.target.is_gpu(), "EP is only supported on GPU."
        comptime assert (
            max_padded_M > 0 and max_padded_M % 32 == 0
        ), "max_padded_M must be a positive multiple of 32"

        var output_tensor = output.to_tile_tensor[.uint8]()
        var scales_tensor = scales.to_tile_tensor()
        var input_tensor = input.to_tile_tensor().as_imm()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            .uint32, mut=False
        ]()

        var gpu_ctx = context
        comptime hw_info = gpu_ctx.default_device_info

        comptime act = get_swiglu_activation[
            activation,
            swiglu_alpha,
            swiglu_limit,
        ]

        comptime fused_silu_fp6 = fused_silu_mxfp6_kernel[
            scales_dtype,
            input_dtype,
            output_tensor.LayoutType,
            scales_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
            fp6_format=FP6Format(FP6_FORMAT),
            activation=act,
        ]

        @inline(.always)
        def description_fn() {imm} -> String:
            # fmt: off
            return String(
                "scales_dtype=", scales_dtype,
                ";input_dtype=", input_dtype,
                ";FP6_FORMAT=", FP6_FORMAT,
                ";activation=", activation,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "ep.fused_silu.mxfp6",
            Trace[TraceLevel.OP]._get_detail_str(description_fn),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu_fp6](
                output_tensor,
                scales_tensor,
                input_tensor,
                row_offsets_tensor,
                Int32(max_padded_M),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )


@extensibility.register("ep.fused_silu.nvfp4")
struct Struct_ep_fused_silu_nvfp4:
    """Registers the `ep.fused_silu.nvfp4` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        activation: StaticString = "silu",
        swiglu_alpha: Float32 = 0.0,
        swiglu_limit: Float32 = 0.0,
    ](
        output: Some[Output & Tensor],
        scales: Some[Output & Tensor],
        input: Some[Input & Tensor],
        row_offsets: Some[Input & Tensor],
        scales_offsets: Some[Input & Tensor],
        input_scales: Some[Input & Tensor],
        context: DeviceContext,
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

        The gated activation is built at compile time from the `activation`,
        `swiglu_alpha` and `swiglu_limit` parameters.
        """
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(scales), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(scales_offsets), TileTensorable)
        comptime assert conforms_to(type_of(input_scales), TileTensorable)
        comptime fp4_dtype = type_of(output).dtype
        comptime scales_dtype = type_of(scales).dtype
        comptime input_dtype = type_of(input).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(scales).rank == 5
        comptime assert type_of(input).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        comptime assert type_of(scales_offsets).rank == 1
        comptime assert type_of(input_scales).rank == 1
        # Ensure this kernel only runs on GPU targets
        comptime assert context.target.is_gpu(), "EP is only supported on GPU."

        var output_tensor = output.to_tile_tensor()
        var scales_tensor = scales.to_tile_tensor()
        var input_tensor = input.to_tile_tensor().as_imm()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            .uint32, mut=False
        ]()
        var scales_offsets_tensor = scales_offsets.to_tile_tensor[
            .uint32, mut=False
        ]()
        var input_scales_tensor = input_scales.to_tile_tensor[
            .float32, mut=False
        ]()

        var gpu_ctx = context
        comptime hw_info = gpu_ctx.default_device_info

        comptime act = get_swiglu_activation[
            activation,
            swiglu_alpha,
            swiglu_limit,
        ]

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
            activation=act,
            OutputEngine=output_tensor.Engine,
            ScalesEngine=scales_tensor.Engine,
            InputEngine=input_tensor.Engine,
            RowOffsetsEngine=row_offsets_tensor.Engine,
            ScalesOffsetsEngine=scales_offsets_tensor.Engine,
            InputScalesEngine=input_scales_tensor.Engine,
        ]

        @inline(.always)
        def description_fn() {imm} -> String:
            # fmt: off
            return String(
                "fp4_dtype=", fp4_dtype,
                ";scales_dtype=", scales_dtype,
                ";input_dtype=", input_dtype,
                ";activation=", activation,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "ep.fused_silu.nvfp4",
            Trace[TraceLevel.OP]._get_detail_str(description_fn),
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

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
Helper functions for Expert Parallelism (EP) Communication Kernels.
"""

from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import is_gpu
from std.math import ceildiv
from layout import DefaultEngine, TensorEngine, TensorLayout, TileTensor
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id
from std.sys.info import simd_width_of, size_of
from std.ffi import external_call, _get_global_or_null

from shmem import shmem_module_init, shmem_my_pe
from shmem.ep_comm import (
    EPLocalSyncCounters,
    TokenFormat,
    dispatch_async_kernel,
    dispatch_wait_kernel,
    dispatch_kernel,
    combine_async_kernel,
    combine_wait_kernel,
    combine_kernel,
    EPCombineStoreFn,
    RouterWeightsFn,
    InputScalesFn,
)


@inline(.always)
def global_cache_insert(key: String, value: OpaquePointer[mut=True, _]):
    """Inserts a pointer into the process-wide compiler runtime global cache.

    Stores `value` under `key` in the KGEN compiler runtime's global
    dictionary. Subsequent calls with the same `key` can retrieve the pointer
    via `_get_global_or_null`. Used to cache lazily-initialized resources
    (such as EP communicator handles) across kernel invocations without
    re-initialization overhead.

    Args:
        key: The string key to store `value` under.
        value: An opaque pointer to the resource to cache.
    """
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(key),
        value,
    )


@inline(.always)
def _ep_dispatch_wait_grid_dim(
    num_tokens: Int,
    n_local_experts: Int,
    sm_count: Int,
) -> Int:
    """Decode-fast-path grid sizing for dispatch_wait.

    `copy_received_tokens_to_output` assigns one local expert per SM via
    `local_expert_id = umod(sm_id, n_local_experts)`, so the active comm-SM
    count must be at least `n_local_experts` for correctness. We scale beyond
    that floor with `num_tokens` so prefill batches still get full parallelism.
    """
    # comm SMs >= n_local_experts (correctness); plus 1 aux SM.
    return min(n_local_experts + max(num_tokens, 0) + 1, sm_count)


@inline(.always)
def _ep_combine_wait_grid_dim(
    num_tokens: Int,
    n_chunks_per_tok: Int,
    n_warps_per_block: Int,
    sm_count: Int,
) -> Int:
    """Decode-fast-path grid sizing for combine_wait.

    `reduce_and_copy_to_output` distributes `num_tokens * n_chunks_per_tok`
    chunks across `n_warps_per_block * (grid_dim.x - 1)` warps. Size the grid
    so each warp handles ~1 chunk in the small-batch case and we cap at the
    hardware SM count for large batches. The `+1` accounts for the aux SM.
    """
    var work_units = max(num_tokens * n_chunks_per_tok, 1)
    var reduce_sms = ceildiv(work_units, max(n_warps_per_block, 1))
    return min(reduce_sms + 1, sm_count)


@inline(.always)
def pack_ptrs_array[
    ptrs_layout: TensorLayout,
    //,
    ptr_type: DType,
    local_rank_only: Bool = False,
    n_gpus_per_node: Int = 1 if local_rank_only else ptrs_layout.static_shape[
        0
    ],
](
    _ptrs: TileTensor[.uint64, ptrs_layout, ...],
    my_rank: Int32,
    out result: Array[
        MutPointer[Scalar[ptr_type], MutUntrackedOrigin], n_gpus_per_node
    ],
):
    """Pack the pointers into an inline array.

    Reads device addresses from `_ptrs` and produces an `Array` of
    `Pointer[Scalar[ptr_type]]` entries to pass to an EP kernel. When
    `local_rank_only` is set, every entry is filled with the address at
    `my_rank` instead of one address per rank.

    Parameters:
        ptrs_layout: Layout of the `_ptrs` input tensor (inferred).
        ptr_type: Element `DType` the packed pointers point to.
        local_rank_only: Whether to replicate only the local rank's pointer
            across all entries (defaults to `False`).
        n_gpus_per_node: Number of entries in the output array, one per local
            GPU (defaults to `1` when `local_rank_only` is set, otherwise
            `ptrs_layout.static_shape[0]`).

    Args:
        _ptrs: 1D `uint64` tensor of device addresses, one per rank.
        my_rank: Rank index of the calling device within the communicator.
    """
    comptime assert _ptrs.flat_rank == 1, "Pointers must be a 1D tensor."

    @inline(.always)
    def ptr_arr_init(
        i: Int,
    ) {imm} -> MutPointer[Scalar[ptr_type], MutUntrackedOrigin]:
        comptime if local_rank_only:
            return MutPointer[Scalar[ptr_type], MutUntrackedOrigin](
                unsafe_from_address=Int(_ptrs[my_rank])
            )
        else:
            return MutPointer[Scalar[ptr_type], MutUntrackedOrigin](
                unsafe_from_address=Int(_ptrs[i])
            )

    var ptr_arr = Array[_, n_gpus_per_node](fill_with=ptr_arr_init)

    return ptr_arr^


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Async Dispatch
# ===-----------------------------------------------------------------------===#


@inline(.always)
def ep_dispatch_async_kernel_api[
    AtomicCountersEngine: TensorEngine,
    InputTokensEngine: TensorEngine,
    TopkIdsEngine: TensorEngine,
    SendPtrsEngine: TensorEngine,
    RecvPtrsEngine: TensorEngine,
    RecvCountPtrsEngine: TensorEngine,
    //,
    token_fmt_type: TokenFormat,
    n_experts: Int,
    max_token_per_rank: Int,
    n_gpus_per_node: Int,
    n_nodes: Int,
    target: StaticString,
    use_shmem: Bool = (n_nodes > 1),
](
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    input_tokens: TileTensor[mut=False, Engine=InputTokensEngine, ...],
    topk_ids: TileTensor[mut=False, .int32, Engine=TopkIdsEngine, ...],
    send_ptrs: TileTensor[.uint64, Engine=SendPtrsEngine, ...],
    recv_ptrs: TileTensor[.uint64, Engine=RecvPtrsEngine, ...],
    recv_count_ptrs: TileTensor[.uint64, Engine=RecvCountPtrsEngine, ...],
    context: DeviceContext,
) raises:
    """Execute the Expert Parallelism async dispatch kernel at unit input scale.

    See the overload taking `input_scales_fn`.

    Parameters:
        AtomicCountersEngine: Storage engine backing the `atomic_counters` tile.
        InputTokensEngine: Storage engine backing the `input_tokens` tile.
        TopkIdsEngine: Storage engine backing the `topk_ids` tile.
        SendPtrsEngine: Storage engine backing the `send_ptrs` tile.
        RecvPtrsEngine: Storage engine backing the `recv_ptrs` tile.
        RecvCountPtrsEngine: Storage engine backing the `recv_count_ptrs` tile.
        token_fmt_type: Token format type.
        n_experts: Total experts across all devices.
        max_token_per_rank: Maximum tokens any device can send.
        n_gpus_per_node: GPUs per physical node.
        n_nodes: Number of physical nodes.
        target: Target.
        use_shmem: Whether to enable SHMEM communication.

    Arguments:
        atomic_counters: EP kernel synchronization counters.
        input_tokens: Tokens to dispatch to experts.
        topk_ids: Expert assignments from router.
        send_ptrs: Send buffer pointers for each local GPU.
        recv_ptrs: Receive buffer pointers for each local GPU.
        recv_count_ptrs: Receive count buffer pointers for each local GPU.
        context: Device context pointer
    """

    # Stands in for the input scales `has_input_scales=False` never reads.
    @inline(.always)
    def no_input_scales[dtype: DType](expert_id: Int) -> Scalar[dtype]:
        return Scalar[dtype](1)

    ep_dispatch_async_kernel_api[
        token_fmt_type,
        n_experts,
        max_token_per_rank,
        n_gpus_per_node,
        n_nodes,
        target,
        use_shmem,
        has_input_scales=False,
    ](
        atomic_counters,
        input_tokens,
        topk_ids,
        send_ptrs,
        recv_ptrs,
        recv_count_ptrs,
        context,
        no_input_scales,
    )


@inline(.always)
def ep_dispatch_async_kernel_api[
    AtomicCountersEngine: TensorEngine,
    InputTokensEngine: TensorEngine,
    TopkIdsEngine: TensorEngine,
    SendPtrsEngine: TensorEngine,
    RecvPtrsEngine: TensorEngine,
    RecvCountPtrsEngine: TensorEngine,
    InputScalesFnType: InputScalesFn,
    //,
    token_fmt_type: TokenFormat,
    n_experts: Int,
    max_token_per_rank: Int,
    n_gpus_per_node: Int,
    n_nodes: Int,
    target: StaticString,
    use_shmem: Bool = (n_nodes > 1),
    has_input_scales: Bool = True,
](
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    input_tokens: TileTensor[mut=False, Engine=InputTokensEngine, ...],
    topk_ids: TileTensor[mut=False, .int32, Engine=TopkIdsEngine, ...],
    send_ptrs: TileTensor[.uint64, Engine=SendPtrsEngine, ...],
    recv_ptrs: TileTensor[.uint64, Engine=RecvPtrsEngine, ...],
    recv_count_ptrs: TileTensor[.uint64, Engine=RecvCountPtrsEngine, ...],
    context: DeviceContext,
    input_scales_fn: InputScalesFnType,
) raises:
    """Execute the Expert Parallelism async dispatch kernel.

    This function launches the dispatch_async_kernel from ep_comm.mojo to
    initiate token distribution across expert devices. In multi-node
    scenarios, all the communication buffers need to be allocated using
    `shmem_malloc`.

    Parameters:
        AtomicCountersEngine: Storage engine backing the `atomic_counters` tile.
        InputTokensEngine: Storage engine backing the `input_tokens` tile.
        TopkIdsEngine: Storage engine backing the `topk_ids` tile.
        SendPtrsEngine: Storage engine backing the `send_ptrs` tile.
        RecvPtrsEngine: Storage engine backing the `recv_ptrs` tile.
        RecvCountPtrsEngine: Storage engine backing the `recv_count_ptrs` tile.
        InputScalesFnType: The type of `input_scales_fn` (inferred).
        token_fmt_type: Token format type.
        n_experts: Total experts across all devices.
        max_token_per_rank: Maximum tokens any device can send.
        n_gpus_per_node: GPUs per physical node.
        n_nodes: Number of physical nodes.
        target: Target.
        use_shmem: Whether to enable SHMEM communication.
        has_input_scales: Whether `input_scales_fn` supplies the input scale.
            The overload without `input_scales_fn` passes `False`.

    Arguments:
        atomic_counters: EP kernel synchronization counters.
        input_tokens: Tokens to dispatch to experts.
        topk_ids: Expert assignments from router.
        send_ptrs: Send buffer pointers for each local GPU.
        recv_ptrs: Receive buffer pointers for each local GPU.
        recv_count_ptrs: Receive count buffer pointers for each local GPU.
        context: Device context pointer
        input_scales_fn: The input scales. It runs on the device, so it must
            copy what it captures.
    """

    comptime assert is_gpu[target](), "EP is only supported on GPU."
    comptime assert (
        input_tokens.static_shape[1] == token_fmt_type.hid_dim
    ), "EP dispatch: input tokens shape doesn't match hidden size."
    comptime assert (
        topk_ids.static_shape[1] == token_fmt_type.top_k
    ), "EP dispatch: topk ids shape doesn't match top k."
    comptime assert (
        send_ptrs.flat_rank == 1
    ), "Send pointers must be a 1D tensor."

    debug_assert[assert_mode="safe"](
        Int(input_tokens.dim(0)) <= max_token_per_rank,
        "Cannot dispatch EP kernel with ",
        input_tokens.dim(0),
        " input tokens when the maximum tokens per rank is ",
        max_token_per_rank,
    )

    var gpu_ctx = context

    comptime n_ranks = n_gpus_per_node * n_nodes
    comptime hw_info = gpu_ctx.default_device_info

    var gpu_id = Int(gpu_ctx.id())
    var my_rank = Int32(gpu_id)

    comptime if n_nodes > 1:
        my_rank = Int32(shmem_my_pe())

    comptime dispatch_async = dispatch_async_kernel[
        input_tokens.dtype,
        hw_info.max_thread_block_size,
        input_tokens.LayoutType,
        topk_ids.LayoutType,
        hw_info.sm_count,
        n_experts,
        n_ranks,
        max_token_per_rank,
        n_gpus_per_node,
        token_fmt_type,
        InputScalesFnType=InputScalesFnType,
        has_input_scales=has_input_scales,
        use_shmem=use_shmem,
    ]

    @inline(.always)
    def description_fn() {imm} -> String:
        # fmt: off
        return String(
            "input_dtype=", input_tokens.dtype,
            "token_transfer_fmt=", token_fmt_type.get_type_name(),
            "n_experts=", n_experts,
            "max_token_per_rank=", max_token_per_rank,
            "n_gpus_per_node=", n_gpus_per_node,
            "n_nodes=", n_nodes,
            "my_rank=", my_rank,
        )
        # fmt: on

    with Trace[TraceLevel.OP, target=target](
        "ep.dispatch_async",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(context),
    ):
        var func = gpu_ctx.compile_function[dispatch_async]()

        comptime if use_shmem:
            var cached_module_key = String(t"EP_DISPATCH_INITED_DEV_{gpu_id}")

            # Don't initialize the module repeatedly
            if not _get_global_or_null(cached_module_key):
                shmem_module_init(func)
                global_cache_insert(
                    cached_module_key,
                    MutPointer[NoneType, MutUntrackedOrigin](
                        unsafe_from_address=1
                    ),
                )

        var send_ptr = MutPointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(send_ptrs[gpu_id])
        )
        var recv_ptrs_arr = pack_ptrs_array[.uint8](recv_ptrs, my_rank)
        var recv_count_ptrs_arr = pack_ptrs_array[.uint64](
            recv_count_ptrs, my_rank
        )
        var ep_counters = EPLocalSyncCounters[n_experts](
            MutPointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=Int(atomic_counters.ptr)
            )
        )

        func._call_with_pack_checked(
            gpu_ctx,
            input_tokens,
            topk_ids,
            send_ptr,
            recv_ptrs_arr,
            recv_count_ptrs_arr,
            ep_counters,
            my_rank,
            host_args=(input_scales_fn,),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Dispatch Wait
# ===-----------------------------------------------------------------------===#


@inline(.always)
def ep_dispatch_wait_kernel_api[
    token_fmt_type: TokenFormat,
    RowOffsetsEngine: TensorEngine,
    ExpertIdsEngine: TensorEngine,
    SrcInfoEngine: TensorEngine,
    RecvPtrsEngine: TensorEngine,
    RecvCountPtrsEngine: TensorEngine,
    AtomicCountersEngine: TensorEngine,
    //,
    n_experts: Int,
    max_token_per_rank: Int,
    n_gpus_per_node: Int,
    n_nodes: Int,
    target: StaticString,
](
    token_handler: token_fmt_type,
    row_offsets: TileTensor[.uint32, Engine=RowOffsetsEngine, ...],
    expert_ids: TileTensor[.int32, Engine=ExpertIdsEngine, ...],
    src_info: TileTensor[.int32, Engine=SrcInfoEngine, ...],
    recv_ptrs: TileTensor[.uint64, Engine=RecvPtrsEngine, ...],
    recv_count_ptrs: TileTensor[.uint64, Engine=RecvCountPtrsEngine, ...],
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    context: DeviceContext,
    num_input_tokens: Int = -1,
) raises:
    """Execute the Expert Parallelism dispatch completion kernel.

    This function launches the dispatch_wait_kernel from ep_comm.mojo to
    complete the token dispatch phase. It waits for all inter-device
    communication to complete, then organizes the received tokens into a
    format suitable for grouped matmul computation.

    Parameters:
        token_fmt_type: Token format type.
        RowOffsetsEngine: Storage engine backing the `row_offsets` tile.
        ExpertIdsEngine: Storage engine backing the `expert_ids` tile.
        SrcInfoEngine: Storage engine backing the `src_info` tile.
        RecvPtrsEngine: Storage engine backing the `recv_ptrs` tile.
        RecvCountPtrsEngine: Storage engine backing the `recv_count_ptrs` tile.
        AtomicCountersEngine: Storage engine backing the `atomic_counters` tile.
        n_experts: Total experts across all devices.
        max_token_per_rank: Maximum tokens any device can send.
        n_gpus_per_node: GPUs per physical node.
        n_nodes: Number of physical nodes.
        target: Target.

    Arguments:
        token_handler: Token handler. Wrapper for the output token tensor.
        row_offsets: Cumulative token counts for grouped matmul.
        expert_ids: Local expert IDs for grouped matmul.
        src_info: Source routing information for combine phase.
        recv_ptrs: Receive buffer pointers for each local GPU.
        recv_count_ptrs: Receive count buffer pointers for each local GPU.
        atomic_counters: EP kernel synchronization counters.
        context: Device context pointer.
        num_input_tokens: Per-rank input token count for this layer. When >= 0
            enables the decode-fast-path grid sizing. Default `-1` keeps the
            full-`sm_count` grid for backwards compatibility.
    """
    comptime n_local_experts = n_experts // (n_gpus_per_node * n_nodes)

    # Ensure this kernel only runs on GPU targets
    comptime assert is_gpu[target](), "EP is only supported on GPU."
    comptime assert (
        recv_ptrs.flat_rank == 1
    ), "Receive pointers must be a 1D tensor."
    comptime assert (
        recv_count_ptrs.flat_rank == 1
    ), "Receive count pointers must be a 1D tensor."

    var gpu_ctx = context
    var gpu_id = Int(gpu_ctx.id())
    var my_rank = Int32(gpu_id)

    comptime if n_nodes > 1:
        my_rank = Int32(shmem_my_pe())

    comptime n_ranks = n_gpus_per_node * n_nodes
    comptime hw_info = gpu_ctx.default_device_info

    comptime dispatch_wait = dispatch_wait_kernel[
        hw_info.max_thread_block_size,
        row_offsets.LayoutType,
        expert_ids.LayoutType,
        src_info.LayoutType,
        hw_info.sm_count,
        n_experts,
        n_ranks,
        max_token_per_rank,
        token_fmt_type,
    ]

    @inline(.always)
    def description_fn() {imm} -> String:
        # fmt: off
        return String(
            "token_fmt_type=", token_fmt_type.get_type_name(),
            "n_experts=", n_experts,
            "max_token_per_rank=", max_token_per_rank,
            "n_gpus_per_node=", n_gpus_per_node,
            "n_nodes=", n_nodes,
            "my_rank=", my_rank,
        )
        # fmt: on

    with Trace[TraceLevel.OP, target=target](
        "ep.dispatch_wait",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(context),
    ):
        var recv_buf_ptr = MutPointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(recv_ptrs[gpu_id])
        )
        var recv_count_ptr = MutPointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=Int(recv_count_ptrs[gpu_id])
        )
        var ep_counters = EPLocalSyncCounters[n_experts](
            MutPointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=Int(atomic_counters.ptr)
            )
        )

        var smem_size = UInt32(token_fmt_type.dispatch_smem_size)
        var grid_dim_x: Int
        comptime if gpu_ctx.target.is_amd_gpu():
            if num_input_tokens >= 0:
                grid_dim_x = _ep_dispatch_wait_grid_dim(
                    num_input_tokens, n_local_experts, hw_info.sm_count
                )
            else:
                grid_dim_x = hw_info.sm_count
        else:
            grid_dim_x = hw_info.sm_count
        gpu_ctx.enqueue_function[dispatch_wait](
            token_handler,
            row_offsets,
            expert_ids,
            src_info,
            recv_buf_ptr,
            recv_count_ptr,
            ep_counters,
            my_rank,
            grid_dim=grid_dim_x,
            block_dim=hw_info.max_thread_block_size,
            shared_mem_bytes=Int(smem_size),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                smem_size
            ),
        )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Fused Dispatch
# ===-----------------------------------------------------------------------===#


@inline(.always)
def ep_fused_dispatch_kernel_api[
    token_fmt_type: TokenFormat,
    dispatch_dtype: DType,
    RowOffsetsEngine: TensorEngine,
    ExpertIdsEngine: TensorEngine,
    SrcInfoEngine: TensorEngine,
    AtomicCountersEngine: TensorEngine,
    InputTokensEngine: TensorEngine,
    TopkIdsEngine: TensorEngine,
    SendPtrsEngine: TensorEngine,
    RecvPtrsEngine: TensorEngine,
    RecvCountPtrsEngine: TensorEngine,
    //,
    n_experts: Int,
    max_token_per_rank: Int,
    n_gpus_per_node: Int,
    n_nodes: Int,
    fused_shared_expert: Bool,
    target: StaticString,
    skip_a2a: Bool = False,
    use_shmem: Bool = (n_nodes > 1),
    allreduce_world_size: Int = 1,
    dispatch_warp_min_tokens: Int = -1,
](
    token_handler: token_fmt_type,
    row_offsets: TileTensor[.uint32, Engine=RowOffsetsEngine, ...],
    expert_ids: TileTensor[.int32, Engine=ExpertIdsEngine, ...],
    src_info: TileTensor[.int32, Engine=SrcInfoEngine, ...],
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    input_tokens: TileTensor[
        mut=False, dispatch_dtype, Engine=InputTokensEngine, ...
    ],
    topk_ids: TileTensor[mut=False, .int32, Engine=TopkIdsEngine, ...],
    send_ptrs: TileTensor[.uint64, Engine=SendPtrsEngine, ...],
    recv_ptrs: TileTensor[.uint64, Engine=RecvPtrsEngine, ...],
    recv_count_ptrs: TileTensor[.uint64, Engine=RecvCountPtrsEngine, ...],
    context: DeviceContext,
) raises:
    """Execute the fused Expert Parallelism dispatch kernel at unit input scale.

    See the overload taking `input_scales_fn`.

    Parameters:
        token_fmt_type: Token format type.
        dispatch_dtype: Data type of the dispatched tokens.
        RowOffsetsEngine: Storage engine backing the `row_offsets` tile.
        ExpertIdsEngine: Storage engine backing the `expert_ids` tile.
        SrcInfoEngine: Storage engine backing the `src_info` tile.
        AtomicCountersEngine: Storage engine backing the `atomic_counters` tile.
        InputTokensEngine: Storage engine backing the `input_tokens` tile.
        TopkIdsEngine: Storage engine backing the `topk_ids` tile.
        SendPtrsEngine: Storage engine backing the `send_ptrs` tile.
        RecvPtrsEngine: Storage engine backing the `recv_ptrs` tile.
        RecvCountPtrsEngine: Storage engine backing the `recv_count_ptrs` tile.
        n_experts: Total experts across all devices.
        max_token_per_rank: Maximum tokens any device can send.
        n_gpus_per_node: GPUs per physical node.
        n_nodes: Number of physical nodes.
        fused_shared_expert: Whether to pack shared expert inputs with
            routed experts' inputs.
        target: Target.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            send tokens within the current device.
        use_shmem: Whether to enable SHMEM communication.
        allreduce_world_size: The world size of the allreduce operation. Only
            needed for skip_a2a. Used to calculate the workload distribution for
            the shared expert (if has one).
        dispatch_warp_min_tokens: From this many tokens a rank on, send
            one token per warp instead of one per CTA, and count each
            expert's rows with one shared-memory histogram pass at any count.
            Same bytes and receive layout. -1 (the default) keeps the stock
            send and count, as does every configuration but single-node NVFP4
            without a fused shared expert.

    Arguments:
        token_handler: Token handler. Wrapper for the output token tensor.
        row_offsets: Row offsets for grouped matmul.
        expert_ids: Expert IDs for grouped matmul.
        src_info: Source routing information for combine phase.
        atomic_counters: EP kernel synchronization counters.
        input_tokens: Tokens to dispatch to experts.
        topk_ids: Expert assignments from router.
        send_ptrs: Send buffer pointers for each local GPU.
        recv_ptrs: Receive buffer pointers for each local GPU.
        recv_count_ptrs: Receive count buffer pointers for each local GPU.
        context: Device context pointer.
    """

    # Stands in for the input scales `has_input_scales=False` never reads.
    @inline(.always)
    def no_input_scales[dtype: DType](expert_id: Int) -> Scalar[dtype]:
        return Scalar[dtype](1)

    ep_fused_dispatch_kernel_api[
        n_experts,
        max_token_per_rank,
        n_gpus_per_node,
        n_nodes,
        fused_shared_expert,
        target,
        skip_a2a,
        use_shmem,
        allreduce_world_size,
        has_input_scales=False,
        dispatch_warp_min_tokens=dispatch_warp_min_tokens,
    ](
        token_handler,
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
        no_input_scales,
    )


@inline(.always)
def ep_fused_dispatch_kernel_api[
    token_fmt_type: TokenFormat,
    dispatch_dtype: DType,
    RowOffsetsEngine: TensorEngine,
    ExpertIdsEngine: TensorEngine,
    SrcInfoEngine: TensorEngine,
    AtomicCountersEngine: TensorEngine,
    InputTokensEngine: TensorEngine,
    TopkIdsEngine: TensorEngine,
    SendPtrsEngine: TensorEngine,
    RecvPtrsEngine: TensorEngine,
    RecvCountPtrsEngine: TensorEngine,
    InputScalesFnType: InputScalesFn,
    //,
    n_experts: Int,
    max_token_per_rank: Int,
    n_gpus_per_node: Int,
    n_nodes: Int,
    fused_shared_expert: Bool,
    target: StaticString,
    skip_a2a: Bool = False,
    use_shmem: Bool = (n_nodes > 1),
    allreduce_world_size: Int = 1,
    has_input_scales: Bool = True,
    dispatch_warp_min_tokens: Int = -1,
](
    token_handler: token_fmt_type,
    row_offsets: TileTensor[.uint32, Engine=RowOffsetsEngine, ...],
    expert_ids: TileTensor[.int32, Engine=ExpertIdsEngine, ...],
    src_info: TileTensor[.int32, Engine=SrcInfoEngine, ...],
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    input_tokens: TileTensor[
        mut=False, dispatch_dtype, Engine=InputTokensEngine, ...
    ],
    topk_ids: TileTensor[mut=False, .int32, Engine=TopkIdsEngine, ...],
    send_ptrs: TileTensor[.uint64, Engine=SendPtrsEngine, ...],
    recv_ptrs: TileTensor[.uint64, Engine=RecvPtrsEngine, ...],
    recv_count_ptrs: TileTensor[.uint64, Engine=RecvCountPtrsEngine, ...],
    context: DeviceContext,
    input_scales_fn: InputScalesFnType,
) raises:
    """Execute the fused Expert Parallelism dispatch kernel.

    This function launches the fused dispatch_kernel from ep_comm.mojo that
    combines both dispatch_async and dispatch_wait functionality in a single
    kernel launch. It distributes input tokens to expert devices based on
    top-k routing decisions, then waits for all tokens to arrive and
    aggregates them for grouped matmul computation.

    Parameters:
        token_fmt_type: Token format type.
        dispatch_dtype: Data type of the dispatched tokens.
        RowOffsetsEngine: Storage engine backing the `row_offsets` tile.
        ExpertIdsEngine: Storage engine backing the `expert_ids` tile.
        SrcInfoEngine: Storage engine backing the `src_info` tile.
        AtomicCountersEngine: Storage engine backing the `atomic_counters` tile.
        InputTokensEngine: Storage engine backing the `input_tokens` tile.
        TopkIdsEngine: Storage engine backing the `topk_ids` tile.
        SendPtrsEngine: Storage engine backing the `send_ptrs` tile.
        RecvPtrsEngine: Storage engine backing the `recv_ptrs` tile.
        RecvCountPtrsEngine: Storage engine backing the `recv_count_ptrs` tile.
        InputScalesFnType: The type of `input_scales_fn` (inferred).
        n_experts: Total experts across all devices.
        max_token_per_rank: Maximum tokens any device can send.
        n_gpus_per_node: GPUs per physical node.
        n_nodes: Number of physical nodes.
        fused_shared_expert: Whether to pack shared expert inputs with
            routed experts' inputs.
        target: Target.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            send tokens within the current device.
        use_shmem: Whether to enable SHMEM communication.
        allreduce_world_size: The world size of the allreduce operation. Only
            needed for skip_a2a. Used to calculate the workload distribution for
            the shared expert (if has one).
        has_input_scales: Whether `input_scales_fn` supplies the input scale.
            The overload without `input_scales_fn` passes `False`.
        dispatch_warp_min_tokens: From this many tokens a rank on, send
            one token per warp instead of one per CTA, and count each
            expert's rows with one shared-memory histogram pass at any count.
            Same bytes and receive layout. -1 (the default) keeps the stock
            send and count, as does every configuration but single-node NVFP4
            without a fused shared expert.

    Arguments:
        token_handler: Token handler. Wrapper for the output token tensor.
        row_offsets: Row offsets for grouped matmul.
        expert_ids: Expert IDs for grouped matmul.
        src_info: Source routing information for combine phase.
        atomic_counters: EP kernel synchronization counters.
        input_tokens: Tokens to dispatch to experts.
        topk_ids: Expert assignments from router.
        send_ptrs: Send buffer pointers for each local GPU.
        recv_ptrs: Receive buffer pointers for each local GPU.
        recv_count_ptrs: Receive count buffer pointers for each local GPU.
        context: Device context pointer.
        input_scales_fn: The input scales. It runs on the device, so it must
            copy what it captures.
    """

    # Ensure this kernel only runs on GPU targets
    comptime assert is_gpu[target](), "EP is only supported on GPU."
    comptime assert dispatch_dtype == .bfloat16
    comptime assert (
        send_ptrs.flat_rank == 1
    ), "Send pointers must be a 1D tensor."

    # Ensure the shape for the input tensors are correct
    comptime assert (
        input_tokens.static_shape[1] == token_fmt_type.hid_dim
    ), "EP dispatch: input tokens shape doesn't match hidden size."
    comptime assert (
        topk_ids.static_shape[1] == token_fmt_type.top_k
    ), "EP dispatch: topk ids shape doesn't match top k."

    debug_assert[assert_mode="safe"](
        Int(input_tokens.dim(0)) <= max_token_per_rank,
        "Cannot dispatch EP kernel with ",
        input_tokens.dim(0),
        " input tokens when the maximum tokens per rank is ",
        max_token_per_rank,
    )

    var gpu_ctx = context
    var gpu_id = Int(gpu_ctx.id())
    var my_rank = Int32(gpu_id)

    comptime if n_nodes > 1:
        my_rank = Int32(shmem_my_pe())

    comptime hw_info = gpu_ctx.default_device_info
    comptime n_ranks = n_gpus_per_node * n_nodes

    comptime fused_dispatch = dispatch_kernel[
        dispatch_dtype,
        hw_info.max_thread_block_size,
        input_tokens.LayoutType,
        topk_ids.LayoutType,
        row_offsets.LayoutType,
        expert_ids.LayoutType,
        src_info.LayoutType,
        hw_info.sm_count,
        n_experts,
        n_ranks,
        max_token_per_rank,
        n_gpus_per_node,  # p2p world size
        token_fmt_type,
        InputScalesFnType=InputScalesFnType,
        fused_shared_expert=fused_shared_expert,
        has_input_scales=has_input_scales,
        skip_a2a=skip_a2a,
        use_shmem=use_shmem,
        allreduce_world_size=allreduce_world_size,
        warp_min_tokens=dispatch_warp_min_tokens,
        InputTokensEngine=input_tokens.Engine,
        TopkIdsEngine=topk_ids.Engine,
        RowOffsetsEngine=row_offsets.Engine,
        ExpertIdsEngine=expert_ids.Engine,
        SrcInfoEngine=src_info.Engine,
    ]

    @inline(.always)
    def description_fn() {imm} -> String:
        # fmt: off
        return String(
            "token_fmt_type=", token_fmt_type.get_type_name(),
            ";n_experts=", n_experts,
            ";max_token_per_rank=", max_token_per_rank,
            ";n_gpus_per_node=", n_gpus_per_node,
            ";n_nodes=", n_nodes,
            ";my_rank=", my_rank,
        )
        # fmt: on

    with Trace[TraceLevel.OP, target=target](
        "ep.dispatch",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(context),
    ):
        var smem_size = UInt32(token_fmt_type.dispatch_smem_size)
        var func = gpu_ctx.compile_function[fused_dispatch](
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                smem_size
            ),
        )

        comptime if use_shmem:
            var cached_module_key = String(
                "EP_FUSED_DISPATCH_INITED_DEV_", gpu_id
            )

            # Don't initialize the module repeatedly
            if not _get_global_or_null(cached_module_key):
                shmem_module_init(func)
                global_cache_insert(
                    cached_module_key,
                    MutPointer[NoneType, MutUntrackedOrigin](
                        unsafe_from_address=1
                    ),
                )

        var send_ptr = MutPointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(send_ptrs[gpu_id])
        )
        # Create inline arrays to store all the p2p accessible pointers
        var recv_ptrs_arr = pack_ptrs_array[
            DType.uint8, local_rank_only=skip_a2a
        ](recv_ptrs, my_rank)
        var recv_count_ptrs_arr = pack_ptrs_array[
            DType.uint64, local_rank_only=skip_a2a
        ](recv_count_ptrs, my_rank)
        var ep_counters = EPLocalSyncCounters[n_experts](
            MutPointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=Int(atomic_counters.ptr)
            )
        )

        # Fused dispatch grid must satisfy both phases:
        #  - async: each token gets one comm SM, after n_signal_sms aux SMs.
        #  - wait: at least n_local_experts comm SMs + 1 aux SM for correctness.
        comptime n_local_experts = n_experts // (n_gpus_per_node * n_nodes)
        var num_input_tokens = Int(input_tokens.dim(0))
        var grid_dim_x: Int
        comptime if gpu_ctx.target.is_amd_gpu():
            grid_dim_x = _ep_dispatch_wait_grid_dim(
                num_input_tokens, n_local_experts, hw_info.sm_count
            )
        else:
            grid_dim_x = hw_info.sm_count
        func._call_with_pack_checked(
            gpu_ctx,
            input_tokens,
            topk_ids,
            token_handler,
            row_offsets,
            expert_ids,
            src_info,
            send_ptr,
            recv_ptrs_arr,
            recv_count_ptrs_arr,
            ep_counters,
            my_rank,
            host_args=(input_scales_fn,),
            grid_dim=grid_dim_x,
            block_dim=hw_info.max_thread_block_size,
            shared_mem_bytes=Int(smem_size),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Combine Async
# ===-----------------------------------------------------------------------===#


@inline(.always)
def ep_combine_async_kernel_api[
    AtomicCountersEngine: TensorEngine,
    InputTokensEngine: TensorEngine,
    SrcInfoEngine: TensorEngine,
    SendPtrsEngine: TensorEngine,
    RecvPtrsEngine: TensorEngine,
    RecvCountPtrsEngine: TensorEngine,
    //,
    combine_dtype: DType,
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    max_token_per_rank: Int,
    n_gpus_per_node: Int,
    n_nodes: Int,
    target: StaticString,
    use_shmem: Bool = (n_nodes > 1),
](
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    input_tokens: TileTensor[
        mut=False, combine_dtype, Engine=InputTokensEngine, ...
    ],
    src_info: TileTensor[mut=False, .int32, Engine=SrcInfoEngine, ...],
    send_ptrs: TileTensor[.uint64, Engine=SendPtrsEngine, ...],
    recv_ptrs: TileTensor[.uint64, Engine=RecvPtrsEngine, ...],
    recv_count_ptrs: TileTensor[.uint64, Engine=RecvCountPtrsEngine, ...],
    context: DeviceContext,
) raises:
    """Execute the Expert Parallelism combine kernel.

    This function launches the combine_async_kernel from ep_comm.mojo to
    initiate sending expert outputs back to their original devices. The kernel
    uses source routing information to determine destinations. This kernel might
    also filter out the shared expert's outputs and store them in a separate
    tensor. In multi-node scenarios, all the communication buffers need to be
    allocated using `shmem_malloc`.

    Parameters:
        AtomicCountersEngine: Storage engine backing the `atomic_counters` tile.
        InputTokensEngine: Storage engine backing the `input_tokens` tile.
        SrcInfoEngine: Storage engine backing the `src_info` tile.
        SendPtrsEngine: Storage engine backing the `send_ptrs` tile.
        RecvPtrsEngine: Storage engine backing the `recv_ptrs` tile.
        RecvCountPtrsEngine: Storage engine backing the `recv_count_ptrs` tile.
        combine_dtype: Data type for tokens during combine phase.
        hidden_size: Model hidden dimension size.
        top_k: Number of experts each token was routed to.
        n_experts: Total experts across all devices.
        max_token_per_rank: Maximum tokens any device can send.
        n_gpus_per_node: GPUs per physical node.
        n_nodes: Number of physical nodes.
        target: Target.
        use_shmem: Whether to enable SHMEM communication.

    Arguments:
        output_tokens: Output tokens for the shared experts.
        atomic_counters: EP kernel synchronization counters.
            Used to coordinate between different thread blocks.
        input_tokens: Expert output tokens to send back to original devices.
        src_info: Source routing information from dispatch phase.
        send_ptrs: Send buffer pointers for each local GPU.
        recv_ptrs: Receive buffer pointers for each local GPU.
        recv_count_ptrs: Receive count buffer pointers for each local GPU.
        context: Device context pointer.
    """

    # Ensure this kernel only runs on GPU targets
    comptime assert is_gpu[target](), "EP is only supported on GPU."
    comptime assert (
        input_tokens.static_shape[1] == hidden_size
    ), "EP combine: input tokens shape doesn't match hidden size."
    comptime assert (
        send_ptrs.flat_rank == 1
    ), "Send pointers must be a 1D tensor."

    var gpu_ctx = context
    var gpu_id = Int(gpu_ctx.id())
    var my_rank = Int32(gpu_id)

    comptime if n_nodes > 1:
        my_rank = Int32(shmem_my_pe())

    comptime hw_info = gpu_ctx.default_device_info
    comptime combine_msg_size = hidden_size * size_of[combine_dtype]()
    comptime n_ranks = n_gpus_per_node * n_nodes

    comptime combine_async = combine_async_kernel[
        combine_dtype,
        hw_info.max_thread_block_size,
        input_tokens.LayoutType,
        src_info.LayoutType,
        hw_info.sm_count,
        top_k,
        n_experts,
        n_ranks,
        combine_msg_size,
        max_token_per_rank,
        n_gpus_per_node,
        use_shmem=use_shmem,
        InputTokensEngine=input_tokens.Engine,
        SrcInfoEngine=src_info.Engine,
    ]

    @inline(.always)
    def description_fn() {imm} -> String:
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
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(context),
    ):
        var func = gpu_ctx.compile_function[combine_async]()

        comptime if use_shmem:
            var cached_module_key = String(t"EP_COMBINE_INITED_DEV_{gpu_id}")

            # Don't initialize the module repeatedly
            if not _get_global_or_null(cached_module_key):
                shmem_module_init(func)
                global_cache_insert(
                    cached_module_key,
                    MutPointer[NoneType, MutUntrackedOrigin](
                        unsafe_from_address=1
                    ),
                )

        var send_ptr = MutPointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(send_ptrs[gpu_id])
        )
        # Create inline arrays to store all the p2p accessible pointers
        var recv_ptrs_arr = pack_ptrs_array[.uint8](recv_ptrs, my_rank)
        var recv_count_ptrs_arr = pack_ptrs_array[.uint64](
            recv_count_ptrs, my_rank
        )
        var ep_counters = EPLocalSyncCounters[n_experts](
            MutPointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=Int(atomic_counters.ptr)
            )
        )

        gpu_ctx.enqueue_function(
            func,
            input_tokens,
            src_info,
            send_ptr,
            recv_ptrs_arr,
            recv_count_ptrs_arr,
            ep_counters,
            my_rank,
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Combine Wait
# ===-----------------------------------------------------------------------===#


@inline(.always)
def ep_combine_wait_kernel_api[
    AtomicCountersEngine: TensorEngine,
    RecvPtrsEngine: TensorEngine,
    RecvCountPtrsEngine: TensorEngine,
    StoreFnType: EPCombineStoreFn,
    //,
    combine_dtype: DType,
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    max_token_per_rank: Int,
    n_gpus_per_node: Int,
    n_nodes: Int,
    target: StaticString,
](
    num_tokens: Int,
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    recv_ptrs: TileTensor[.uint64, Engine=RecvPtrsEngine, ...],
    recv_count_ptrs: TileTensor[.uint64, Engine=RecvCountPtrsEngine, ...],
    context: DeviceContext,
    store_fn: StoreFnType,
) raises:
    """Execute the Expert Parallelism combine completion kernel, unweighted.

    Each routed expert's output is stored on its own. See the overload taking
    `router_weights_fn`.

    Parameters:
        AtomicCountersEngine: Storage engine backing the `atomic_counters` tile.
        RecvPtrsEngine: Storage engine backing the `recv_ptrs` tile.
        RecvCountPtrsEngine: Storage engine backing the `recv_count_ptrs` tile.
        StoreFnType: The type of `store_fn` (inferred).
        combine_dtype: Data type for tokens during combine phase.
        hidden_size: Model hidden dimension size.
        top_k: Number of experts each token was routed to.
        n_experts: Total experts across all devices.
        max_token_per_rank: Maximum tokens any device can receive.
        n_gpus_per_node: GPUs per physical node.
        n_nodes: Number of physical nodes.
        target: Target.

    Arguments:
        num_tokens: Number of output tokens.
        atomic_counters: EP kernel synchronization counters.
            Used to coordinate between different thread blocks.
        recv_ptrs: Receive buffer pointers for each local GPU.
        recv_count_ptrs: Receive count buffer pointers for each local GPU.
        context: Device context pointer.
        store_fn: Stores the output tokens, called with
            `(token_idx, topk_idx, elem)`. It runs on the device, so it must
            copy what it captures.
    """

    # Stands in for the router weights `has_router_weights=False` never reads.
    @inline(.always)
    def no_router_weights[
        width: Int
    ](token_idx: Int, topk_id: Int) -> SIMD[.float32, width]:
        return SIMD[.float32, width](1)

    ep_combine_wait_kernel_api[
        combine_dtype,
        hidden_size,
        top_k,
        n_experts,
        max_token_per_rank,
        n_gpus_per_node,
        n_nodes,
        target,
        has_router_weights=False,
    ](
        num_tokens,
        atomic_counters,
        recv_ptrs,
        recv_count_ptrs,
        context,
        store_fn,
        no_router_weights,
    )


@inline(.always)
def ep_combine_wait_kernel_api[
    AtomicCountersEngine: TensorEngine,
    RecvPtrsEngine: TensorEngine,
    RecvCountPtrsEngine: TensorEngine,
    StoreFnType: EPCombineStoreFn,
    RouterWeightsFnType: RouterWeightsFn,
    //,
    combine_dtype: DType,
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    max_token_per_rank: Int,
    n_gpus_per_node: Int,
    n_nodes: Int,
    target: StaticString,
    has_router_weights: Bool = True,
](
    num_tokens: Int,
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    recv_ptrs: TileTensor[.uint64, Engine=RecvPtrsEngine, ...],
    recv_count_ptrs: TileTensor[.uint64, Engine=RecvCountPtrsEngine, ...],
    context: DeviceContext,
    store_fn: StoreFnType,
    router_weights_fn: RouterWeightsFnType,
) raises:
    """Execute the Expert Parallelism combine completion kernel.

    This function launches the combine_wait_kernel from ep_comm.mojo to
    complete the token combine phase. It waits for all inter-device
    communication to complete, then computes the weighted sum of routed
    expert outputs for each token.

    Parameters:
        AtomicCountersEngine: Storage engine backing the `atomic_counters` tile.
        RecvPtrsEngine: Storage engine backing the `recv_ptrs` tile.
        RecvCountPtrsEngine: Storage engine backing the `recv_count_ptrs` tile.
        StoreFnType: The type of `store_fn` (inferred).
        RouterWeightsFnType: The type of `router_weights_fn` (inferred).
        combine_dtype: Data type for tokens during combine phase.
        hidden_size: Model hidden dimension size.
        top_k: Number of experts each token was routed to.
        n_experts: Total experts across all devices.
        max_token_per_rank: Maximum tokens any device can receive.
        n_gpus_per_node: GPUs per physical node.
        n_nodes: Number of physical nodes.
        target: Target.
        has_router_weights: Whether the routed expert outputs are weighted by
            `router_weights_fn` and summed. The overload without
            `router_weights_fn` passes `False`.

    Arguments:
        num_tokens: Number of output tokens.
        atomic_counters: EP kernel synchronization counters.
            Used to coordinate between different thread blocks.
        recv_ptrs: Receive buffer pointers for each local GPU.
        recv_count_ptrs: Receive count buffer pointers for each local GPU.
        context: Device context pointer.
        store_fn: Stores the output tokens. Called with `(token_idx, elem)`
            when `has_router_weights`, and with
            `(token_idx, topk_idx, elem)` otherwise. It runs on the device,
            so it must copy what it captures.
        router_weights_fn: The router weights. It runs on the device, so it
            must copy what it captures.
    """

    # Ensure this kernel only runs on GPU targets
    comptime assert is_gpu[target](), "EP is only supported on GPU."
    comptime assert (
        recv_ptrs.flat_rank == 1
    ), "Receive pointers must be a 1D tensor."
    comptime assert (
        recv_count_ptrs.flat_rank == 1
    ), "Receive count pointers must be a 1D tensor."

    var gpu_ctx = context
    var gpu_id = Int(gpu_ctx.id())
    var my_rank = Int32(gpu_id)

    comptime if n_nodes > 1:
        my_rank = Int32(shmem_my_pe())

    comptime hw_info = gpu_ctx.default_device_info
    comptime combine_msg_size = hidden_size * size_of[combine_dtype]()
    comptime n_ranks = n_gpus_per_node * n_nodes

    comptime combine_wait = combine_wait_kernel[
        combine_dtype,
        hw_info.max_thread_block_size,
        hw_info.sm_count,
        top_k,
        n_experts,
        n_ranks,
        combine_msg_size,
        max_token_per_rank,
        StoreFnType,
        RouterWeightsFnType,
        has_router_weights,
    ]

    @inline(.always)
    def description_fn() {imm} -> String:
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
        "ep.combine_wait",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(context),
    ):
        var recv_buf_ptr = MutPointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(recv_ptrs[gpu_id])
        )
        var recv_count_ptr = MutPointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=Int(recv_count_ptrs[gpu_id])
        )
        var ep_counters = EPLocalSyncCounters[n_experts](
            MutPointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=Int(atomic_counters.ptr)
            )
        )

        comptime dst_simd_width = simd_width_of[combine_dtype]()
        comptime n_chunks_per_tok = hidden_size // (
            hw_info.warp_size * dst_simd_width
        )
        comptime n_warps_per_block = (
            hw_info.max_thread_block_size // hw_info.warp_size
        )
        var grid_dim_x: Int
        comptime if gpu_ctx.target.is_amd_gpu():
            grid_dim_x = _ep_combine_wait_grid_dim(
                num_tokens,
                n_chunks_per_tok,
                n_warps_per_block,
                hw_info.sm_count,
            )
        else:
            grid_dim_x = hw_info.sm_count
        gpu_ctx.enqueue_function[combine_wait](
            Int32(num_tokens),
            recv_buf_ptr,
            recv_count_ptr,
            ep_counters,
            my_rank,
            host_args=(store_fn, router_weights_fn),
            grid_dim=grid_dim_x,
            block_dim=hw_info.max_thread_block_size,
        )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Fused Combine
# ===-----------------------------------------------------------------------===#


@inline(.always)
def ep_fused_combine_kernel_api[
    combine_dtype: DType,
    AtomicCountersEngine: TensorEngine,
    InputTokensEngine: TensorEngine,
    SrcInfoEngine: TensorEngine,
    SendPtrsEngine: TensorEngine,
    RecvPtrsEngine: TensorEngine,
    RecvCountPtrsEngine: TensorEngine,
    StoreFnType: EPCombineStoreFn,
    //,
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    max_token_per_rank: Int,
    n_gpus_per_node: Int,
    n_nodes: Int,
    target: StaticString,
    fused_shared_expert: Bool = False,
    skip_a2a: Bool = False,
    use_shmem: Bool = (n_nodes > 1),
    allreduce_world_size: Int = 1,
    *,
    # The output goes through `store_fn`, so there is no output tile to infer
    # the engine from.
    OutputTokensEngine: TensorEngine = DefaultEngine[element_width=1],
](
    num_tokens: Int,
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    input_tokens: TileTensor[
        mut=False, combine_dtype, Engine=InputTokensEngine, ...
    ],
    src_info: TileTensor[mut=False, .int32, Engine=SrcInfoEngine, ...],
    send_ptrs: TileTensor[.uint64, Engine=SendPtrsEngine, ...],
    recv_ptrs: TileTensor[.uint64, Engine=RecvPtrsEngine, ...],
    recv_count_ptrs: TileTensor[.uint64, Engine=RecvCountPtrsEngine, ...],
    context: DeviceContext,
    store_fn: StoreFnType,
    topk_ids_p: Optional[ImmPointer[Int32, ImmUntrackedOrigin]] = None,
) raises:
    """Execute the fused Expert Parallelism combine kernel, unweighted.

    Each routed expert's output is stored on its own. See the overload taking
    `router_weights_fn`.

    Parameters:
        combine_dtype: Data type for tokens during combine phase.
        AtomicCountersEngine: Storage engine backing the `atomic_counters` tile.
        InputTokensEngine: Storage engine backing the `input_tokens` tile.
        SrcInfoEngine: Storage engine backing the `src_info` tile.
        SendPtrsEngine: Storage engine backing the `send_ptrs` tile.
        RecvPtrsEngine: Storage engine backing the `recv_ptrs` tile.
        RecvCountPtrsEngine: Storage engine backing the `recv_count_ptrs` tile.
        StoreFnType: The type of `store_fn` (inferred).
        hidden_size: Model hidden dimension size.
        top_k: Number of experts each token was routed to.
        n_experts: Total experts across all devices.
        max_token_per_rank: Maximum tokens any device can receive.
        n_gpus_per_node: GPUs per physical node.
        n_nodes: Number of physical nodes.
        target: Target.
        fused_shared_expert: Whether to add shared expert outputs to the
            combined routed expert outputs.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            send tokens within the current device.
        use_shmem: Whether to enable SHMEM communication.
        allreduce_world_size: The world size of the allreduce operation. Only
            needed for skip_a2a. Used to calculate the workload distribution for
            the shared expert (if has one).
        OutputTokensEngine: Storage engine backing the output tokens tile.

    Arguments:
        num_tokens: Number of output tokens.
        atomic_counters: EP kernel synchronization counters.
        input_tokens: Expert output tokens to send back.
        src_info: Source routing information from dispatch phase.
        send_ptrs: Send buffer pointers for each local GPU.
        recv_ptrs: Receive buffer pointers for each local GPU.
        recv_count_ptrs: Receive count buffer pointers for each local GPU.
        context: Device context pointer.
        store_fn: Stores the output tokens, called with
            `(token_idx, topk_idx, elem)`. It runs on the device, so it must
            copy what it captures.
        topk_ids_p: Pointer to the top-k IDs for each token, only required if
            skip_a2a is True.
    """

    # Stands in for the router weights `has_router_weights=False` never reads.
    @inline(.always)
    def no_router_weights[
        width: Int
    ](token_idx: Int, topk_id: Int) -> SIMD[.float32, width]:
        return SIMD[.float32, width](1)

    ep_fused_combine_kernel_api[
        hidden_size,
        top_k,
        n_experts,
        max_token_per_rank,
        n_gpus_per_node,
        n_nodes,
        target,
        fused_shared_expert,
        skip_a2a,
        use_shmem,
        allreduce_world_size,
        has_router_weights=False,
        OutputTokensEngine=OutputTokensEngine,
    ](
        num_tokens,
        atomic_counters,
        input_tokens,
        src_info,
        send_ptrs,
        recv_ptrs,
        recv_count_ptrs,
        context,
        store_fn,
        no_router_weights,
        topk_ids_p,
    )


@inline(.always)
def ep_fused_combine_kernel_api[
    combine_dtype: DType,
    AtomicCountersEngine: TensorEngine,
    InputTokensEngine: TensorEngine,
    SrcInfoEngine: TensorEngine,
    SendPtrsEngine: TensorEngine,
    RecvPtrsEngine: TensorEngine,
    RecvCountPtrsEngine: TensorEngine,
    StoreFnType: EPCombineStoreFn,
    RouterWeightsFnType: RouterWeightsFn,
    //,
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    max_token_per_rank: Int,
    n_gpus_per_node: Int,
    n_nodes: Int,
    target: StaticString,
    fused_shared_expert: Bool = False,
    skip_a2a: Bool = False,
    use_shmem: Bool = (n_nodes > 1),
    allreduce_world_size: Int = 1,
    has_router_weights: Bool = True,
    *,
    # The output goes through `store_fn`, so there is no output tile to infer
    # the engine from.
    OutputTokensEngine: TensorEngine = DefaultEngine[element_width=1],
](
    num_tokens: Int,
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    input_tokens: TileTensor[
        mut=False, combine_dtype, Engine=InputTokensEngine, ...
    ],
    src_info: TileTensor[mut=False, .int32, Engine=SrcInfoEngine, ...],
    send_ptrs: TileTensor[.uint64, Engine=SendPtrsEngine, ...],
    recv_ptrs: TileTensor[.uint64, Engine=RecvPtrsEngine, ...],
    recv_count_ptrs: TileTensor[.uint64, Engine=RecvCountPtrsEngine, ...],
    context: DeviceContext,
    store_fn: StoreFnType,
    router_weights_fn: RouterWeightsFnType,
    topk_ids_p: Optional[ImmPointer[Int32, ImmUntrackedOrigin]] = None,
) raises:
    """Execute the fused Expert Parallelism combine kernel.

    This function launches the fused combine_kernel from ep_comm.mojo that
    combines both combine_async and combine_wait functionality in a single
    kernel launch. It sends expert outputs back to their original devices,
    then waits for all transfers to complete and computes the weighted sum
    of routed expert outputs for each token.

    Parameters:
        combine_dtype: Data type for tokens during combine phase.
        AtomicCountersEngine: Storage engine backing the `atomic_counters` tile.
        InputTokensEngine: Storage engine backing the `input_tokens` tile.
        SrcInfoEngine: Storage engine backing the `src_info` tile.
        SendPtrsEngine: Storage engine backing the `send_ptrs` tile.
        RecvPtrsEngine: Storage engine backing the `recv_ptrs` tile.
        RecvCountPtrsEngine: Storage engine backing the `recv_count_ptrs` tile.
        StoreFnType: The type of `store_fn` (inferred).
        RouterWeightsFnType: The type of `router_weights_fn` (inferred).
        hidden_size: Model hidden dimension size.
        top_k: Number of experts each token was routed to.
        n_experts: Total experts across all devices.
        max_token_per_rank: Maximum tokens any device can receive.
        n_gpus_per_node: GPUs per physical node.
        n_nodes: Number of physical nodes.
        target: Target.
        fused_shared_expert: Whether to add shared expert outputs to the
            combined routed expert outputs.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            send tokens within the current device.
        use_shmem: Whether to enable SHMEM communication.
        allreduce_world_size: The world size of the allreduce operation. Only
            needed for skip_a2a. Used to calculate the workload distribution for
            the shared expert (if has one).
        has_router_weights: Whether the routed expert outputs are weighted by
            `router_weights_fn` and summed. The overload without
            `router_weights_fn` passes `False`.
        OutputTokensEngine: Storage engine backing the output tokens tile.

    Arguments:
        num_tokens: Number of output tokens.
        atomic_counters: EP kernel synchronization counters.
        input_tokens: Expert output tokens to send back.
        src_info: Source routing information from dispatch phase.
        send_ptrs: Send buffer pointers for each local GPU.
        recv_ptrs: Receive buffer pointers for each local GPU.
        recv_count_ptrs: Receive count buffer pointers for each local GPU.
        context: Device context pointer.
        store_fn: Stores the combined output tokens. Called with
            `(token_idx, elem)` when `has_router_weights`, and with
            `(token_idx, topk_idx, elem)` otherwise. It runs on the device,
            so it must copy what it captures.
        router_weights_fn: The router weights. It runs on the device, so it
            must copy what it captures.
        topk_ids_p: Pointer to the top-k IDs for each token, only required if
            skip_a2a is True.
    """

    # Ensure this kernel only runs on GPU targets
    comptime assert is_gpu[target](), "EP is only supported on GPU."
    # Ensure the shape for the tensors are correct
    comptime assert (
        input_tokens.static_shape[1] == hidden_size
    ), "EP combine: input tokens shape doesn't match hidden size."
    comptime assert (
        send_ptrs.flat_rank == 1
    ), "Send pointers must be a 1D tensor."

    var gpu_ctx = context
    var gpu_id = Int(gpu_ctx.id())
    var my_rank = Int32(gpu_id)

    comptime if n_nodes > 1:
        my_rank = Int32(shmem_my_pe())

    comptime assert not (
        skip_a2a and use_shmem
    ), "skip_a2a and use_shmem cannot be True at the same time."
    comptime if skip_a2a:
        if not topk_ids_p:
            raise "topk_ids_p is required when skip_a2a is True."

    comptime hw_info = gpu_ctx.default_device_info
    comptime combine_msg_size = hidden_size * size_of[combine_dtype]()
    comptime n_ranks = n_gpus_per_node * n_nodes

    comptime fused_combine = combine_kernel[
        combine_dtype,
        hw_info.max_thread_block_size,
        input_tokens.LayoutType,
        src_info.LayoutType,
        hw_info.sm_count,
        top_k,
        n_experts,
        n_ranks,
        combine_msg_size,
        max_token_per_rank,
        n_gpus_per_node,
        StoreFnType,
        RouterWeightsFnType,
        has_router_weights,
        fused_shared_expert=fused_shared_expert,
        skip_a2a=skip_a2a,
        use_shmem=use_shmem,
        allreduce_world_size=allreduce_world_size,
        InputTokensEngine=input_tokens.Engine,
        SrcInfoEngine=src_info.Engine,
    ]

    @inline(.always)
    def description_fn() {imm} -> String:
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
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(context),
    ):
        var func = gpu_ctx.compile_function[fused_combine]()

        comptime if use_shmem:
            var cached_module_key = String(
                "EP_FUSED_COMBINE_INITED_DEV_", gpu_id
            )

            # Don't initialize the module repeatedly
            if not _get_global_or_null(cached_module_key):
                shmem_module_init(func)
                global_cache_insert(
                    cached_module_key,
                    MutPointer[NoneType, MutUntrackedOrigin](
                        unsafe_from_address=1
                    ),
                )

        var send_ptr = MutPointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(send_ptrs[gpu_id])
        )
        # Create inline arrays to store all the p2p accessible pointers
        var recv_ptrs_arr = pack_ptrs_array[
            DType.uint8, local_rank_only=skip_a2a
        ](recv_ptrs, my_rank)
        var recv_count_ptrs_arr = pack_ptrs_array[
            DType.uint64, local_rank_only=skip_a2a
        ](recv_count_ptrs, my_rank)
        var ep_counters = EPLocalSyncCounters[n_experts](
            MutPointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=Int(atomic_counters.ptr)
            )
        )

        # Fused combine grid must satisfy:
        #  - async send_tokens_back: one block per expert (n_experts iterations
        #    with stride grid_dim.x), so any grid >= 1 is correct.
        #  - wait + reduce: needs num_tokens * n_chunks_per_tok work units
        #    distributed across n_warps_per_block * (grid_dim - 1) warps.
        # Take the max to be safe; cap at sm_count.
        comptime dst_simd_width = simd_width_of[combine_dtype]()
        comptime n_chunks_per_tok = hidden_size // (
            hw_info.warp_size * dst_simd_width
        )
        comptime n_warps_per_block = (
            hw_info.max_thread_block_size // hw_info.warp_size
        )
        var num_input_tokens = Int(input_tokens.dim(0))
        var grid_dim_x: Int
        comptime if gpu_ctx.target.is_amd_gpu():
            grid_dim_x = _ep_combine_wait_grid_dim(
                num_input_tokens,
                n_chunks_per_tok,
                n_warps_per_block,
                hw_info.sm_count,
            )
        else:
            grid_dim_x = hw_info.sm_count
        func._call_with_pack_checked(
            gpu_ctx,
            input_tokens,
            src_info,
            Int32(num_tokens),
            send_ptr,
            recv_ptrs_arr,
            recv_count_ptrs_arr,
            ep_counters,
            topk_ids_p,
            my_rank,
            host_args=(store_fn, router_weights_fn),
            grid_dim=grid_dim_x,
            block_dim=hw_info.max_thread_block_size,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

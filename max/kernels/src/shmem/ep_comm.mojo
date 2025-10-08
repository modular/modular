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

from math import align_up, ceildiv
from os.atomic import Atomic
from sys.info import align_of, simd_width_of, size_of

import gpu.warp as warp
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from gpu.intrinsics import Scope, load_acquire, store_release
from gpu.memory import AddressSpace
from gpu.sync import syncwarp
from layout import Layout, LayoutTensor, RuntimeLayout, RuntimeTuple
from layout.int_tuple import (
    UNKNOWN_VALUE,
    IntTuple,
    _get_index_type,
    _get_layout_type,
)
from memory import stack_allocation
from memory.unsafe import bitcast
from shmem import SHMEM_SIGNAL_SET, SHMEMScope, shmem_put_nbi, shmem_signal_op

from utils.index import IndexList, StaticTuple

alias RtTuple_2 = RuntimeTuple[
    IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE), element_type = DType.int32
]
alias RtTuple_3 = RuntimeTuple[
    IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
    element_type = DType.int32,
]
alias RtTuple_4 = RuntimeTuple[
    IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
    element_type = DType.int32,
]

alias EP_DATA_READY_FLAG = 1 << 10


@fieldwise_init
@register_passable("trivial")
struct EPMsgConfig:
    var dtype: DType
    var hid_dim: Int
    var top_k: Int
    var alignment: Int

    fn token_size(self) -> Int:
        return align_up(self.hid_dim * self.dtype.size_of(), self.alignment)

    fn src_info_size(self) -> Int:
        return align_up(size_of[Int32](), self.alignment)

    fn topk_info_size(self) -> Int:
        return align_up(size_of[UInt16]() * self.top_k, self.alignment)

    fn msg_size(self) -> Int:
        return self.token_size() + self.src_info_size() + self.topk_info_size()

    fn src_info_offset(self) -> Int:
        return self.token_size()

    fn topk_info_offset(self) -> Int:
        return self.token_size() + self.src_info_size()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn dispatch_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: Layout,
    topk_ids_layout: Layout,
    n_sms: Int,
    n_aux_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
](
    input_tokens: LayoutTensor[
        input_type, input_tokens_layout, ImmutableAnyOrigin
    ],
    topk_ids: LayoutTensor[DType.int32, topk_ids_layout, ImmutableAnyOrigin],
    send_buf_p: UnsafePointer[UInt8],
    recv_buf_p: UnsafePointer[UInt8],
    recv_count_p: UnsafePointer[UInt64],
    atomic_counter: UnsafePointer[Int32],
    my_rank: Int32,
):
    """
    Dispatch tokens to experts on remote ranks based on the top-k expert IDs.
    This kernel utilizes the non-blocking SHMEM API, and would return immediately
    after initiating the communication. The communication is considered complete
    after calling the `dispatch_cb_kernel`.

    Parameters:
        input_type: The type of the input tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        topk_ids_layout: The layout of the top-k expert IDs.
        n_sms: The total number of SMs in the device.
        n_aux_sms: The number SMs that are used for counting the number of tokens
            for each expert, and for signaling the completion of the communication.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        msg_bytes: This is the total number of bytes we need to send for each token.
        max_tokens_per_rank: The maximum number of tokens per rank.

    Args:
        input_tokens: The input tokens to be dispatched.
        topk_ids: The top-k expert IDs for each token.
        send_buf_p: The pointer to the send buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(max_tokens_per_rank, msg_bytes)`.
        recv_buf_p: The pointer to the receive buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)`.
        recv_count_p: The pointer to the receive count buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(n_local_experts, n_ranks)`.
        atomic_counter: The pointer to the atomic counter.
        my_rank: The rank of the current device.
    """

    alias n_local_experts = n_experts // n_ranks
    alias n_warps = num_threads // WARP_SIZE
    alias n_comm_sms = n_sms - n_aux_sms
    constrained[
        n_local_experts <= n_warps,
        "EP dispatch: number of experts per rank must be less than or equal to "
        + String(n_warps),
    ]()

    alias src_simd_width = simd_width_of[input_type]()
    alias byte_simd_width = simd_width_of[DType.uint8]()

    alias top_k = topk_ids.shape[1]()
    alias hid_dim = input_tokens.shape[1]()
    alias msg_config = EPMsgConfig(
        input_type,
        hid_dim,
        top_k,
        align_of[SIMD[DType.uint8, byte_simd_width]](),
    )
    constrained[
        msg_bytes == msg_config.msg_size(),
        "EP dispatch: input shape doesn't match message size.",
    ]()

    var send_buf_layout = RuntimeLayout[
        Layout.row_major(max_tokens_per_rank, msg_bytes),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()
    alias recv_layout_static = Layout.row_major(
        n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes
    )
    var recv_buf_layout = RuntimeLayout[
        recv_layout_static,
        element_type = _get_layout_type(
            recv_layout_static, AddressSpace.GENERIC
        ),
        linear_idx_type = _get_index_type(
            recv_layout_static, AddressSpace.GENERIC
        ),
    ]()
    var recv_count_layout = RuntimeLayout[
        Layout.row_major(n_local_experts, n_ranks),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    var tid = thread_idx.x
    var num_tokens = input_tokens.dim[0]()

    # The reserved counter is incremented once a warp is ready to send.
    # The finished counter is incremented once the token is sent.
    var expert_reserved_counter = atomic_counter
    var expert_finished_counter = atomic_counter + n_experts

    # The auxiliary SMs are used for counting counting the number of tokens
    # that need to be sent to each expert. It also monitors the completion of
    # the communication for each expert.
    if block_idx.x < UInt(n_aux_sms):
        var expert_idx: Int32 = block_idx.x * UInt(n_warps) + warp_id()
        var expert_count: Int32 = 0

        if expert_idx < n_experts:
            for i in range(lane_id(), num_tokens * top_k, WARP_SIZE):
                if topk_ids.ptr[i] == expert_idx:
                    expert_count += 1

            expert_count = warp.sum(expert_count)

            if lane_id() == 0:
                # Wait until all the tokens for the expert have been sent.
                while (
                    load_acquire[scope = Scope.GPU](
                        expert_finished_counter + expert_idx
                    )
                    != expert_count
                ):
                    pass

                var dst_rank = expert_idx // n_local_experts
                var dst_expert_local_idx = expert_idx % n_local_experts

                var dst_recv_count_ptr = recv_count_p.offset(
                    recv_count_layout(
                        RtTuple_2(Int(dst_expert_local_idx), Int(my_rank))
                    )
                )

                # This signal operation is sent using the same RC as the one used
                # for token transfer. Since RC guarantees the message is delivered
                # in order, the remote device can confirm all the tokens for the
                # expert has been received once the signal operation is received.
                shmem_signal_op(
                    dst_recv_count_ptr,
                    UInt64(expert_count),
                    SHMEM_SIGNAL_SET,
                    dst_rank,
                )

                expert_reserved_counter[expert_idx] = 0
                expert_finished_counter[expert_idx] = 0

    # All the other SMs are used for sending the tokens to the experts. A token will
    # first be copied to the send buffer (so the NIC can see it), and then be sent
    # to the remote device.
    else:
        for token_idx in range(
            block_idx.x - UInt(n_aux_sms), num_tokens, n_comm_sms
        ):
            # First, all threads in the block copy the input token to the send buffer.
            alias _align = align_of[SIMD[DType.uint8, byte_simd_width]]()
            var curr_send_buf_ptr = send_buf_p.offset(
                send_buf_layout(RtTuple_2(token_idx, 0))
            )

            for i in range(tid, hid_dim // src_simd_width, num_threads):
                curr_send_buf_ptr.store[
                    width=byte_simd_width, alignment=_align
                ](
                    i * byte_simd_width,
                    bitcast[DType.uint8, byte_simd_width](
                        input_tokens.aligned_load[src_simd_width](
                            token_idx, i * src_simd_width
                        )
                    ),
                )

            if tid < UInt(top_k):
                # Store all the top-k expert IDs in current token's message.
                # The remote device will use the expert ID to determine a token's
                # top-k id.
                # Cast the expert ID to a 16-bit integer to save space.
                var top_k_idx = topk_ids.load[width=1](token_idx, tid)
                curr_send_buf_ptr.store[
                    width = size_of[UInt16](),
                    alignment = align_of[DType.uint16](),
                ](
                    msg_config.topk_info_offset()
                    + tid * UInt(size_of[UInt16]()),
                    bitcast[DType.uint8, size_of[UInt16]()](UInt16(top_k_idx)),
                )

                # Store the source token index in current token's message.
                if tid == 0:
                    curr_send_buf_ptr.store[
                        width = size_of[Int32](),
                        alignment = align_of[DType.int32](),
                    ](
                        msg_config.src_info_offset(),
                        bitcast[DType.uint8, size_of[Int32]()](
                            Int32(token_idx)
                        ),
                    )

            barrier()

            # We set up `n_local_experts` Reliable Communications (RCs) for each remote
            # device. We would like to use the same RC for each expert. However, NVSHMEM
            # does not allow us to explicitly specify the RC for each transfer. Instead,
            # we set the environment variable `NVSHMEM_IBGDA_RC_MAP_BY=warp` so that the RC
            # is selected by the warp ID using round-robin. We can then control the RC
            # for each expert by using the correct warp.
            alias n_rc_groups = n_warps // n_local_experts
            var rc_group_id = warp_id() // UInt(n_local_experts)
            var rc_map_offset: Int32 = (
                block_idx.x * UInt(n_warps) + warp_id()
            ) % UInt(n_local_experts)

            # If the RC group ID is greater than the number of RC groups, we skip the
            # communication.
            if rc_group_id >= UInt(n_rc_groups):
                continue

            for i in range(rc_group_id, top_k, n_rc_groups):
                var target_expert = topk_ids.load[width=1](token_idx, i)
                var dst_rank = target_expert // n_local_experts
                var dst_expert_local_idx = target_expert % n_local_experts

                if rc_map_offset == dst_expert_local_idx:
                    # First reserve a slot for the token.
                    var slot_idx: Int32 = 0
                    if lane_id() == 0:
                        slot_idx = Atomic.fetch_add(
                            expert_reserved_counter + target_expert, 1
                        )
                    slot_idx = warp.shuffle_idx(slot_idx, 0)

                    var dst_recv_buf_ptr = recv_buf_p.offset(
                        recv_buf_layout(
                            RtTuple_4(
                                Int(dst_expert_local_idx),
                                Int(my_rank),
                                Int(slot_idx),
                                0,
                            )
                        )
                    )
                    shmem_put_nbi[kind = SHMEMScope.warp](
                        dst_recv_buf_ptr,
                        curr_send_buf_ptr,
                        UInt(msg_bytes),
                        dst_rank,
                    )
                    syncwarp()

                    # Signal the completion of current token.
                    if lane_id() == 0:
                        _ = Atomic.fetch_add(
                            expert_finished_counter + target_expert, 1
                        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn dispatch_cb_kernel[
    output_type: DType,
    num_threads: Int,
    output_tokens_layout: Layout,
    row_offsets_layout: Layout,
    expert_ids_layout: Layout,
    src_info_layout: Layout,
    n_sms: Int,
    n_aux_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
](
    output_tokens: LayoutTensor[
        output_type, output_tokens_layout, MutableAnyOrigin
    ],
    row_offsets: LayoutTensor[
        DType.uint32, row_offsets_layout, MutableAnyOrigin
    ],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, MutableAnyOrigin],
    src_info: LayoutTensor[DType.int32, src_info_layout, MutableAnyOrigin],
    recv_buf_p: UnsafePointer[UInt8],
    recv_count_p: UnsafePointer[UInt64],
    atomic_counter: UnsafePointer[Int32],
    my_rank: Int32,
):
    """
    This kernel is called after the `dispatch_kernel` to complete the communication.
    It will keep polling the receive count buffer, and once the count is no longer
    MAX_FINITE, it can confirm that the communication is complete from a remote rank.

    The kernel will also aggregate the tokens from all the experts, and store them in
    the output tensor using a ragged representation.

    Parameters:
        output_type: The type of the output tokens.
        num_threads: The number of threads in the block.
        output_tokens_layout: The layout of the output tokens.
        row_offsets_layout: The layout of the row offsets.
        expert_ids_layout: The layout of the expert IDs.
        src_info_layout: The layout of the source token info.
        n_sms: The total number of SMs in the device.
        n_aux_sms: The number of auxiliary SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The number of experts in the device.
        n_ranks: The number of ranks.
        msg_bytes: The number of bytes in the message for each token.
        max_tokens_per_rank: The maximum number of tokens per rank.

    Args:
        output_tokens: The tensor to store the output tokens.
        row_offsets: The row offsets to be updated. Will be consumed by the
            `grouped_matmul` kernel.
        expert_ids: The expert IDs to be updated. Will be consumed by the
            `grouped_matmul` kernel.
        src_info: The source token info to be updated. Once the expert
            computation is complete, tokens will be send back to the original
            rank using information in this tensor.
        recv_buf_p: The pointer to the receive buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)`.
        recv_count_p: The pointer to the receive count buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(n_local_experts, n_ranks)`.
        atomic_counter: The pointer to the atomic counter.
        my_rank: The rank of the current device.
    """
    alias n_local_experts = n_experts // n_ranks
    alias n_warps = num_threads // WARP_SIZE
    alias n_comm_sms = n_sms - n_aux_sms

    alias hid_dim = output_tokens.shape[1]()
    alias dst_simd_width = simd_width_of[output_type]()
    alias byte_simd_width = simd_width_of[DType.uint8]()
    alias msg_config = EPMsgConfig(
        output_type,
        hid_dim,
        top_k,
        align_of[SIMD[DType.uint8, byte_simd_width]](),
    )
    constrained[
        msg_bytes == msg_config.msg_size(),
        "EP dispatch: input shape doesn't match message size.",
    ]()
    constrained[
        n_local_experts <= n_warps,
        "EP dispatch: local experts per device should be less than "
        + String(WARP_SIZE),
    ]()

    alias recv_layout_static = Layout.row_major(
        n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes
    )
    var recv_buf_layout = RuntimeLayout[
        recv_layout_static,
        element_type = _get_layout_type(
            recv_layout_static, AddressSpace.GENERIC
        ),
        linear_idx_type = _get_index_type(
            recv_layout_static, AddressSpace.GENERIC
        ),
    ]()
    var recv_count_layout = RuntimeLayout[
        Layout.row_major(n_local_experts, n_ranks),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    # The first SM is used for checking if any of a local expert has received
    # tokens from all the remote ranks. It will also calculate the offset where
    # the tokens start in the output tensor.
    if block_idx.x < UInt(n_aux_sms):
        var shared_mem = stack_allocation[
            1, DType.uint32, address_space = AddressSpace.SHARED
        ]()
        if thread_idx.x == 0:
            shared_mem[] = 0
        barrier()

        var local_expert_id = warp_id()
        if local_expert_id >= UInt(n_local_experts):
            return

        alias scan_round = ceildiv(n_ranks, WARP_SIZE)
        var prefix_sum_arr = stack_allocation[
            scan_round, DType.uint32, address_space = AddressSpace.LOCAL
        ]()
        var local_expert_token_count: UInt32 = 0

        # We need to scan the receive count buffer for each rank to get the total
        # number of tokens for the local expert. Also, we calculate the prefix sum
        # to get the offset where each rank ends in the output tensor.
        @parameter
        for round_i in range(scan_round):
            var target_rank = lane_id() + UInt(round_i * WARP_SIZE)
            var expert_rank_offset = recv_count_layout(
                RtTuple_2(Int(local_expert_id), Int(target_rank))
            )

            if target_rank < UInt(n_ranks):
                var target_count_ptr = recv_count_p.offset(expert_rank_offset)
                var token_count = load_acquire[scope = Scope.GPU](
                    target_count_ptr
                )
                while token_count == UInt64.MAX_FINITE:
                    token_count = load_acquire[scope = Scope.GPU](
                        target_count_ptr
                    )

                prefix_sum_arr[round_i] = UInt32(token_count)
            syncwarp()
            prefix_sum_arr[round_i] = warp.prefix_sum(prefix_sum_arr[round_i])
            syncwarp()
            prefix_sum_arr[round_i] += local_expert_token_count
            syncwarp()
            local_expert_token_count = warp.shuffle_idx(
                prefix_sum_arr[round_i], WARP_SIZE - 1
            )

        local_expert_token_count = warp.shuffle_idx(
            prefix_sum_arr[scan_round - 1], (n_ranks - 1) % WARP_SIZE
        )

        # Conduct a atomic add to get how many experts have already completed the
        # communication, and the offset where the previous expert end in the output
        # tensor.
        var expert_idx_and_offsets: UInt32 = 0
        if lane_id() == 0:
            expert_idx_and_offsets = Atomic.fetch_add(
                shared_mem, local_expert_token_count | 0x01000000
            )

        # It is unlikely a rank will receive more than 16777216 tokens.
        expert_idx_and_offsets = warp.broadcast(expert_idx_and_offsets)
        var expert_idx = expert_idx_and_offsets >> 24
        var prev_expert_offset = expert_idx_and_offsets & 0x00FFFFFF

        @parameter
        for round_i in range(scan_round):
            var target_rank = lane_id() + UInt(round_i * WARP_SIZE)
            var expert_rank_offset = recv_count_layout(
                RtTuple_2(Int(local_expert_id), Int(target_rank))
            )

            if target_rank < UInt(n_ranks):
                atomic_counter.store(
                    expert_rank_offset * 2,
                    Int32(
                        EP_DATA_READY_FLAG
                        + prev_expert_offset
                        + prefix_sum_arr[round_i]
                    ),
                )

        if lane_id() == 0:
            expert_ids[Int(expert_idx)] = local_expert_id
            row_offsets[Int(expert_idx) + 1] = (
                prev_expert_offset + local_expert_token_count
            )

            if expert_idx == 0:
                row_offsets[0] = 0

    # All the other SMs are used for copying the tokens to the output tensor.
    # The compute resources are partitioned into multiple work groups (wg), and
    # each work group is responsible for copying tokens for a single expert from
    # a remote rank.
    else:
        alias n_wg_per_sm = ceildiv(n_experts, n_comm_sms)
        alias wg_size = n_warps // n_wg_per_sm
        alias wg_threads = wg_size * WARP_SIZE

        var wg_idx = warp_id() // UInt(wg_size)
        var global_wg_idx = (block_idx.x - UInt(n_aux_sms)) * UInt(
            n_wg_per_sm
        ) + wg_idx
        var warp_id_in_wg = warp_id() % UInt(wg_size)

        if wg_idx >= UInt(n_wg_per_sm) or global_wg_idx >= UInt(n_experts):
            return

        var local_expert_id = global_wg_idx % UInt(n_local_experts)
        var target_rank = global_wg_idx // UInt(n_local_experts)
        var expert_rank_offset = recv_count_layout(
            RtTuple_2(Int(local_expert_id), Int(target_rank))
        )

        # Wait until the auxiliary SM has signaled that the data is ready, and
        # provided the offset where the tokens end in the output tensor.
        var offset_ptr = atomic_counter.offset(expert_rank_offset * 2)
        var output_offset = load_acquire[scope = Scope.GPU](offset_ptr)
        while output_offset < EP_DATA_READY_FLAG:
            output_offset = load_acquire[scope = Scope.GPU](offset_ptr)
        output_offset -= EP_DATA_READY_FLAG

        var token_count = Int32(recv_count_p.load(expert_rank_offset))
        output_offset -= token_count

        for token_idx in range(warp_id_in_wg, token_count, wg_size):
            alias _align = align_of[SIMD[DType.uint8, byte_simd_width]]()
            var token_pos = Int(token_idx + output_offset)
            var recv_buf_ptr = recv_buf_p.offset(
                recv_buf_layout(
                    RtTuple_4(
                        Int(local_expert_id),
                        Int(target_rank),
                        Int(token_idx),
                        0,
                    )
                )
            )

            for i in range(lane_id(), hid_dim // dst_simd_width, WARP_SIZE):
                output_tokens.aligned_store[width=dst_simd_width](
                    token_pos,
                    i * dst_simd_width,
                    bitcast[output_type, dst_simd_width](
                        recv_buf_ptr.load[
                            width=byte_simd_width,
                            invariant=True,
                            alignment=_align,
                        ](
                            i * byte_simd_width,
                        )
                    ),
                )

            if lane_id() < UInt(top_k):
                # Load top-k expert IDs from the token's message.
                var src_topk_idx = bitcast[DType.uint16, 1](
                    recv_buf_ptr.load[width = size_of[UInt16]()](
                        msg_config.topk_info_offset()
                        + lane_id() * UInt(size_of[UInt16]()),
                    )
                )
                var global_expert_idx = (
                    my_rank * n_local_experts + local_expert_id
                )
                if global_expert_idx == Int32(src_topk_idx):
                    # Store the source token index and the top-k id.
                    var src_idx = bitcast[DType.int32, 1](
                        recv_buf_ptr.load[width = size_of[Int32]()](
                            msg_config.src_info_offset()
                        )
                    )

                    src_info[token_pos, 0] = src_idx
                    src_info[token_pos, 1] = lane_id()

        barrier()
        if lane_id() == 0 and warp_id_in_wg == 0:
            recv_count_p.store(expert_rank_offset, UInt64.MAX_FINITE)
            offset_ptr.store(1, token_count)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn combine_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: Layout,
    src_info_layout: Layout,
    n_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
](
    input_tokens: LayoutTensor[
        input_type, input_tokens_layout, ImmutableAnyOrigin
    ],
    src_info: LayoutTensor[DType.int32, src_info_layout, ImmutableAnyOrigin],
    send_buf_p: UnsafePointer[UInt8],
    recv_buf_p: UnsafePointer[UInt8],
    recv_count_p: UnsafePointer[UInt64],
    atomic_counter: UnsafePointer[Int32],
    my_rank: Int32,
):
    """
    Send tokens to the original rank based on the src_info tensor.
    This kernel utilizes the non-blocking SHMEM API, and would return immediately
    after initiating the communication. The communication is considered complete
    after calling the `combine_cb_kernel`.

    Parameters:
        input_type: The type of the input tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        src_info_layout: The layout of the source token info.
        n_sms: The total number of SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        msg_bytes: This is the total number of bytes we need to send for each token.
        max_tokens_per_rank: The maximum number of tokens per rank.

    Args:
        input_tokens: The tokens to be sent back to the original rank.
        src_info: The source token info tensor of shape
            `(n_local_experts * n_ranks * max_tokens_per_rank, 2)`. The first
            column stores a token's position in the original rank's tensor, and
            the second column stores the top-k ID for the token.
        send_buf_p: The pointer to the send buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(n_local_experts * n_ranks * max_tokens_per_rank, msg_bytes)`.
        recv_buf_p: The pointer to the receive buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(max_tokens_per_rank, top_k, msg_bytes, msg_bytes)`.
        recv_count_p: The pointer to the receive count buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(n_local_experts, n_ranks)`.
        atomic_counter: The pointer to the atomic counter.
        my_rank: The rank of the current device.
    """
    alias n_local_experts = n_experts // n_ranks
    alias n_warps = num_threads // WARP_SIZE

    alias src_simd_width = simd_width_of[input_type]()
    alias byte_simd_width = simd_width_of[DType.uint8]()

    alias hid_dim = input_tokens.shape[1]()

    constrained[
        msg_bytes == hid_dim * size_of[Scalar[input_type]](),
        "EP combine: input shape doesn't match message size.",
    ]()
    constrained[
        msg_bytes % byte_simd_width == 0,
        "EP combine: message size must be divisible by "
        + String(byte_simd_width),
    ]()

    alias send_layout_static = Layout.row_major(
        n_local_experts * n_ranks * max_tokens_per_rank, msg_bytes
    )
    var send_buf_layout = RuntimeLayout[
        send_layout_static,
        element_type = _get_layout_type(
            send_layout_static, AddressSpace.GENERIC
        ),
        linear_idx_type = _get_index_type(
            send_layout_static, AddressSpace.GENERIC
        ),
    ]()
    var recv_buf_layout = RuntimeLayout[
        Layout.row_major(max_tokens_per_rank, top_k, msg_bytes),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()
    var recv_count_layout = RuntimeLayout[
        Layout.row_major(n_local_experts, n_ranks),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    var tid = Int(thread_idx.x)
    var sm_id = Int(block_idx.x)

    # Each rank holds `n_local_experts` experts, and for each expert, it needs to
    # send back different tokens to `n_ranks` remote ranks. We use one block
    # per-expert-per-rank to send back the tokens.
    for global_idx in range(sm_id, n_experts, n_sms):
        var local_expert_id = global_idx % n_local_experts
        var target_rank = global_idx // n_local_experts
        var expert_rank_offset = recv_count_layout(
            RtTuple_2(Int(local_expert_id), Int(target_rank))
        )

        # The tokens are sent back to the original rank using the same RC as the
        # one they come from.
        var rc_map_offset = (sm_id * n_warps + warp_id()) % n_local_experts

        # Info for where the tokens for the current expert and rank start and end
        # are stored in the atomic counter by the `dispatch_cb_kernel`.
        alias DATA_READY_FLAG = 1024
        var token_end_count = atomic_counter.load[
            width=2,
            alignment = align_of[SIMD[DType.int32, 2]](),
            invariant=True,
        ](2 * expert_rank_offset)
        var token_end = token_end_count[0] - DATA_READY_FLAG
        var token_start = token_end - token_end_count[1]

        for token_idx in range(token_start, token_end):
            var src_token_info = src_info.aligned_load[2](Int(token_idx), 0)
            var src_idx = src_token_info[0]
            var src_topk_idx = src_token_info[1]

            # First, all threads in the block copy the input token to the send buffer.
            alias _align = align_of[SIMD[DType.uint8, byte_simd_width]]()
            var curr_send_buf_ptr = send_buf_p.offset(
                send_buf_layout(RtTuple_2(Int(token_idx), 0))
            )

            for i in range(tid, hid_dim // src_simd_width, num_threads):
                curr_send_buf_ptr.store[
                    width=byte_simd_width, alignment=_align
                ](
                    i * byte_simd_width,
                    bitcast[DType.uint8, byte_simd_width](
                        input_tokens.aligned_load[src_simd_width](
                            Int(token_idx), i * src_simd_width
                        )
                    ),
                )
            barrier()

            # Send the token to the original rank.
            var dst_recv_buf_ptr = recv_buf_p.offset(
                recv_buf_layout(RtTuple_3(Int(src_idx), Int(src_topk_idx), 0))
            )
            if (
                warp_id() < UInt(n_local_experts)
                and local_expert_id == rc_map_offset
            ):
                shmem_put_nbi[kind = SHMEMScope.warp](
                    dst_recv_buf_ptr,
                    curr_send_buf_ptr,
                    UInt(msg_bytes),
                    target_rank,
                )
        barrier()

        # Once all the tokens for the current expert and rank have been sent,
        # signal the completion of the communication.
        if (
            warp_id() < UInt(n_local_experts)
            and local_expert_id == rc_map_offset
        ):
            if lane_id() == 0:
                var global_expert_idx = (
                    my_rank * n_local_experts + local_expert_id
                )
                shmem_signal_op(
                    recv_count_p.offset(global_expert_idx),
                    UInt64(token_end - token_start),
                    SHMEM_SIGNAL_SET,
                    target_rank,
                )

                atomic_counter.store[
                    width=2, alignment = align_of[SIMD[DType.int32, 2]]()
                ](expert_rank_offset * 2, 0)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn combine_cb_kernel[
    output_type: DType,
    num_threads: Int,
    output_tokens_layout: Layout,
    n_sms: Int,
    n_aux_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
](
    output_tokens: LayoutTensor[
        output_type, output_tokens_layout, MutableAnyOrigin
    ],
    recv_buf_p: UnsafePointer[UInt8],
    recv_count_p: UnsafePointer[UInt64],
    atomic_counter: UnsafePointer[Int32],
    my_rank: Int32,
):
    """
    This kernel is called after the `combine_kernel` to complete the communication.
    It will keep polling the receive count buffer, and once the count is no longer
    MAX_FINITE, it can confirm that the communication is complete from a remote rank.

    Parameters:
        output_type: The type of the output tokens.
        num_threads: The number of threads in the block.
        output_tokens_layout: The layout of the output tokens.
        n_sms: The total number of SMs in the device.
        n_aux_sms: The number of auxiliary SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The number of experts in the device.
        n_ranks: The number of ranks.
        msg_bytes: The number of bytes in the message for each token.
        max_tokens_per_rank: The maximum number of tokens per rank.

    Args:
        output_tokens: The tensor to store the output tokens.
        recv_buf_p: The pointer to the receive buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(max_tokens_per_rank, top_k, msg_bytes)`.
        recv_count_p: The pointer to the receive count buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(n_local_experts, n_ranks)`.
        atomic_counter: The pointer to the atomic counter.
        my_rank: The rank of the current device.
    """

    alias n_local_experts = n_experts // n_ranks
    alias n_warps = num_threads // WARP_SIZE
    alias n_red_sms = n_sms - n_aux_sms

    alias dst_simd_width = simd_width_of[output_type]()
    alias byte_simd_width = simd_width_of[DType.uint8]()

    alias hid_dim = output_tokens.shape[2]()

    constrained[
        msg_bytes == hid_dim * size_of[Scalar[output_type]](),
        "EP combine: output shape doesn't match message size.",
    ]()
    constrained[
        msg_bytes % byte_simd_width == 0,
        "EP combine: message size must be divisible by "
        + String(byte_simd_width),
    ]()

    var recv_buf_layout = RuntimeLayout[
        Layout.row_major(max_tokens_per_rank, top_k, msg_bytes),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()
    var recv_count_layout = RuntimeLayout[
        Layout.row_major(n_local_experts, n_ranks),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    alias DATA_READY_FLAG = 1024

    # `num_tokens` is the total number of tokens before the EP communication. The
    # actual number of tokens we receive is `num_tokens * top_k`.
    var num_tokens = output_tokens.dim[0]()
    var tid = Int(thread_idx.x)
    var sm_id = Int(block_idx.x)

    # The first SM is used for checking if we have received tokens from all the
    # remote ranks.
    if sm_id < n_aux_sms:
        if tid < n_experts:
            var target_count_ptr = recv_count_p.offset(tid)
            while (
                load_acquire[scope = Scope.GPU](target_count_ptr)
                == UInt64.MAX_FINITE
            ):
                pass

            target_count_ptr[] = UInt64.MAX_FINITE
        barrier()

        # Once all the tokens have been received, set flags for other SMs to copy
        # the tokens to the output tensor.
        if tid < n_red_sms:
            atomic_counter.store(n_aux_sms + tid, DATA_READY_FLAG)

    # All the other SMs are used for copying the tokens to the output tensor.
    else:
        if tid == 0:
            while (
                load_acquire[scope = Scope.GPU](atomic_counter.offset(sm_id))
                != DATA_READY_FLAG
            ):
                pass

            # Reset the atomic counter for the next round.
            atomic_counter.store(sm_id, 0)
        barrier()

        for token_idx in range(sm_id - n_aux_sms, num_tokens, n_red_sms):
            # The output tensor is of shape `(num_tokens, top_k, hid_dim)`.
            var output_token_tensor = output_tokens.slice[:, :, (1, 2)](
                token_idx
            )

            # Copy the received tokens from all the experts.
            @parameter
            for topk_id in range(top_k):
                var recv_buf_ptr = recv_buf_p.offset(
                    recv_buf_layout(RtTuple_3(token_idx, topk_id, 0))
                )

                for i in range(tid, hid_dim // dst_simd_width, num_threads):
                    alias _align = align_of[
                        SIMD[DType.uint8, byte_simd_width]
                    ]()
                    output_token_tensor.aligned_store[width=dst_simd_width](
                        topk_id,
                        i * dst_simd_width,
                        bitcast[output_type, dst_simd_width](
                            recv_buf_ptr.load[
                                width=byte_simd_width,
                                invariant=True,
                                alignment=_align,
                            ](
                                i * byte_simd_width,
                            )
                        ),
                    )

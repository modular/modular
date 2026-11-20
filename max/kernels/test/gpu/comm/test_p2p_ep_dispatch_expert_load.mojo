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

# Checks the expert-load tally that dispatch wait keeps for EPLB, through the
# kernel APIs the graph ops call, on both the split (async + wait) and the
# fused dispatch. Each step adds every local expert's routed-token count to the
# row of the step's layer; totals run across steps without a reset and the
# other layer's row is left alone. Blockwise FP8 pads each expert's rows for
# the grouped matmul, and the tally must count tokens, not padded rows.

from std.math import align_up
from std.random import randint, seed

from comm.sync import enable_p2p
from max.gpu.host import DeviceBuffer, DeviceContext, default_accelerator
from layout import TileTensor, Idx, row_major
from shmem.ep import (
    ep_dispatch_async_kernel_api,
    ep_dispatch_wait_kernel_api,
    ep_fused_dispatch_kernel_api,
)
from shmem.ep_comm import (
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    EPLocalSyncCounters,
    ExpertLoadTally,
    TokenFormat,
)
from std.memory.alloc import unsafe_alloc
from std.testing import assert_equal, assert_true


def legalize_topk_ids[
    n_experts: Int, top_k: Int
](topk_ids: MutPointer[Int32, MutUntrackedOrigin], n_tokens: Int):
    for tok_id in range(n_tokens):
        var topk_ids_for_token = topk_ids.unsafe_offset(tok_id * top_k)

        def is_duplicate() {imm} -> Int:
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    if topk_ids_for_token[i] == topk_ids_for_token[j]:
                        return i
            return -1

        var duplicate_idx = is_duplicate()
        while duplicate_idx != -1:
            randint(
                topk_ids_for_token.unsafe_offset(duplicate_idx),
                1,
                0,
                n_experts - 1,
            )
            duplicate_idx = is_duplicate()


trait LoadTestFormat(Deinitable, Movable):
    """A token format and the receive buffers its handler writes to."""

    comptime TokenFormatType: TokenFormat
    # Rows each expert's group is padded to for the grouped matmul.
    comptime m_padding: Int

    def __init__(out self, list_of_ctx: List[DeviceContext[]]) raises:
        ...

    def handler(self, dev_idx: Int, out result: Self.TokenFormatType):
        ...


struct BF16Format[hidden_size: Int, top_k: Int, max_recv_num_tokens: Int](
    LoadTestFormat
):
    comptime output_layout = row_major(
        (Idx[Self.max_recv_num_tokens], Idx[Self.hidden_size])
    )
    comptime TokenFormatType = BF16TokenFormat[
        output_layout=type_of(Self.output_layout),
        Self.hidden_size,
        Self.top_k,
    ]
    comptime m_padding = 1

    var output_bufs: List[DeviceBuffer[.bfloat16]]

    def __init__(out self, list_of_ctx: List[DeviceContext[]]) raises:
        self.output_bufs = List[DeviceBuffer[.bfloat16]]()
        for ctx in list_of_ctx:
            self.output_bufs.append(
                ctx.enqueue_create_buffer[.bfloat16](
                    Self.max_recv_num_tokens * Self.hidden_size
                )
            )

    def handler(self, dev_idx: Int, out result: Self.TokenFormatType):
        result = Self.TokenFormatType(
            TileTensor(
                ptr=self.output_bufs[dev_idx].unsafe_ptr(),
                layout=Self.output_layout,
            )
        )


struct FP8Format[hidden_size: Int, top_k: Int, max_recv_num_tokens: Int](
    LoadTestFormat
):
    comptime group_size = 128
    comptime output_layout = row_major(
        (Idx[Self.max_recv_num_tokens], Idx[Self.hidden_size])
    )
    comptime scales_layout = row_major(
        (
            Self.hidden_size // Self.group_size,
            Idx[Self.max_recv_num_tokens],
        )
    )
    comptime TokenFormatType = BlockwiseFP8TokenFormat[
        fp8_dtype=DType.float8_e4m3fn,
        scales_dtype=DType.float32,
        output_layout=type_of(Self.output_layout),
        scales_layout=type_of(Self.scales_layout),
        Self.hidden_size,
        Self.top_k,
    ]
    comptime m_padding = Self.TokenFormatType.expert_m_padding

    var output_bufs: List[DeviceBuffer[DType.float8_e4m3fn]]
    var scales_bufs: List[DeviceBuffer[DType.float32]]

    def __init__(out self, list_of_ctx: List[DeviceContext[]]) raises:
        self.output_bufs = List[DeviceBuffer[DType.float8_e4m3fn]]()
        self.scales_bufs = List[DeviceBuffer[DType.float32]]()
        for ctx in list_of_ctx:
            self.output_bufs.append(
                ctx.enqueue_create_buffer[DType.float8_e4m3fn](
                    Self.max_recv_num_tokens * Self.hidden_size
                )
            )
            self.scales_bufs.append(
                ctx.enqueue_create_buffer[DType.float32](
                    Self.max_recv_num_tokens
                    * Self.hidden_size
                    // Self.group_size
                )
            )

    def handler(self, dev_idx: Int, out result: Self.TokenFormatType):
        result = Self.TokenFormatType(
            TileTensor(
                ptr=self.output_bufs[dev_idx].unsafe_ptr(),
                layout=Self.output_layout,
            ),
            TileTensor(
                ptr=self.scales_bufs[dev_idx].unsafe_ptr(),
                layout=Self.scales_layout,
            ),
        )


def test_expert_load[
    FormatT: LoadTestFormat,
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_tokens_per_rank: Int,
    n_layers: Int,
    n_steps: Int,
    fused_dispatch: Bool,
](list_of_ctx: List[DeviceContext[]]) raises:
    comptime input_type = DType.bfloat16
    comptime n_local_experts = n_experts // n_ranks
    comptime max_recv_num_tokens = n_experts * n_tokens_per_rank
    # Above 2^32, so a 32-bit tally or a reset would show.
    comptime tally_start = UInt64(1) << 33

    comptime token_fmt_type = FormatT.TokenFormatType
    var formats = FormatT(list_of_ctx)
    comptime msg_bytes = token_fmt_type.msg_size()
    comptime counters_size = EPLocalSyncCounters[n_experts].total_size()

    print(
        "Running ep_dispatch expert-load test:",
        token_fmt_type.get_type_name(),
        "fused:",
        fused_dispatch,
        "n_experts:",
        n_experts,
        "n_ranks:",
        n_ranks,
        "n_tokens_per_rank:",
        n_tokens_per_rank,
    )

    # fmt: off
    var send_bufs = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var recv_bufs = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var recv_count_bufs = List[DeviceBuffer[.uint64]](capacity=n_ranks)
    var atomic_counter_bufs = List[DeviceBuffer[.int32]](capacity=n_ranks)
    var topk_bufs = List[DeviceBuffer[.int32]](capacity=n_ranks)
    var input_bufs = List[DeviceBuffer[input_type]](capacity=n_ranks)
    var row_offsets_bufs = List[DeviceBuffer[.uint32]](capacity=n_ranks)
    var expert_ids_bufs = List[DeviceBuffer[.int32]](capacity=n_ranks)
    var src_info_bufs = List[DeviceBuffer[.int32]](capacity=n_ranks)
    var tally_bufs = List[DeviceBuffer[.uint64]](capacity=n_ranks)
    var layer_idx_bufs = List[DeviceBuffer[.int32]](capacity=n_ranks)

    for i in range(n_ranks):
        var ctx = list_of_ctx[i]
        send_bufs.append(ctx.enqueue_create_buffer[.uint8](n_tokens_per_rank * msg_bytes))
        recv_bufs.append(ctx.enqueue_create_buffer[.uint8](max_recv_num_tokens * msg_bytes))
        recv_count_bufs.append(ctx.enqueue_create_buffer[.uint64](n_experts))
        atomic_counter_bufs.append(ctx.enqueue_create_buffer[.int32](counters_size))
        topk_bufs.append(ctx.enqueue_create_buffer[.int32](n_tokens_per_rank * top_k))
        input_bufs.append(ctx.enqueue_create_buffer[input_type](n_tokens_per_rank * hidden_size))
        row_offsets_bufs.append(ctx.enqueue_create_buffer[.uint32](n_local_experts + 1))
        expert_ids_bufs.append(ctx.enqueue_create_buffer[.int32](n_local_experts))
        src_info_bufs.append(ctx.enqueue_create_buffer[.int32](max_recv_num_tokens * 2))
        tally_bufs.append(ctx.enqueue_create_buffer[.uint64](n_layers * n_local_experts))
        layer_idx_bufs.append(ctx.enqueue_create_buffer[.int32](1))

        ctx.enqueue_memset(recv_count_bufs[i], UInt64.MAX_FINITE)
        ctx.enqueue_memset(atomic_counter_bufs[i], Int32(0))
        ctx.enqueue_memset(input_bufs[i], Scalar[input_type](0))
        ctx.enqueue_memset(tally_bufs[i], tally_start)

    var send_ptrs = unsafe_alloc[UInt64](n_ranks)
    var recv_ptrs = unsafe_alloc[UInt64](n_ranks)
    var recv_count_ptrs = unsafe_alloc[UInt64](n_ranks)
    for i in range(n_ranks):
        send_ptrs[i] = UInt64(Int(send_bufs[i].unsafe_ptr()))
        recv_ptrs[i] = UInt64(Int(recv_bufs[i].unsafe_ptr()))
        recv_count_ptrs[i] = UInt64(Int(recv_count_bufs[i].unsafe_ptr()))

    var ptrs_layout = row_major[n_ranks]()
    var send_ptrs_tensor = TileTensor(ptr=send_ptrs.as_unsafe_any_origin(), layout=ptrs_layout)
    var recv_ptrs_tensor = TileTensor(ptr=recv_ptrs.as_unsafe_any_origin(), layout=ptrs_layout)
    var recv_count_ptrs_tensor = TileTensor(ptr=recv_count_ptrs.as_unsafe_any_origin(), layout=ptrs_layout)
    # fmt: on

    var all_topk_ids = unsafe_alloc[Int32](n_ranks * n_tokens_per_rank * top_k)
    var host_layer_idx = unsafe_alloc[Int32](1)
    var host_tally = unsafe_alloc[UInt64](n_layers * n_local_experts)
    var host_row_offsets = unsafe_alloc[UInt32](n_local_experts + 1)
    var expected = unsafe_alloc[UInt64](n_ranks * n_layers * n_local_experts)
    for j in range(n_ranks * n_layers * n_local_experts):
        expected[j] = tally_start

    @inline(.always)
    def run_step(
        dev_idx: Int,
    ) raises {
        mut row_offsets_bufs,
        mut expert_ids_bufs,
        mut src_info_bufs,
        mut atomic_counter_bufs,
        imm,
    }:
        var ctx = list_of_ctx[dev_idx]
        var format_handler = formats.handler(dev_idx)
        var expert_load = ExpertLoadTally(
            MutPointer[UInt64, MutUntrackedOrigin](
                unsafe_from_address=Int(tally_bufs[dev_idx].unsafe_ptr())
            ),
            ImmPointer[Int32, ImmUntrackedOrigin](
                unsafe_from_address=Int(layer_idx_bufs[dev_idx].unsafe_ptr())
            ),
        )
        var counters = TileTensor(
            ptr=atomic_counter_bufs[dev_idx]
            .unsafe_ptr()
            .as_unsafe_any_origin(),
            layout=row_major[counters_size](),
        )
        var input_tokens = TileTensor[mut=False](
            ptr=input_bufs[dev_idx].unsafe_ptr().as_unsafe_any_origin(),
            layout=row_major((n_tokens_per_rank, Idx[hidden_size])),
        )
        var topk_ids = TileTensor[mut=False](
            ptr=topk_bufs[dev_idx].unsafe_ptr().as_unsafe_any_origin(),
            layout=row_major(n_tokens_per_rank, Idx[top_k]),
        )
        var row_offsets = TileTensor(
            ptr=row_offsets_bufs[dev_idx].unsafe_ptr().as_unsafe_any_origin(),
            layout=row_major[n_local_experts + 1](),
        )
        var expert_ids = TileTensor(
            ptr=expert_ids_bufs[dev_idx].unsafe_ptr().as_unsafe_any_origin(),
            layout=row_major[n_local_experts](),
        )
        var src_info = TileTensor(
            ptr=src_info_bufs[dev_idx].unsafe_ptr().as_unsafe_any_origin(),
            layout=row_major((Idx[max_recv_num_tokens], Idx[2])),
        )
        comptime if fused_dispatch:
            ep_fused_dispatch_kernel_api[
                n_experts,
                n_tokens_per_rank,
                n_ranks,
                1,
                False,  # fused_shared_expert
                use_shmem=False,
            ](
                format_handler,
                row_offsets,
                expert_ids,
                src_info,
                counters,
                input_tokens,
                topk_ids,
                send_ptrs_tensor,
                recv_ptrs_tensor,
                recv_count_ptrs_tensor,
                ctx,
                expert_load,
            )
        else:
            ep_dispatch_async_kernel_api[
                token_fmt_type,
                n_experts,
                n_tokens_per_rank,
                n_ranks,
                1,
                use_shmem=False,
            ](
                counters,
                input_tokens,
                topk_ids,
                send_ptrs_tensor,
                recv_ptrs_tensor,
                recv_count_ptrs_tensor,
                ctx,
            )
            ep_dispatch_wait_kernel_api[
                n_experts,
                n_tokens_per_rank,
                n_ranks,
                1,
            ](
                format_handler,
                row_offsets,
                expert_ids,
                src_info,
                recv_ptrs_tensor,
                recv_count_ptrs_tensor,
                counters,
                ctx,
                expert_load=expert_load,
            )

    # Without an expert whose count is off the padding, the FP8 case would
    # not tell tokens from padded rows.
    var padded_an_expert = False
    for step in range(n_steps):
        var layer = step % n_layers
        host_layer_idx[0] = Int32(layer)
        for dev_idx in range(n_ranks):
            var topk = all_topk_ids.unsafe_offset(
                dev_idx * n_tokens_per_rank * top_k
            )
            seed(dev_idx + step * n_ranks)
            randint(topk, n_tokens_per_rank * top_k, 0, n_experts - 1)
            legalize_topk_ids[n_experts, top_k](topk, n_tokens_per_rank)
            list_of_ctx[dev_idx].enqueue_copy(topk_bufs[dev_idx], topk)
            list_of_ctx[dev_idx].enqueue_copy(
                layer_idx_bufs[dev_idx], host_layer_idx
            )
        # Peers write into each device's receive counts, so every device
        # must hold its reset before any device starts sending.
        for dev_idx in range(n_ranks):
            list_of_ctx[dev_idx].synchronize()

        for dev_idx in range(n_ranks):
            run_step(dev_idx)

        for dev_idx in range(n_ranks):
            var ctx = list_of_ctx[dev_idx]
            ctx.enqueue_copy(host_tally, tally_bufs[dev_idx])
            ctx.enqueue_copy(host_row_offsets, row_offsets_bufs[dev_idx])
            ctx.synchronize()

            var first_expert = dev_idx * n_local_experts
            for e in range(n_local_experts):
                var step_count = 0
                for t in range(n_ranks * n_tokens_per_rank * top_k):
                    if Int(all_topk_ids[t]) == first_expert + e:
                        step_count += 1
                expected[
                    (dev_idx * n_layers + layer) * n_local_experts + e
                ] += UInt64(step_count)
                if step_count % FormatT.m_padding != 0:
                    padded_an_expert = True
                assert_equal(
                    align_up(step_count, FormatT.m_padding),
                    Int(host_row_offsets[e + 1] - host_row_offsets[e]),
                    String("row offsets, dev ", dev_idx, " expert ", e),
                )

            for j in range(n_layers * n_local_experts):
                assert_equal(
                    host_tally[j],
                    expected[dev_idx * n_layers * n_local_experts + j],
                    String(
                        "tally, step ",
                        step,
                        " dev ",
                        dev_idx,
                        " layer ",
                        j // n_local_experts,
                        " expert ",
                        j % n_local_experts,
                    ),
                )

        for dev_idx in range(n_ranks):
            var ctx = list_of_ctx[dev_idx]
            ctx.enqueue_memset(atomic_counter_bufs[dev_idx], Int32(0))
            ctx.enqueue_memset(recv_count_bufs[dev_idx], UInt64.MAX_FINITE)

    for dev_idx in range(n_ranks):
        list_of_ctx[dev_idx].synchronize()
    comptime if FormatT.m_padding > 1:
        assert_true(padded_an_expert, "no expert count needed padding")
    print("expert-load tally matches after", n_steps, "steps")


def main() raises:
    comptime test_gpu_counts = (2, 4, 8)

    if not enable_p2p():
        raise Error("Cannot enable P2P Mem Access!")

    comptime assert (
        default_accelerator().is_nvidia_gpu()
        or default_accelerator().is_amd_gpu()
    ), "Only NVIDIA and AMD GPUs are supported"

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime num_gpus = test_gpu_counts[gpu_idx]
        if DeviceContext.number_of_devices() != num_gpus:
            continue

        var ctx = List[DeviceContext[]]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        comptime n_experts = num_gpus * 32
        comptime max_recv = n_experts * 64
        comptime for fused in [False, True]:
            test_expert_load[
                BF16Format[3584, 8, max_recv],
                hidden_size=3584,
                top_k=8,
                n_experts=n_experts,
                n_ranks=num_gpus,
                n_tokens_per_rank=64,
                n_layers=2,
                n_steps=5,
                fused_dispatch=fused,
            ](ctx)
            test_expert_load[
                FP8Format[7168, 8, max_recv],
                hidden_size=7168,
                top_k=8,
                n_experts=n_experts,
                n_ranks=num_gpus,
                n_tokens_per_rank=64,
                n_layers=2,
                n_steps=5,
                fused_dispatch=fused,
            ](ctx)

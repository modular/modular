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

from std.math import ceildiv

from std.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.random import rand
from std.memory import alloc
from state_space.gated_delta_conv1d import gated_delta_conv1d_fwd_gpu
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index, IndexList


def run_gated_delta_conv1d_gpu[
    dtype: DType,
    KERNEL_SIZE: Int,
](
    batch_size: Int,
    total_seq_len: Int,
    conv_dim: Int,
    seq_lengths: IndexList,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Test gated_delta_conv1d_fwd_gpu kernel against CPU reference.

    Allocates inputs on host, copies to device, runs GPU kernel, copies
    results back, and compares against a scalar CPU reference implementation.

    Args:
        batch_size: Number of sequences in the batch.
        total_seq_len: Total flattened sequence length (sum of seq_lengths).
        conv_dim: Number of convolution channels.
        seq_lengths: Per-sequence lengths summing to total_seq_len.
        ctx: GPU device context.
        rtol: Relative tolerance for GPU vs CPU comparison.
    """
    comptime CONV1D_BLOCK_DIM = 128
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    var state_len = KERNEL_SIZE - 1

    # ── Allocate host tensors ──────────────────────────────────────────────────
    # qkv_input_ragged: [total_seq_len, conv_dim]
    var qkv_input_heap = alloc[Scalar[dtype]](total_seq_len * conv_dim)
    var qkv_input_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        qkv_input_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, conv_dim)),
    )

    # conv_weight: [conv_dim, KERNEL_SIZE]
    var conv_weight_heap = alloc[Scalar[dtype]](conv_dim * KERNEL_SIZE)
    var conv_weight_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        conv_weight_heap,
        RuntimeLayout[layout_2d].row_major(Index(conv_dim, KERNEL_SIZE)),
    )

    # conv_state_in: [batch_size, conv_dim, state_len], zeroed
    var conv_state_in_heap = alloc[Scalar[dtype]](
        batch_size * conv_dim * state_len
    )
    var conv_state_in_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        conv_state_in_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(batch_size, conv_dim, state_len)
        ),
    ).fill(0)

    # input_row_offsets: [batch_size + 1]
    var input_row_offsets_heap = alloc[Scalar[DType.uint32]](batch_size + 1)
    var input_row_offsets_h = LayoutTensor[
        DType.uint32, layout_1d, MutAnyOrigin
    ](
        input_row_offsets_heap,
        RuntimeLayout[layout_1d].row_major(Index(batch_size + 1)),
    )
    var cumsum = 0
    input_row_offsets_h.ptr.store(0, Scalar[DType.uint32](0))
    for b in range(batch_size):
        cumsum += seq_lengths[b]
        input_row_offsets_h.ptr.store(b + 1, Scalar[DType.uint32](cumsum))

    # conv_output_gpu: [total_seq_len, conv_dim] — receives GPU results
    var conv_output_gpu_heap = alloc[Scalar[dtype]](total_seq_len * conv_dim)
    var conv_output_gpu_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        conv_output_gpu_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, conv_dim)),
    ).fill(0)

    # conv_state_out_gpu: [batch_size, conv_dim, state_len] — receives GPU state
    var conv_state_out_gpu_heap = alloc[Scalar[dtype]](
        batch_size * conv_dim * state_len
    )
    var conv_state_out_gpu_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        conv_state_out_gpu_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(batch_size, conv_dim, state_len)
        ),
    ).fill(0)

    # conv_output_cpu / conv_state_out_cpu: for CPU reference
    var conv_output_cpu_heap = alloc[Scalar[dtype]](total_seq_len * conv_dim)
    var conv_output_cpu_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        conv_output_cpu_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, conv_dim)),
    ).fill(0)

    var conv_state_out_cpu_heap = alloc[Scalar[dtype]](
        batch_size * conv_dim * state_len
    )
    var conv_state_out_cpu_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        conv_state_out_cpu_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(batch_size, conv_dim, state_len)
        ),
    ).fill(0)

    # ── Fill inputs with random values ─────────────────────────────────────────
    rand[dtype](qkv_input_h.ptr, qkv_input_h.size())
    rand[dtype](conv_weight_h.ptr, conv_weight_h.size())

    # ── Allocate device buffers and copy inputs ────────────────────────────────
    var qkv_input_device = ctx.enqueue_create_buffer[dtype](
        total_seq_len * conv_dim
    )
    var conv_weight_device = ctx.enqueue_create_buffer[dtype](
        conv_dim * KERNEL_SIZE
    )
    var conv_state_in_device = ctx.enqueue_create_buffer[dtype](
        batch_size * conv_dim * state_len
    )
    var input_row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var conv_output_device = ctx.enqueue_create_buffer[dtype](
        total_seq_len * conv_dim
    )
    var conv_state_out_device = ctx.enqueue_create_buffer[dtype](
        batch_size * conv_dim * state_len
    )

    with ctx.push_context():
        ctx.enqueue_copy(qkv_input_device, qkv_input_h.ptr)
        ctx.enqueue_copy(conv_weight_device, conv_weight_h.ptr)
        ctx.enqueue_copy(conv_state_in_device, conv_state_in_h.ptr)
        ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_h.ptr)

    # ── Build device TileTensors for kernel call ───────────────────────────────
    var qkv_input_tt = TileTensor(
        qkv_input_device, row_major(Idx(total_seq_len), Idx(conv_dim))
    )
    var conv_weight_tt = TileTensor(
        conv_weight_device, row_major(Idx(conv_dim), Idx(KERNEL_SIZE))
    )
    var conv_state_in_tt = TileTensor(
        conv_state_in_device,
        row_major(Idx(batch_size), Idx(conv_dim), Idx(state_len)),
    )
    var input_row_offsets_tt = TileTensor(
        input_row_offsets_device, row_major(Idx(batch_size + 1))
    )
    var conv_output_tt = TileTensor(
        conv_output_device, row_major(Idx(total_seq_len), Idx(conv_dim))
    )
    var conv_state_out_tt = TileTensor(
        conv_state_out_device,
        row_major(Idx(batch_size), Idx(conv_dim), Idx(state_len)),
    )

    # ── Strides (row-major seqlen-first) ───────────────────────────────────────
    var qkv_input_seqlen_stride: UInt32 = UInt32(conv_dim)
    var qkv_input_channel_stride: UInt32 = 1
    var conv_weight_channel_stride: UInt32 = UInt32(KERNEL_SIZE)
    var conv_weight_offset_stride: UInt32 = 1
    var conv_state_batch_stride: UInt32 = UInt32(conv_dim * state_len)
    var conv_state_channel_stride: UInt32 = UInt32(state_len)
    var conv_state_slot_stride: UInt32 = 1
    var conv_output_seqlen_stride: UInt32 = UInt32(conv_dim)
    var conv_output_channel_stride: UInt32 = 1

    # ── Launch GPU kernel ──────────────────────────────────────────────────────
    var compiled_func = ctx.compile_function[
        gated_delta_conv1d_fwd_gpu[
            dtype,
            KERNEL_SIZE,
            CONV1D_BLOCK_DIM,
            qkv_input_tt.LayoutType,
            conv_weight_tt.LayoutType,
            conv_state_in_tt.LayoutType,
            input_row_offsets_tt.LayoutType,
            conv_output_tt.LayoutType,
            conv_state_out_tt.LayoutType,
        ],
        gated_delta_conv1d_fwd_gpu[
            dtype,
            KERNEL_SIZE,
            CONV1D_BLOCK_DIM,
            qkv_input_tt.LayoutType,
            conv_weight_tt.LayoutType,
            conv_state_in_tt.LayoutType,
            input_row_offsets_tt.LayoutType,
            conv_output_tt.LayoutType,
            conv_state_out_tt.LayoutType,
        ],
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            batch_size,
            total_seq_len,
            conv_dim,
            qkv_input_tt,
            conv_weight_tt,
            conv_state_in_tt,
            input_row_offsets_tt,
            conv_output_tt,
            conv_state_out_tt,
            qkv_input_seqlen_stride,
            qkv_input_channel_stride,
            conv_weight_channel_stride,
            conv_weight_offset_stride,
            conv_state_batch_stride,
            conv_state_channel_stride,
            conv_state_slot_stride,
            conv_output_seqlen_stride,
            conv_output_channel_stride,
            grid_dim=(batch_size, ceildiv(conv_dim, CONV1D_BLOCK_DIM)),
            block_dim=(CONV1D_BLOCK_DIM,),
        )

    # ── Copy GPU results back to host ──────────────────────────────────────────
    with ctx.push_context():
        ctx.enqueue_copy(conv_output_gpu_h.ptr, conv_output_device)
        ctx.enqueue_copy(conv_state_out_gpu_h.ptr, conv_state_out_device)
    ctx.synchronize()

    # ── CPU reference implementation ───────────────────────────────────────────
    comptime KERNEL_SIZE_MINUS_ONE = KERNEL_SIZE - 1

    for batch_item_idx in range(batch_size):
        var sequence_start = Int(input_row_offsets_h.ptr.load(batch_item_idx))
        var sequence_end = Int(input_row_offsets_h.ptr.load(batch_item_idx + 1))
        var sequence_length = sequence_end - sequence_start

        for conv_channel_idx in range(conv_dim):
            for token_pos in range(sequence_length):
                var flat_token_idx = sequence_start + token_pos
                var conv_sum: Scalar[dtype] = 0

                comptime for kernel_offset_k in range(KERNEL_SIZE):
                    var lookback = token_pos - (
                        KERNEL_SIZE_MINUS_ONE - kernel_offset_k
                    )
                    var input_value: Scalar[dtype] = 0

                    if lookback >= 0:
                        input_value = qkv_input_h.ptr.load(
                            UInt32(sequence_start + lookback)
                            * qkv_input_seqlen_stride
                            + UInt32(conv_channel_idx)
                            * qkv_input_channel_stride
                        )
                    else:
                        var state_slot = KERNEL_SIZE_MINUS_ONE + lookback
                        if state_slot >= 0:
                            input_value = conv_state_in_h.ptr.load(
                                UInt32(batch_item_idx) * conv_state_batch_stride
                                + UInt32(conv_channel_idx)
                                * conv_state_channel_stride
                                + UInt32(state_slot) * conv_state_slot_stride
                            )

                    var weight = conv_weight_h.ptr.load(
                        UInt32(conv_channel_idx) * conv_weight_channel_stride
                        + UInt32(kernel_offset_k) * conv_weight_offset_stride
                    )
                    conv_sum = conv_sum + input_value * weight

                conv_output_cpu_h.ptr.store(
                    UInt32(flat_token_idx) * conv_output_seqlen_stride
                    + UInt32(conv_channel_idx) * conv_output_channel_stride,
                    conv_sum,
                )

            comptime for state_slot_j in range(KERNEL_SIZE_MINUS_ONE):
                var source_pos = (
                    sequence_length - KERNEL_SIZE_MINUS_ONE + state_slot_j
                )
                var state_value: Scalar[dtype] = 0

                if source_pos >= 0:
                    state_value = qkv_input_h.ptr.load(
                        UInt32(sequence_start + source_pos)
                        * qkv_input_seqlen_stride
                        + UInt32(conv_channel_idx) * qkv_input_channel_stride
                    )
                else:
                    var old_slot = KERNEL_SIZE_MINUS_ONE + source_pos
                    if old_slot >= 0:
                        state_value = conv_state_in_h.ptr.load(
                            UInt32(batch_item_idx) * conv_state_batch_stride
                            + UInt32(conv_channel_idx)
                            * conv_state_channel_stride
                            + UInt32(old_slot) * conv_state_slot_stride
                        )

                conv_state_out_cpu_h.ptr.store(
                    UInt32(batch_item_idx) * conv_state_batch_stride
                    + UInt32(conv_channel_idx) * conv_state_channel_stride
                    + UInt32(state_slot_j) * conv_state_slot_stride,
                    state_value,
                )

    # ── Compare GPU vs CPU outputs ─────────────────────────────────────────────
    var output_size = total_seq_len * conv_dim
    for i in range(output_size):
        assert_almost_equal(
            conv_output_gpu_h.ptr[i],
            conv_output_cpu_h.ptr[i],
            rtol=rtol,
        )

    var state_size = batch_size * conv_dim * state_len
    for i in range(state_size):
        assert_almost_equal(
            conv_state_out_gpu_h.ptr[i],
            conv_state_out_cpu_h.ptr[i],
            rtol=rtol,
        )

    # ── Cleanup ────────────────────────────────────────────────────────────────
    qkv_input_heap.free()
    conv_weight_heap.free()
    conv_state_in_heap.free()
    input_row_offsets_heap.free()
    conv_output_gpu_heap.free()
    conv_state_out_gpu_heap.free()
    conv_output_cpu_heap.free()
    conv_state_out_cpu_heap.free()


# =============================================================================
# Test functions
# =============================================================================


def test_gated_delta_conv1d_kernel_size_4_single_sequence() raises:
    """Test kernel_size=4 with a single sequence."""
    var ctx = DeviceContext()
    run_gated_delta_conv1d_gpu[DType.float32, 4](
        batch_size=1,
        total_seq_len=10,
        conv_dim=8,
        seq_lengths=Index(10),
        ctx=ctx,
    )


def test_gated_delta_conv1d_kernel_size_4_multiple_sequences() raises:
    """Test kernel_size=4 with multiple variable-length sequences."""
    var ctx = DeviceContext()
    run_gated_delta_conv1d_gpu[DType.float32, 4](
        batch_size=3,
        total_seq_len=15,
        conv_dim=16,
        seq_lengths=Index(5, 7, 3),
        ctx=ctx,
    )


def test_gated_delta_conv1d_kernel_size_4_large_conv_dim() raises:
    """Test kernel_size=4 with a conv_dim larger than the thread block."""
    var ctx = DeviceContext()
    run_gated_delta_conv1d_gpu[DType.float32, 4](
        batch_size=2,
        total_seq_len=40,
        conv_dim=256,
        seq_lengths=Index(20, 20),
        ctx=ctx,
    )


def test_gated_delta_conv1d_kernel_size_4_short_sequences() raises:
    """Test kernel_size=4 with sequences shorter than the kernel."""
    var ctx = DeviceContext()
    run_gated_delta_conv1d_gpu[DType.float32, 4](
        batch_size=3,
        total_seq_len=6,
        conv_dim=8,
        seq_lengths=Index(2, 3, 1),
        ctx=ctx,
    )


def test_gated_delta_conv1d_kernel_size_3() raises:
    """Test kernel_size=3."""
    var ctx = DeviceContext()
    run_gated_delta_conv1d_gpu[DType.float32, 3](
        batch_size=2,
        total_seq_len=20,
        conv_dim=16,
        seq_lengths=Index(8, 12),
        ctx=ctx,
    )


def test_gated_delta_conv1d_kernel_size_5() raises:
    """Test kernel_size=5."""
    var ctx = DeviceContext()
    run_gated_delta_conv1d_gpu[DType.float32, 5](
        batch_size=1,
        total_seq_len=10,
        conv_dim=16,
        seq_lengths=Index(10),
        ctx=ctx,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

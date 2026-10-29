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

"""Keep the BF16 sparse MLA capture key aligned with the launched split grid."""

from max.gpu.host import (
    DeviceAttribute,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
    default_accelerator,
)
from nn.attention.gpu.mha_decode_partition_heuristic import (
    amd_sparse_mla_bf16_num_partitions,
)
from nn.attention.gpu.mla_decode_dispatch_scalars import (
    mla_decode_dispatch_scalars,
)
from nn.attention.gpu.mla_graph import (
    _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS,
    _amd_sparse_mla_compact_indices,
)
from layout import TileTensor, row_major
from std.testing import assert_equal

comptime stride = 2051
comptime row_bytes = 1152


# MI355X's CU count, which the expected splits below assume.
comptime mi355x_cus = 256


def _check_key(
    ctx: DeviceContext,
    heads: Int,
    batch: Int,
    q_len: Int,
    want_bf16: Int,
    want_fp8: Int,
) raises:
    """Checks the capture key on a BF16 and an FP8 cache against the split
    each arm launches, pinned at MI355X's CU count so a change to either
    side shows up here rather than agreeing by construction."""
    var sm_count = ctx.device().get_attribute(
        DeviceAttribute.MULTIPROCESSOR_COUNT
    )
    comptime context_length = 80000
    var _, _, bf16_key = mla_decode_dispatch_scalars(
        batch, context_length, q_len, heads, False, ctx, stride
    )
    var _, _, fp8_key = mla_decode_dispatch_scalars(
        batch, context_length, q_len, heads, True, ctx, stride
    )
    var bf16_launch = amd_sparse_mla_bf16_num_partitions(
        batch * q_len, stride, heads, sm_count
    )
    assert_equal(bf16_key, bf16_launch)
    if sm_count == mi355x_cus:
        assert_equal(bf16_key, want_bf16)
        assert_equal(fp8_key, want_fp8)


def _check_replay(ctx: DeviceContext, batch: Int) raises:
    var index_count = batch * stride
    var indices_host = ctx.enqueue_create_host_buffer[.int32](index_count)
    var topk_host = ctx.enqueue_create_host_buffer[.int32](batch)
    var indices = ctx.enqueue_create_buffer[.int32](index_count)
    var topk = ctx.enqueue_create_buffer[.int32](batch)
    var compact = ctx.enqueue_create_buffer[.int32](index_count)
    var lengths = ctx.enqueue_create_buffer[.uint32](batch)
    var zero = ctx.enqueue_create_buffer[.uint8](row_bytes)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            rec.enqueue_function[
                _amd_sparse_mla_compact_indices[
                    _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS
                ]
            ](
                indices.unsafe_ptr().as_unsafe_any_origin(),
                topk.unsafe_ptr().as_unsafe_any_origin(),
                compact.unsafe_ptr().as_unsafe_any_origin(),
                lengths.unsafe_ptr().as_unsafe_any_origin(),
                Int32(stride),
                zero.unsafe_ptr().as_unsafe_any_origin(),
                Int32(row_bytes),
                grid_dim=(batch, 1, 1),
                block_dim=(_AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS, 1, 1),
            )

    var graph = DeviceGraph.create(ctx, build)
    var indices_rows = TileTensor(indices_host, row_major(batch, stride))
    comptime assert indices_rows.flat_rank == 2
    for replay in range(3):
        for i in range(index_count):
            indices_host[i] = -1
        for row in range(batch):
            topk_host[row] = Int32(
                0 if replay == 2 else (4 if replay == 0 else 2)
            )
            if replay == 0:
                indices_rows[row, 0] = 1
                indices_rows[row, 2] = 3
            elif replay == 1:
                indices_rows[row, 0] = 7
                indices_rows[row, 1] = 9
        ctx.enqueue_copy(indices, indices_host)
        ctx.enqueue_copy(topk, topk_host)
        zero.enqueue_fill(UInt8(255))
        graph.replay()
        var got = ctx.enqueue_create_host_buffer[.int32](index_count)
        var got_lengths = ctx.enqueue_create_host_buffer[.uint32](batch)
        var got_zero = ctx.enqueue_create_host_buffer[.uint8](row_bytes)
        ctx.enqueue_copy(got, compact)
        ctx.enqueue_copy(got_lengths, lengths)
        ctx.enqueue_copy(got_zero, zero)
        ctx.synchronize()
        var got_rows = TileTensor(got, row_major(batch, stride))
        comptime assert got_rows.flat_rank == 2
        for row in range(batch):
            assert_equal(got_lengths[row], UInt32(0 if replay == 2 else 2))
            if replay < 2:
                assert_equal(got_rows[row, 0], Int32(1 if replay == 0 else 7))
                assert_equal(got_rows[row, 1], Int32(3 if replay == 0 else 9))
            assert_equal(got_rows[row, 2], -1)
        for i in range(row_bytes):
            assert_equal(got_zero[i], UInt8(0))
        _ = got^
        _ = got_lengths^
        _ = got_zero^


def main() raises:
    comptime if default_accelerator().is_amd_gpu():
        with DeviceContext() as ctx:
            # (heads, batch, q_len, BF16-cache split, FP8-cache split). At 16
            # heads or fewer the split takes its other branch.
            _check_key(ctx, 32, 1, 1, 16, 16)
            _check_key(ctx, 32, 1, 4, 16, 16)
            _check_key(ctx, 32, 8, 1, 16, 16)
            _check_key(ctx, 32, 8, 4, 4, 8)
            _check_key(ctx, 32, 32, 1, 4, 8)
            _check_key(ctx, 32, 32, 4, 2, 2)
            _check_key(ctx, 32, 128, 1, 2, 2)
            _check_key(ctx, 32, 128, 4, 1, 1)
            _check_key(ctx, 32, 257, 1, 1, 1)
            _check_key(ctx, 32, 257, 4, 1, 1)
            _check_key(ctx, 16, 1, 1, 32, 64)
            _check_key(ctx, 16, 1, 4, 32, 16)
            _check_key(ctx, 16, 8, 1, 32, 32)
            _check_key(ctx, 16, 8, 4, 16, 16)
            _check_key(ctx, 16, 32, 1, 16, 8)
            _check_key(ctx, 16, 32, 4, 4, 4)
            _check_key(ctx, 16, 128, 1, 4, 2)
            _check_key(ctx, 16, 128, 4, 1, 1)
            _check_key(ctx, 16, 257, 1, 1, 1)
            _check_key(ctx, 16, 257, 4, 1, 1)
            _check_replay(ctx, 1)
            _check_replay(ctx, 32)
            _check_replay(ctx, 257)

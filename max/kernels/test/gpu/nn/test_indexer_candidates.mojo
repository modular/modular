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
"""Check candidate ordering, padding, and sparse-column remapping."""

from std.builtin.sort import sort
from std.collections import List
from std.math import clamp
from std.testing import assert_equal
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from nn.indexer_candidates import (
    indexer_expand_candidates,
    indexer_remap_sparse_columns,
)


def _host(ctx: DeviceContext, size: Int) raises -> HostBuffer[.int32]:
    var buffer = ctx.enqueue_create_host_buffer[.int32](size)
    for i in range(size):
        buffer[i] = Int32.MIN
    return buffer^


def _upload(
    ctx: DeviceContext, host: HostBuffer[.int32]
) raises -> DeviceBuffer[.int32]:
    var buffer = ctx.enqueue_create_buffer[.int32](len(host))
    ctx.enqueue_copy(dst_buf=buffer, src_buf=host)
    return buffer^


def _compare(
    ctx: DeviceContext,
    actual: DeviceBuffer[.int32],
    expected: HostBuffer[.int32],
) raises:
    var host = ctx.enqueue_create_host_buffer[.int32](len(actual))
    ctx.enqueue_copy(dst_buf=host, src_buf=actual)
    ctx.synchronize()
    for i in range(len(expected)):
        assert_equal(host[i], expected[i], String("offset=", i))


def _case[
    candidates: Int, ratio: Int, block_kv: Int
](ctx: DeviceContext, rows: Int, seed: Int) raises:
    comptime size = candidates * ratio
    comptime input_stride = candidates + 3
    comptime output_stride = size + 7
    comptime bound_stride = 2
    comptime selected = 513
    comptime selected_stride = selected + 7
    comptime width = size * block_kv - 3
    var input = _host(ctx, rows * input_stride + 17)
    var starts = _host(ctx, rows * bound_stride + 17)
    var stops = _host(ctx, rows * bound_stride + 17)
    var expected = _host(ctx, rows * output_stride + 17)
    var expected_ends = _host(ctx, rows * bound_stride + 17)
    var columns = _host(ctx, rows * selected_stride + 17)
    var expected_remap = _host(ctx, rows * selected_stride + 17)
    for row in range(rows):
        var start = row * 3 + seed % block_kv
        var length = (row * 17 + seed) % (size * block_kv + 1)
        if row % 7 == 2:
            length = 7 * block_kv - 1
        elif row % 7 == 4:
            length = size * block_kv
        elif row % 7 == 5:
            length = 0
        elif row % 7 == 6:
            length = -1
        starts[row * bound_stride] = Int32(start)
        stops[row * bound_stride] = Int32(start + length)
        var valid = List[Int32]()
        for i in range(candidates):
            var candidate = (i * 37 + seed * 13) % (candidates * 2)
            if row % 7 == 0 or i % 11 == 10:
                candidate = -1
            elif row % 7 == 1:
                candidate = candidates - 1 - i
            elif row % 7 == 2:
                candidate = 3
            elif row % 7 == 4:
                candidate = i
            input[row * input_stride + i] = Int32(candidate)
            for part in range(ratio):
                var expanded = candidate * ratio + part
                if expanded >= 0 and expanded * block_kv < max(length, 0):
                    valid.append(Int32(expanded))
        sort(valid)
        var base = start // block_kv
        var last = Int(valid[len(valid) - 1]) if len(valid) > 0 else 0
        for i in range(size):
            expected[row * output_stride + i] = Int32(base) + (
                valid[i] if i < len(valid) else Int32(last)
            )
        var end = 0
        if len(valid) > 0:
            end = (len(valid) - 1) * block_kv + clamp(
                length - last * block_kv, 1, block_kv
            )
        expected_ends[row * bound_stride] = Int32(end)
        for i in range(selected):
            var column = (i * 13 + seed) % (width + 3)
            if i % 11 == 0:
                column = -1
            elif i % 11 == 1:
                column = width
            elif i % 11 == 2:
                column = Int(Int32.MIN)
            columns[row * selected_stride + i] = Int32(column)
            var position = Int32(-1)
            if 0 <= column and column < width:
                position = (
                    expected[row * output_stride + column // block_kv]
                    - Int32(base)
                ) * Int32(block_kv) + Int32(column % block_kv)
            expected_remap[row * selected_stride + i] = position
    var d_input = _upload(ctx, input)
    var d_starts = _upload(ctx, starts)
    var d_stops = _upload(ctx, stops)
    var d_columns = _upload(ctx, columns)
    var d_output = ctx.enqueue_create_buffer[.int32](len(expected))
    var d_ends = ctx.enqueue_create_buffer[.int32](len(expected_ends))
    var d_remap = ctx.enqueue_create_buffer[.int32](len(expected_remap))
    d_output.enqueue_fill(Int32.MIN)
    d_ends.enqueue_fill(Int32.MIN)
    d_remap.enqueue_fill(Int32.MIN)
    indexer_expand_candidates[candidates, ratio, block_kv](
        TileTensor(d_input, row_major(rows, input_stride)),
        TileTensor(d_starts, row_major(rows, bound_stride)),
        TileTensor(d_stops, row_major(rows, bound_stride)),
        TileTensor(d_output, row_major(rows, output_stride)),
        TileTensor(d_ends, row_major(rows, bound_stride)),
        ctx,
    )
    _compare(ctx, d_output, expected)
    _compare(ctx, d_ends, expected_ends)
    indexer_remap_sparse_columns[block_kv](
        TileTensor(d_columns, row_major(rows, selected_stride)),
        TileTensor(d_output, row_major(rows, output_stride)),
        TileTensor(d_starts, row_major(rows, bound_stride)),
        TileTensor(d_remap, row_major(rows, selected_stride)),
        selected,
        width,
        ctx,
    )
    _compare(ctx, d_remap, expected_remap)


def main() raises:
    var ctx = DeviceContext()
    comptime for ci in range(3):
        comptime candidates = 8 if ci == 0 else (128 if ci == 1 else 2048)
        comptime for ratio in range(1, 3):
            comptime for bi in range(2):
                comptime block_kv = 8 << bi
                for rows in [0, 1, 13]:
                    for seed in [1, 19, 2026]:
                        _case[candidates, ratio, block_kv](ctx, rows, seed)
    print("PASS 108 candidate and remap cases")

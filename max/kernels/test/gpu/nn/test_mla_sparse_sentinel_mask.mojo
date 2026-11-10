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

"""Checks the physical-sentinel mask word with both load algorithms."""

from std.testing import assert_equal
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from layout import Coord, TensorEngine, TensorLayout, TileTensor, row_major
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    _sparse_physical_sentinel_bits,
)

comptime PATTERNS = 6
comptime BASES = 4
comptime CASES = 33 * PATTERNS * BASES
comptime THREADS = 128


def _mask_kernel[
    InputLayout: TensorLayout,
    OutputLayout: TensorLayout,
    InputEngine: TensorEngine,
    OutputEngine: TensorEngine,
](
    indices: TileTensor[.int32, InputLayout, MutAnyOrigin, Engine=InputEngine],
    output: TileTensor[
        .uint32, OutputLayout, MutAnyOrigin, Engine=OutputEngine
    ],
    stride: Int64,
):
    var tid = Int(thread_idx.x)
    var case_idx = Int(block_idx.x) * (THREADS // 32) + tid // 32
    var readable = case_idx // (PATTERNS * BASES)
    var base = case_idx % BASES
    var row = indices.ptr_at_offset(Coord(case_idx * Int(stride) + base))
    # Every warp calls both modes, including lanes beyond a short readable tail.
    var old_bits = _sparse_physical_sentinel_bits[False](row, readable)
    var ballot_bits = _sparse_physical_sentinel_bits[True](row, readable)
    var dst = (case_idx * 32 + (tid & 31)) * 2
    output[dst] = old_bits
    output[dst + 1] = ballot_bits


def _run(stride: Int, ctx: DeviceContext) raises:
    var host = ctx.enqueue_create_host_buffer[.int32](CASES * stride)
    for i in range(CASES * stride):
        host[i] = -1
    for case_idx in range(CASES):
        var pattern = (case_idx // BASES) % PATTERNS
        var row = case_idx * stride + case_idx % BASES
        for j in range(32):
            var value: Int32 = -1
            if pattern == 1:
                value = Int32.MAX if j & 1 else Int32(0)
            elif pattern == 2 and j % 2 == 0:
                value = Int32(j)
            elif pattern == 3 and j != 31:
                value = Int32.MAX
            elif pattern == 4 and j == 17:
                value = 0
            elif pattern == 5:
                value = Int32.MIN if j % 3 == 0 else Int32(j * 104729)
            host[row + j] = value

    var indices = ctx.enqueue_create_buffer[.int32](CASES * stride)
    ctx.enqueue_copy(indices, host)
    var result = ctx.enqueue_create_buffer[.uint32](CASES * 32 * 2)
    var input_tt = TileTensor(indices, row_major((CASES * stride,)))
    var output_tt = TileTensor(result, row_major((CASES * 32 * 2,)))
    comptime kernel = _mask_kernel[
        type_of(input_tt).LayoutType,
        type_of(output_tt).LayoutType,
        type_of(input_tt).Engine,
        type_of(output_tt).Engine,
    ]
    ctx.enqueue_function[kernel](
        input_tt,
        output_tt,
        Int64(stride),
        grid_dim=CASES // (THREADS // 32),
        block_dim=THREADS,
    )
    var actual = ctx.enqueue_create_host_buffer[.uint32](CASES * 32 * 2)
    ctx.enqueue_copy(actual, result)
    ctx.synchronize()
    for case_idx in range(CASES):
        var readable = case_idx // (PATTERNS * BASES)
        var row = case_idx * stride + case_idx % BASES
        var expected: UInt32 = 0
        for j in range(readable):
            if host[row + j] >= 0:
                expected |= UInt32(1) << UInt32(j)
        for lane in range(32):
            var dst = (case_idx * 32 + lane) * 2
            assert_equal(actual[dst], expected)
            assert_equal(actual[dst + 1], expected)
    _ = indices
    _ = result
    _ = host


def main() raises:
    with DeviceContext() as ctx:
        _run(37, ctx)
        _run(65, ctx)

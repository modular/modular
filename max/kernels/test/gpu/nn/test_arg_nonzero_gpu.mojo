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

from layout import TileTensor, col_major, row_major
from max.gpu.host import DeviceContext
from nn.arg_nonzero import _arg_nonzero_gpu
from std.math import nan
from std.testing import assert_equal, TestSuite


def test_arg_nonzero_gpu_order_and_float_semantics() raises:
    comptime input_layout = row_major[2, 4]()
    comptime output_layout = row_major[4, 2]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](8)
        input_host[0] = 0.0
        input_host[1] = -0.0
        input_host[2] = nan[.float32]()
        input_host[3] = -2.0
        input_host[4] = 0.0
        input_host[5] = 3.0
        input_host[6] = 0.0
        input_host[7] = 4.0

        var input_device = ctx.enqueue_create_buffer[.float32](8)
        var output_device = ctx.enqueue_create_buffer[.int64](8)
        ctx.enqueue_copy(input_device, input_host)

        _arg_nonzero_gpu(
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.int64](8)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        # Scalar expectations: SIMD constants must have power-of-two length.
        var expected = Array[Int, 8](fill={})
        expected[1] = 2
        expected[2] = 0
        expected[3] = 3
        expected[4] = 1
        expected[5] = 1
        expected[6] = 1
        expected[7] = 3
        for i in range(8):
            assert_equal(Int(output_host[i]), expected[i])


def test_arg_nonzero_gpu_rank3_integer() raises:
    comptime input_layout = row_major[2, 2, 3]()
    comptime output_layout = row_major[3, 3]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.int32](12)
        for i in range(12):
            input_host[i] = 0
        input_host[1] = -1
        input_host[6] = 2
        input_host[11] = 3

        var input_device = ctx.enqueue_create_buffer[.int32](12)
        var output_device = ctx.enqueue_create_buffer[.int64](9)
        ctx.enqueue_copy(input_device, input_host)

        _arg_nonzero_gpu(
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.int64](9)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        # Scalar expectations: SIMD constants must have power-of-two length.
        # Row-major [3,3] rows: (0,0,1), (1,0,0), (1,1,2).
        var expected = Array[Int, 9](fill={})
        expected[2] = 1
        expected[3] = 1
        expected[6] = 1
        expected[7] = 1
        expected[8] = 2
        for i in range(9):
            assert_equal(Int(output_host[i]), expected[i])


def test_arg_nonzero_gpu_multi_tile() raises:
    """Several tiles: the per-tile offset scan must order rows across blocks.

    700 elements over 3 tiles (BLOCK_SIZE 256, tile_len 234): nonzeros at
    233, 234, 467, 468, 699 in [700] pair tiles 0/1 and 1/2 across the
    tile boundary, and 699 exercises the final short tile.
    """
    comptime input_layout = row_major[700]()
    comptime output_layout = row_major[5, 1]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.int32](700)
        for i in range(700):
            input_host[i] = 0
        input_host[233] = 1
        input_host[234] = 2
        input_host[467] = 3
        input_host[468] = 4
        input_host[699] = 5

        var input_device = ctx.enqueue_create_buffer[.int32](700)
        var output_device = ctx.enqueue_create_buffer[.int64](5)
        ctx.enqueue_copy(input_device, input_host)

        _arg_nonzero_gpu(
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.int64](5)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var expected = Array[Int, 5](fill={})
        expected[0] = 233
        expected[1] = 234
        expected[2] = 467
        expected[3] = 468
        expected[4] = 699
        for i in range(5):
            assert_equal(Int(output_host[i]), expected[i])


def test_arg_nonzero_gpu_all_nonzero_at_tile_limit() raises:
    """65536 elements (BLOCK_SIZE * BLOCK_SIZE): every element is nonzero.

    Before the chunked tile walk, each block read only its first
    BLOCK_SIZE elements; tile_len is exactly BLOCK_SIZE here, so this
    input pins the exact boundary where larger inputs start dropping
    elements, and checks the full count/offsets/emit pipeline end to
    end with one row per element.
    """
    comptime numel = 65536
    comptime input_layout = row_major[numel]()
    comptime output_layout = row_major[numel, 1]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.int32](numel)
        for i in range(numel):
            input_host[i] = 1

        var input_device = ctx.enqueue_create_buffer[.int32](numel)
        var output_device = ctx.enqueue_create_buffer[.int64](numel)
        ctx.enqueue_copy(input_device, input_host)

        _arg_nonzero_gpu(
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.int64](numel)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        for i in range(numel):
            assert_equal(Int(output_host[i]), i)


def test_arg_nonzero_gpu_first_element_past_tile_block() raises:
    """65537 elements with nonzeros at flat 256 and 65536.

    Both indices were silently skipped before the chunked tile walk: 256
    is the first element past a full BLOCK_SIZE chunk, and 65536 is the
    single element beyond the 256x256 grid coverage.
    """
    comptime numel = 65537
    comptime input_layout = row_major[numel]()
    comptime output_layout = row_major[2, 1]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.int32](numel)
        for i in range(numel):
            input_host[i] = 0
        input_host[256] = 1
        input_host[65536] = 2

        var input_device = ctx.enqueue_create_buffer[.int32](numel)
        var output_device = ctx.enqueue_create_buffer[.int64](2)
        ctx.enqueue_copy(input_device, input_host)

        _arg_nonzero_gpu(
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.int64](2)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        assert_equal(Int(output_host[0]), 256)
        assert_equal(Int(output_host[1]), 65536)


def test_arg_nonzero_gpu_scattered_within_tiles() raises:
    """100000 elements with nonzeros scattered past each tile's first block.

    With 256 tiles of 391 elements, every nonzero here lies past offset
    255 within its tile, i.e. inside the second BLOCK_SIZE chunk that
    was never read before the chunked walk.
    """
    comptime numel = 100000
    comptime input_layout = row_major[numel]()
    comptime output_layout = row_major[5, 1]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.int32](numel)
        for i in range(numel):
            input_host[i] = 0
        input_host[300] = 1
        input_host[700] = 2
        input_host[12345] = 3
        input_host[65537] = 4
        input_host[99999] = 5

        var input_device = ctx.enqueue_create_buffer[.int32](numel)
        var output_device = ctx.enqueue_create_buffer[.int64](5)
        ctx.enqueue_copy(input_device, input_host)

        _arg_nonzero_gpu(
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.int64](5)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var expected = Array[Int, 5](fill={})
        expected[0] = 300
        expected[1] = 700
        expected[2] = 12345
        expected[3] = 65537
        expected[4] = 99999
        for i in range(5):
            assert_equal(Int(output_host[i]), expected[i])


def test_arg_nonzero_gpu_rejects_column_major_input() raises:
    """A non-row-major input must raise, not misdecode coordinates.

    The kernels walk memory linearly and decode coordinates from row-major
    strides. The graph compiler materializes strided views before they reach
    this op, so a column-major input is unreachable in practice; this test
    pins the loud failure if that guarantee ever regresses.
    """
    comptime input_layout = col_major[2, 4]()
    comptime output_layout = row_major[4, 2]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](8)
        for i in range(8):
            input_host[i] = 0.0
        input_host[0] = 1.0

        var input_device = ctx.enqueue_create_buffer[.float32](8)
        var output_device = ctx.enqueue_create_buffer[.int64](8)
        ctx.enqueue_copy(input_device, input_host)

        var raised = False
        try:
            _arg_nonzero_gpu(
                TileTensor(input_device, input_layout),
                TileTensor(output_device, output_layout),
                ctx,
            )
        except Error:
            raised = True
        assert_equal(raised, True)


def main() raises:
    var suite = TestSuite()
    suite.test[test_arg_nonzero_gpu_order_and_float_semantics]()
    suite.test[test_arg_nonzero_gpu_rank3_integer]()
    suite.test[test_arg_nonzero_gpu_multi_tile]()
    suite.test[test_arg_nonzero_gpu_all_nonzero_at_tile_limit]()
    suite.test[test_arg_nonzero_gpu_first_element_past_tile_block]()
    suite.test[test_arg_nonzero_gpu_scattered_within_tiles]()
    suite.test[test_arg_nonzero_gpu_rejects_column_major_input]()
    suite^.run()

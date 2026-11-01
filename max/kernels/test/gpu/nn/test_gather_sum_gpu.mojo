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

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.gather_scatter import gather_sum
from std.testing import assert_equal, TestSuite


def test_gather_sum_gpu_basic() raises:
    """Gather rows then sum over the multi-hot dimension.

    input = [[0..11] as 4x3], indices = [[0, 2], [1, 3]]:
    output[0] = input[0] + input[2], output[1] = input[1] + input[3].
    """
    comptime table_layout = row_major[4, 3]()
    comptime indices_layout = row_major[2, 2]()
    comptime output_layout = row_major[2, 3]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](12)
        for i in range(12):
            input_host[i] = Float32(i)

        var indices_host = ctx.enqueue_create_host_buffer[.int32](4)
        indices_host[0] = 0
        indices_host[1] = 2
        indices_host[2] = 1
        indices_host[3] = 3

        var input_device = ctx.enqueue_create_buffer[.float32](12)
        var indices_device = ctx.enqueue_create_buffer[.int32](4)
        var output_device = ctx.enqueue_create_buffer[.float32](6)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(indices_device, indices_host)

        gather_sum[target="gpu"](
            TileTensor(output_device, output_layout),
            TileTensor(input_device, table_layout),
            TileTensor(indices_device, indices_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](6)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        # row 0: input[0] + input[2] = [0,1,2] + [6,7,8] = [6, 8, 10]
        # row 1: input[1] + input[3] = [3,4,5] + [9,10,11] = [12, 14, 16]
        var expected = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        for i in range(6):
            assert_equal(output_host[i], Float32(expected[i]))


def test_gather_sum_gpu_single_index() raises:
    """Single index per row — sum is a plain row copy."""
    comptime table_layout = row_major[3, 2]()
    comptime indices_layout = row_major[2, 1]()
    comptime output_layout = row_major[2, 2]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](6)
        input_host[0] = Float32(10.0)
        input_host[1] = Float32(20.0)
        input_host[2] = Float32(30.0)
        input_host[3] = Float32(40.0)
        input_host[4] = Float32(50.0)
        input_host[5] = Float32(60.0)

        var indices_host = ctx.enqueue_create_host_buffer[.int32](2)
        indices_host[0] = 2
        indices_host[1] = 0

        var input_device = ctx.enqueue_create_buffer[.float32](6)
        var indices_device = ctx.enqueue_create_buffer[.int32](2)
        var output_device = ctx.enqueue_create_buffer[.float32](4)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(indices_device, indices_host)

        gather_sum[target="gpu"](
            TileTensor(output_device, output_layout),
            TileTensor(input_device, table_layout),
            TileTensor(indices_device, indices_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](4)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        assert_equal(output_host[0], Float32(50.0))
        assert_equal(output_host[1], Float32(60.0))
        assert_equal(output_host[2], Float32(10.0))
        assert_equal(output_host[3], Float32(20.0))


def test_gather_sum_gpu_int64_data() raises:
    """Integer input dtype exercises the generic-dtype kernel path."""
    comptime table_layout = row_major[4, 2]()
    comptime indices_layout = row_major[2, 2]()
    comptime output_layout = row_major[2, 2]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.int64](8)
        for i in range(8):
            input_host[i] = Int64(i)

        var indices_host = ctx.enqueue_create_host_buffer[.int32](4)
        indices_host[0] = 0
        indices_host[1] = 1
        indices_host[2] = 2
        indices_host[3] = 3

        var input_device = ctx.enqueue_create_buffer[.int64](8)
        var indices_device = ctx.enqueue_create_buffer[.int32](4)
        var output_device = ctx.enqueue_create_buffer[.int64](4)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(indices_device, indices_host)

        gather_sum[target="gpu"](
            TileTensor(output_device, output_layout),
            TileTensor(input_device, table_layout),
            TileTensor(indices_device, indices_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.int64](4)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        # row 0: input[0] + input[1] = [0,1] + [2,3] = [2, 4]
        # row 1: input[2] + input[3] = [4,5] + [6,7] = [10, 12]
        assert_equal(Int(output_host[0]), 2)
        assert_equal(Int(output_host[1]), 4)
        assert_equal(Int(output_host[2]), 10)
        assert_equal(Int(output_host[3]), 12)


def test_gather_sum_gpu_negative_indices() raises:
    """Negative indices normalize against the table size (row-major wrap)."""
    comptime table_layout = row_major[4, 3]()
    comptime indices_layout = row_major[2, 2]()
    comptime output_layout = row_major[2, 3]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](12)
        for i in range(12):
            input_host[i] = Float32(i)

        # -4 == 0 and -1 == 3 after normalization.
        var indices_host = ctx.enqueue_create_host_buffer[.int32](4)
        indices_host[0] = -4
        indices_host[1] = -1
        indices_host[2] = 0
        indices_host[3] = 3

        var input_device = ctx.enqueue_create_buffer[.float32](12)
        var indices_device = ctx.enqueue_create_buffer[.int32](4)
        var output_device = ctx.enqueue_create_buffer[.float32](6)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(indices_device, indices_host)

        gather_sum[target="gpu"](
            TileTensor(output_device, output_layout),
            TileTensor(input_device, table_layout),
            TileTensor(indices_device, indices_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](6)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        # row 0: input[0] + input[3] = [0,1,2] + [9,10,11] = [9, 11, 13]
        # row 1: input[0] + input[3] = same (duplicate ids are legal).
        var expected = [9.0, 11.0, 13.0, 9.0, 11.0, 13.0]
        for i in range(6):
            assert_equal(output_host[i], Float32(expected[i]))


def test_gather_sum_gpu_rank3_output() raises:
    """Rank-3 output ([batch, 1, emb_dim]) matches the fused reduce.add
    result, which keeps the reduced axis at size 1."""
    comptime table_layout = row_major[4, 3]()
    comptime indices_layout = row_major[2, 2]()
    comptime output_layout = row_major[2, 1, 3]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](12)
        for i in range(12):
            input_host[i] = Float32(i)

        var indices_host = ctx.enqueue_create_host_buffer[.int32](4)
        indices_host[0] = 0
        indices_host[1] = 2
        indices_host[2] = 1
        indices_host[3] = 3

        var input_device = ctx.enqueue_create_buffer[.float32](12)
        var indices_device = ctx.enqueue_create_buffer[.int32](4)
        var output_device = ctx.enqueue_create_buffer[.float32](6)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(indices_device, indices_host)

        gather_sum[target="gpu"](
            TileTensor(output_device, output_layout),
            TileTensor(input_device, table_layout),
            TileTensor(indices_device, indices_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](6)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var expected = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        for i in range(6):
            assert_equal(output_host[i], Float32(expected[i]))


def test_gather_sum_gpu_large_embedding_dim() raises:
    """An emb_dim larger than one block span exercises the k-stride loop
    across multiple block iterations, with an unaligned tail."""
    comptime batch = 3
    comptime multi_hot = 4
    comptime table_size = 10
    comptime emb_dim = 700

    comptime table_layout = row_major[table_size, emb_dim]()
    comptime indices_layout = row_major[batch, multi_hot]()
    comptime output_layout = row_major[batch, emb_dim]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](
            table_size * emb_dim
        )
        for i in range(table_size * emb_dim):
            input_host[i] = Float32(i % 17)

        var indices_host = ctx.enqueue_create_host_buffer[.int32](
            batch * multi_hot
        )
        for i in range(batch * multi_hot):
            indices_host[i] = Int32((i * 3) % table_size)

        var input_device = ctx.enqueue_create_buffer[.float32](
            table_size * emb_dim
        )
        var indices_device = ctx.enqueue_create_buffer[.int32](
            batch * multi_hot
        )
        var output_device = ctx.enqueue_create_buffer[.float32](batch * emb_dim)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(indices_device, indices_host)

        gather_sum[target="gpu"](
            TileTensor(output_device, output_layout),
            TileTensor(input_device, table_layout),
            TileTensor(indices_device, indices_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](
            batch * emb_dim
        )
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        for b in range(batch):
            for k in range(emb_dim):
                var expected = Float32(0)
                for j in range(multi_hot):
                    var idx = Int((b * multi_hot + j) * 3) % table_size
                    expected += Float32((idx * emb_dim + k) % 17)
                assert_equal(
                    output_host[b * emb_dim + k],
                    expected,
                )


def main() raises:
    var suite = TestSuite()
    suite.test[test_gather_sum_gpu_basic]()
    suite.test[test_gather_sum_gpu_single_index]()
    suite.test[test_gather_sum_gpu_int64_data]()
    suite.test[test_gather_sum_gpu_negative_indices]()
    suite.test[test_gather_sum_gpu_rank3_output]()
    suite.test[test_gather_sum_gpu_large_embedding_dim]()
    suite^.run()

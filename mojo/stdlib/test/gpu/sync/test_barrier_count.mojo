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

from std.gpu import thread_idx
from std.gpu.globals import WARP_SIZE
from std.gpu.host import DeviceContext
from std.gpu.sync import barrier_count
from std.testing import assert_equal


def barrier_count_kernel[
    active: Int,
](output: UnsafePointer[Int32, MutAnyOrigin]):
    var predicate = thread_idx.x < UInt(active)
    output[thread_idx.x] = barrier_count(predicate)


def test_barrier_count_half() raises:
    comptime BLOCK = WARP_SIZE * 2
    comptime ACTIVE = BLOCK // 2

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[DType.int32](BLOCK)
        ctx.enqueue_function_experimental[barrier_count_kernel[ACTIVE]](
            buf, grid_dim=1, block_dim=BLOCK
        )
        var host = ctx.enqueue_create_host_buffer[DType.int32](BLOCK)
        ctx.enqueue_copy(host.unsafe_ptr(), buf)
        ctx.synchronize()

        for i in range(BLOCK):
            assert_equal(host[i], Int32(ACTIVE))


def test_barrier_count_extremes() raises:
    comptime BLOCK = WARP_SIZE * 2

    with DeviceContext() as ctx:
        var all_true = ctx.enqueue_create_buffer[DType.int32](BLOCK)
        ctx.enqueue_function_experimental[barrier_count_kernel[BLOCK]](
            all_true, grid_dim=1, block_dim=BLOCK
        )

        var none_true = ctx.enqueue_create_buffer[DType.int32](BLOCK)
        ctx.enqueue_function_experimental[barrier_count_kernel[0]](
            none_true, grid_dim=1, block_dim=BLOCK
        )

        var all_true_host = ctx.enqueue_create_host_buffer[DType.int32](BLOCK)
        ctx.enqueue_copy(all_true_host.unsafe_ptr(), all_true)

        var none_true_host = ctx.enqueue_create_host_buffer[DType.int32](BLOCK)
        ctx.enqueue_copy(none_true_host.unsafe_ptr(), none_true)

        ctx.synchronize()

        for i in range(BLOCK):
            assert_equal(all_true_host[i], Int32(BLOCK))
            assert_equal(none_true_host[i], Int32(0))

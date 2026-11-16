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

from internal_utils import InitializationType, init_vector_launch
from max.gpu.host import DeviceContext
from std.testing import assert_equal


def test_init_vector_short_buffer() raises:
    """``init_vector_launch`` fills a buffer shorter than its grid correctly.

    The launch sizes its grid as ``ceildiv(ceildiv(length, 4), 256)`` blocks of
    256 threads and every thread fills four elements a stride apart, so a
    buffer of 16 elements is handed 256 threads with 16 elements to write.
    Every one of the four stores has to be guarded: an unguarded store of a
    thread with nothing left to fill lands up to ``3 * stride`` elements past
    the end, which compute-sanitizer reports as an invalid write when the test
    runs with ``--//:gpu_disable_memory_manager=true`` (each buffer is then its
    own allocation, so out-of-bounds stores and reads are visible).
    """
    comptime dtype = DType.float32
    comptime length = 16

    with DeviceContext() as ctx:
        var device = ctx.enqueue_create_buffer[dtype](length)
        init_vector_launch[dtype](
            device, length, InitializationType.arange, ctx
        )

        var host = ctx.enqueue_create_host_buffer[dtype](length)
        ctx.enqueue_copy(host, device)
        ctx.synchronize()

        # ``arange`` stores the thread's own index at its first position, and
        # thread ``t`` owns element ``t`` of a buffer this short.
        for i in range(length):
            assert_equal(host[i], Scalar[dtype](Float64(i)))


def main() raises:
    test_init_vector_short_buffer()

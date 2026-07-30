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

"""Regression test for the AMD GEVM grid writing past its output view."""

from layout import TileTensor, row_major
from linalg.gemv import gemv_gpu
from std.gpu.host import DeviceContext
from std.testing import assert_equal


def main() raises:
    comptime N = 64
    comptime K = 64
    comptime GUARD_VALUE = Float32(-7)

    var a_host = alloc[Float32](K)
    # Pad B so the buggy extra blocks read deterministic values rather than
    # relying on neighboring allocator contents.
    var b_host = alloc[Float32]((K + 3) * N)
    # Expose only the first N elements through C's TileTensor view. The
    # remaining 3 * N elements are a canary for out-of-bounds GEVM writes.
    var c_host = alloc[Float32](4 * N)

    for i in range(K):
        a_host[i] = 1
    for i in range((K + 3) * N):
        b_host[i] = 1
    for i in range(4 * N):
        c_host[i] = GUARD_VALUE

    with DeviceContext() as ctx:
        var a_device = ctx.enqueue_create_buffer[DType.float32](K)
        var b_device = ctx.enqueue_create_buffer[DType.float32]((K + 3) * N)
        var c_device = ctx.enqueue_create_buffer[DType.float32](4 * N)

        ctx.enqueue_copy(a_device, a_host)
        ctx.enqueue_copy(b_device, b_host)
        ctx.enqueue_copy(c_device, c_host)

        var a = TileTensor(a_device, row_major(1, K)).as_immut()
        var b = TileTensor(b_device, row_major(K, N)).as_immut()
        var c = TileTensor(c_device, row_major(1, N))
        gemv_gpu(c, a, b, ctx)

        ctx.enqueue_copy(c_host, c_device)
        ctx.synchronize()

    for i in range(N):
        assert_equal(c_host[i], Float32(K))
    for i in range(N, 4 * N):
        assert_equal(
            c_host[i],
            GUARD_VALUE,
            "GEVM wrote beyond its 1xN output",
        )

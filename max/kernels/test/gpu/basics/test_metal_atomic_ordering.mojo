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

# Built for apple-m4-metal4 (BUILD copt): other Metal targets reject these orders.

from std.atomic import Atomic, Ordering
from std.testing import assert_equal

from max.gpu import thread_idx
from max.gpu.host import DeviceContext

comptime _THREADS = 256
comptime _ORDERS = 4


def _kernel[origin: MutOrigin](counters: MutPointer[Int32, origin]):
    _ = Atomic[Int32].fetch_add[ordering=Ordering.ACQUIRE](
        counters.unsafe_offset(0), Int32(1)
    )
    _ = Atomic[Int32].fetch_add[ordering=Ordering.RELEASE](
        counters.unsafe_offset(1), Int32(1)
    )
    _ = Atomic[Int32].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](
        counters.unsafe_offset(2), Int32(1)
    )
    _ = Atomic[Int32].fetch_add[ordering=Ordering.SEQUENTIAL](
        counters.unsafe_offset(3), Int32(1)
    )

    var p = counters.unsafe_offset(4)
    # Plain load: Metal's atomic load lowering is untested.
    var expected = p[]
    while not Atomic[Int32].compare_exchange[
        success_ordering=Ordering.ACQUIRE_RELEASE,
        failure_ordering=Ordering.ACQUIRE,
        weak=True,
    ](p, expected, expected + 1):
        pass


def main() raises:
    with DeviceContext() as ctx:
        var counters = ctx.enqueue_create_buffer[DType.int32](_ORDERS + 1)
        counters.enqueue_fill(0)
        ctx.enqueue_function[_kernel[origin_of(counters)]](
            counters, grid_dim=4, block_dim=_THREADS
        )
        with counters.map_to_host() as host:
            for i in range(_ORDERS + 1):
                assert_equal(
                    host[i], Int32(4 * _THREADS), msg=String("slot ", i)
                )
        print("PASS")

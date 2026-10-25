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
#
# Positive control for the compute-sanitizer `synccheck` lane.
#
# The kernel is the negative control's two-warpgroup handoff with `barrier()`
# in place of `named_barrier`. CUDA requires every thread of the block to reach
# the same `barrier()` call, so this is a real defect, and the findings
# classifier must keep synccheck's report of it. Run it with the negative
# control and expect findings here and none there:
#
#   CS=max/kernels/test/gpu/compute_sanitizer
#   $CS/run_sanitizer.sh synccheck \
#     //$CS:negative_control_synccheck_named_barrier.mojo.test \
#     //$CS:positive_control_synccheck_barrier.mojo.test
#
# Tagged `manual`, so it never runs in the normal suite.

from std.memory import unsafe_stack_allocation

from max.gpu import thread_idx
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier

comptime PAIR_THREADS = 2 * WARPGROUP_SIZE


def two_warpgroup_barrier(dst: MutPointer[Int32, MutAnyOrigin]):
    var tid = Int(thread_idx.x)
    var wg = tid // WARPGROUP_SIZE
    var slot = tid % WARPGROUP_SIZE
    var published = unsafe_stack_allocation[
        WARPGROUP_SIZE, Int32, address_space=.SHARED
    ]()

    if wg == 1:
        published[unsafe_offset=slot] = Int32(slot + 1)
        barrier()
        dst[unsafe_offset=tid] = Int32(slot + 1)
        return

    var acc = Int32(0)
    for i in range(slot + 1):
        acc += Int32(i)
    barrier()
    dst[unsafe_offset=tid] = published[unsafe_offset=slot] + acc


def main() raises:
    with DeviceContext() as ctx:
        var dst = ctx.enqueue_create_buffer[.int32](PAIR_THREADS)
        ctx.enqueue_function[two_warpgroup_barrier](
            dst,
            grid_dim=(1,),
            block_dim=(PAIR_THREADS,),
        )
        ctx.synchronize()

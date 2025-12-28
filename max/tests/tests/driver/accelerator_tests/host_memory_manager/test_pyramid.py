# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


from conftest import MiB, align_down, alloc_pinned
from max.driver import Accelerator
from max.support import to_human_readable_bytes


def test_pyramid(memory_manager_config: None) -> None:
    print("====== test_pyramid")

    # Allocate 1x90MiB, 2x45MiB, 3x30MiB, ..., 30x3MiB in a pyramid pattern.
    total_size = 90 * MiB
    for num_buffers in range(1, 31):
        # Compute the size of each buffer, aligning down to the nearest multiple of 256 KiB.
        # Unaligned allocations may result in memory fragmentation.
        size_per_buffer = align_down(total_size // num_buffers)
        size_per_buffer_str = to_human_readable_bytes(size_per_buffer)

        print(
            f"====== Allocating {num_buffers} buffers of size {size_per_buffer_str}"
        )
        bufs = [alloc_pinned(size_per_buffer) for _ in range(num_buffers)]

        print("====== Freeing buffers")
        # Clear the list of buffers to decr ref_cnt of DeviceBuffers.
        # As now ref_cnt==0, this calls enqueueEventHandler() for async free.
        bufs.clear()

        # This explicit sync is necessary to ensure that the buffers are returned
        # to the memory manager.
        # Interally, this calls checkPendingWork() which polls for ready events
        # and runs the above handlers we enqueued earlier. These handlers free
        # the buffers.
        Accelerator().synchronize()

        print("====== Done")
        print()

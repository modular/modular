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

"""Implements `parallel_memcpy`.

You can import these APIs from the `algorithm` package. For example:

```mojo
from algorithm import parallel_memcpy
```
"""

from math import ceildiv

from memory import memcpy
from runtime.asyncrt import parallelism_level


fn parallel_memcpy[
    type: DType
](
    dest: UnsafePointer[Scalar[type]],
    src: UnsafePointer[Scalar[type]],
    count: Int,
    count_per_task: Int,
    num_tasks: Int,
):
    """Copies `count` elements from a memory buffer `src` to `dest` in parallel
    by spawning `num_tasks` tasks each copying `count_per_task` elements.

    Parameters:
        type: The element dtype.

    Args:
        dest: The destination buffer.
        src: The source buffer.
        count: Number of elements in the buffer.
        count_per_task: Task size.
        num_tasks: Number of tasks to run in parallel.
    """

    @parameter
    @always_inline
    fn _parallel_copy(thread_id: Int):
        var begin = count_per_task * thread_id
        var end = min(
            count_per_task * (thread_id + 1),
            count,
        )
        if begin >= count:
            return
        var to_copy = end - begin
        if to_copy <= 0:
            return

        memcpy(dest.offset(begin), src.offset(begin), to_copy)

    sync_parallelize[_parallel_copy](num_tasks)


fn parallel_memcpy[
    type: DType,
](
    dest: UnsafePointer[Scalar[type]],
    src: UnsafePointer[Scalar[type]],
    count: Int,
):
    """Copies `count` elements from a memory buffer `src` to `dest` in parallel.

    Parameters:
        type: The element type.

    Args:
        dest: The destination pointer.
        src: The source pointer.
        count: The number of elements to copy.
    """

    # TODO: Find a heuristic to replace the magic number.
    alias min_work_per_task = 1024
    alias min_work_for_parallel = 4 * min_work_per_task

    # If number of elements to be copied is less than minimum preset (4048),
    # then use default memcpy.
    if count < min_work_for_parallel:
        memcpy(dest, src, count)
    else:
        var work_units = ceildiv(count, min_work_per_task)
        var num_tasks = min(work_units, parallelism_level())
        var work_block_size = ceildiv(work_units, num_tasks)

        parallel_memcpy(
            dest,
            src,
            count,
            work_block_size * min_work_per_task,
            num_tasks,
        )

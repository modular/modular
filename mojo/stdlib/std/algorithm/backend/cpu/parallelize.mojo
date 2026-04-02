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
"""CPU implementations of parallelization functions."""

from std.math import ceildiv
from std.os import abort
from std.utils.index import IndexList

from std.runtime import tracing
from std.runtime.asyncrt import TaskGroup, parallelism_level
from std.runtime.tracing import Trace, TraceLevel

from std.utils.numerics import FlushDenormals


# ===-----------------------------------------------------------------------===#
# Parallelize
# ===-----------------------------------------------------------------------===#


@always_inline
def sync_parallelize[
    origins: OriginSet,
    //,
    func: def(Int) raises capturing[origins] -> None,
](num_work_items: Int):
    """Executes func(0) ... func(num_work_items-1) as parallel sub-tasks,
    and returns when all are complete.

    TODO: Currently exceptions raised by func will cause a trap rather than
          be propagated back to the caller.

    Parameters:
        origins: The capture origins.
        func: The function to invoke.

    Args:
        num_work_items: Number of parallel tasks.
    """

    # The try/except here is required to satisfy the non-raising
    # `unified register_passable -> None` signature. The unified overload's
    # inner `func_wrapped` has its own try/except for the same reason, but
    # that outer catch is unreachable since abort() here terminates first.
    def func_unified(i: Int) unified register_passable {}:
        try:
            func(i)
        except e:
            abort(String(e))

    sync_parallelize(func_unified, num_work_items)


@always_inline
def sync_parallelize[
    FuncType: def(Int) unified register_passable -> None,
](func: FuncType, num_work_items: Int):
    """Executes func(0) ... func(num_work_items-1) as parallel sub-tasks,
    and returns when all are complete.

    Parameters:
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        num_work_items: Number of parallel tasks.
    """
    # We have no tasks, so do nothing.
    if num_work_items <= 0:
        # No-op
        return

    # If profiling is enabled, and the caller's thread has an active profile
    # entry, each sub-task will also be profiled with a reference back to the
    # parent. Otherwise parent_id will be zero.
    var parent_id = tracing.get_current_trace_id[TraceLevel.THREAD]()

    @parameter
    @always_inline
    def func_wrapped(i: Int):
        with FlushDenormals():
            try:
                with Trace[TraceLevel.THREAD, target=StaticString("cpu")](
                    "task", task_id=i, parent_id=parent_id
                ):
                    func(i)
            except e:
                abort(String(e))

    if num_work_items == 1:
        # Just run inline.
        func_wrapped(0)
        return

    @always_inline
    @parameter
    async def task_fn(i: Int):
        func_wrapped(i)

    # Run sub-tasks using the 'default' runtime. If the caller is part of
    # Mojo kernel executing within the Modular Inference Engine then the
    # default runtime will be that established by the engine. Otherwise a
    # suitable runtime will be created if it does not already exist.
    var num_threads = parallelism_level()
    var num_per_lq_tasks, num_global_queue_tasks = divmod(
        num_work_items, num_threads
    )
    var tg = TaskGroup()
    var count = 0
    for _ in range(num_per_lq_tasks):
        for j in range(num_threads):
            tg._create_task(task_fn(count), j)
            count += 1
    for _ in range(num_global_queue_tasks):
        tg.create_task(task_fn(count))
        count += 1

    # execute Nth task inline. When using local queues, we need to know
    # this threads tid so that we do not push tasks into its queue.
    # This involves plumbing workerIDTLS from the threadpool. It may be
    # worth to do this. Until then we schedule all tasks through addTask
    tg.wait()


@always_inline
def parallelize[
    origins: OriginSet, //, func: def(Int) capturing[origins] -> None
](num_work_items: Int):
    """Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel, and
    returns when all are complete.

    Parameters:
        origins: The capture origins.
        func: The function to invoke.

    Args:
        num_work_items: Number of parallel tasks.
    """

    def func_unified(i: Int) unified register_passable {}:
        func(i)

    _parallelize_impl(func_unified, num_work_items, parallelism_level())


@always_inline
def parallelize[
    origins: OriginSet, //, func: def(Int) capturing[origins] -> None
](num_work_items: Int, num_workers: Int):
    """Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel, and
    returns when all are complete.

    Parameters:
        origins: The capture origins.
        func: The function to invoke.

    Args:
        num_work_items: Number of parallel tasks.
        num_workers: The number of workers to use for execution.
    """

    def func_unified(i: Int) unified register_passable {}:
        func(i)

    _parallelize_impl(func_unified, num_work_items, num_workers)


@always_inline
def parallelize[
    FuncType: def(Int) unified register_passable -> None,
](func: FuncType, num_work_items: Int):
    """Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel, and
    returns when all are complete.

    Parameters:
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        num_work_items: Number of parallel tasks.
    """
    _parallelize_impl(func, num_work_items, parallelism_level())


@always_inline
def parallelize[
    FuncType: def(Int) unified register_passable -> None,
](func: FuncType, num_work_items: Int, num_workers: Int):
    """Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel, and
    returns when all are complete.

    Parameters:
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        num_work_items: Number of parallel tasks.
        num_workers: The number of workers to use for execution.
    """
    _parallelize_impl(func, num_work_items, num_workers)


@always_inline
def _parallelize_impl[
    FuncType: def(Int) unified register_passable -> None,
](func: FuncType, num_work_items: Int, num_workers: Int):
    """Distributes work items across workers by coalescing consecutive items
    into chunks and executing them in parallel via `sync_parallelize`.

    Parameters:
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        num_work_items: Number of parallel tasks.
        num_workers: The number of workers to use for execution.
    """
    assert num_workers > 0, "Number of workers must be positive"
    # Calculate how many items are picked up by each worker.
    var chunk_size, extra_items = divmod(num_work_items, num_workers)

    # We coalesce consecutive groups of work items into a single dispatch by
    # using the coarse_grained_func below.
    @always_inline
    def coarse_grained_func(
        thread_idx: Int,
    ) unified {read func, read chunk_size, read extra_items,}:
        # Calculate the consecutive range of work items this invocation is
        # responsible for.
        var start_idx = thread_idx * chunk_size + min(thread_idx, extra_items)
        for i in range(chunk_size + Int(thread_idx < extra_items)):
            func(start_idx + i)

    sync_parallelize(coarse_grained_func, num_workers)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
def _get_num_workers(problem_size: Int, grain_size: Int = 32768) -> Int:
    """Returns a number of workers to run in parallel for given problem_size,
    accounting for the available worker threads of the current runtime.

    Args:
        problem_size: The number of parallel tasks.
        grain_size: Minimum number of elements to warrant an additional thread.

    Returns:
        The number of workers to run in parallel.
    """
    # default grain_size copied from https://github.com/pytorch/pytorch/blob/20dfce591ce88bc957ffcd0c8dc7d5f7611a4a3b/aten/src/ATen/TensorIterator.h#L86
    # Ensure at least one worker is always returned to avoid division by zero.
    return max(1, min(parallelism_level(), ceildiv(problem_size, grain_size)))


# ===-----------------------------------------------------------------------===#
# parallelize_over_rows
# ===-----------------------------------------------------------------------===#


def parallelize_over_rows[
    func: def(Int, Int) capturing[_] -> None
](shape: IndexList, axis: Int, grain_size: Int):
    """Parallelize func over non-axis dims of shape.

    Parameters:
        func: Function to call on range of rows.

    Args:
        shape: Shape to parallelize over.
        axis: Rows are slices along the axis dimension of shape.
        grain_size: The minimum number of elements to warrant using an additional thread.
    """

    def func_unified(start: Int, end: Int) unified register_passable {}:
        func(start, end)

    parallelize_over_rows(func_unified, shape, axis, grain_size)


def parallelize_over_rows[
    FuncType: def(Int, Int) unified register_passable -> None,
](func: FuncType, shape: IndexList, axis: Int, grain_size: Int):
    """Parallelize func over non-axis dims of shape.

    Parameters:
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: Shape to parallelize over.
        axis: Rows are slices along the axis dimension of shape.
        grain_size: The minimum number of elements to warrant using an additional thread.
    """
    # If we know we will have no work, return early
    if shape[axis] == 0:
        return

    var total_size = shape.flattened_length()
    var num_rows = total_size // shape[axis]

    var num_workers = min(
        num_rows,
        _get_num_workers(total_size, grain_size),
    )
    var chunk_size = ceildiv(num_rows, num_workers)

    @always_inline
    def task_func(
        task_id: Int,
    ) unified {read func, read chunk_size, read num_rows,}:
        var start_row = task_id * chunk_size
        var end_row = min((task_id + 1) * chunk_size, num_rows)

        func(start_row, end_row)

    sync_parallelize(task_func, num_workers)

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
"""Implements higher-order functions.

You can import these APIs from the `algorithm` package. For example:

```mojo
from algorithm import map
```
"""

from collections.string.string_slice import get_static_string
from math import align_down, ceildiv, clamp
from os import abort
import sys

from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from gpu.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu
from runtime import tracing
from runtime.asyncrt import DeviceContextPtr, TaskGroup, parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, IndexList
from utils.numerics import FlushDenormals
from utils.static_tuple import StaticTuple
from pathlib import Path

from gpu.host.info import B200

# ===-----------------------------------------------------------------------===#
# Map
# ===-----------------------------------------------------------------------===#


@always_inline
fn map[
    origins: OriginSet, //, func: fn (Int) capturing [origins] -> None
](size: Int):
    """Maps a function over a range from 0 to size.

    Parameters:
        origins: The capture origins.
        func: Function to map.

    Args:
        size: The number of elements.
    """
    for i in range(size):
        func(i)


# ===-----------------------------------------------------------------------===#
# Vectorize
# ===-----------------------------------------------------------------------===#


@always_inline
fn vectorize[
    origins: OriginSet, //,
    func: fn[width: Int] (Int) capturing [origins] -> None,
    simd_width: Int,
    /,
    *,
    unroll_factor: Int = 1,
](size: Int):
    """Simplifies SIMD optimized loops by mapping a function across a range from
    0 to `size`, incrementing by `simd_width` at each step. The remainder of
    `size % simd_width` will run in separate iterations.

    Parameters:
        origins: The capture origins.
        func: The function that will be called in the loop body.
        simd_width: The SIMD vector width.
        unroll_factor: The unroll factor for the main loop (Default 1).

    Args:
        size: The upper limit for the loop.

    The below example demonstrates how you could improve the performance of a
    loop, by setting multiple values at the same time using SIMD registers on
    the machine:

    ```mojo
    from algorithm.functional import vectorize
    from sys import simd_width_of

    # The amount of elements to loop through
    alias size = 10
    # How many Dtype.int32 elements fit into the SIMD register (4 on 128bit)
    alias simd_width = simd_width_of[DType.int32]()  # assumed to be 4 in this example

    fn main():
        var p = UnsafePointer[Int32].alloc(size)

        # @parameter allows the closure to capture the `p` pointer
        @parameter
        fn closure[width: Int](i: Int):
            print("storing", width, "els at pos", i)
            p.store[width=width](i, i)

        vectorize[closure, simd_width](size)
        print(p.load[width=simd_width]())
        print(p.load[width=simd_width](simd_width))
    ```

    On a machine with a SIMD register size of 128, this will set 4xInt32 values
    on each iteration. The remainder of 10 % 4 is 2, so those last two elements
    will be set in two separate iterations:

    ```plaintext
    storing 4 els at pos 0
    storing 4 els at pos 4
    storing 1 els at pos 8
    storing 1 els at pos 9
    [0, 0, 0, 0, 4, 4, 4, 4, 8, 9]
    ```

    You can also unroll the loop to potentially improve performance at the cost
    of binary size:

    ```
    vectorize[closure, width, unroll_factor=2](size)
    ```

    In the generated assembly the function calls will be repeated, resulting in
    fewer arithmetic, comparison, and conditional jump operations. The assembly
    would look like this in pseudocode:

    ```
    closure[4](0)
    closure[4](4)
    # Remainder loop won't unroll unless `size` is passed as a parameter
    for i in range(8, 10):
        closure[1](i)
        closure[1](i)
    ```

    You can pass `size` as a parameter if it's compile time known to reduce the
    iterations for the remainder. This only occurs if the remainder is an
    exponent of 2 (2, 4, 8, 16, ...). The remainder loop will still unroll for
    performance improvements if not an exponent of 2.
    """
    constrained[simd_width > 0, "simd width must be > 0"]()
    constrained[unroll_factor > 0, "unroll factor must be > 0"]()
    debug_assert(size >= 0, "size must be >= 0")

    alias unrolled_simd_width = simd_width * unroll_factor
    var simd_end = align_down(UInt(size), UInt(simd_width))
    var unrolled_end = align_down(UInt(size), UInt(unrolled_simd_width))

    for unrolled_idx in range(0, unrolled_end, unrolled_simd_width):

        @parameter
        for idx in range(unroll_factor):
            func[simd_width](unrolled_idx + idx * simd_width)

    @parameter
    if unroll_factor > 1:
        for simd_idx in range(unrolled_end, simd_end, simd_width):
            func[simd_width](simd_idx)

    for i in range(simd_end, size):
        func[1](i)


@always_inline
fn vectorize[
    origins: OriginSet, //,
    func: fn[width: Int] (Int) capturing [origins] -> None,
    simd_width: Int,
    /,
    *,
    size: Int,
    unroll_factor: Int = size if sys.is_gpu() else 1,
]():
    """Simplifies SIMD optimized loops by mapping a function across a range from
    0 to `size`, incrementing by `simd_width` at each step. The remainder of
    `size % simd_width` will run in a single iteration if it's an exponent of
    2.

    Parameters:
        origins: The capture origins.
        func: The function that will be called in the loop body.
        simd_width: The SIMD vector width.
        size: The upper limit for the loop.
        unroll_factor: The unroll factor for the main loop (Default 1).

    The below example demonstrates how you could improve the performance of a
    loop, by setting multiple values at the same time using SIMD registers on
    the machine:

    ```mojo
    from algorithm.functional import vectorize
    from sys import simd_width_of

    # The amount of elements to loop through
    alias size = 10
    # How many Dtype.int32 elements fit into the SIMD register (4 on 128bit)
    alias simd_width = simd_width_of[DType.int32]()  # assumed to be 4 in this example

    fn main():
        var p = UnsafePointer[Int32].alloc(size)

        # @parameter allows the closure to capture the `p` pointer
        @parameter
        fn closure[width: Int](i: Int):
            print("storing", width, "els at pos", i)
            p.store[width=width](i, i)

        vectorize[closure, simd_width](size)
        print(p.load[width=simd_width]())
        print(p.load[width=simd_width](simd_width))
    ```

    On a machine with a SIMD register size of 128, this will set 4xInt32 values
    on each iteration. The remainder of 10 % 4 is 2, so those last two elements
    will be set in a single iteration:

    ```plaintext
    storing 4 els at pos 0
    storing 4 els at pos 4
    storing 2 els at pos 8
    [0, 0, 0, 0, 4, 4, 4, 4, 8, 8]
    ```

    If the remainder is not an exponent of 2 (2, 4, 8, 16 ...) there will be a
    separate iteration for each element. However passing `size` as a parameter
    also allows the loop for the remaining elements to be unrolled.

    You can also unroll the main loop to potentially improve performance at the
    cost of binary size:

    ```
    vectorize[closure, width, size=size, unroll_factor=2]()
    ```

    In the generated assembly the function calls will be repeated, resulting in
    fewer arithmetic, comparison, and conditional jump operations. The assembly
    would look like this in pseudocode:

    ```
    closure[4](0)
    closure[4](4)
    closure[2](8)
    ```
    """
    constrained[simd_width > 0, "simd width must be > 0"]()
    constrained[unroll_factor > 0, "unroll factor must be > 0"]()
    constrained[size >= 0, "size must be >= 0"]()

    alias unrolled_simd_width = simd_width * unroll_factor
    alias simd_end = align_down(size, simd_width)
    alias unrolled_end = align_down(size, unrolled_simd_width)

    @parameter
    for unrolled_idx in range(0, unrolled_end, unrolled_simd_width):

        @parameter
        for idx in range(unroll_factor):
            func[simd_width](unrolled_idx + idx * simd_width)

    @parameter
    if unroll_factor > 1:
        for simd_idx in range(unrolled_end, simd_end, simd_width):
            func[simd_width](simd_idx)

    @parameter
    if size > simd_end:

        @parameter
        if (size - simd_end).is_power_of_two():
            func[size - simd_end](simd_end)
        else:

            @parameter
            for i in range(simd_end, size):
                func[1](i)


# ===-----------------------------------------------------------------------===#
# Parallelize
# ===-----------------------------------------------------------------------===#


@always_inline
fn sync_parallelize[
    origins: OriginSet, //, func: fn (Int) capturing [origins] -> None
](num_work_items: Int):
    """Executes func(0) ... func(num_work_items-1) as parallel sub-tasks,
    and returns when all are complete.

    Parameters:
        origins: The capture origins.
        func: The function to invoke.

    Args:
        num_work_items: Number of parallel tasks.
    """

    @always_inline
    @parameter
    fn func_wrapper(i: Int) raises:
        func(i)

    # Defer to the raising overload.
    sync_parallelize[func_wrapper](num_work_items)


@always_inline
fn sync_parallelize[
    origins: OriginSet, //,
    func: fn (Int) raises capturing [origins] -> None,
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
    fn func_wrapped(i: Int):
        with FlushDenormals():
            with Trace[TraceLevel.THREAD, target = StaticString("cpu")](
                "task", task_id=i, parent_id=parent_id
            ):
                try:
                    func(i)
                except e:
                    abort(String(e))

    if num_work_items == 1:
        # Just run inline.
        func_wrapped(0)
        return

    @always_inline
    @parameter
    async fn task_fn(i: Int):
        func_wrapped(i)

    # Run sub-tasks using the 'default' runtime. If the caller is part of
    # Mojo kernel executing within the Modular Inference Engine then the
    # default runtime will be that established by the engine. Otherwise a
    # suitable runtime will be created if it does not already exist.
    var num_threads = parallelism_level()
    var num_per_lq_tasks = num_work_items // num_threads
    var num_global_queue_tasks = num_work_items % num_threads
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
fn parallelize[
    origins: OriginSet, //, func: fn (Int) capturing [origins] -> None
](num_work_items: Int):
    """Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel, and
    returns when all are complete.

    Parameters:
        origins: The capture origins.
        func: The function to invoke.

    Args:
        num_work_items: Number of parallel tasks.
    """

    _parallelize_impl[func](num_work_items, parallelism_level())


@always_inline
fn parallelize[
    origins: OriginSet, //, func: fn (Int) capturing [origins] -> None
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

    _parallelize_impl[func](num_work_items, num_workers)


@always_inline
fn _parallelize_impl[
    origins: OriginSet, //, func: fn (Int) capturing [origins] -> None
](num_work_items: Int, num_workers: Int):
    debug_assert(num_workers > 0, "Number of workers must be positive")
    # Calculate how many items are picked up by each worker.
    var chunk_size = num_work_items // num_workers
    # Calculate how many workers need to add an extra item to their work.
    var extra_items = num_work_items % num_workers

    # We coalesce consecutive groups of work items into a single dispatch by
    # using the coarse_grained_func below.
    @always_inline
    @parameter
    fn coarse_grained_func(thread_idx: Int):
        # Calculate the consecutive range of work items this invocation is
        # responsible for.
        var start_idx = thread_idx * chunk_size + min(thread_idx, extra_items)
        for i in range(chunk_size + Int(thread_idx < extra_items)):
            func(start_idx + i)

    sync_parallelize[coarse_grained_func](num_workers)


# ===-----------------------------------------------------------------------===#
# tile
# ===-----------------------------------------------------------------------===#

alias Static1DTileUnitFunc = fn[width: Int] (Int) capturing [_] -> None
"""
Signature of a 1d tiled function that performs some work with a static tile size
and an offset. i.e. func<tile_size: Int> (offset: Int)
"""

alias Dynamic1DTileUnitFunc = fn (Int, Int) capturing [_] -> None
"""
Signature of a 1d tiled function that performs some work with a dynamic tile size
  and an offset. i.e. func(offset: Int, tile_size: Int)
"""


alias BinaryTile1DTileUnitFunc = fn[width: Int] (Int, Int) capturing [_] -> None
"""
Signature of a tiled function that performs some work with a dynamic tile size
and a secondary static tile size.
"""


@always_inline
fn tile[
    workgroup_function: Static1DTileUnitFunc, tile_size_list: VariadicList[Int]
](offset: Int, upperbound: Int):
    """A generator that launches work groups in specified list of tile sizes.

    A workgroup function is a function that can process a configurable
    consecutive "tile" of workload. E.g.
      `work_on[3](5)`
    should launch computation on item 5,6,7, and should be semantically
    equivalent to
      `work_on[1](5)`, `work_on[1](6)`, `work_on[1](7)`.

    This generator will try to proceed with the given list of tile sizes on the
    listed order. E.g.
        `tile[func, (3,2,1)](offset, upperbound)`
    will try to call `func[3]` starting from offset until remaining work is less
    than 3 from upperbound and then try `func[2]`, and then `func[1]`, etc.

    Parameters:
        workgroup_function: Workgroup function that processes one tile of
          workload.
        tile_size_list: List of tile sizes to launch work.

    Args:
        offset: The initial index to start the work from.
        upperbound: The runtime upperbound that the work function should not
          exceed.
    """

    # Initialize where to start on the overall work load.
    var current_offset: Int = offset

    @parameter
    for tile_size in tile_size_list:
        # Process work with the tile size until there's not enough remaining work
        #  to fit in a tile.
        while current_offset <= upperbound - tile_size:
            workgroup_function[tile_size](current_offset)
            current_offset += tile_size


@always_inline
fn tile[
    workgroup_function: Dynamic1DTileUnitFunc,
](offset: Int, upperbound: Int, tile_size_list: VariadicList[Int]):
    """A generator that launches work groups in specified list of tile sizes.

    This is the version of tile generator for the case where work_group function
    can take the tile size as a runtime value.

    Parameters:
        workgroup_function: Workgroup function that processes one tile of
          workload.

    Args:
        offset: The initial index to start the work from.
        upperbound: The runtime upperbound that the work function should not
          exceed.
        tile_size_list: List of tile sizes to launch work.
    """
    # Initialize the work_idx with the starting offset.
    var work_idx = offset
    # Iterate on the list of given tile sizes.
    for tile_idx in range(len(tile_size_list)):
        var tile_size = tile_size_list[tile_idx]
        # Launch workloads on the current tile sizes until cannot proceed.
        while work_idx <= upperbound - tile_size:
            workgroup_function(work_idx, tile_size)
            work_idx += tile_size
    # Clean up the remaining workload with a residue tile that exactly equals to
    #  the remaining workload size.
    # Note: This is the key difference from the static version of tile
    #  generator.
    if work_idx < upperbound:
        workgroup_function(work_idx, upperbound - work_idx)


@always_inline
fn tile[
    secondary_tile_size_list: VariadicList[Int],
    secondary_cleanup_tile: Int,
    workgroup_function: BinaryTile1DTileUnitFunc,
](
    offset: Int,
    upperbound: Int,
    primary_tile_size_list: VariadicList[Int],
    primary_cleanup_tile: Int,
):
    """A generator that launches work groups in specified list of tile sizes
    until the sum of primary_tile_sizes has exceeded the upperbound.

    Parameters:
        secondary_tile_size_list: List of static tile sizes to launch work.
        secondary_cleanup_tile: Last static tile to use when primary tile sizes
          don't fit exactly within the upperbound.
        workgroup_function: Workgroup function that processes one tile of
          workload.

    Args:
        offset: The initial index to start the work from.
        upperbound: The runtime upperbound that the work function should not
          exceed.
        primary_tile_size_list: List of dynamic tile sizes to launch work.
        primary_cleanup_tile: Last dynamic tile to use when primary tile sizes
          don't fit exactly within the upperbound.
    """
    var work_idx = offset
    alias num_tiles = len(secondary_tile_size_list)

    @parameter
    for i in range(num_tiles):
        alias secondary_tile_size = secondary_tile_size_list[i]
        var primary_tile_size = primary_tile_size_list[i]

        while work_idx <= upperbound - primary_tile_size:
            workgroup_function[secondary_tile_size](work_idx, primary_tile_size)
            work_idx += primary_tile_size

    # launch the last cleanup tile
    if work_idx < upperbound:
        workgroup_function[secondary_cleanup_tile](
            work_idx, primary_cleanup_tile
        )


# ===-----------------------------------------------------------------------===#
# tile2d
# ===-----------------------------------------------------------------------===#


alias Static2DTileUnitFunc = fn[tile_x: Int, tile_y: Int] (Int, Int) capturing [
    _
] -> None
"""
Signature of a 2d tiled function that performs some work with a static tile size
and an offset. i.e.
func<tile_size_x: Int, tile_size_y: Int> (offset_x: Int, offset_y: Int)
"""


@always_inline
fn tile[
    workgroup_function: Static2DTileUnitFunc,
    tile_sizes_x: VariadicList[Int],
    tile_sizes_y: VariadicList[Int],
](offset_x: Int, offset_y: Int, upperbound_x: Int, upperbound_y: Int):
    """Launches workgroup_function using the largest tile sizes possible in each
    dimension, starting from the x and y offset, until the x and y upperbounds
    are reached.

    Parameters:
        workgroup_function: Function that is invoked for each tile and offset.
        tile_sizes_x: List of tile sizes to use for the first parameter of workgroup_function.
        tile_sizes_y: List of tile sizes to use for the second parameter of workgroup_function.

    Args:
        offset_x: Initial x offset passed to workgroup_function.
        offset_y: Initial y offset passed to workgroup_function.
        upperbound_x: Max offset in x dimension passed to workgroup function.
        upperbound_y: Max offset in y dimension passed to workgroup function.
    """
    # Initialize where to start on the overall work load.
    var current_offset_y: Int = offset_y

    @parameter
    for tile_size_y in tile_sizes_y:
        while current_offset_y <= upperbound_y - tile_size_y:
            var current_offset_x = offset_x

            @parameter
            for tile_size_x in tile_sizes_x:
                while current_offset_x <= upperbound_x - tile_size_x:
                    workgroup_function[tile_size_x, tile_size_y](
                        current_offset_x, current_offset_y
                    )
                    current_offset_x += tile_size_x

            current_offset_y += tile_size_y


# ===-----------------------------------------------------------------------===#
# Unswitch
# ===-----------------------------------------------------------------------===#

# Signature of a function that unswitch can take.
alias SwitchedFunction = fn[sw: Bool] () raises capturing [_] -> None

# Version of unswitch supporting 2 predicates.
alias SwitchedFunction2 = fn[sw0: Bool, sw1: Bool] () capturing [_] -> None


@always_inline
fn unswitch[switched_func: SwitchedFunction](dynamic_switch: Bool) raises:
    """Performs a functional unswitch transformation.

    Unswitch is a simple pattern that is similar idea to loop unswitching
    pass but extended to functional patterns. The pattern facilitates the
    following code transformation that reduces the number of branches in the
    generated code

    Before:

        for i in range(...)
            if i < xxx:
                ...

    After:

        if i < ...
            for i in range(...)
                ...
        else
            for i in range(...)
                if i < xxx:
                    ...

    This unswitch function generalizes that pattern with the help of meta
    parameters and can be used to perform both loop unswitching and other
    tile predicate lifting like in simd and amx.

    TODO: Generalize to support multiple predicates.
    TODO: Once nested lambdas compose well should make unswitch compose with
    tile in an easy way.

    Parameters:
        switched_func: The function containing the inner loop logic that can be
          unswitched.

    Args:
        dynamic_switch: The dynamic condition that enables the unswitched code
          path.
    """
    if dynamic_switch:
        switched_func[True]()
    else:
        switched_func[False]()


@always_inline
fn unswitch[
    switched_func: fn[sw: Bool] () capturing [_] -> None
](dynamic_switch: Bool):
    """Performs a functional unswitch transformation.

    Unswitch is a simple pattern that is similar idea to loop unswitching
    pass but extended to functional patterns. The pattern facilitates the
    following code transformation that reduces the number of branches in the
    generated code

    Before:

        for i in range(...)
            if i < xxx:
                ...

    After:

        if i < ...
            for i in range(...)
                ...
        else
            for i in range(...)
                if i < xxx:
                    ...

    This unswitch function generalizes that pattern with the help of meta
    parameters and can be used to perform both loop unswitching and other
    tile predicate lifting like in simd and amx.

    TODO: Generalize to support multiple predicates.
    TODO: Once nested lambdas compose well should make unswitch compose with
    tile in an easy way.

    Parameters:
        switched_func: The function containing the inner loop logic that can be
          unswitched.

    Args:
        dynamic_switch: The dynamic condition that enables the unswitched code
          path.
    """

    if dynamic_switch:
        switched_func[True]()
    else:
        switched_func[False]()


@always_inline
fn unswitch[
    switched_func: SwitchedFunction2
](dynamic_switch_a: Bool, dynamic_switch_b: Bool):
    """Performs a functional 2-predicates unswitch transformation.

    Parameters:
        switched_func: The function containing the inner loop logic that has 2
          predicates which can be unswitched.

    Args:
        dynamic_switch_a: The first dynamic condition that enables the outer
          unswitched code path.
        dynamic_switch_b: The second dynamic condition that enables the inner
          unswitched code path.
    """
    # TODO: This could be a lot easier to write once parameter names can be
    #  removed.
    if dynamic_switch_a:

        @always_inline
        @parameter
        fn switched_a_true[static_switch: Bool]():
            switched_func[True, static_switch]()

        unswitch[switched_a_true](dynamic_switch_b)
    else:

        @always_inline
        @parameter
        fn switched_a_false[static_switch: Bool]():
            switched_func[False, static_switch]()

        unswitch[switched_a_false](dynamic_switch_b)


# ===-----------------------------------------------------------------------===#
# TileWithUnswitch
# ===-----------------------------------------------------------------------===#

alias Static1DTileUnswitchUnitFunc = fn[width: Int, sw: Bool] (
    Int, Int
) capturing [_] -> None
"""
Signature of a tiled function that performs some work with a static tile size
  and an offset. i.e. func<tile_size: Int> (offset: Int)
"""

alias Static1DTileUnitFuncWithFlag = fn[width: Int, flag: Bool] (
    Int
) capturing [_] -> None


@always_inline("nodebug")
fn tile_and_unswitch[
    workgroup_function: Static1DTileUnswitchUnitFunc,
    tile_size_list: VariadicList[Int],
](offset: Int, upperbound: Int):
    """Performs time and unswitch functional transformation.

    A variant of static tile given a workgroup function that can be unswitched.
    This generator is a fused version of tile and unswitch, where the static
    unswitch is true throughout the "inner" portion of the workload and is
    false only on the residue tile.

    Parameters:
        workgroup_function: Workgroup function that processes one tile of
          workload.
        tile_size_list: List of tile sizes to launch work.

    Args:
        offset: The initial index to start the work from.
        upperbound: The runtime upperbound that the work function should not
          exceed.
    """

    # Initialize where to start on the overall work load.
    var current_offset = offset
    var remaining = upperbound - offset

    @parameter
    for tile_size in tile_size_list:
        # Process work with the tile size until there's not enough remaining work
        #  to fit in a tile.
        while remaining >= tile_size:
            workgroup_function[tile_size, True](current_offset, upperbound)
            current_offset += tile_size
            remaining -= tile_size

    # Use the last tile size to process the residue.
    if remaining > 0:
        workgroup_function[tile_size_list[len(tile_size_list) - 1], False](
            current_offset, upperbound
        )


alias Dynamic1DTileUnswitchUnitFunc = fn[sw: Bool] (Int, Int, Int) capturing [
    _
] -> None


@always_inline
fn tile_and_unswitch[
    workgroup_function: Dynamic1DTileUnswitchUnitFunc,
](offset: Int, upperbound: Int, tile_size_list: VariadicList[Int]):
    """Performs time and unswitch functional transformation.

    A variant of dynamic tile given a workgroup function that can be
    unswitched. This generator is a fused version of tile and unswitch, where
    the static unswitch is true throughout the "inner" portion of the workload
    and is false only on the residue tile.

    Parameters:
        workgroup_function: Workgroup function that processes one tile of
          workload.

    Args:
        offset: The initial index to start the work from.
        upperbound: The runtime upperbound that the work function should not exceed.
        tile_size_list: List of tile sizes to launch work.
    """

    # Initialize where to start on the overall work load.
    var current_offset: Int = offset

    for idx in range(len(tile_size_list)):
        # Get the tile size to proceed with.
        var tile_size = tile_size_list[idx]

        # Process work with the tile size until there's not enough remaining work
        #  to fit in a tile.
        while current_offset <= upperbound - tile_size:
            workgroup_function[True](
                current_offset, upperbound, tile_size_list[idx]
            )
            current_offset += tile_size

    # Use the last tile size to process the residue.
    if current_offset < upperbound:
        workgroup_function[False](
            current_offset,
            upperbound,
            tile_size_list[len(tile_size_list) - 1],
        )


@always_inline
fn tile_middle_unswitch_boundaries[
    work_fn: Static1DTileUnitFuncWithFlag,
    middle_tile_sizes: VariadicList[Int],
    left_tile_size: Int = 1,  # No tiling by default.
    right_tile_size: Int = 1,  # No tiling by default.
](
    left_boundary_start: Int,
    left_boundary_end: Int,
    right_boundary_start: Int,
    right_boundary_end: Int,
):
    """Divides 1d iteration space into three parts and tiles them with different
    steps.

    The 1d iteration space is divided into:
        1. [left_boundary_start, left_boundary_end), effected by left boundary.
        2. [left_boundary_end, right_boundary_start), not effected by any boundary.
        3. [right_boundary_start, right_boundary_end), effected by right boundary.

    work_fn's switch is true for the left and right boundaries, implying boundary
    conditions like padding in convolution. The middle part is tiled with static
    tile sizes with the switch as false.

    Parameters:
        work_fn: Work function that processes one tile of workload.
        middle_tile_sizes: List of tile sizes for the middle part.
        left_tile_size: Tile size for the left boundary region.
        right_tile_size: Tile size for the right boundary region.

    Args:
        left_boundary_start: Start index of the left boundary.
        left_boundary_end: End index of the left boundary.
        right_boundary_start: Start index of the right boundary.
        right_boundary_end: End index of the right boundary.

    `middle_tile_sizes` should be in descending order for optimal performance.
    (Larger tile size appeared later in the list fails the while-loop.)
    """

    var offset = left_boundary_start

    # Handle the edge case where filter window is so large that every input
    # point is effected by padding.
    var min_boundary_end = min(left_boundary_end, right_boundary_end)

    # Left boundary region.
    while offset < min_boundary_end:
        work_fn[left_tile_size, True](offset)
        offset += left_tile_size

    # Middle
    @parameter
    for tile_size in middle_tile_sizes:
        while offset <= right_boundary_start - tile_size:
            work_fn[tile_size, False](offset)
            offset += tile_size

    # Right boundary region.
    while offset < right_boundary_end:
        work_fn[right_tile_size, True](offset)
        offset += right_tile_size


alias Static1DTileUnitFuncWithFlags = fn[
    width: Int, left_flag: Bool, right_flag: Bool
] (Int) capturing [_] -> None


@always_inline
fn tile_middle_unswitch_boundaries[
    work_fn: Static1DTileUnitFuncWithFlags,
    tile_size: Int,
    size: Int,
]():
    """Tile 1d iteration space with boundary conditions at both ends.

    This generator is primarily for convolution with static shapes. `work_fn`'s
    flags hints the function to handle padding at the boundary. The size is the
    static output row size, i.e., WO dimension.

    Parameters:
        work_fn: Work function that updates one tile. It has two flags for
            left and right boundaries, respectively.
        tile_size: 1D Tile size.
        size: Iteration range is [0, size).
    """

    # Tile size covers the entire range, e.g., using 14x2 register tile for
    # 14x14 image. Both sides of the tile has boundary conditions.
    @parameter
    if size <= tile_size:
        work_fn[size, True, True](0)
    else:
        # Set bounds of tile sizes on boundaries. E.g. for 7x7 image and
        # tile_size = 6, it's better to use tile_sizes 4 and 3 than using
        # 6 and 1 since the it's tricky to handle padding with very small
        # tile size.
        alias tile_size_lbound = min(tile_size, size // 2)
        alias tile_size_rbound = min(tile_size, size - size // 2)

        var offset = 0

        # left boundary
        work_fn[tile_size_lbound, True, False](offset)

        # middle
        @always_inline
        @parameter
        fn update_middle[_tile_size: Int](_offset: Int):
            work_fn[_tile_size, False, False](_offset)

        alias num_middle_points = size - tile_size_lbound - tile_size_rbound
        alias remainder = num_middle_points % tile_size
        # `tile` can't handle zero tile size.
        alias tile_size_remainder = remainder if remainder > 0 else 1

        tile[update_middle, VariadicList[Int](tile_size, tile_size_remainder)](
            tile_size_lbound, size - tile_size_rbound
        )

        # right boundary
        work_fn[tile_size_rbound, False, True](size - tile_size_rbound)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _get_num_workers(problem_size: Int, grain_size: Int = 32768) -> Int:
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


@always_inline
fn _get_start_indices_of_nth_subvolume[
    rank: Int, //, subvolume_rank: Int = 1
](n: Int, shape: IndexList[rank, **_]) -> __type_of(shape):
    """Converts a flat index into the starting ND indices of the nth subvolume
    with rank `subvolume_rank`.

    For example:
        - `_get_start_indices_of_nth_subvolume[0](n, shape)` will return
        the starting indices of the nth element in shape.
        - `_get_start_indices_of_nth_subvolume[1](n, shape)` will return
        the starting indices of the nth row in shape.
        - `_get_start_indices_of_nth_subvolume[2](n, shape)` will return
        the starting indices of the nth horizontal slice in shape.

    The ND indices will iterate from right to left. I.E

        shape = (20, 5, 2, N)
        _get_start_indices_of_nth_subvolume[1](1, shape) = (0, 0, 1, 0)
        _get_start_indices_of_nth_subvolume[1](5, shape) = (0, 2, 1, 0)
        _get_start_indices_of_nth_subvolume[1](50, shape) = (5, 0, 0, 0)
        _get_start_indices_of_nth_subvolume[1](56, shape) = (5, 1, 1, 0)

    Parameters:
        rank: The rank of the ND index.
        subvolume_rank: The rank of the subvolume under consideration.

    Args:
        n: The flat index to convert (the nth subvolume to retrieve).
        shape: The shape of the ND space we are converting into.

    Returns:
        Constructed ND-index.
    """

    constrained[
        subvolume_rank <= rank,
        "subvolume rank cannot be greater than indices rank",
    ]()
    constrained[subvolume_rank >= 0, "subvolume rank must be non-negative"]()

    # fast impls for common cases
    @parameter
    if rank == 2 and subvolume_rank == 1:
        return __type_of(shape)(n, 0)

    @parameter
    if rank - 1 == subvolume_rank:
        var out = __type_of(shape)(0)
        out[0] = n
        return out

    @parameter
    if rank == subvolume_rank:
        return __type_of(shape)(0)

    var out = __type_of(shape)()
    var curr_index = n

    @parameter
    for i in reversed(range(rank - subvolume_rank)):
        out[i] = curr_index._positive_rem(shape[i])
        curr_index = curr_index._positive_div(shape[i])

    return out


# TODO(KERN-637) - optimize this algorithm for UInt rather than delegating
# to the Int overload.
@always_inline
fn _get_start_indices_of_nth_subvolume_uint[
    rank: Int, //,
    subvolume_rank: UInt = 1,
](n: UInt, shape: IndexList[rank, **_]) -> __type_of(shape):
    """Converts a flat index into the starting ND indices of the nth subvolume
    with rank `subvolume_rank`.

    For example:
        - `_get_start_indices_of_nth_subvolume[0](n, shape)` will return
        the starting indices of the nth element in shape.
        - `_get_start_indices_of_nth_subvolume[1](n, shape)` will return
        the starting indices of the nth row in shape.
        - `_get_start_indices_of_nth_subvolume[2](n, shape)` will return
        the starting indices of the nth horizontal slice in shape.

    The ND indices will iterate from right to left. I.E

        shape = (20, 5, 2, N)
        _get_start_indices_of_nth_subvolume[1](1, shape) = (0, 0, 1, 0)
        _get_start_indices_of_nth_subvolume[1](5, shape) = (0, 2, 1, 0)
        _get_start_indices_of_nth_subvolume[1](50, shape) = (5, 0, 0, 0)
        _get_start_indices_of_nth_subvolume[1](56, shape) = (5, 1, 1, 0)

    Parameters:
        rank: The rank of the ND index.
        subvolume_rank: The rank of the subvolume under consideration.

    Args:
        n: The flat index to convert (the nth subvolume to retrieve).
        shape: The shape of the ND space we are converting into.

    Returns:
        Constructed ND-index.
    """
    return _get_start_indices_of_nth_subvolume[Int(subvolume_rank)](
        Int(n), shape
    )


# ===-----------------------------------------------------------------------===#
# Elementwise
# ===-----------------------------------------------------------------------===#


@always_inline
fn elementwise[
    func: fn[width: Int, rank: Int, alignment: Int = 1] (
        IndexList[rank]
    ) capturing [_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    _trace_description: StaticString = "",
](shape: Int) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: Do not invoke the function using asynchronous calls.
        target: The target to run on.
        _trace_description: Description of the trace.

    Args:
        shape: The shape of the buffer.
    """

    elementwise[
        func,
        simd_width=simd_width,
        use_blocking_impl=use_blocking_impl,
        target=target,
        _trace_description=_trace_description,
    ](Index(shape))


@always_inline
fn elementwise[
    rank: Int, //,
    func: fn[width: Int, rank: Int, alignment: Int = 1] (
        IndexList[rank]
    ) capturing [_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    _trace_description: StaticString = "",
](shape: IndexList[rank, **_]) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: Do not invoke the function using asynchronous calls.
        target: The target to run on.
        _trace_description: Description of the trace.

    Args:
        shape: The shape of the buffer.
    """

    constrained[
        is_cpu[target](),
        (
            "the target must be CPU use the elementwise which takes the"
            " DeviceContext to be able to use the GPU version"
        ),
    ]()

    _elementwise_impl_cpu[
        func, simd_width, use_blocking_impl=use_blocking_impl
    ](shape)


@always_inline
fn elementwise[
    func: fn[width: Int, rank: Int, alignment: Int = 1] (
        IndexList[rank]
    ) capturing [_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    _trace_description: StaticString = "",
](shape: Int, context: DeviceContext) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: Do not invoke the function using asynchronous calls.
        target: The target to run on.
        _trace_description: Description of the trace.

    Args:
        shape: The shape of the buffer.
        context: The device context to use.
    """

    elementwise[
        func,
        simd_width=simd_width,
        use_blocking_impl=use_blocking_impl,
        target=target,
    ](Index(shape), context)


@always_inline
fn elementwise[
    rank: Int, //,
    func: fn[width: Int, rank: Int, alignment: Int = 1] (
        IndexList[rank]
    ) capturing [_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    _trace_description: StaticString = "",
](shape: IndexList[rank, **_], context: DeviceContext) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: Do not invoke the function using asynchronous calls.
        target: The target to run on.
        _trace_description: Description of the trace.

    Args:
        shape: The shape of the buffer.
        context: The device context to use.
    """

    _elementwise_impl[
        func, simd_width, use_blocking_impl=use_blocking_impl, target=target
    ](shape, context)


@always_inline
fn elementwise[
    rank: Int, //,
    func: fn[width: Int, rank: Int, alignment: Int = 1] (
        IndexList[rank]
    ) capturing [_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    _trace_description: StaticString = "",
](shape: IndexList[rank, **_], context: DeviceContextPtr) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: Do not invoke the function using asynchronous calls.
        target: The target to run on.
        _trace_description: Description of the trace.

    Args:
        shape: The shape of the buffer.
        context: The device context to use.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        var shape_str = trace_arg("shape", shape)
        var vector_width_str = String("vector_width=", simd_width)

        return ";".join(
            shape_str,
            vector_width_str,
        )

    # Intern the kind string as a static string so we don't allocate.
    alias kind = get_static_string[
        "elementwise",
        ("(" + _trace_description + ")" if _trace_description else ""),
    ]()

    with Trace[TraceLevel.OP, target=target](
        kind,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):

        @parameter
        if is_gpu[target]():
            _elementwise_impl_gpu[func, simd_width = UInt(simd_width)](
                shape, context[]
            )
        else:
            _elementwise_impl_cpu[
                func, simd_width, use_blocking_impl=use_blocking_impl
            ](shape)


@always_inline
fn _elementwise_impl[
    rank: Int, //,
    func: fn[width: Int, rank: Int, alignment: Int = 1] (
        IndexList[rank]
    ) capturing [_] -> None,
    simd_width: Int,
    /,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
](shape: IndexList[rank, **_], context: DeviceContext) raises:
    @parameter
    if is_cpu[target]():
        _elementwise_impl_cpu[
            func, simd_width, use_blocking_impl=use_blocking_impl
        ](shape)
    else:
        _elementwise_impl_gpu[func, UInt(simd_width)](
            shape,
            context,
        )


@always_inline
fn _elementwise_impl_cpu[
    rank: Int, //,
    func: fn[width: Int, rank: Int, alignment: Int = 1] (
        IndexList[rank]
    ) capturing [_] -> None,
    simd_width: Int,
    /,
    *,
    use_blocking_impl: Bool = False,
](shape: IndexList[rank, **_]):
    alias impl = _elementwise_impl_cpu_1d if rank == 1 else _elementwise_impl_cpu_nd
    impl[func, simd_width, use_blocking_impl=use_blocking_impl](shape)


@always_inline
fn _elementwise_impl_cpu_1d[
    rank: Int, //,
    func: fn[width: Int, rank: Int, alignment: Int = 1] (
        IndexList[rank]
    ) capturing [_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool,
](shape: IndexList[rank, **_]):
    """Executes `func[width, rank](indices)`, possibly using sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: If true the functions execute without sub-tasks.

    Args:
        shape: The shape of the buffer.
    """
    constrained[rank == 1, "Specialization for 1D"]()

    alias unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    var problem_size = shape.flattened_length()

    @parameter
    if use_blocking_impl:

        @always_inline
        @parameter
        fn blocking_task_fun[simd_width: Int](idx: Int):
            func[simd_width, rank](idx)

        vectorize[blocking_task_fun, simd_width, unroll_factor=unroll_factor](
            problem_size
        )
        return

    var num_workers = _get_num_workers(problem_size)
    var chunk_size = ceildiv(problem_size, num_workers)

    @always_inline
    @parameter
    fn task_func(i: Int):
        var start_offset = i * chunk_size
        var end_offset = min((i + 1) * chunk_size, problem_size)
        var len = end_offset - start_offset

        @always_inline
        @parameter
        fn func_wrapper[simd_width: Int](idx: Int):
            var offset = start_offset + idx
            func[simd_width, rank](offset)

        vectorize[func_wrapper, simd_width, unroll_factor=unroll_factor](len)

    sync_parallelize[task_func](num_workers)


@always_inline
fn _elementwise_impl_cpu_nd[
    rank: Int, //,
    func: fn[width: Int, rank: Int, alignment: Int = 1] (
        IndexList[rank]
    ) capturing [_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool,
](shape: IndexList[rank, **_]):
    """Executes `func[width, rank](indices)`, possibly using sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns
    when all sub-tasks have completed.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: If true this is a blocking op.

    Args:
        shape: The shape of the buffer.
    """
    constrained[rank > 1, "Specialization for ND where N > 1"]()

    alias unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    # Strategy: we parallelize over all dimensions except the innermost and
    # vectorize over the innermost dimension. We unroll the innermost dimension
    # by a factor of unroll_factor.

    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    var total_size: Int = shape.flattened_length()

    @parameter
    if use_blocking_impl:

        @always_inline
        @parameter
        fn blocking_task_fn(i: Int):
            var indices = _get_start_indices_of_nth_subvolume(i, shape)

            @always_inline
            @parameter
            fn func_wrapper[simd_width: Int](idx: Int):
                # The inner most dimension is vectorized, so we set it
                # to the index offset.
                indices[rank - 1] = idx
                func[simd_width, rank](indices.canonicalize())

            # We vectorize over the innermost dimension.
            vectorize[func_wrapper, simd_width, unroll_factor=unroll_factor](
                shape[rank - 1]
            )

        map[blocking_task_fn](total_size // shape[rank - 1])

        return

    var num_workers = _get_num_workers(total_size)
    var parallelism_size = total_size // shape[rank - 1]
    var chunk_size = ceildiv(parallelism_size, num_workers)

    @always_inline
    @parameter
    fn task_func(i: Int):
        var start_parallel_offset = i * chunk_size
        var end_parallel_offset = min((i + 1) * chunk_size, parallelism_size)

        var len = end_parallel_offset - start_parallel_offset
        if len <= 0:
            return

        for parallel_offset in range(
            start_parallel_offset, end_parallel_offset
        ):
            var indices = _get_start_indices_of_nth_subvolume(
                parallel_offset, shape
            )

            @always_inline
            @parameter
            fn func_wrapper[simd_width: Int](idx: Int):
                # The inner most dimension is vectorized, so we set it
                # to the index offset.
                indices[rank - 1] = idx
                func[simd_width, rank](indices.canonicalize())

            # We vectorize over the innermost dimension.
            vectorize[func_wrapper, simd_width, unroll_factor=unroll_factor](
                shape[rank - 1]
            )

    sync_parallelize[task_func](num_workers)


@always_inline
fn _elementwise_impl_gpu[
    rank: Int, //,
    func: fn[width: Int, rank: Int, alignment: Int = 1] (
        IndexList[rank]
    ) capturing [_] -> None,
    simd_width: UInt,
](shape: IndexList[rank, **_], ctx: DeviceContext) raises:
    """Executes `func[width, rank](indices)` as sub-tasks for a suitable
    combination of width and indices so as to cover shape on the GPU.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.

    Args:
        shape: The shape of the buffer.
        ctx: The pointer to DeviceContext.
    """

    # optimized implementation inspired by https://archive.md/Tye9y#selection-1101.2-1151.3

    alias hw_info = ctx.default_device_info

    alias registers_per_thread = 255
    alias num_waves = 32
    alias registers_per_block = hw_info.max_registers_per_block
    alias sm_count: UInt = UInt(hw_info.sm_count)
    alias threads_per_sm: UInt = UInt(hw_info.threads_per_sm)

    constrained[
        sm_count > 0 and threads_per_sm > 0,
        "the sm_count and thread_count must be known",
    ]()

    # split between packed and tail regions of input
    var length: UInt = UInt(shape.flattened_length())
    var num_packed_elems = length // simd_width
    var unpacked_tail_length = length % simd_width
    var packed_region_length = length - unpacked_tail_length

    alias block_size_unrounded = registers_per_block // registers_per_thread

    # when testing other elementwise kernels, they appear to also use 128 as the block size on blackwell specifcally
    alias block_size = 128 if ctx.default_device_info is B200 else block_size_unrounded - (
        block_size_unrounded % 2
    )

    var num_blocks = clamp(
        ceildiv(num_packed_elems, UInt(block_size)),
        1,
        sm_count * threads_per_sm // block_size * num_waves,
    )

    @__copy_capture(
        num_packed_elems, unpacked_tail_length, packed_region_length
    )
    @parameter
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](block_size)
    )
    fn _elementwise_gpu_kernel[*, block_size: UInt, handle_uneven_simd: Bool]():
        # process the packed region
        var tid = thread_idx.x + block_size * block_idx.x

        @parameter
        if PDLLevel() == PDLLevel.OVERLAP_AT_BEGINNING:
            launch_dependent_grids()

        @parameter
        if PDLLevel() > PDLLevel.OFF:
            wait_on_dependent_grids()

        for idx in range(
            tid,
            num_packed_elems,
            block_size * grid_dim.x,
        ):
            var start_indices = _get_start_indices_of_nth_subvolume_uint[0](
                idx * simd_width, shape
            )

            @parameter
            if handle_uneven_simd:
                if start_indices[rank - 1] + simd_width >= shape[rank - 1]:

                    @parameter
                    for off in range(Int(simd_width)):
                        func[1, rank](
                            _get_start_indices_of_nth_subvolume_uint[0](
                                UInt(idx * simd_width + off),
                                shape,
                            ).canonicalize()
                        )
                else:
                    func[Int(simd_width), rank](start_indices.canonicalize())
            else:
                # The alignment is by number of elements, which will be converted to
                # number of bytes by graph compiler.
                func[Int(simd_width), rank, Int(simd_width)](
                    start_indices.canonicalize()
                )

        # process the tail region
        if tid < unpacked_tail_length:
            var index_tup = _get_start_indices_of_nth_subvolume_uint[0](
                packed_region_length + tid, shape
            ).canonicalize()
            func[1, rank](index_tup)

        @parameter
        if PDLLevel() == PDLLevel.OVERLAP_AT_END:
            launch_dependent_grids()

    if shape[rank - 1] % simd_width == 0:
        ctx.enqueue_function[
            _elementwise_gpu_kernel[
                block_size = UInt(block_size), handle_uneven_simd=False
            ]
        ](
            grid_dim=Int(num_blocks),
            block_dim=Int(block_size),
            attributes=pdl_launch_attributes(),
        )
    else:
        ctx.enqueue_function[
            _elementwise_gpu_kernel[
                block_size = UInt(block_size), handle_uneven_simd=True
            ]
        ](
            grid_dim=Int(num_blocks),
            block_dim=Int(block_size),
            attributes=pdl_launch_attributes(),
        )


# ===-----------------------------------------------------------------------===#
# parallelize_over_rows
# ===-----------------------------------------------------------------------===#


fn parallelize_over_rows[
    func: fn (Int, Int) capturing [_] -> None
](shape: IndexList, axis: Int, grain_size: Int):
    """Parallelize func over non-axis dims of shape.

    Parameters:
        func: Function to call on range of rows.

    Args:
        shape: Shape to parallelize over.
        axis: Rows are slices along the axis dimension of shape.
        grain_size: The minimum number of elements to warrant using an additional thread.
    """
    var total_size = shape.flattened_length()
    var num_rows = total_size // shape[axis]

    var num_workers = min(
        num_rows,
        _get_num_workers(total_size, grain_size),
    )
    var chunk_size = ceildiv(num_rows, num_workers)

    @always_inline
    @parameter
    fn task_func(task_id: Int):
        var start_row = task_id * chunk_size
        var end_row = min((task_id + 1) * chunk_size, num_rows)

        func(start_row, end_row)

    sync_parallelize[task_func](num_workers)


# ===-----------------------------------------------------------------------===#
# stencil
# ===-----------------------------------------------------------------------===#

alias stencil = _stencil_impl_cpu
alias stencil_gpu = _stencil_impl_gpu


fn _stencil_impl_cpu[
    shape_element_type: DType,
    input_shape_element_type: DType, //,
    rank: Int,
    stencil_rank: Int,
    stencil_axis: IndexList[stencil_rank, **_],
    simd_width: Int,
    dtype: DType,
    map_fn: fn (IndexList[stencil_rank, **_]) capturing [_] -> (
        IndexList[stencil_rank, **_],
        IndexList[stencil_rank, **_],
    ),
    map_strides: fn (dim: Int) capturing [_] -> Int,
    load_fn: fn[simd_width: Int, dtype: DType] (
        IndexList[rank, **_]
    ) capturing [_] -> SIMD[dtype, simd_width],
    compute_init_fn: fn[simd_width: Int] () capturing [_] -> SIMD[
        dtype, simd_width
    ],
    compute_fn: fn[simd_width: Int] (
        IndexList[rank, **_],
        SIMD[dtype, simd_width],
        SIMD[dtype, simd_width],
    ) capturing [_] -> SIMD[dtype, simd_width],
    compute_finalize_fn: fn[simd_width: Int] (
        IndexList[rank, **_], SIMD[dtype, simd_width]
    ) capturing [_] -> None,
](
    shape: IndexList[rank, element_type=shape_element_type],
    input_shape: IndexList[rank, element_type=input_shape_element_type],
):
    """Computes stencil operation in parallel.

    Computes output as a function that processes input stencils, stencils are
    computed as a continuous region for each output point that is determined
    by map_fn : map_fn(y) -> lower_bound, upper_bound. The boundary conditions
    for regions that fail out of the input domain are handled by load_fn.


    Parameters:
        shape_element_type: The element dtype of the shape.
        input_shape_element_type: The element dtype of the input shape.
        rank: Input and output domain rank.
        stencil_rank: Rank of stencil subdomain slice.
        stencil_axis: Stencil subdomain axes.
        simd_width: The SIMD vector width to use.
        dtype: The input and output data dtype.
        map_fn: A function that a point in the output domain to the input co-domain.
        map_strides: A function that returns the stride for the dim.
        load_fn: A function that loads a vector of simd_width from input.
        compute_init_fn: A function that initializes vector compute over the stencil.
        compute_fn: A function the process the value computed for each point in the stencil.
        compute_finalize_fn: A function that finalizes the computation of a point in the output domain given a stencil.

    Args:
        shape: The shape of the output buffer.
        input_shape: The shape of the input buffer.
    """
    constrained[rank == 4, "Only stencil of rank-4 supported"]()
    constrained[
        stencil_axis[0] == 1 and stencil_axis[1] == 2,
        "Only stencil spatial axes [1, 2] are supported",
    ]()

    var total_size = shape.flattened_length()

    var num_workers = _get_num_workers(total_size)
    var parallelism_size = total_size // shape[rank - 1]
    var chunk_size = ceildiv(parallelism_size, num_workers)

    alias unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    @always_inline
    @parameter
    fn task_func(i: Int):
        var start_parallel_offset = i * chunk_size
        var end_parallel_offset = min((i + 1) * chunk_size, parallelism_size)

        var len = end_parallel_offset - start_parallel_offset
        if len <= 0:
            return

        for parallel_offset in range(
            start_parallel_offset, end_parallel_offset
        ):
            var indices = _get_start_indices_of_nth_subvolume(
                parallel_offset, shape
            )

            @always_inline
            @parameter
            fn func_wrapper[simd_width: Int](idx: Int):
                indices[rank - 1] = idx
                var stencil_indices = IndexList[
                    stencil_rank, element_type = stencil_axis.element_type
                ](indices[stencil_axis[0]], indices[stencil_axis[1]])
                var bounds = map_fn(stencil_indices)
                var lower_bound = bounds[0]
                var upper_bound = bounds[1]
                var step_i = map_strides(0)
                var step_j = map_strides(1)
                var result = compute_init_fn[simd_width]()
                var input_height = input_shape[1]
                var input_width = input_shape[2]

                # In the below loops, each part corresponds to either a padded
                # side (A, B, C, D), or the main part of the input buffer (X)
                # For the padded parts we do not need to load pad value
                # (e.g., neginf for max_pool, or 0 for avg_pool) because result
                # is initialized by compute_init_fn() to the appropriate value
                # (e.g., neginf or 0 in the max_pool/avg_pool cases).
                # Loading and calculating the result for boundary locations
                # (A-D) as in:
                #   var val = pad_value
                #   result = compute_fn[simd_width](point_idx, result, val)
                # would therefore make no difference to not doing it at
                # all.

                # NOTE: The above works when padding is constant and invariant
                # to input coordinates.

                #  AAAAAAAA
                #  AAAAAAAA
                #  BBXXXXDD
                #  BBXXXXDD
                #  BBXXXXDD
                #  BBXXXXDD
                #  CCCCCCCC
                #  CCCCCCCC

                # Calculation for lower_bound below takes into account dilation
                # across rows dimension.
                # Will be the zero if dilation is 1, or the closest point >0 if
                # dilation > 1
                if lower_bound[0] < 0:
                    var mul_i = ceildiv(-lower_bound[0], step_i)
                    lower_bound[0] = lower_bound[0] + mul_i * step_i
                if lower_bound[1] < 0:
                    var mul_j = ceildiv(-lower_bound[1], step_j)
                    lower_bound[1] = lower_bound[1] + mul_j * step_j

                # Part X (inner part)
                for i in range(
                    lower_bound[0],
                    min(input_height, upper_bound[0]),
                    step_i,
                ):
                    for j in range(
                        lower_bound[1],
                        min(input_width, upper_bound[1]),
                        step_j,
                    ):
                        var point_idx = IndexList[
                            rank, element_type=shape_element_type
                        ](indices[0], i, j, indices[3])

                        var val = load_fn[simd_width, dtype](point_idx)
                        result = compute_fn[simd_width](point_idx, result, val)

                compute_finalize_fn[simd_width](indices, result)

            vectorize[func_wrapper, simd_width, unroll_factor=unroll_factor](
                shape[rank - 1]
            )

    sync_parallelize[task_func](num_workers)


fn _stencil_impl_gpu[
    shape_element_type: DType,
    input_shape_element_type: DType, //,
    rank: Int,
    stencil_rank: Int,
    stencil_axis: IndexList[stencil_rank, **_],
    simd_width: Int,
    dtype: DType,
    map_fn: fn (IndexList[stencil_rank, **_]) capturing [_] -> (
        IndexList[stencil_rank, **_],
        IndexList[stencil_rank, **_],
    ),
    map_strides: fn (dim: Int) capturing [_] -> Int,
    load_fn: fn[simd_width: Int, dtype: DType] (
        IndexList[rank, **_]
    ) capturing [_] -> SIMD[dtype, simd_width],
    compute_init_fn: fn[simd_width: Int] () capturing [_] -> SIMD[
        dtype, simd_width
    ],
    compute_fn: fn[simd_width: Int] (
        IndexList[rank, **_],
        SIMD[dtype, simd_width],
        SIMD[dtype, simd_width],
    ) capturing [_] -> SIMD[dtype, simd_width],
    compute_finalize_fn: fn[simd_width: Int] (
        IndexList[rank, **_], SIMD[dtype, simd_width]
    ) capturing [_] -> None,
](
    ctx: DeviceContext,
    shape: IndexList[rank, element_type=shape_element_type],
    input_shape: IndexList[rank, element_type=input_shape_element_type],
) raises:
    """(Naive implementation) Computes stencil operation in parallel on GPU.

    Parameters:
        shape_element_type: The element dtype of the shape.
        input_shape_element_type: The element dtype of the input shape.
        rank: Input and output domain rank.
        stencil_rank: Rank of stencil subdomain slice.
        stencil_axis: Stencil subdomain axes.
        simd_width: The SIMD vector width to use.
        dtype: The input and output data dtype.
        map_fn: A function that a point in the output domain to the input co-domain.
        map_strides: A function that returns the stride for the dim.
        load_fn: A function that loads a vector of simd_width from input.
        compute_init_fn: A function that initializes vector compute over the stencil.
        compute_fn: A function the process the value computed for each point in the stencil.
        compute_finalize_fn: A function that finalizes the computation of a point in the output domain given a stencil.

    Args:
        ctx: The DeviceContext to use for GPU execution.
        shape: The shape of the output buffer.
        input_shape: The shape of the input buffer.
    """
    constrained[rank == 4, "Only stencil of rank-4 supported"]()
    constrained[
        stencil_axis[0] == 1 and stencil_axis[1] == 2,
        "Only stencil spatial axes [1, 2] are supported",
    ]()

    # GPU kernel implementation
    @always_inline
    @parameter
    fn stencil_kernel():
        # Get thread indices
        var tid_x = thread_idx.x
        var tid_y = thread_idx.y
        var bid_x = block_idx.x
        var bid_y = block_idx.y
        var bid_z = block_idx.z

        # Calculate global indices
        var x = bid_x * block_dim.x + tid_x
        var y = bid_y * block_dim.y + tid_y

        # Calculate batch and channel from bid_z
        var batch_idx = bid_z // shape[3]
        var channel = bid_z % shape[3]

        # Early exit if outside bounds
        if x >= UInt(shape[2]) or y >= UInt(shape[1]):
            return

        # Create output point indices with computed batch and channel
        var indices = IndexList[rank, element_type=shape_element_type](
            Int(batch_idx), Int(y), Int(x), Int(channel)
        )

        # Process stencil for this point
        var stencil_indices = IndexList[
            stencil_rank, element_type = stencil_axis.element_type
        ](indices[stencil_axis[0]], indices[stencil_axis[1]])
        var bounds = map_fn(stencil_indices)
        var lower_bound = bounds[0]
        var upper_bound = bounds[1]
        var step_i = map_strides(0)
        var step_j = map_strides(1)
        var result = compute_init_fn[simd_width]()
        var input_height = input_shape[1]
        var input_width = input_shape[2]

        # Handle boundary conditions
        if lower_bound[0] < 0:
            var mul_i = ceildiv(-lower_bound[0], step_i)
            lower_bound[0] = lower_bound[0] + mul_i * step_i
        if lower_bound[1] < 0:
            var mul_j = ceildiv(-lower_bound[1], step_j)
            lower_bound[1] = lower_bound[1] + mul_j * step_j

        # Process stencil window
        for i in range(
            lower_bound[0],
            min(input_height, upper_bound[0]),
            step_i,
        ):
            for j in range(
                lower_bound[1],
                min(input_width, upper_bound[1]),
                step_j,
            ):
                var point_idx = IndexList[
                    rank, element_type=shape_element_type
                ](indices[0], i, j, indices[3])
                var val = load_fn[simd_width, dtype](point_idx)
                result = compute_fn[simd_width](point_idx, result, val)

        compute_finalize_fn[simd_width](indices, result)

    # Calculate grid and block dimensions
    var block_dim = (32, 32, 1)
    var grid_dim = (
        ceildiv(shape[2], block_dim[0]),  # width
        ceildiv(shape[1], block_dim[1]),  # height
        shape[0] * shape[3],  # batch_size * num_channels
    )

    # Compile and launch kernel
    ctx.enqueue_function[stencil_kernel](grid_dim=grid_dim, block_dim=block_dim)

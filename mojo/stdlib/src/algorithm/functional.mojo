# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements higher-order functions."""

from Assert import assert_param
from Index import StaticIntTuple
from List import VariadicList
from LLCL import (
    num_cores,
    Runtime,
    OutputChainPtr,
    OwningOutputChainPtr,
    AsyncTaskGroupPtr,
)
from math import div_ceil, min, max
from numerics import FlushDenormals
from Range import range
from TargetInfo import triple_is_nvidia_cuda

# ===----------------------------------------------------------------------===#
# Map
# ===----------------------------------------------------------------------===#


@always_inline
fn map[func: fn (Int) capturing -> None](size: Int):
    """Maps a function over a range from 0 to size.

    Parameters:
        func: Function to map.

    Args:
        size: The number of elements.
    """
    for i in range(size):
        func(i)


# ===----------------------------------------------------------------------===#
# unroll
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll[
    count: Int,
    func: fn[idx: Int] () capturing -> None,
]():
    """Repeatedly evaluates a function `count` times.

    Parameters:
        count: A number of repetitions.
        func: The function to evaluate. The function should take a single Int
          argument.
    """
    _unroll_impl[0, count, func]()


@always_inline
fn _unroll_impl[
    idx: Int,
    count: Int,
    func: fn[idx: Int] () capturing -> None,
]():
    @parameter
    if idx < count:
        func[idx]()
        _unroll_impl[idx + 1, count, func]()


# ===----------------------------------------------------------------------===#
# unroll
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll[
    dim0: Int,
    dim1: Int,
    func: fn[idx0: Int, idx1: Int] () capturing -> None,
]():
    """Repeatedly evaluates a 2D nested loop.

    Parameters:
        dim0: The first dimension size.
        dim1: The second dimension size.
        func: The function to evaluate. The function should take two Int
          arguments.
    """

    @always_inline
    @parameter
    fn outer_func_wrapper[idx0: Int]():
        @always_inline
        @parameter
        fn inner_func_wrapper[idx1: Int]():
            func[idx0, idx1]()

        unroll[dim1, inner_func_wrapper]()

    unroll[dim0, outer_func_wrapper]()


# ===----------------------------------------------------------------------===#
# unroll
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll[
    dim0: Int,
    dim1: Int,
    dim2: Int,
    func: fn[idx0: Int, idx1: Int, idx2: Int] () capturing -> None,
]():
    """Repeatedly evaluates a 3D nested loop.

    Parameters:
        dim0: The first dimension size.
        dim1: The second dimension size.
        dim2: The second dimension size.
        func: The function to evaluate. The function should take three Int
          arguments.
    """

    @always_inline
    @parameter
    fn func_wrapper[idx0: Int, idx1: Int]():
        alias _idx1 = idx1 // dim2
        alias _idx2 = idx1 % dim2
        func[idx0, _idx1, _idx2]()

    unroll[dim0, dim1 * dim2, func_wrapper]()


# ===----------------------------------------------------------------------===#
# Vectorize
# ===----------------------------------------------------------------------===#


@always_inline
fn vectorize[
    simd_width: Int,
    func: fn[width: Int] (Int) capturing -> None,
](size: Int):
    """Maps a function which is parametrized over a simd_width over a range
    from 0 to size in simd fashion.

    Parameters:
        simd_width: The SIMD vector width.
        func: The function for the loop body.

    Args:
        size: The total loop count.
    """
    assert_param[simd_width > 0, "simd width must be > 0"]()
    vectorize_unroll[simd_width, 1, func](size)


fn _variadic_get(
    a: __mlir_type[`!kgen.variadic<`, fn (Int) capturing -> NoneType, `>`],
    idx: Int,
) -> fn (Int) capturing -> NoneType:
    return __mlir_op.`pop.variadic.get`(a, idx.value)


@always_inline
fn vectorize_unroll[
    simd_width: Int,
    unroll_factor: Int,
    func: fn[width: Int] (Int) capturing -> NoneType,
](size: Int):
    """Maps a function which is parametrized over a simd_width over a range
    from 0 to size in simd fashion and unroll the loop by unroll_factor.

    Parameters:
        simd_width: The SIMD vector width.
        unroll_factor: The unroll factor for the main loop.
        func: The function for the loop body.

    Args:
        size: The total loop count.
    """
    assert_param[simd_width > 0, "simd width must be > 0"]()
    assert_param[unroll_factor > 0, "unroll factor must be > 0"]()

    alias unrolled_simd_width = simd_width * unroll_factor
    let vector_end_unrolled_simd = (
        size // unrolled_simd_width
    ) * unrolled_simd_width
    let vector_end_simd = (size // simd_width) * simd_width

    # Explicitly fork on all available candidates for `func[simd_width]`.
    # We could've just called it from the loops below, but making it explicit
    # allows us to completely control how the search space is generated.
    # This way there is just one multi-versioning happening in this function
    # and it is this `kgen.param.fork`.
    alias vector_func_impls = __mlir_attr[
        `#kgen.param.expr<get_all_impls,`,
        func[simd_width],
        `> :`,
        __mlir_type[`!kgen.variadic<`, fn (Int) capturing -> NoneType, `>`],
    ]
    # TODO: `kgen.param.fork` and `get_all_impls` invocations should have
    # nice looking wrappers.
    __mlir_op.`kgen.param.fork`[
        paramDecl : __mlir_attr[
            `#kgen<param.decl result_hidden :`,
            fn (Int) capturing -> NoneType,
            `>`,
        ],
        values : __mlir_attr[
            vector_func_impls,
            `: !kgen.variadic<`,
            fn (Int) capturing -> NoneType,
            `>`,
        ],
    ]()
    alias vector_func_impl = __mlir_attr[
        `#kgen.param.decl.ref<"result_hidden"> :`,
        fn (Int) capturing -> NoneType,
    ]

    # For scalar version we will just get the first available implementation.
    # If we do `kgen.param.fork` here, we will blow up the search space
    # quadratically.
    alias scalar_func_impls = __mlir_attr[
        `#kgen.param.expr<get_all_impls,`,
        func[1],
        `> :`,
        __mlir_type[`!kgen.variadic<`, fn (Int) capturing -> NoneType, `>`],
    ]
    alias scalar_func_impl = _variadic_get(scalar_func_impls, 0)

    @always_inline
    @parameter
    fn unrolled_func(unrolled_simd_idx: Int):
        @always_inline
        @parameter
        fn unroll_iter[idx: Int]():
            vector_func_impl(unrolled_simd_idx + idx * simd_width)

        unroll[unroll_factor, unroll_iter]()

    for unrolled_simd_idx in range(
        0, vector_end_unrolled_simd, unrolled_simd_width
    ):
        unrolled_func(unrolled_simd_idx)

    @parameter
    if unroll_factor != 1:
        for simd_idx in range(
            vector_end_unrolled_simd, vector_end_simd, simd_width
        ):
            vector_func_impl(simd_idx)

    for i in range(vector_end_simd, size):
        scalar_func_impl(i)


# ===----------------------------------------------------------------------===#
# Parallelize
# ===----------------------------------------------------------------------===#


@always_inline
fn async_parallelize[
    func: fn (Int) capturing -> None
](out_chain: OutputChainPtr, num_work_items: Int):
    """Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel and
    returns immediately. The out_chain will be marked as ready only when all
    sub-tasks have completed.

    Execute func(0) ... func(num_work_items-1) as sub-tasks in parallel and
    mark out_chain as ready when all functions have returned. This function
    will return when the sub-tasks have been scheduled but not necessarily
    completed. The runtime may execute the sub-tasks in any order and with any
    degree of concurrency.

    All free variables in func must be "async safe". Currently this means:
     - The variable must be bound by a by-val function argument (ie no &),
       or let binding.
     - The variable's type must be "async safe", ie is marked as
       @register_passable and any internal pointers are to memory with
       lifetime at least until out_chain is ready. In practice, this means
       only pointers to buffers held alive by the runtime.
    Consider using sync_parallelize if this requirement is too onerous.

    If num_work_items is 0 then the out_chain is marked as ready
    before async_parallelize returns. If num_work_items is 1 then func(0) may
    still be executed as a sub-task.

    Parameters:
        func: The function to invoke.

    Args:
        out_chain: Out chain onto which to signal completion.
        num_work_items: Number of parallel tasks.
    """

    # We have no tasks, so do nothing.
    if num_work_items == 0:
        # No-op
        out_chain.mark_ready()
        return

    # If there is a single task, consider executing it in the host thread.
    # If the runtime has only 1 thread, executing inline is guaranteed to
    # reduce launch overhead by executing immediately.
    # If the runtime has more than 1 thread, executing inline may be suboptimal
    # since it may block other kernel launches that can then execute in parallel.
    # TODO (#14524): Add heuristic to determine when inlining the func is
    # appropriate when the runtime has more than 1 thread.
    if num_work_items == 1 and out_chain.get_runtime().parallelism_level() == 1:
        with FlushDenormals():
            func(0)
        out_chain.mark_ready()
        return

    @always_inline
    @parameter
    async fn task_fn(i: Int):
        with FlushDenormals():
            func(i)

    var atg = AsyncTaskGroupPtr(num_work_items, out_chain)
    for i in range(num_work_items):
        let coroutine: Coroutine[NoneType] = task_fn(i)
        atg.add_task(coroutine ^)


@always_inline
fn sync_parallelize[
    func: fn (Int) capturing -> None
](out_chain: OutputChainPtr, num_work_items: Int):
    """Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel.
    Marks out_chain as ready and returns only when all sub-tasks have
    completed.

    Execute func(0) ... func(num_work_items-1) as sub-tasks in parallel,
    and return only when they have all functions have returned. The runtime
    may execute the sub-tasks in any order and with any degree of concurrency.
    The out_chain will be marked as ready before returning.

    Parameters:
        func: The function to invoke.

    Args:
        out_chain: Out chain onto which to signal completion.
        num_work_items: Number of parallel tasks.
    """

    # CAUTION CAUTION CAUTION: out_chain may be the overall model's final
    # output chain, in which case emplacing it may signal that the model has
    # finished and the overall model runtime can be torn down. However,
    # this call is still nested inside a Mojo and C++ call stack which may
    # still depend on the runtime. Thus we must be very careful to emplace
    # out_chain in a way that its waiters will be run only after this call
    # has returned all the way back to the worker thread loop. We do this
    # using a fresh chain.
    let local_chain = OwningOutputChainPtr(out_chain.get_runtime())
    async_parallelize[func](local_chain.borrow(), num_work_items)
    local_chain.wait()
    out_chain.mark_ready()


@always_inline
fn parallelize[func: fn (Int) capturing -> None]():
    """Executes func(0) ... func(N-1) as sub-tasks in parallel and returns when
    all are complete. N is chosen to be the number of physical processors on the
    system.

    Execute func(0) ... func(N-1) as sub-tasks in parallel. This function will
    return only after all the sub-tasks have completed.

    CAUTION: Creates and destroys a local runtime! Do not use from kernels!

    Parameters:
        func: The function to invoke.
    """
    return parallelize[func](num_cores())


@always_inline
fn parallelize[func: fn (Int) capturing -> None](num_work_items: Int):
    """Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel and
    returns when all are complete.

    Execute func(0) ... func(num_work_items-1) as sub-tasks in parallel. This
    function will return only after all the sub-tasks have completed.

    CAUTION: Creates and destroys a local runtime! Do not use from kernels!

    Parameters:
        func: The function to invoke.

    Args:
        num_work_items: Number of parallel tasks.
    """
    let core_count = num_cores()
    let chunk_size = max(div_ceil(num_work_items, core_count), 1)

    @always_inline
    @parameter
    fn coarsed_func(thread_idx: Int):
        for i in range(
            chunk_size * thread_idx,
            min(chunk_size * (thread_idx + 1), num_work_items),
        ):
            func(i)

    with Runtime(core_count) as rt:
        let out_chain = OwningOutputChainPtr(rt)
        async_parallelize[coarsed_func](out_chain.borrow(), core_count)
        out_chain.wait()


# ===----------------------------------------------------------------------===#
# tile
# ===----------------------------------------------------------------------===#

alias Static1DTileUnitFunc = fn[width: Int] (Int) capturing -> None
"""
Signature of a 1d tiled function that performs some work with a static tile size
and an offset. i.e. func<tile_size: Int> (offset: Int)
"""

alias Dynamic1DTileUnitFunc = fn (Int, Int) capturing -> None
"""
Signature of a 1d tiled function that performs some work with a dynamic tile size
  and an offset. i.e. func(offset: Int, tile_size: Int)
"""


alias BinaryTile1DTileUnitFunc = fn[width: Int] (Int, Int) capturing -> None
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
      work_on[3](5)
    should launch computation on item 5,6,7, and should be semantically
    equivalent to
      work_on[1](5), work_on[1](6), work_on[1](7).

    This generator will try to proceed with the given list of tile sizes on the
    listed order. E.g.
        tile [func, (3,2,1)](offset, upperbound)
    will try to call func[3] starting from offset until remaining work is less
    than 3 from upperbound and then try func[2], and then func[1] etc.

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

    @always_inline
    @parameter
    fn static_tile_impl[idx: Int]():
        # Get the tile size to proceed with.
        alias tile_size = tile_size_list[idx]

        # Process work with the tile size until there's not enough remaining work
        #  to fit in a tile.
        while current_offset <= upperbound - tile_size:
            workgroup_function[tile_size](current_offset)
            current_offset += tile_size

    unroll[tile_size_list.__len__(), static_tile_impl]()


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
    for tile_idx in range(tile_size_list.__len__()):
        let tile_size = tile_size_list[tile_idx]
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
    alias num_tiles = secondary_tile_size_list.__len__()

    @always_inline
    @parameter
    fn static_tile_impl[idx: Int]():
        alias secondary_tile_size = secondary_tile_size_list[idx]
        let primary_tile_size = primary_tile_size_list[idx]

        while work_idx <= upperbound - primary_tile_size:
            workgroup_function[secondary_tile_size](work_idx, primary_tile_size)
            work_idx += primary_tile_size

    unroll[num_tiles, static_tile_impl]()

    # launch the last cleanup tile
    if work_idx < upperbound:
        workgroup_function[secondary_cleanup_tile](
            work_idx, primary_cleanup_tile
        )


# ===----------------------------------------------------------------------===#
# tile2d
# ===----------------------------------------------------------------------===#


alias Static2DTileUnitFunc = fn[tile_x: Int, tile_y: Int] (
    Int, Int
) capturing -> None
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
    var current_offset_x: Int = offset_x
    var current_offset_y: Int = offset_y

    alias num_tiles_x = tile_sizes_x.__len__()
    alias num_tiles_y = tile_sizes_y.__len__()

    @always_inline
    @parameter
    fn tile_on_y[idx_y: Int]():
        alias tile_size_y = tile_sizes_y[idx_y]
        while current_offset_y <= upperbound_y - tile_size_y:
            current_offset_x = offset_x

            @always_inline
            @parameter
            fn tile_on_x[idx_x: Int]():
                alias tile_size_x = tile_sizes_x[idx_x]
                while current_offset_x <= upperbound_x - tile_size_x:
                    workgroup_function[tile_size_x, tile_size_y](
                        current_offset_x, current_offset_y
                    )
                    current_offset_x += tile_size_x

            unroll[num_tiles_x, tile_on_x]()
            current_offset_y += tile_size_y

    unroll[num_tiles_y, tile_on_y]()


# ===----------------------------------------------------------------------===#
# Unswitch
# ===----------------------------------------------------------------------===#

# Signature of a function that unswitch can take.
alias SwitchedFunction = fn[sw: Bool] () capturing -> None

# Version of unswitch supporting 2 predicates.
alias SwitchedFunction2 = fn[sw0: Bool, sw1: Bool] () capturing -> None


@always_inline
fn unswitch[switched_func: SwitchedFunction](dynamic_switch: Bool):
    """Performs a functional unswitch transformation.

    Unswitch is a simple pattern that is similar idea to loop unswitching
    pass but extended to functional patterns. The pattern facilitates the
    following code transformation that reduces the number of branches in the
    generated code

        Before:
        ```
            for i in range(...)
                if i < xxx:
                    ...
        ```

        After:

        ```
            if i < ...
                for i in range(...)
                    ...
            else
                for i in range(...)
                    if i < xxx:
                        ...
        ```

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


# ===----------------------------------------------------------------------===#
# TileWithUnswitch
# ===----------------------------------------------------------------------===#

alias Static1DTileUnswitchUnitFunc = fn[width: Int, sw: Bool] (
    Int, Int
) capturing -> None
"""
Signature of a tiled function that performs some work with a static tile size
  and an offset. i.e. func<tile_size: Int> (offset: Int)
"""


@always_inline
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
    var current_offset: Int = offset

    @always_inline
    @parameter
    fn static_tile_impl[idx: Int]():
        # Get the tile size to proceed with.
        let tile_size = tile_size_list[idx]

        # Process work with the tile size until there's not enough remaining work
        #  to fit in a tile.
        while current_offset <= upperbound - tile_size:
            workgroup_function[tile_size_list[idx], True](
                current_offset, upperbound
            )
            current_offset += tile_size

    unroll[tile_size_list.__len__(), static_tile_impl]()

    # Use the last tile size to process the residue.
    if current_offset < upperbound:
        workgroup_function[tile_size_list[tile_size_list.__len__() - 1], False](
            current_offset, upperbound
        )


alias Dynamic1DTileUnswitchUnitFunc = fn[sw: Bool] (
    Int, Int, Int
) capturing -> None


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

    for idx in range(tile_size_list.__len__()):
        # Get the tile size to proceed with.
        let tile_size = tile_size_list[idx]

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
            tile_size_list[tile_size_list.__len__() - 1],
        )


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_num_workers(problem_size: Int, runtime: Runtime) -> Int:
    """Returns a number of workers to run in parallel.

    Args:
        problem_size: The number of parallel tasks.
        runtime: Runtime object.

    Returns:
        The number of workers to run in parallel.
    """
    # Minimum number of elements to warrant an additional thread.
    # copied from https://github.com/pytorch/pytorch/blob/20dfce591ce88bc957ffcd0c8dc7d5f7611a4a3b/aten/src/ATen/TensorIterator.h#L86
    # TODO: refine this heuristic. It may not be appropriate for more compute-heavy
    # ops like gelu.
    # Ensure at least one worker is always returned to avoid division by zero.
    alias GRAIN_SIZE = 32768
    return max(
        1, min(runtime.parallelism_level(), div_ceil(problem_size, GRAIN_SIZE))
    )


# ===----------------------------------------------------------------------===#
# Elementwise
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_start_indices_of_nth_subvolume[
    rank: Int, subvolume_rank: Int
](n: Int, shape: StaticIntTuple[rank]) -> StaticIntTuple[rank]:
    """Converts a flat index into the starting ND indices of the nth subvolume
    with rank subvolume_rank.

    For example:
        - _get_start_indices_of_nth_subvolume[3, 0](n, shape) will return
        the starting indices of the nth element in shape.
        - _get_start_indices_of_nth_subvolume[3, 1](n, shape) will return
        the starting indices of the nth row in shape.
        - _get_start_indices_of_nth_subvolume[3, 2](n, shape) will return
        the starting indices of the nth horizontal slice in shape.

    The ND indices will iterate from right to left. I.E

    shape = (20, 5, 2, N)
    _get_start_indices_of_nth_subvolume[4, 1](1, shape) = (0, 0, 1, 0)
    _get_start_indices_of_nth_subvolume[4, 1](5, shape) = (0, 2, 1, 0)
    _get_start_indices_of_nth_subvolume[4, 1](50, shape) = (5, 0, 0, 0)
    _get_start_indices_of_nth_subvolume[4, 1](56, shape) = (5, 1, 1, 0)

    Parameters:
        rank: The rank of the ND index.
        subvolume_rank: The rank of the subvolume under consideration.

    Args:
        n: The flat index to convert (the nth subvolume to retrieve).
        shape: The shape of the ND space we are converting into.

    Returns:
        Constructed ND-index.
    """

    assert_param[
        subvolume_rank <= rank,
        "subvolume rank cannot be greater than indices rank",
    ]()
    assert_param[subvolume_rank >= 0, "subvolume rank must be non-negative"]()

    # fast impls for common cases
    @parameter
    if rank == 2 and subvolume_rank == 1:
        return StaticIntTuple[rank](n, 0)

    @parameter
    if rank - 1 == subvolume_rank:
        var out = StaticIntTuple[rank](0)
        out[0] = n
        return out

    @parameter
    if rank == subvolume_rank:
        return StaticIntTuple[rank](0)

    var out = StaticIntTuple[rank](0)
    var curr_index = n

    @always_inline
    @parameter
    fn compute_shape[idx: Int]():
        alias i = rank - 1 - idx - subvolume_rank
        out[i] = curr_index % shape[i]
        curr_index //= shape[i]

    unroll[rank - subvolume_rank, compute_shape]()

    return out


@always_inline
@adaptive
fn elementwise[
    rank: Int,
    simd_width: Int,
    func: fn[width: Int, rank: Int] (StaticIntTuple[rank]) capturing -> None,
](shape: StaticIntTuple[rank], out_chain: OutputChainPtr):
    """Executes func[width, rank](indices) as sub-tasks for a suitable
    combination of width and indices so as to cover shape.

    Parameters:
        rank: The rank of the buffer.
        simd_width: The SIMD vector width to use.
        func: The body function.

    Args:
        shape: The shape of the buffer.
        out_chain: The our chain to attach results to.
    """

    _elementwise_impl[
        rank,
        simd_width,
        False
        # On CUDA devices, we do not want to launch threads, so we use the
        # blocking API
        or triple_is_nvidia_cuda(),
        func,
    ](shape, out_chain)


@always_inline
@adaptive
fn _elementwise_impl[
    rank: Int,
    simd_width: Int,
    use_blocking_impl: Bool,
    func: fn[width: Int, rank: Int] (StaticIntTuple[rank]) capturing -> None,
](shape: StaticIntTuple[rank], out_chain: OutputChainPtr):
    """Executes func[width, rank](indices) as sub-tasks for a suitable
    combination of width and indices so as to cover shape.

    Parameters:
        rank: The rank of the buffer.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: If true this is a blocking op.
        func: The body function.

    Args:
        shape: The shape of the buffer.
        out_chain: The our chain to attach results to.
    """
    assert_param[rank == 1, "Specialization for 1D"]()

    alias unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    let problem_size = shape.flattened_length()

    @parameter
    if use_blocking_impl:

        @always_inline
        @parameter
        fn blocking_task_fun[simd_width: Int](idx: Int):
            func[simd_width, rank](idx)

        vectorize_unroll[
            simd_width,
            unroll_factor,
            blocking_task_fun,
        ](problem_size)
        return

    let num_workers = _get_num_workers(problem_size, out_chain.get_runtime())
    let chunk_size = div_ceil(problem_size, num_workers)

    @always_inline
    @parameter
    fn task_func(i: Int):
        let start_offset = i * chunk_size
        let end_offset = min((i + 1) * chunk_size, problem_size)
        let len = end_offset - start_offset

        @always_inline
        @parameter
        fn func_wrapper[simd_width: Int](idx: Int):
            let offset = start_offset + idx
            func[simd_width, rank](offset)

        vectorize_unroll[
            simd_width,
            unroll_factor,
            func_wrapper,
        ](len)

    async_parallelize[task_func](out_chain, num_workers)


@always_inline
@adaptive
fn _elementwise_impl[
    rank: Int,
    simd_width: Int,
    use_blocking_impl: Bool,
    func: fn[width: Int, rank: Int] (StaticIntTuple[rank]) capturing -> None,
](shape: StaticIntTuple[rank], out_chain: OutputChainPtr):
    """Executes func[width, rank](indices) as sub-tasks for a suitable
    combination of width and indices so as to cover shape.

    All free vars in func must be "async safe", see async_parallelize.

    Parameters:
        rank: The rank of the buffer.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: If true this is a blocking op.
        func: The body function.

    Args:
        shape: The shape of the buffer.
        out_chain: The our chain to attach results to.
    """

    assert_param[rank > 1, "Specialization for ND where N > 1"]()

    alias unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    # Strategy: we parallelize over all dimensions except the innermost and
    # vectorize over the innermost dimension. We unroll the innermost dimension
    # by a factor of unroll_factor.

    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    let total_size: Int = shape.flattened_length()

    @parameter
    if use_blocking_impl:

        @always_inline
        @parameter
        fn blocking_task_fn(i: Int):
            var indices = _get_start_indices_of_nth_subvolume[rank, 1](i, shape)

            @always_inline
            @parameter
            fn func_wrapper[simd_width: Int](idx: Int):
                # The inner most dimension is vectorized, so we set it
                # to the index offset.
                indices[rank - 1] = idx
                func[simd_width, rank](indices)

            # We vectorize over the innermost dimension.
            vectorize_unroll[
                simd_width,
                unroll_factor,
                func_wrapper,
            ](shape[rank - 1])

        map[blocking_task_fn](total_size // shape[rank - 1])

        return

    let num_workers = _get_num_workers(total_size, out_chain.get_runtime())
    let parallelism_size = total_size // shape[rank - 1]
    let chunk_size = div_ceil(parallelism_size, num_workers)

    @always_inline
    @parameter
    fn task_func(i: Int):
        let start_parallel_offset = i * chunk_size
        let end_parallel_offset = min((i + 1) * chunk_size, parallelism_size)

        let len = end_parallel_offset - start_parallel_offset
        if len <= 0:
            return

        for parallel_offset in range(
            start_parallel_offset, end_parallel_offset
        ):
            var indices = _get_start_indices_of_nth_subvolume[rank, 1](
                parallel_offset, shape
            )

            @always_inline
            @parameter
            fn func_wrapper[simd_width: Int](idx: Int):
                # The inner most dimension is vectorized, so we set it
                # to the index offset.
                indices[rank - 1] = idx
                func[simd_width, rank](indices)

            # We vectorize over the innermost dimension.
            vectorize_unroll[
                simd_width,
                unroll_factor,
                func_wrapper,
            ](shape[rank - 1])

    async_parallelize[task_func](out_chain, num_workers)

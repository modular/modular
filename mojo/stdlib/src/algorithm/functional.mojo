# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements higher-order functions."""

from Assert import assert_param_msg
from Coroutine import Coroutine
from Index import StaticIntTuple
from List import VariadicList
from LLCL import Runtime, OutputChainPtr, AsyncTaskGroupPtr
from Math import div_ceil, min, max
from Range import range

alias InlinedFixedVectorLength = 64

# ===----------------------------------------------------------------------===#
# Map
# ===----------------------------------------------------------------------===#


@always_inline
fn map[
    func: __mlir_type[`!kgen.signature<(`, Int, ` borrow) -> `, NoneType, `>`],
](size: Int):
    """Map a function over a range from 0 to size.

    Parameters:
        func: Function to map.

    Args:
        size: number of elements.
    """
    for i in range(size):
        func(i)


# ===----------------------------------------------------------------------===#
# unroll
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll[
    count: Int,
    func: __mlir_type[`!kgen.signature<<`, Int, `>() -> `, NoneType, `>`],
]():
    """Reateadly evaluate a function `count` times.

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
    func: __mlir_type[`!kgen.signature<<`, Int, `>() -> `, NoneType, `>`],
]():
    @parameter
    if idx < count:
        func[idx]()
        _unroll_impl[idx + 1, count, func]()


# ===----------------------------------------------------------------------===#
# unroll2
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll2[
    dim0: Int,
    dim1: Int,
    func: __mlir_type[
        `!kgen.signature<<`, Int, `, `, Int, `>() -> `, NoneType, `>`
    ],
]():
    """Repeateadly evaluate a 2D nested loop.

    Parameters:
        dim0: The first dimension size.
        dim1: The second dimension size.
        func: The function to evaluate. The function should take two Int
          arguments.
    """

    @always_inline
    fn func_wrapper[idx: Int]():
        alias idx0 = idx // dim1
        alias idx1 = idx % dim1
        func[idx0, idx1]()

    unroll[dim0 * dim1, func_wrapper]()


# ===----------------------------------------------------------------------===#
# unroll3
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll3[
    dim0: Int,
    dim1: Int,
    dim2: Int,
    func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `, `,
        Int,
        `, `,
        Int,
        `>() -> `,
        NoneType,
        `>`,
    ],
]():
    """Repeateadly evaluate a 3D nested loop.

    Parameters:
        dim0: The first dimension size.
        dim1: The second dimension size.
        dim2: The second dimension size.
        func: The function to evaluate. The function should take three Int
          arguments.
    """

    @always_inline
    fn func_wrapper[idx0: Int, idx1: Int]():
        alias _idx1 = idx1 // dim2
        alias _idx2 = idx1 % dim2
        func[idx0, _idx1, _idx2]()

    unroll2[dim0, dim1 * dim2, func_wrapper]()


# ===----------------------------------------------------------------------===#
# Vectorize
# ===----------------------------------------------------------------------===#


alias fn_sig_type = __mlir_type[
    `!kgen.signature<(`,
    Int,
    `) -> `,
    NoneType,
    `>`,
]

alias fn_simd_sig_type = __mlir_type[
    `!kgen.signature<<`,
    Int,
    `>(`,
    Int,
    ` borrow) -> `,
    NoneType,
    `>`,
]


@always_inline
fn vectorize[
    simd_width: Int,
    func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `>(`,
        Int,
        ` borrow) -> `,
        NoneType,
        `>`,
    ],
](size: Int):
    """Map a function which is parametrized over a simd_width over a range
    from 0 to size in simd fashion.

    Parameters:
        simd_width: The SIMD vector width.
        func: The function for the loop body.

    Args:
        size: The total loop count.
    """
    assert_param_msg[simd_width > 0, "simd width must be > 0"]()
    vectorize_unroll[simd_width, 1, func](size)


fn _variadic_get(
    a: __mlir_type[`!kgen.variadic<`, fn_sig_type, `>`], idx: Int
) -> fn_sig_type:
    return __mlir_op.`pop.variadic.get`(a, idx.__as_mlir_index())


@always_inline
fn vectorize_unroll[
    simd_width: Int,
    unroll_factor: Int,
    func: fn_simd_sig_type,
](size: Int):
    """Map a function which is parametrized over a simd_width over a range
    from 0 to size in simd fashion and unroll the loop by unroll_factor.

    Parameters:
        simd_width: The SIMD vector width.
        unroll_factor: The unroll factor for the main loop.
        func: The function for the loop body.

    Args:
        size: The total loop count.
    """
    assert_param_msg[simd_width > 0, "simd width must be > 0"]()
    assert_param_msg[unroll_factor > 0, "unroll factor must be > 0"]()

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
        __mlir_type[`!kgen.variadic<`, fn_sig_type, `>`],
    ]
    # TODO: `kgen.param.fork` and `get_all_impls` invocations should have
    # nice looking wrappers.
    __mlir_op.`kgen.param.fork`[
        paramDecl : __mlir_attr[
            `#kgen<param.decl result_hidden :`, fn_sig_type, `>`
        ],
        values : __mlir_attr[
            vector_func_impls, `: !kgen.variadic<`, fn_sig_type, `>`
        ],
    ]()
    alias vector_func_impl = __mlir_attr[
        `#kgen.param.decl.ref<"result_hidden"> :`, fn_sig_type
    ]

    # For scalar version we will just get the first available implementation.
    # If we do `kgen.param.fork` here, we will blow up the search space
    # quadratically.
    alias scalar_func_impls = __mlir_attr[
        `#kgen.param.expr<get_all_impls,`,
        func[1],
        `> :`,
        __mlir_type[`!kgen.variadic<`, fn_sig_type, `>`],
    ]
    alias scalar_func_impl = _variadic_get(scalar_func_impls, 0)

    @always_inline
    fn unrolled_func(unrolled_simd_idx: Int):
        @always_inline
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
    func: __mlir_type[`!kgen.signature<(`, Int, ` borrow) -> `, NoneType, `>`],
](out_chain: OutputChainPtr, num_work_items: Int):
    """Execute func(0) ... func(num_work_items-1) as sub-tasks in parallel.

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

    If num_work_items is 0 then the out_chain is marked as ready
    before async_parallelize returns. If num_work_items is 1 then func(0) is
    still executed as a sub-task.

    Parameters:
        func: The function to invoke.

    Args:
        out_chain: Out chain to attach results too.
        num_work_items: Number of parallel tasks.
    """

    # We have no tasks, so do nothing.
    if num_work_items == 0:
        # No-op
        out_chain.mark_ready()
        return

    @always_inline
    async fn task_fn(i: Int):
        func(i)

    var atg = AsyncTaskGroupPtr(num_work_items, out_chain)
    for i in range(num_work_items):
        let coroutine: Coroutine[NoneType] = task_fn(i)
        atg.add_task(coroutine)


# ===----------------------------------------------------------------------===#
# invoke
# ===----------------------------------------------------------------------===#


fn invoke(func: __mlir_type.`() -> ()`):
    """Invoke a function.

    Args:
        func: The function to invoke.
    """
    __mlir_op.`pop.call_indirect`[_type:[]](func)


fn invoke(
    func: __mlir_type.`() -> (!pop.pointer<scalar<si8>>)`,
) -> __mlir_type.`!pop.pointer<scalar<si8>>`:
    """Invoke a function.

    Args:
        func: The function to invoke.

    Returns:
        The result returned by the function.
    """
    return __mlir_op.`pop.call_indirect`[
        _type : [__mlir_type.`!pop.pointer<scalar<si8>>`]
    ](func)


fn invoke[
    arg_type: AnyType
](func: __mlir_type[`(`, arg_type, `) -> ()`], arg: arg_type):
    """Invoke a parametrized function.

    Parameter:
        arg_type: The type of the function input.

    Args:
        func: The function to invoke.
        arg: The argument to pass to the function.
    """
    __mlir_op.`pop.call_indirect`[_type:[]](func, arg)


fn invoke[
    result_type: AnyType,
    arg_type: AnyType,
](
    func: __mlir_type[`(`, arg_type, `) -> (`, result_type, `)`], arg: arg_type
) -> result_type:
    """Invoke a parametrized function.

    Parameter:
        result_type: The type of the function result.
        arg_type: The type of the function input.

    Args:
        func: The function to invoke.
        arg: The argument to pass to the function.

    Returns:
        The result returned by the function.
    """
    return __mlir_op.`pop.call_indirect`[_type:[result_type]](func, arg)


fn invoke[
    result_type: AnyType,
    arg1_type: AnyType,
    arg2_type: AnyType,
](
    func: __mlir_type[
        `(`, arg1_type, `,`, arg2_type, `) -> (`, result_type, `)`
    ],
    arg1: arg1_type,
    arg2: arg2_type,
) -> result_type:
    """Invoke a parametrized function.

    Parameter:
        result_type: The type of the function result.
        arg1_type: The type of the 1st function input.
        arg2_type: The type of the 2nd function input.

    Args:
        func: The function to invoke.
        arg1: The 1st argument to pass to the function.
        arg2: The 2nd argument to pass to the function.

    Returns:
        The result returned by the function.
    """
    return __mlir_op.`pop.call_indirect`[_type:[result_type]](func, arg1, arg2)


fn invoke[
    result_type: AnyType,
    arg1_type: AnyType,
    arg2_type: AnyType,
    arg3_type: AnyType,
](
    func: __mlir_type[
        `(`,
        arg1_type,
        `,`,
        arg2_type,
        `,`,
        arg3_type,
        `) -> (`,
        result_type,
        `)`,
    ],
    arg1: arg1_type,
    arg2: arg2_type,
    arg3: arg3_type,
) -> result_type:
    """Invoke a parametrized function.

    Parameter:
        result_type: The type of the function result.
        arg1_type: The type of the 1st function input.
        arg2_type: The type of the 2nd function input.
        arg3_type: The type of the 3rd function input.

    Args:
        func: The function to invoke.
        arg1: The 1st argument to pass to the function.
        arg2: The 2nd argument to pass to the function.
        arg3: The 3rd argument to pass to the function.

    Returns:
        The result returned by the function.
    """
    return __mlir_op.`pop.call_indirect`[_type:[result_type]](
        func, arg1, arg2, arg3
    )


# ===----------------------------------------------------------------------===#
# tile
# ===----------------------------------------------------------------------===#

"""
Signature of a tiled function that performs some work with a static tile size
and an offset. i.e. func<tile_size: Int> (offset: Int)
"""
alias Static1DTileUnitFunc = __mlir_type[
    `!kgen.signature<<`, Int, `>(`, Int, ` borrow) -> `, NoneType, `>`
]

"""
Signature of a tiled function that performs some work with a dynamic tile size
  and an offset. i.e. func(offset: Inttile_size: Int)
"""
alias Dynamic1DTileUnitFunc = __mlir_type[
    `!kgen.signature<(`, Int, ` borrow,`, Int, ` borrow) -> `, NoneType, `>`
]

"""
Signature of a tiled function that performs some work with a dynamic tile size
and a secondary static tile size.
"""
alias BinaryTile1DTileUnitFunc = __mlir_type[
    `!kgen.signature<<`,
    Int,
    `>(`,
    Int,
    ` borrow,`,
    Int,
    ` borrow) -> `,
    NoneType,
    `>`,
]


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
        workgroup_function: workgroup function that processes one tile of
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
        workgroup_function: workgroup function that processes one tile of
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
    """A generator that launches work groups in specified list of tile sizes until the
    sum of primary_tile_sizes has exceeded the upperbound.

    Parameters:
        secondary_tile_size_list: List of static tile sizes to launch work.
        secondary_cleanup_tile: Last static tile to use when primary tile sizes
          don't fit exactly within the upperbound.
        workgroup_function: workgroup function that processes one tile of
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
# NullaryClosure
# ===----------------------------------------------------------------------===#


struct NullaryClosure[result_type: AnyType]:
    """A struct representing a 0-arguments closure."""

    alias closure_type = __mlir_type[`!pop.closure<() -> `, result_type, `>`]
    var value: closure_type

    @always_inline("nodebug")
    fn __init__(self&, value: closure_type):
        """Create a nullary closure.

        Arguments:
          value: the closure value.

        Returns:
          The nullary closure.
        """
        self.value = value

    @always_inline("nodebug")
    fn __copyinit__(self&, existing: Self):
        """Clone a nullary closure.

        Arguments:
          self: the value to clone.

        Returns:
          A new nullary closure.
        """
        self.value = existing.value

    @always_inline("nodebug")
    fn __call__(self) -> result_type:
        """Call a nullary closure.

        Returns:
          The closure result.
        """
        return __mlir_op.`pop.call_indirect`[_type:result_type](self.value)


struct UnaryClosure[
    input_type: AnyType,
    result_type: AnyType,
]:
    """A struct representing a single argument closure."""

    alias closure_type = __mlir_type[
        `!pop.closure<(`, input_type, `) -> `, result_type, `>`
    ]
    var value: closure_type

    @always_inline("nodebug")
    fn __init__(self&, value: closure_type):
        """Create a unary closure.

        Arguments:
          value: the closure value.

        Returns:
          The unary closure.
        """
        self.value = value

    @always_inline("nodebug")
    fn __copyinit__(self&, existing: Self):
        """Clone a unary closure.

        Arguments:
          self: the value to clone.

        Returns:
          A new unary closure.
        """
        self.value = existing.value

    @always_inline("nodebug")
    fn __call__(self, input: input_type) -> result_type:
        """Call a unary closure.

        Arguments:
          input: the input to the unary closure

        Returns:
          The unary closure result.
        """
        return __mlir_op.`pop.call_indirect`[_type:result_type](
            self.value, input
        )


struct BinaryClosure[
    lhs_type: AnyType,
    rhs_type: AnyType,
    result_type: AnyType,
]:
    """A struct representing a two arguments closure."""

    alias closure_type = __mlir_type[
        `!pop.closure<(`, lhs_type, `, `, rhs_type, `) -> `, result_type, `>`
    ]
    var value: closure_type

    @always_inline("nodebug")
    fn __init__(self&, value: closure_type):
        """Create a binary closure.

        Arguments:
          value: the closure value.

        Returns:
          The binary closure.
        """
        self.value = value

    @always_inline("nodebug")
    fn __copyinit__(self&, existing: Self):
        """Clone a binary closure.

        Arguments:
          self: the value to clone.

        Returns:
          A new binary closure.
        """
        self.value = existing.value

    @always_inline("nodebug")
    fn __call__(self, lhs: lhs_type, rhs: rhs_type) -> result_type:
        """Call a binary closure.

        Arguments:
          lhs: the first input to the binary closure.
          rhs: the second input to the binary closure.

        Returns:
          The binary closure result.
        """
        return __mlir_op.`pop.call_indirect`[_type:result_type](
            self.value, lhs, rhs
        )


struct TernaryClosure[
    arg1_type: AnyType,
    arg2_type: AnyType,
    arg3_type: AnyType,
    result_type: AnyType,
]:
    """A struct representing a three arguments closure."""

    alias closure_type = __mlir_type[
        `!pop.closure<(`,
        arg1_type,
        `, `,
        arg2_type,
        `, `,
        arg3_type,
        `) -> `,
        result_type,
        `>`,
    ]
    var value: closure_type

    @always_inline("nodebug")
    fn __init__(self&, value: closure_type):
        """Create a Ternary closure.

        Arguments:
          value: the closure value.

        Returns:
          The Ternary closure.
        """
        self.value = value

    @always_inline("nodebug")
    fn __copyinit__(self&, existing: Self):
        """Clone a Ternary closure.

        Arguments:
          self: the value to clone.

        Returns:
          A new Ternary closure.
        """
        self.value = existing.value

    @always_inline("nodebug")
    fn __call__(
        self,
        arg1: arg1_type,
        arg2: arg2_type,
        arg3: arg3_type,
    ) -> result_type:
        """Call a Ternary closure.

        Arguments:
          arg1: the first input to the Ternary closure.
          arg2: the second input to the Ternary closure.
          arg3: the third input to the Ternary closure.

        Returns:
          The Ternary closure result.
        """
        return __mlir_op.`pop.call_indirect`[_type:result_type](
            self.value, arg1, arg2, arg3
        )


# ===----------------------------------------------------------------------===#
# Unswitch
# ===----------------------------------------------------------------------===#

# Signature of a function that unswitch can take.
alias SwitchedFunction = __mlir_type[
    `!kgen.signature<<`, Bool, `>() -> `, NoneType, `>`
]

# Version of unswitch supporting 2 predicates.
alias SwitchedFunction2 = __mlir_type[
    `!kgen.signature<<`,
    Bool,
    `,`,
    Bool,
    `>() -> `,
    NoneType,
    `>`,
]


@always_inline
fn unswitch[switched_func: SwitchedFunction](dynamic_switch: Bool):
    """Perform a functional unswitch transformation.

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
    This unswitch function genralizes that pattern with the help of meta parame-
    ters and can be used to perform both loop unswitching and other tile predic-
    ate lifting like in simd and amx.

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
    """Perform a functional 2-predicates unswitch transformation.

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
        fn switched_a_true[static_switch: Bool]():
            switched_func[True, static_switch]()

        unswitch[switched_a_true](dynamic_switch_b)
    else:

        @always_inline
        fn switched_a_false[static_switch: Bool]():
            switched_func[False, static_switch]()

        unswitch[switched_a_false](dynamic_switch_b)


# ===----------------------------------------------------------------------===#
# TileWithUnswitch
# ===----------------------------------------------------------------------===#

"""
Signature of a tiled function that performs some work with a static tile size
  and an offset. i.e. func<tile_size: Int> (offset: Int)
"""
alias Static1DTileUnswitchUnitFunc = __mlir_type[
    `!kgen.signature<<`,
    Int,
    `, `,
    Bool,
    `>(`,
    Int,
    ` borrow,`,
    Int,
    ` borrow) -> `,
    NoneType,
    `>`,
]


@always_inline
fn tile_and_unswitch[
    workgroup_function: Static1DTileUnswitchUnitFunc,
    tile_size_list: VariadicList[Int],
](offset: Int, upperbound: Int):
    """Perform time and unswitch functional transformation.

    A variant of static tile given a workgroup function that can be unswitched.
    This generator is a fused version of tile and unswitch, where the static
    unswitch is true throughout the "inner" portion of the workload and is
    false only on the residue tile.

    Parameters:
        workgroup_function: workgroup function that processes one tile of
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


alias Dynamic1DTileUnswitchUnitFunc = __mlir_type[
    `!kgen.signature<<`,
    Bool,
    `>(`,
    Int,
    ` borrow,`,
    Int,
    ` borrow,`,
    Int,
    ` borrow) -> `,
    NoneType,
    `>`,
]


@always_inline
fn tile_and_unswitch[
    workgroup_function: Dynamic1DTileUnswitchUnitFunc,
](offset: Int, upperbound: Int, tile_size_list: VariadicList[Int]):
    """Perform time and unswitch functional transformation.

    A variant of dynamic tile given a workgroup function that can be
    unswitched. This generator is a fused version of tile and unswitch, where
    the static unswitch is true throughout the "inner" portion of the workload
    and is false only on the residue tile.

    Parameters:
        workgroup_function: workgroup function that processes one tile of
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
fn get_num_workers(problem_size: Int, runtime: Runtime) -> Int:
    """Return a number of workers to run in parallel.

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
@adaptive
fn elementwise[
    rank: Int,
    simd_width: Int,
    unroll_factor: Int,
    func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        Int,
        `>(`,
        StaticIntTuple[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, Int]
        ],
        ` borrow) -> `,
        NoneType,
        `>`,
    ],
](shape: StaticIntTuple[rank], out_chain: OutputChainPtr):
    """Execute func[width, rank](indices) as sub-tasks for a suitable
    combination of width and indices so as to cover shape.

    Parameters:
        rank: The rank of the buffer.
        simd_width: The SIMD vector width to use.
        unroll_factor: The unroll factor to use.
        func: The body function.

    Args:
        shape: The shape of the buffer.
        out_chain: The our chain to attach results to.
    """
    assert_param_msg[rank == 1, "Specialization for 1D"]()

    let problem_size = shape.flattened_length()
    let num_workers = get_num_workers(problem_size, out_chain.get_runtime())
    let chunk_size = div_ceil(problem_size, num_workers)

    @always_inline
    fn task_func(i: Int):
        let start_offset = i * chunk_size
        let end_offset = min((i + 1) * chunk_size, problem_size)

        let len = end_offset - start_offset

        if len <= 0:
            return

        @always_inline
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
fn _get_nd_indices_from_flat_index[
    rank: Int
](flat_index: Int, shape: StaticIntTuple[rank]) -> StaticIntTuple[rank]:
    """Converts a flat index into ND indices.

    The ND indices will iterate from right to left. I.E

    shape = (20, 5, 2, N)
    _get_nd_indices_from_flat_index(1, shape) = (0, 0, 1, 0)
    _get_nd_indices_from_flat_index(5, shape) = (0, 2, 1, 0)
    _get_nd_indices_from_flat_index(50, shape) = (5, 0, 0, 0)
    _get_nd_indices_from_flat_index(56, shape) = (5, 1, 1, 0)

    We ignore the Nth dimension to allow that to be traversed in the elementwise
    function.

    Parameters:
        rank: The rank of the ND index.

    Args:
        flat_index: The flat index to convert.
        shape: The shape of the ND space we are converting into.

    Returns:
        Constructed ND-index.
    """

    # The inner dimensions ([outer, outer, inner]) are not traversed here.

    @parameter
    if rank == 2:
        return StaticIntTuple[rank](flat_index, 0)

    var out = StaticIntTuple[rank]()
    var curr_index = flat_index

    @always_inline
    fn compute_shape[idx: Int]():
        alias i = rank - idx - 2
        out[i] = curr_index % shape[i]
        curr_index //= shape[i]

    unroll[rank - 1, compute_shape]()
    out[rank - 1] = 0

    return out


@always_inline
@adaptive
fn elementwise[
    rank: Int,
    simd_width: Int,
    unroll_factor: Int,
    func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        Int,
        `>(`,
        StaticIntTuple[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, Int]
        ],
        ` borrow) -> `,
        NoneType,
        `>`,
    ],
](shape: StaticIntTuple[rank], out_chain: OutputChainPtr):
    """Execute func[width, rank](indices) as sub-tasks for a suitable
    combination of width and indices so as to cover shape.

    All free vars in func must be "async safe", see async_parallelize.

    Parameters:
        rank: The rank of the buffer.
        simd_width: The SIMD vector width to use.
        unroll_factor: The unroll factor to use.
        func: The body function.

    Args:
        shape: The shape of the buffer.
        out_chain: The our chain to attach results to.
    """

    assert_param_msg[rank > 1, "Specialization for ND where N > 1"]()

    # Stategy: we parallelize over all dimensions except the innermost and
    # vectorize over the innermost dimension. We unroll the innermost dimension
    # by a factor of unroll_factor.

    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    let total_size: Int = shape.flattened_length()
    let num_workers = get_num_workers(total_size, out_chain.get_runtime())

    let parallelism_size: Int = total_size // shape[rank - 1]
    let chunk_size = div_ceil(parallelism_size, num_workers)

    @always_inline
    fn task_func(i: Int):
        let start_parallel_offset = i * chunk_size
        let end_parallel_offset = min((i + 1) * chunk_size, parallelism_size)

        let len = end_parallel_offset - start_parallel_offset
        if len <= 0:
            return

        for parallel_offset in range(
            start_parallel_offset, end_parallel_offset
        ):
            var indices = _get_nd_indices_from_flat_index[rank](
                parallel_offset, shape
            )

            @always_inline
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

# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, assert_param_bool_msg
from Coroutine import Coroutine
from DType import DType
from Index import StaticIntTuple
from Int import Int
from List import VariadicList
from LLCL import Runtime, TaskGroup
from Math import div_ceil, min, max
from Range import range
from SIMD import SIMD
from Vector import InlinedFixedVector

# ===----------------------------------------------------------------------===#
# Map
# ===----------------------------------------------------------------------===#


@always_inline
fn map[
    func: __mlir_type[`!kgen.signature<(`, Int, ` borrow) -> !lit.none>`],
](size: Int):
    """
    Map a function over a range from 0 to size.
    """
    for i in range(size):
        func(i)


# ===----------------------------------------------------------------------===#
# unroll
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll[
    count: Int,
    func: __mlir_type[`!kgen.signature<<idx: `, Int, `>() -> !lit.none>`],
]():
    """
    Reateadly evaluate a function `count` times.
    """
    _unroll_impl[0, count, func]()


@always_inline
fn _unroll_impl[
    idx: Int,
    count: Int,
    func: __mlir_type[`!kgen.signature<<idx: `, Int, `>() -> !lit.none>`],
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
    rows: Int,
    cols: Int,
    func: __mlir_type[
        `!kgen.signature<<idx0: `, Int, `,idx1: `, Int, `>() -> !lit.none>`
    ],
]():
    """
    Reateadly evaluate a 2D nested loop where the outer iteration is rows and
    the inner iteration is `cols`.
    """

    @always_inline
    fn func_wrapper[idx: Int]():
        alias idx0 = idx // cols
        alias idx1 = idx % cols
        func[idx0, idx1]()

    unroll[rows * cols, func_wrapper]()


# ===----------------------------------------------------------------------===#
# Vectorize
# ===----------------------------------------------------------------------===#


alias fn_sig_type = __mlir_type[
    `!kgen.signature<(`,
    Int,
    `) -> !lit.none>`,
]

alias fn_simd_sig_type = __mlir_type[
    `!kgen.signature<<simd_width:`,
    Int,
    `>(`,
    Int,
    ` borrow) -> !lit.none>`,
]


@always_inline
fn vectorize[
    simd_width: Int,
    func: __mlir_type[
        `!kgen.signature<<simd_width:`,
        Int,
        `>(`,
        Int,
        ` borrow) -> !lit.none>`,
    ],
](size: Int):
    """Map a function which is parametrized over a simd_width over a range
    from 0 to size in simd fashion.
    """
    assert_param_bool_msg[simd_width > 0, "simd width must be > 0"]()
    vectorize_unroll[simd_width, 1, func](size)


fn _variadic_get(
    a: __mlir_type[`!kgen.variadic<`, fn_sig_type, `>`],
    idx: __mlir_type.`index`,
) -> fn_sig_type:
    return __mlir_op.`pop.variadic.get`(a, idx)


@always_inline
fn vectorize_unroll[
    simd_width: Int,
    unroll_factor: Int,
    func: fn_simd_sig_type,
](size: Int):
    """Map a function which is parametrized over a simd_width over a range
    from 0 to size in simd fashion and unroll the loop by unroll_factor.
    """
    assert_param_bool_msg[simd_width > 0, "simd width must be > 0"]()
    assert_param_bool_msg[unroll_factor > 0, "unroll factor must be > 0"]()

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

    for simd_idx in range(
        vector_end_unrolled_simd, vector_end_simd, simd_width
    ):
        vector_func_impl(simd_idx)

    for i in range(vector_end_simd, size):
        scalar_func_impl(i)


# ===----------------------------------------------------------------------===#
# parallelForEachN
# ===----------------------------------------------------------------------===#

alias none = __mlir_type.`!lit.none`
alias InlinedFixedVectorLength = 64


@always_inline
fn parallelForEachNChain[
    args_type: __mlir_type.`!kgen.mlirtype`,
    func: __mlir_type[
        `!kgen.signature<(`,
        Int,
        ` borrow,`,
        args_type,
        ` borrow) async -> !lit.none>`,
    ],
](
    total_count: Int,
    args: args_type,
    tasks&: InlinedFixedVector[InlinedFixedVectorLength, Coroutine[none]],
    tg&: TaskGroup,
):
    for i in range(total_count):
        let task: Coroutine[__mlir_type.`!lit.none`] = func(i, args)
        tg.create_task[none](task)
        tasks.append(task)


@always_inline
fn parallelForEachN[
    args_type: __mlir_type.`!kgen.mlirtype`,
    func: __mlir_type[
        `!kgen.signature<(`,
        Int,
        ` borrow,`,
        args_type,
        ` borrow) -> !lit.none>`,
    ],
](rt: Runtime, total_count: Int, args: args_type):
    # We have no tasks, so do nothing.
    if total_count == 0:
        return

    # Only have a single task, just run it on the main thread.
    if total_count == 1:
        func(0, args)
        return

    @always_inline
    async fn task_fn(i: Int, args: args_type):
        func(i, args)

    var tasks = InlinedFixedVector[InlinedFixedVectorLength, Coroutine[none]](
        total_count - 1
    )
    var tg = TaskGroup(rt)
    parallelForEachNChain[args_type, task_fn](total_count - 1, args, tasks, tg)

    func(total_count - 1, args)

    tg.wait()
    tg.__del__()
    for j in range(tasks.__len__()):
        tasks[j].__del__()
    tasks.__del__()


# ===----------------------------------------------------------------------===#
# Parallelize
# ===----------------------------------------------------------------------===#


@always_inline
fn parallelize[
    func: __mlir_type[`!kgen.signature<(`, Int, ` borrow) -> !lit.none>`],
](rt: Runtime, num_work_items: Int):
    # We have no tasks, so do nothing.
    if num_work_items == 0:
        return

    # Only have a single task, just run it on the main thread.
    if num_work_items == 1:
        func(0)
        return

    @always_inline
    async fn task_fn(i: Int):
        func(i)

    var tasks = InlinedFixedVector[InlinedFixedVectorLength, Coroutine[none]](
        num_work_items - 1
    )
    var tg = TaskGroup(rt)
    for i in range(num_work_items - 1):
        let task: Coroutine[__mlir_type.`!lit.none`] = task_fn(i)
        tg.create_task[none](task)
        tasks.append(task)

    func(num_work_items - 1)

    tg.wait()
    tg.__del__()
    for j in range(tasks.__len__()):
        tasks[j].__del__()
    tasks.__del__()


# ===----------------------------------------------------------------------===#
# invoke
# ===----------------------------------------------------------------------===#


fn invoke(func: __mlir_type.`() -> ()`):
    __mlir_op.`pop.call_indirect`[_type:[]](func)


fn invoke(
    func: __mlir_type.`() -> (!pop.pointer<scalar<si8>>)`,
) -> __mlir_type.`!pop.pointer<scalar<si8>>`:
    return __mlir_op.`pop.call_indirect`[
        _type : [__mlir_type.`!pop.pointer<scalar<si8>>`]
    ](func)


fn invoke[
    arg_type: __mlir_type.`!kgen.mlirtype`
](func: __mlir_type[`(`, arg_type, `) -> ()`], arg: arg_type):
    __mlir_op.`pop.call_indirect`[_type:[]](func, arg)


fn invoke[
    result_type: __mlir_type.`!kgen.mlirtype`,
    arg_type: __mlir_type.`!kgen.mlirtype`,
](
    func: __mlir_type[`(`, arg_type, `) -> (`, result_type, `)`], arg: arg_type
) -> result_type:
    return __mlir_op.`pop.call_indirect`[_type:[result_type]](func, arg)


fn invoke[
    result_type: __mlir_type.`!kgen.mlirtype`,
    arg1_type: __mlir_type.`!kgen.mlirtype`,
    arg2_type: __mlir_type.`!kgen.mlirtype`,
](
    func: __mlir_type[
        `(`, arg1_type, `,`, arg2_type, `) -> (`, result_type, `)`
    ],
    arg1: arg1_type,
    arg2: arg2_type,
) -> result_type:
    return __mlir_op.`pop.call_indirect`[_type:[result_type]](func, arg1, arg2)


fn invoke[
    result_type: __mlir_type.`!kgen.mlirtype`,
    arg1_type: __mlir_type.`!kgen.mlirtype`,
    arg2_type: __mlir_type.`!kgen.mlirtype`,
    arg3_type: __mlir_type.`!kgen.mlirtype`,
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
    `!kgen.signature<<tile_size:`, Int, `>(`, Int, ` borrow) -> !lit.none>`
]

"""
Signature of a tiled function that performs some work with a dynamic tile size
  and an offset. i.e. func(offset: Inttile_size: Int)
"""
alias Dynamic1DTileUnitFunc = __mlir_type[
    `!kgen.signature<(`, Int, ` borrow,`, Int, ` borrow) -> !lit.none>`
]


@always_inline
fn tile[
    workgroup_function: Static1DTileUnitFunc, tile_size_list: VariadicList[Int]
](offset: Int, upperbound: Int):
    """A generator that launches work groups in specified list of tile sizes.

    A workgroup function is a function that can process a configurable
      consecutive "tile" of workload.
      e.g. work_on[3](5) should launch computation on item 5,6,7, and should be
      semantically equivalent to work_on[1](5),work_on[1](6),work_on[1](7).

    This generator will try to proceed with the given list of tile sizes on the
     listed order. E.g.
        tile [func, (3,2,1)](offset, upperbound)
    will try to call func[3] starting from offset until remaining work is less
      than 3 from upperbound and then try func[2], and then func[1] etc.

    Args:
        workgroup_function(Static1DTileUnitFunc): workgroup function that processes one
            tile of workload.
        tile_size_list(VariadicList[Int]): List of tile sizes to launch work.
        offset(Int): The initial index to start the work from.
        upperbound(Int): The runtime upperbound that the work function should not exceed.
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

    Args:
        workgroup_function(Dynamic1DTileUnitFunc): workgroup function that processes one
            tile of workload.
        tile_size_list(VariadicList[Int]): List of tile sizes to launch work.
        offset(Int): The initial index to start the work from.
        upperbound(Int): The runtime upperbound that the work function should not exceed.
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


# ===----------------------------------------------------------------------===#
# NullaryClosure
# ===----------------------------------------------------------------------===#


struct NullaryClosure[result_type: __mlir_type.`!kgen.mlirtype`]:
    alias closure_type = __mlir_type[`!pop.closure<() -> `, result_type, `>`]
    var value: closure_type

    @always_inline("nodebug")
    fn __init__(value: closure_type) -> Self:
        """Create a nullary closure.

        Arguments:
          value: the closure value

        Returns:
          The nullary closure.
        """
        return Self {value: value}

    @always_inline("nodebug")
    fn __copy__(self) -> Self:
        """Clone a nullary closure.

        Arguments:
          self: the value to clone

        Returns:
          A new nullary closure.
        """
        return Self {value: self.value}

    @always_inline("nodebug")
    fn __call__(self) -> result_type:
        """Call a nullary closure.

        Returns:
          The closure result.
        """
        return __mlir_op.`pop.call_indirect`[_type:result_type](self.value)


struct UnaryClosure[
    input_type: __mlir_type.`!kgen.mlirtype`,
    result_type: __mlir_type.`!kgen.mlirtype`,
]:
    alias closure_type = __mlir_type[
        `!pop.closure<(`, input_type, `) -> `, result_type, `>`
    ]
    var value: closure_type

    @always_inline("nodebug")
    fn __init__(value: closure_type) -> Self:
        """Create a unary closure.

        Arguments:
          value: the closure value

        Returns:
          The unary closure.
        """
        return Self {value: value}

    @always_inline("nodebug")
    fn __copy__(self) -> Self:
        """Clone a unary closure.

        Arguments:
          self: the value to clone

        Returns:
          A new unary closure.
        """
        return Self {value: self.value}

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
    lhs_type: __mlir_type.`!kgen.mlirtype`,
    rhs_type: __mlir_type.`!kgen.mlirtype`,
    result_type: __mlir_type.`!kgen.mlirtype`,
]:
    alias closure_type = __mlir_type[
        `!pop.closure<(`, lhs_type, `, `, rhs_type, `) -> `, result_type, `>`
    ]
    var value: closure_type

    @always_inline("nodebug")
    fn __init__(value: closure_type) -> Self:
        """Create a binary closure.

        Arguments:
          value: the closure value

        Returns:
          The binary closure.
        """
        return Self {value: value}

    @always_inline("nodebug")
    fn __copy__(self) -> Self:
        """Clone a binary closure.

        Arguments:
          self: the value to clone

        Returns:
          A new binary closure.
        """
        return Self {value: self.value}

    @always_inline("nodebug")
    fn __call__(self, lhs: lhs_type, rhs: rhs_type) -> result_type:
        """Call a binary closure.

        Arguments:
          lhs: the first input to the binary closure
          rhs: the second input to the binary closure

        Returns:
          The binary closure result.
        """
        return __mlir_op.`pop.call_indirect`[_type:result_type](
            self.value, lhs, rhs
        )


# ===----------------------------------------------------------------------===#
# Unswitch
# ===----------------------------------------------------------------------===#

# Signature of a function that unswitch can take.
alias SwitchedFunction = __mlir_type[
    `!kgen.signature<<static_switch:`, Bool, `>() -> !lit.none>`
]

# Version of unswitch supporting 2 predicates.
alias SwitchedFunction2 = __mlir_type[
    `!kgen.signature<<static_switch0:`,
    Bool,
    `, static_switch1:`,
    Bool,
    `>() -> !lit.none>`,
]


@always_inline
fn unswitch[switched_func: SwitchedFunction](dynamic_switch: Bool):
    """Unswitch is a simple pattern that is similar idea to loop unswitching
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

        Args:
            switched_func (SwitchedFunction): The function containing the inner
                loop logic that can be unswitched.

            dynamic_switch (Bool): The dynamic condition that enables the unswi-
                tched code path.
    """
    if dynamic_switch:
        switched_func[True]()
    else:
        switched_func[False]()


@always_inline
fn unswitch[
    switched_func: SwitchedFunction2
](dynamic_switch_a: Bool, dynamic_switch_b: Bool):
    """This is a version of unswitch pattern that takes 2 predicates.

    Args:
        switched_func (SwitchedFunction2): The function containing the inner
            loop logic that has 2 predicates which can be unswitched.

        dynamic_switch_a (Bool): The first dynamic condition that enables the
            outer unswitched code path.

        dynamic_switch_b (Bool): The second dynamic condition that enables
            the inner unswitched code path.
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
    `!kgen.signature<<tile_size:`,
    Int,
    `, static_switch:`,
    Bool,
    `>(`,
    Int,
    ` borrow,`,
    Int,
    ` borrow) -> !lit.none>`,
]


@always_inline
fn tile_and_unswitch[
    workgroup_function: Static1DTileUnswitchUnitFunc,
    tile_size_list: VariadicList[Int],
](offset: Int, upperbound: Int):
    """A variant of static tile given a workgroup function that can be
    unswitched. This generator is a fused version of tile and unswitch, where
    the static unswitch is true throughout the "inner" portion of the workload
    and is false only on the residue tile.

    Args:
        workgroup_function(Static1DTileUnitFunc): workgroup function that processes one
            tile of workload.
        tile_size_list(VariadicList[Int]): List of tile sizes to launch work.
        offset(Int): The initial index to start the work from.
        upperbound(Int): The runtime upperbound that the work function should not exceed.
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


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


fn get_num_workers(problem_size: Int, runtime: Runtime) -> Int:
    # Minimum number of elements to warrant an additional thread.
    # copied from https://github.com/pytorch/pytorch/blob/20dfce591ce88bc957ffcd0c8dc7d5f7611a4a3b/aten/src/ATen/TensorIterator.h#L86
    # TODO: refine this heuristic. It may not be appropriate for more compute-heavy
    # ops like gelu.
    alias GRAIN_SIZE = 32768
    return min(runtime.parallelism_level(), div_ceil(problem_size, GRAIN_SIZE))


# ===----------------------------------------------------------------------===#
# Elementwise
# ===----------------------------------------------------------------------===#


@always_inline
fn elementwise[
    rank: __mlir_type.index,
    simd_width: Int,
    unroll_factor: Int,
    func: __mlir_type[
        `!kgen.signature<<`,
        `simd_width: `,
        Int,
        `,`,
        `rank: `,
        __mlir_type.index,
        `>(`,
        StaticIntTuple[__mlir_attr[`#kgen.param.decl.ref<"rank">: index`]],
        ` borrow) -> !lit.none>`,
    ],
](shape: StaticIntTuple[rank], runtime: Runtime,):
    let problem_size = shape.flattened_length()
    let num_workers = get_num_workers(problem_size, runtime)
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

    parallelize[task_func](runtime, num_workers)

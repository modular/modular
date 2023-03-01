# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, assert_param_bool_msg
from Coroutine import Coroutine

from Int import Int
from LLCL import Runtime, TaskGroup
from Range import range
from Vector import InlinedFixedVector
from List import VariadicList

# ===----------------------------------------------------------------------===#
# Map
# ===----------------------------------------------------------------------===#


@always_inline
fn map[
    func: __mlir_type[`!kgen.signature<(`, Int, `) -> !lit.none>`],
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
    count: __mlir_type.index,
    func: __mlir_type[`!kgen.signature<<idx>() -> !lit.none>`],
]():
    """
    Reateadly evaluate a function `count` times.
    """
    _unroll_impl[0, count, func]()


@always_inline
@adaptive
fn _unroll_impl[
    idx: __mlir_type.index,
    count: __mlir_type.index,
    func: __mlir_type[`!kgen.signature<<idx>() -> !lit.none>`],
]():
    assert_param[idx >= count]()


@always_inline
@adaptive
fn _unroll_impl[
    idx: __mlir_type.index,
    count: __mlir_type.index,
    func: __mlir_type[`!kgen.signature<<idx>() -> !lit.none>`],
]():
    assert_param[idx < count]()
    func[idx]()
    _unroll_impl[idx + 1, count, func]()


# ===----------------------------------------------------------------------===#
# Vectorize
# ===----------------------------------------------------------------------===#


@always_inline
fn vectorize[
    simd_width: __mlir_type.index,
    func: __mlir_type[
        `!kgen.signature<<simd_width>(`,
        Int,
        `) -> !lit.none>`,
    ],
](size: Int):
    """Map a function which is parametrized over a simd_width over a range
    from 0 to size in simd fashion.
    """
    let vector_end = (size // simd_width) * simd_width
    for simd_idx in range(0, vector_end, simd_width):
        func[simd_width](simd_idx)
    for i in range(vector_end, size):
        func[1](i)


@always_inline
fn vectorize_unroll[
    simd_width: __mlir_type.index,
    unroll_factor: __mlir_type.index,
    func: __mlir_type[
        `!kgen.signature<<simd_width>(`,
        Int,
        `) -> !lit.none>`,
    ],
](size: Int):
    """Map a function which is parametrized over a simd_width over a range
    from 0 to size in simd fashion and unroll the loop by unroll_factor.
    """
    alias unrolled_simd_width = simd_width * unroll_factor
    let vector_end_unrolled_simd = (
        size // unrolled_simd_width
    ) * unrolled_simd_width
    let vector_end_simd = (size // simd_width) * simd_width

    @always_inline
    fn unrolled_func(unrolled_simd_idx: Int):
        @always_inline
        fn unroll_iter[idx: __mlir_type.index]():
            func[simd_width](unrolled_simd_idx + idx * simd_width)

        unroll[unroll_factor, unroll_iter]()

    for unrolled_simd_idx in range(
        0, vector_end_unrolled_simd, unrolled_simd_width
    ):
        unrolled_func(unrolled_simd_idx)

    for simd_idx in range(
        vector_end_unrolled_simd, vector_end_simd, simd_width
    ):
        func[simd_width](simd_idx)

    for i in range(vector_end_simd, size):
        func[1](i)


# ===----------------------------------------------------------------------===#
# Parallelize
# ===----------------------------------------------------------------------===#

alias none = __mlir_type.`!lit.none`
alias InlinedFixedVectorLength = 64


fn parallelForEachNChain[
    args_type: __mlir_type.`!kgen.mlirtype`,
    func: __mlir_type[
        `!kgen.signature<(`, Int, `,`, args_type, `) async -> !lit.none>`
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


fn parallelForEachN[
    args_type: __mlir_type.`!kgen.mlirtype`,
    func: __mlir_type[
        `!kgen.signature<(`, Int, `,`, args_type, `) -> !lit.none>`
    ],
](rt: Runtime, total_count: Int, args: args_type):
    # We have no tasks, so do nothing.
    if total_count == 0:
        return

    # Only have a single task, just run it on the main thread.
    if total_count == 1:
        func(0, args)
        return

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
        tasks.__getitem__(j).__del__()
    tasks.__del__()


@always_inline
fn div_ceil(numerator: Int, denominator: Int) -> Int:
    return (numerator + denominator - 1) // denominator


@always_inline
fn parallelize[
    func: __mlir_type[`!kgen.signature<(`, Int, `,`, Int, `) -> !lit.none>`],
](num_work_items: Int, size: Int):
    let chunk_size = div_ceil(size, num_work_items)
    for start in range(0, size, chunk_size):
        let end = Int.min(start + chunk_size, size)
        func(start, end)


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
    `!kgen.signature<<tile_size:`, Int, `>(`, Int, `) -> !lit.none>`
]

"""
Signature of a tiled function that performs some work with a dynamic tile size
  and an offset. i.e. func(offset: Inttile_size: Int)
"""
alias Dynamic1DTileUnitFunc = __mlir_type[
    `!kgen.signature<(`, Int, `,`, Int, `) -> !lit.none>`
]


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
    fn static_tile_impl[idx: __mlir_type.index]():
        # Get the tile size to proceed with.
        let tile_size = tile_size_list[idx]

        # Process work with the tile size until there's not enough remaining work
        #  to fit in a tile.
        while current_offset <= upperbound - tile_size:
            workgroup_function[tile_size_list[idx]](current_offset)
            current_offset += tile_size

    unroll[tile_size_list.size().__as_mlir_index(), static_tile_impl]()


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
    for tile_idx in range(tile_size_list.size()):
        let tile_size = tile_size_list[tile_idx]
        # Launch workloads on the current tile sizes until cannot proceed.
        while work_idx <= upperbound - tile_size:
            workgroup_function(work_idx, tile_size)
            work_idx += tile_size
    # Clean up the remaining workload with a residue tile that exactly equals to
    #  the remaining workload size.
    # Note: This is the key difference from the static version of tile
    #  generator.
    workgroup_function(work_idx, upperbound - work_idx)


# ===----------------------------------------------------------------------===#
# NullaryClosure
# ===----------------------------------------------------------------------===#


struct NullaryClosure[result_type: __mlir_type.`!kgen.mlirtype`]:
    alias closure_type = __mlir_type[`!pop.closure<() -> `, result_type, `>`]
    var value: closure_type

    fn __new__(value: closure_type) -> Self:
        """Create a nullary closure.

        Arguments:
          value: the closure value

        Returns:
          The nullary closure.
        """
        return Self {value: value}

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

    fn __new__(value: closure_type) -> Self:
        """Create a unary closure.

        Arguments:
          value: the closure value

        Returns:
          The unary closure.
        """
        return Self {value: value}

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

    fn __new__(value: closure_type) -> Self:
        """Create a binary closure.

        Arguments:
          value: the closure value

        Returns:
          The binary closure.
        """
        return Self {value: value}

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

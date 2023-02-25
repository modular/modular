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

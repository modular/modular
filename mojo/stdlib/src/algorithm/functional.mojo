# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, assert_param_bool_msg
from Coroutine import Coroutine
from Bool import Bool
from Int import Int
from LLCL import Runtime, TaskGroup
from Range import range
from Vector import UnsafeFixedVector

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
    var i: Int = 0
    while i < size:
        func(i)
        i += 1


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
@interface
fn _unroll_impl[
    idx: __mlir_type.index,
    count: __mlir_type.index,
    func: __mlir_type[`!kgen.signature<<idx>() -> !lit.none>`],
]():
    ...


@always_inline
@implements(_unroll_impl)
fn _unroll_impl_base[
    idx: __mlir_type.index,
    count: __mlir_type.index,
    func: __mlir_type[`!kgen.signature<<idx>() -> !lit.none>`],
]():
    assert_param[idx >= count]()


@always_inline
@implements(_unroll_impl)
fn _unroll_impl_iter[
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
    """Map a function which is parametrized over a simd_Width over a range
    from 0 to size in simd fashion.
    """
    var i: Int = 0
    let vector_end = (size // simd_width) * simd_width
    while i < vector_end:
        func[simd_width](i)
        i += simd_width
    i = vector_end
    while i < size:
        func[1](i)
        i += 1


# ===----------------------------------------------------------------------===#
# Parallelize
# ===----------------------------------------------------------------------===#

alias none = __mlir_type.`!lit.none`

# TODO: Because of the inability to share Runtimes across libraries we cannot
# call run a true parallelForEachN here. Instead of crashing, we run sequential,
# will revert this once the Runtime ids issue is fixed.
alias DISABLE_MULTI_THREADING: Bool = True


fn parallelForEachNChain[
    args_type: __mlir_type.`!kgen.mlirtype`,
    func: __mlir_type[
        `!kgen.signature<(`, Int, `,`, args_type, `) async -> !lit.none>`
    ],
](
    total_count: Int,
    args: args_type,
    tasks&: UnsafeFixedVector[Coroutine[none]],
    tg&: TaskGroup,
):
    for i in range(total_count):
        let task: Coroutine[__mlir_type.`!lit.none`] = func(i, args)
        tg.create_task[none](task)
        tasks.append(task)


@interface
fn parallelForEachN[
    args_type: __mlir_type.`!kgen.mlirtype`,
    func: __mlir_type[
        `!kgen.signature<(`, Int, `,`, args_type, `) -> !lit.none>`
    ],
](rt: Runtime, total_count: Int, args: args_type):
    ...


@implements(parallelForEachN)
fn parallelForEachN_disabled[
    args_type: __mlir_type.`!kgen.mlirtype`,
    func: __mlir_type[
        `!kgen.signature<(`, Int, `,`, args_type, `) -> !lit.none>`
    ],
](rt: Runtime, total_count: Int, args: args_type):
    assert_param_bool_msg[
        DISABLE_MULTI_THREADING == True,
        "multi-threading is disabled. This is temporary solution.",
    ]()

    if total_count == 0:
        return

    for idx in range(total_count):
        func(idx, args)


@implements(parallelForEachN)
fn parallelForEachN_enabled[
    args_type: __mlir_type.`!kgen.mlirtype`,
    func: __mlir_type[
        `!kgen.signature<(`, Int, `,`, args_type, `) -> !lit.none>`
    ],
](rt: Runtime, total_count: Int, args: args_type):
    assert_param_bool_msg[
        DISABLE_MULTI_THREADING == False, "running in multiple threads!!!"
    ]()

    if total_count == 0:
        return

    var tg: TaskGroup
    var tasks: UnsafeFixedVector[Coroutine[none]]
    var b = Bool(False)
    if total_count > 1:

        async fn task_fn(i: Int, args: args_type):
            func(i, args)

        tasks = UnsafeFixedVector[Coroutine[none]](total_count - 1)
        tg = TaskGroup(rt)
        parallelForEachNChain[args_type, task_fn](
            total_count - 1, args, tasks, tg
        )
        b = True

    func(total_count - 1, args)

    if b:
        tg.wait()
        tg.__del__()
        for j in range(tasks.size):
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
    var start: Int = 0
    while start < size:
        let end = Int.min(start + chunk_size, size)
        func(start, end)
        start += chunk_size

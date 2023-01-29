# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from Bool import Bool
from Int import Int
from LLCL import Future, Runtime, TaskGroup
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


fn parallelForEachNChain[
    argsType: __mlir_type.`!kgen.mlirtype`,
    func: __mlir_type[
        `!kgen.signature<(`, Int, `,`, argsType, `) async -> !lit.none>`
    ],
](
    rt: Runtime,
    totalCount: Int,
    args: argsType,
    tasks&: UnsafeFixedVector[Future[none]],
    tg&: TaskGroup,
):
    for i in range(totalCount):
        let task = rt.init_and_run[none](func(i, args))
        tasks.append(task)
        tg.add_task[none](task)


fn parallelForEachN[
    argsType: __mlir_type.`!kgen.mlirtype`,
    func: __mlir_type[
        `!kgen.signature<(`, Int, `,`, argsType, `) -> !lit.none>`
    ],
](rt: Runtime, totalCount: Int, args: argsType):
    if totalCount == 0:
        return

    var tg: TaskGroup
    var tasks: UnsafeFixedVector[Future[none]]
    var b = Bool(False)
    if totalCount > 1:

        async fn task_fn(i: Int, args: argsType):
            func(i, args)

        tasks = UnsafeFixedVector[Future[none]](totalCount - 1)
        tg = TaskGroup(rt)
        parallelForEachNChain[argsType, task_fn](
            rt, totalCount - 1, args, tasks, tg
        )
        b = True

    func(totalCount - 1, args)

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

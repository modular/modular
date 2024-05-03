# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the low level concurrency library."""

from os import PathLike
from os.atomic import Atomic
from sys import external_call
from sys.ffi import _get_global, _get_global_or_null
from sys.info import num_physical_cores
from sys.param_env import is_defined

from builtin.coroutine import _coro_resume_fn
from memory.unsafe import DTypePointer, Pointer

from utils import StringRef

from .tracing import TraceLevel, is_mojo_profiling_disabled

# ===----------------------------------------------------------------------===#
# AsyncContext
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct Chain(Boolable):
    """A proxy for the C++ runtime's AsyncValueRef<Chain> type."""

    # Actually an AsyncValueRef<Chain>, which is just an AsyncValue*
    var storage: Pointer[Int]

    fn __init__(inout self):
        self.storage = Pointer[Int].get_null()

    fn __bool__(self) -> Bool:
        return self.storage != Pointer[Int].get_null()


# ===----------------------------------------------------------------------===#
# SpinWaiter
# ===----------------------------------------------------------------------===#


struct SpinWaiter:
    """A proxy for the C++ runtime's SpinWaiter type."""

    # Pointer to the underlying SpinWaiter instance
    var storage: Pointer[NoneType]

    fn __init__(inout self: Self):
        """Initializes a SpinWaiter instance."""
        self.storage = external_call[
            "KGEN_CompilerRT_LLCL_InitializeSpinWaiter",
            Pointer[NoneType],
        ]()

    fn __del__(owned self: Self):
        """Destroys the SpinWaiter instance."""
        external_call["KGEN_CompilerRT_LLCL_DestroySpinWaiter", NoneType](
            self.storage
        )

    fn wait(self: Self):
        """Blocks the current task for a duration determined by the underlying
        policy."""
        external_call["KGEN_CompilerRT_LLCL_SpinWaiter_Wait", NoneType](
            self.storage
        )


@register_passable("trivial")
struct AsyncContext:
    """This struct models the coroutine context contained in every coroutine
    instance. The struct consists of a unary callback function that accepts a
    pointer argument. It is invoked with the second struct field, which is an
    opaque pointer. This struct is essentially a completion callback closure
    that is invoked by a coroutine when it completes and its results are made
    available.

    In async execution, a task's completion callback is to set its async token
    to available.
    """

    alias callback_fn_type = fn (Pointer[NoneType], Chain) -> None

    var callback: Self.callback_fn_type
    var chain: Chain

    @staticmethod
    fn get_chain(ctx: Pointer[AsyncContext]) -> Pointer[Chain]:
        return Reference(ctx[].chain).get_legacy_pointer()

    @staticmethod
    fn complete(hdl: Pointer[NoneType], ch: Chain):
        var tmp = ch
        _async_complete(Pointer[Chain].address_of(tmp))


# ===----------------------------------------------------------------------===#
# LLCL C Shims
# ===----------------------------------------------------------------------===#


fn _init_llcl_chain(rt: Runtime, chain: Pointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_InitializeChain", NoneType](
        rt.ptr, chain.address
    )


fn _del_llcl_chain(chain: Pointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_DestroyChain", NoneType](chain.address)


fn _async_and_then(hdl: __mlir_type.`!kgen.pointer<i8>`, chain: Pointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_AndThen", NoneType](
        _coro_resume_fn, chain.address, hdl
    )


fn _async_execute[
    type: AnyRegType
](handle: Coroutine[type]._handle_type, rt: Runtime, desired_worker_id: Int,):
    external_call["KGEN_CompilerRT_LLCL_Execute", NoneType](
        _coro_resume_fn, handle, rt.ptr, desired_worker_id
    )


fn _async_wait(chain: Pointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_Wait", NoneType](chain.address)


fn _async_complete(chain: Pointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_Complete", NoneType](chain.address)


# ===----------------------------------------------------------------------===#
# Global Runtime
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_current_runtime() -> Pointer[NoneType]:
    """Returns the current runtime. The runtime is either created by the
    surrounding mojo tool (mojo-repl, mojo-jit, ...) or by the entry main
    function.
    """
    return external_call[
        "KGEN_CompilerRT_LLCL_GetCurrentRuntime", Pointer[NoneType]
    ]()


# ===----------------------------------------------------------------------===#
# Runtime
# ===----------------------------------------------------------------------===#


@register_passable
struct Runtime:
    alias ptr_type = Pointer[NoneType]
    var ptr: Self.ptr_type
    var owning: Bool

    # TODO: Probably don't want the runtime to be implicitly copyable.
    @always_inline("nodebug")
    fn __copyinit__(self) -> Self:
        return Self {ptr: self.ptr, owning: False}

    fn __init__() -> Runtime:
        """Construct an LLCL Runtime with the same number of threads as
        processor cores.
        """
        var ptr = _get_current_runtime()
        return Runtime(ptr, owning=False)

    fn __init__(threads: Int) -> Runtime:
        """Construct an LLCL Runtime with the specified number of threads."""

        # If the request was for a runtime with the default number of threads,
        # then we can just return the global runtime. Should this be
        # parallelism_level() instead of num_physical_cores()?
        if threads == num_physical_cores():
            return Runtime()

        var ptr = external_call[
            "KGEN_CompilerRT_LLCL_CreateRuntime", Self.ptr_type
        ](threads)
        return Runtime(ptr, owning=True)

    fn __init__[T: PathLike](threads: Int, profile_filename: T) -> Runtime:
        """Construct an LLCL Runtime with the specified number of threads
        that writes tracing events to profile_filename.
        """
        var path = profile_filename.__fspath__()
        var ptr = external_call[
            "KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile", Self.ptr_type
        ](threads, path._strref_dangerous())
        path._strref_keepalive()
        return Runtime(ptr, owning=True)

    fn __init__(ptr: Self.ptr_type, owning: Bool) -> Runtime:
        return Runtime {ptr: ptr, owning: owning}

    fn __enter__(self) -> Self:
        return self

    fn __exit__(self):
        """Destroys the LLCL Runtime. Note that this must be explicitly called
        when the Runtime goes out of the context.
        """
        self._destroy()

    fn __exit__(self, err: Error) -> Bool:
        """Destroys the LLCL Runtime within a raise-able env. Return True if
        the error is empty."""
        self._destroy()
        return not err

    fn _destroy(self):
        """Destroys the LLCL Runtime. Note that this must be explicitly called
        when the Runtime goes out of the context.
        """
        if self.owning:
            external_call["KGEN_CompilerRT_LLCL_DestroyRuntime", NoneType](
                self.ptr
            )

    fn parallelism_level(self) -> Int:
        """Gets the parallelism level of the Runtime."""
        return int(
            external_call[
                "KGEN_CompilerRT_LLCL_ParallelismLevel",
                Int32,
            ](self.ptr)
        )

    fn create_task[
        type: AnyRegType
    ](
        self,
        owned handle: Coroutine[type],
        desired_worker_id: Int = -1,
    ) -> Task[type]:
        """Run the coroutine as a task on the LLCL Runtime."""
        var ctx = handle._get_ctx[AsyncContext]()
        _init_llcl_chain(self, AsyncContext.get_chain(ctx))
        ctx[].callback = AsyncContext.complete
        _async_execute[type](handle._handle, self, desired_worker_id)
        return Task[type](handle^)

    fn run[type: AnyRegType](self, owned handle: Coroutine[type]) -> type:
        var t = self.create_task(handle^)
        var result = t.wait()
        return result


# ===----------------------------------------------------------------------===#
# Task
# ===----------------------------------------------------------------------===#


struct Task[type: AnyRegType]:
    var handle: Coroutine[type]

    fn __init__(inout self, owned handle: Coroutine[type]):
        self.handle = handle^

    fn get(self) -> type:
        """Get the task's result value."""
        return self.handle.get()

    fn __del__(owned self):
        """Destroy the memory associated with a task. This must be manually
        called when a task goes out of scope.
        """
        var ctx: Pointer[AsyncContext] = self.handle._get_ctx[AsyncContext]()
        var chainPtr: Pointer[Chain] = AsyncContext.get_chain(ctx)
        _del_llcl_chain(chainPtr)
        _ = self.handle^

    @always_inline
    fn __await__(self) -> type:
        """Suspend the current async function until the task completes and its
        result becomes available. This function must be force inlined into the
        calling async function.
        """
        var cur_hdl = __mlir_op.`co.opaque_handle`()

        __mlir_region await_body():
            _async_and_then(
                cur_hdl,
                AsyncContext.get_chain(self.handle._get_ctx[AsyncContext]()),
            )
            __mlir_op.`co.await.end`()

        __mlir_op.`co.await`[_region = "await_body".value]()
        return self.get()

    fn wait(self) -> type:
        """Block the current thread until the future value becomes available."""
        _async_wait(
            AsyncContext.get_chain(self.handle._get_ctx[AsyncContext]())
        )
        return self.get()


# ===----------------------------------------------------------------------===#
# TaskGroup
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct TaskGroupContext:
    alias tg_callback_fn_type = fn (Pointer[NoneType], inout TaskGroup) -> None

    var callback: Self.tg_callback_fn_type
    var task_group: Pointer[TaskGroup]


@register_passable
struct TaskGroupTask[type: AnyRegType]:
    """A task that belongs to a TaskGroup. This object retains ownership of the
    underlying coroutine handle, which can be used to query the results of the
    task once the taskgroup completes.
    """

    var handle: Coroutine[type]

    fn __init__(inout self, owned handle: Coroutine[type]):
        self.handle = handle^

    fn get(self) -> type:
        """Get the task's result value. This should only be called once the
        task result is available. I.e., when the taskgroup completes.

        Returns:
            The task's result value.
        """
        return self.handle.get()


struct TaskGroup:
    var counter: Atomic[DType.index]
    var chain: Chain
    var rt: Runtime

    fn __init__(inout self, rt: Runtime):
        var chain = Chain()
        _init_llcl_chain(rt, Pointer[Chain].address_of(chain))
        self.counter = 1
        self.chain = chain
        self.rt = rt

    fn __del__(owned self):
        _del_llcl_chain(Pointer[Chain].address_of(self.chain))

    @always_inline
    fn _counter_decr(inout self) -> Int:
        var prev: Int = self.counter.fetch_sub(1).value
        return prev - 1

    @staticmethod
    fn _task_complete_callback(hdl: Pointer[NoneType], inout tg: TaskGroup):
        tg._task_complete()

    fn _task_complete(inout self: TaskGroup):
        if self._counter_decr() == 0:
            _async_complete(Pointer[Chain].address_of(self.chain))

    fn create_task[
        type: AnyRegType
    ](
        inout self,
        owned task: Coroutine[type],
        desired_worker_id: Int = -1,
    ) -> TaskGroupTask[type]:
        self.counter += 1
        task._get_ctx[TaskGroupContext]().store(
            TaskGroupContext {
                callback: Self._task_complete_callback,
                task_group: Pointer[TaskGroup].address_of(self),
            }
        )
        _async_execute[type](task._handle, self.rt, desired_worker_id)
        return task^

    @staticmethod
    fn await_body_impl(
        hdl: __mlir_type.`!kgen.pointer<i8>`, inout task_group: TaskGroup
    ):
        _async_and_then(hdl, Pointer[Chain].address_of(task_group.chain))
        task_group._task_complete()

    @always_inline
    fn __await__(inout self):
        var cur_hdl = __mlir_op.`co.opaque_handle`()

        __mlir_region await_body():
            Self.await_body_impl(cur_hdl, self)
            __mlir_op.`co.await.end`()

        __mlir_op.`co.await`[_region = "await_body".value]()

    fn wait(inout self):
        self._task_complete()
        _async_wait(Pointer[Chain].address_of(self.chain))


# ===----------------------------------------------------------------------===#
# TaskGroupTaskList
# ===----------------------------------------------------------------------===#


struct TaskGroupTaskList[type: AnyRegType](Sized):
    """Container to hold a set of TaskGroupTasks alive until they all complete.
    """

    var data: Pointer[TaskGroupTask[type]]
    var size: Int

    fn __init__(inout self, num_work_items: Int):
        self.data = Pointer[TaskGroupTask[type]].alloc(num_work_items)
        self.size = 0

    fn add(inout self, owned hdl: TaskGroupTask[type]):
        __get_address_as_uninit_lvalue(self.data.offset(self.size).address) = (
            hdl^
        )
        self.size += 1

    fn __getitem__(self, i: Int) -> Pointer[TaskGroupTask[type]]:
        """Returns a pointer to the TaskGroupTask at the specified index.

        Note that the returned pointer can only be dereferenced while the
        TaskGroupTaskList is in scope as the underlying tasks get destroyed
        in the destructor of the TaskGroupTaskList.
        Remove this implementation once #35168 lands since Lists should suffice
        """
        debug_assert(i < self.size, "index must be within bounds")
        var hdl = (__get_address_as_owned_value(self.data.offset(i).address))
        var ret_val = Pointer[TaskGroupTask[type]].address_of(hdl)
        return ret_val

    fn __len__(self) -> Int:
        return self.size

    fn __del__(owned self):
        for i in range(self.size):
            _ = __get_address_as_owned_value(self.data.offset(i).address)
        self.data.free()


# ===----------------------------------------------------------------------===#
# MojoCallContext
# ===----------------------------------------------------------------------===#


@register_passable
struct MojoCallContextPtr:
    """A pointer to a C++ MojoCallContext struct, which is used by the Modular
    C++ runtime to coordinate execution with Mojo kernels.
    """

    # Actually a KGEN::MojoCallContext*
    alias ptr_type = DTypePointer[DType.invalid]
    var ptr: Self.ptr_type

    @always_inline
    fn __init__() -> MojoCallContextPtr:
        return MojoCallContextPtr {ptr: DTypePointer[DType.invalid].get_null()}

    @always_inline
    fn __init__(ptr: Self.ptr_type) -> MojoCallContextPtr:
        """Casts a raw pointer to our MojoCallContextPtr."""
        return MojoCallContextPtr {ptr: ptr}

    @always_inline
    fn complete(self):
        """Indicates to the C++ runtime that the async kernel has finished."""
        external_call[
            "KGEN_CompilerRT_LLCL_MojoCallContext_Complete", NoneType
        ](
            self.ptr,
        )

    @always_inline
    fn set_to_error(self, message: StringLiteral):
        """Indicates to the C++ runtime that the kernel has failed."""
        var strref = StringRef(message)
        external_call[
            "KGEN_CompilerRT_LLCL_MojoCallContext_SetToError", NoneType
        ](
            self.ptr,
            strref.data,
            strref.length,
        )

    @always_inline
    fn set_to_error(self, err: Error):
        """Indicates to the C++ runtime that the kernel has failed."""
        var str = err.__str__()
        var strref = str._strref_dangerous()
        external_call[
            "KGEN_CompilerRT_LLCL_MojoCallContext_SetToError", NoneType
        ](
            self.ptr,
            strref.data,
            strref.length,
        )
        str._strref_keepalive()


# ===----------------------------------------------------------------------===#
# MojoCallTask
# ===----------------------------------------------------------------------===#


@register_passable
struct _MojoCallTaskContext:
    """Coroutine context for calls to async Mojo kernels from the C++ runtime.
    """

    var callback: fn (owned Coroutine[NoneType], MojoCallContextPtr) -> None
    var payload: MojoCallContextPtr

    @staticmethod
    fn callback_impl(owned coro: Coroutine[NoneType], ctx: MojoCallContextPtr):
        # Just emplace the chain when the async function completes.
        ctx.complete()


@register_passable
struct MojoCallTask:
    """This struct adapts a Mojo async function call to tie back to the C++
    runtime's LLCL Chain AsyncValue.
    """

    var coro: Coroutine[NoneType]

    @always_inline
    fn __init__(
        owned coro: Coroutine[NoneType], ctx: MojoCallContextPtr
    ) -> Self:
        var result = Self {coro: coro^}
        # Set the callback and payload.
        result.coro._get_ctx[_MojoCallTaskContext]().store(
            _MojoCallTaskContext {
                callback: _MojoCallTaskContext.callback_impl, payload: ctx.ptr
            }
        )
        return result^

    @always_inline
    fn __call__(owned self):
        """Hot-start the task by resuming the coroutine. It will execute up
        to its first await point. In this way kernels can optionally setup
        sub-tasks on the calling C++ runtime thread and return.
        """
        # Because the caller may return while the coroutine executes on another
        # thread, forget the lifetime of `self`. It will be deleted in its
        # callback, which receives `Task` as owned. Both of these must be 'var'
        # definitions, despite the resulting warning.
        var self_lit_ref = Reference(self).value
        var coro = self.coro^
        var hdl = coro._handle
        __mlir_op.`lit.ownership.mark_destroyed`(Reference(coro).value)
        __mlir_op.`lit.ownership.mark_destroyed`(self_lit_ref)
        __mlir_op.`co.resume`(hdl)


# ===----------------------------------------------------------------------===#
# MojoCallRaisingTask
# ===----------------------------------------------------------------------===#


# FIXME(#26008): async raising functions are temporarily disabled. Work around
# this by returning an explicit result type.
@register_passable
struct _OptionalError:
    alias type = __mlir_type[`!kgen.variant<`, Error, `, none>`]
    var value: Self.type

    fn __init__(inout self):
        self.value = __mlir_op.`kgen.variant.create`[
            _type = Self.type, index = Int(1).value
        ](__mlir_attr.`#kgen.none`)

    fn __init__(inout self, owned existing: Error):
        self.value = __mlir_op.`kgen.variant.create`[
            _type = Self.type, index = Int(0).value
        ](existing^)

    fn is_error(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Int(0).value](self.value)

    fn consume_as_error(owned self) -> Error:
        return __mlir_op.`kgen.variant.take`[index = Int(0).value](self.value)


@register_passable
struct _MojoCallRaisingTaskContext:
    """Coroutine context for calls to raising async Mojo kernels from the
    C++ runtime.
    """

    var callback: fn (
        owned Coroutine[_OptionalError], MojoCallContextPtr
    ) -> None
    var payload: MojoCallContextPtr

    @staticmethod
    fn callback_impl(
        owned coro: Coroutine[_OptionalError], ctx: MojoCallContextPtr
    ):
        # Handle failure of the coroutine, which must be propagated back
        # by setting the chain to the error state.
        var err = coro.get()
        if err.is_error():
            ctx.set_to_error((err^).consume_as_error())
        else:
            ctx.complete()


@register_passable
struct MojoCallRaisingTask:
    """This struct adapts a Mojo raising async function call to tie back to the
    C++ runtime's LLCL Chain AsyncValue.
    """

    var coro: Coroutine[_OptionalError]

    @always_inline
    fn __init__(
        owned coro: Coroutine[_OptionalError], ctx: MojoCallContextPtr
    ) -> Self:
        var result = Self {coro: coro^}
        result.coro._get_ctx[_MojoCallRaisingTaskContext]().store(
            _MojoCallRaisingTaskContext {
                callback: _MojoCallRaisingTaskContext.callback_impl,
                payload: ctx.ptr,
            }
        )
        return result^

    @always_inline
    fn __call__(owned self):
        """Hot-start the task by resuming the coroutine. It will execute up
        to its first await point. In this way kernels can optionally setup
        sub-tasks on the calling C++ runtime thread and return.
        """
        # Because the caller may return while the coroutine executes on another
        # thread, forget the lifetime of `self`. It will be deleted in its
        # callback, which receives `Task` as owned. Both of these must be 'var'
        # definitions, despite the resulting warning.
        var self_lit_ref = Reference(self).value
        var coro = self.coro^
        var hdl = coro._handle
        __mlir_op.`lit.ownership.mark_destroyed`(Reference(coro).value)
        __mlir_op.`lit.ownership.mark_destroyed`(self_lit_ref)
        __mlir_op.`co.resume`(hdl)

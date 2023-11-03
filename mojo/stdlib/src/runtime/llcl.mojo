# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the low level concurrency library."""

from os.atomic import Atomic
from sys import external_call
from sys.param_env import is_defined

from builtin.coroutine import _coro_resume_fn
from gpu.host import Stream
from gpu.host.stream import _StreamImpl
from memory.unsafe import DTypePointer, Pointer
from runtime.tracing import TraceLevel, is_mojo_profiling_disabled

from sys.ffi import _get_global

# ===----------------------------------------------------------------------===#
# num_cores
# ===----------------------------------------------------------------------===#


@always_inline
fn num_cores() -> Int:
    """Returns the number of cores on the system.

    Returns:
        Int: The number of cores on the system.
    """
    return __mlir_op.`pop.external_call`[
        func = "KGEN_CompilerRT_CoreCount".value,
        resAttrs = __mlir_attr.`[{llvm.noundef}]`,
        funcAttrs = __mlir_attr.`["willreturn"]`,
        memory = __mlir_attr[
            `#llvm.memory_effects<other = read, `,
            `argMem = read, `,
            `inaccessibleMem = read>`,
        ],
        _type = __mlir_type.index,
    ]()


# ===----------------------------------------------------------------------===#
# AsyncContext
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct Chain:
    """This is an opaque instance of the C++ class `AsyncValueRef<Chain>`, whose
    size is the same as the pointer width.
    """

    var storage: Pointer[Int]

    fn __init__() -> Self:
        return Self {storage: Pointer[Int].get_null()}

    fn __bool__(self) -> Bool:
        return self.storage != Pointer[Int].get_null()


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

    alias callback_fn_type = fn (Chain) -> None

    var callback: fn (Chain) -> None
    var chain: Chain

    @staticmethod
    fn get_chain(ctx: Pointer[AsyncContext]) -> Pointer[Chain]:
        return __get_lvalue_as_address(
            __get_address_as_lvalue(ctx.address).chain
        )

    @staticmethod
    fn complete(ch: Chain):
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


fn _async_execute[type: AnyType](handle: Coroutine[type], rt: Runtime):
    external_call["KGEN_CompilerRT_LLCL_Execute", NoneType](
        _coro_resume_fn, handle._handle, rt.ptr
    )


fn _async_wait(chain: Pointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_Wait", NoneType](chain.address)


fn _async_complete(chain: Pointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_Complete", NoneType](chain.address)


# ===----------------------------------------------------------------------===#
# Global Runtime
# ===----------------------------------------------------------------------===#


fn _init_global_runtime(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    """Intialize the global runtime. This is a singleton that handle the common
    case where the runtime has the same number of threads as the number of cores.
    """
    return external_call[
        "KGEN_CompilerRT_LLCL_CreateRuntime", Pointer[NoneType]
    ](num_cores())


fn _destroy_global_runtime(ptr: Pointer[NoneType]):
    """Destroy the global runtime if ever used."""
    external_call["KGEN_CompilerRT_LLCL_DestroyRuntime", NoneType](ptr)


@always_inline
fn _get_global_runtime() -> Runtime:
    """Gets or creats the global runtime."""
    return _get_global[
        "Runtime", _init_global_runtime, _destroy_global_runtime
    ]()


# ===----------------------------------------------------------------------===#
# Runtime
# ===----------------------------------------------------------------------===#


# FIXME(traits): This shouldn't be a register_passable type but we need this
# until we have traits for proper parametric types.
@register_passable
struct Runtime:
    alias ptr_type = Pointer[NoneType]
    var ptr: Self.ptr_type

    # TODO: Probably don't want the runtime to be implicitly copyable.
    @always_inline("nodebug")
    fn __copyinit__(self) -> Self:
        return Self {ptr: self.ptr}

    fn __init__() -> Runtime:
        """Construct an LLCL Runtime with the same number of threads as
        processor cores.
        """
        return _get_global_runtime()

    fn __init__(num_threads: Int) -> Runtime:
        """Construct an LLCL Runtime with the specified number of threads."""

        # If the request was for a runtime with the default number of cores,
        # then we can just return the global runtime.
        if num_threads == num_cores():
            return Runtime()

        return external_call[
            "KGEN_CompilerRT_LLCL_CreateRuntime", Self.ptr_type
        ](num_threads)

    fn __init__(num_threads: Int, profileFilename: StringRef) -> Runtime:
        """Construct an LLCL Runtime with the specified number of threads
        that writes tracing events to profileFilename.
        """
        return external_call[
            "KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile", Self.ptr_type
        ](
            num_threads,
            profileFilename.data,
            profileFilename.length.value,
        )

    fn __init__(ptr: Self.ptr_type) -> Runtime:
        return Runtime {ptr: ptr}

    fn __enter__(self) -> Self:
        return self

    fn __exit__(self):
        """Destroys the LLCL Runtime. Note that this must be explicitly called
        when the Runtime goes out of the context.
        """
        self._destroy()

    fn _destroy(self):
        """Destroys the LLCL Runtime. Note that this must be explicitly called
        when the Runtime goes out of the context.
        """
        if _get_global_runtime().ptr != self.ptr:
            external_call["KGEN_CompilerRT_LLCL_DestroyRuntime", NoneType](
                self.ptr
            )

    fn parallelism_level(self) -> Int:
        """Gets the parallelism level of the Runtime."""
        return external_call[
            "KGEN_CompilerRT_LLCL_ParallelismLevel",
            Int32,
        ](self.ptr).to_int()

    fn create_task[
        type: AnyType
    ](self, owned handle: Coroutine[type]) -> Task[type]:
        """Run the coroutine as a task on the LLCL Runtime."""
        let ctx = handle.get_ctx[AsyncContext]()
        _init_llcl_chain(self, AsyncContext.get_chain(ctx))
        __get_address_as_lvalue(ctx.address).callback = AsyncContext.complete
        _async_execute(handle, self)
        return Task[type](handle ^)

    fn run[type: AnyType](self, owned handle: Coroutine[type]) -> type:
        let t = self.create_task(handle ^)
        let result = t.wait()
        return result


# ===----------------------------------------------------------------------===#
# Task
# ===----------------------------------------------------------------------===#


struct Task[type: AnyType]:
    var handle: Coroutine[type]

    fn __init__(inout self, owned handle: Coroutine[type]):
        self.handle = handle ^

    fn get(self) -> type:
        """Get the task's result value."""
        return self.handle.get()

    fn __del__(owned self):
        """Destroy the memory associated with a task. This must be manually
        called when a task goes out of scope.
        """
        let ctx: Pointer[AsyncContext] = self.handle.get_ctx[AsyncContext]()
        let chainPtr: Pointer[Chain] = AsyncContext.get_chain(ctx)
        _del_llcl_chain(chainPtr)
        _ = self.handle ^

    @always_inline
    fn __await__(self) -> type:
        """Suspend the current async function until the task completes and its
        result becomes available. This function must be force inlined into the
        calling async function.
        """
        let cur_hdl = __mlir_op.`pop.coroutine.opaque_handle`()

        __mlir_region await_body():
            _async_and_then(
                cur_hdl,
                AsyncContext.get_chain(self.handle.get_ctx[AsyncContext]()),
            )
            __mlir_op.`pop.coroutine.await.end`()

        __mlir_op.`pop.coroutine.await`[_region = "await_body".value]()
        return self.get()

    fn wait(self) -> type:
        """Block the current thread until the future value becomes available."""
        _async_wait(AsyncContext.get_chain(self.handle.get_ctx[AsyncContext]()))
        return self.get()


# ===----------------------------------------------------------------------===#
# TaskGroup
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct TaskGroupContext:
    alias tg_callback_fn_type = fn (inout TaskGroup) -> None

    var callback: Self.tg_callback_fn_type
    var task_group: Pointer[TaskGroup]


struct TaskGroupTask[type: AnyType]:
    """A task that belongs to a taskgroup. This object retains ownership of the
    underlying coroutine handle, which can be used to query the results of the
    task once the taskgroup completes.
    """

    var handle: Coroutine[type]

    fn __init__(inout self, owned handle: Coroutine[type]):
        self.handle = handle ^

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
        let prev: Int = self.counter.fetch_sub(1).value
        return prev - 1

    fn _task_complete(inout self):
        if self._counter_decr() == 0:
            _async_complete(Pointer[Chain].address_of(self.chain))

    fn create_task[
        type: AnyType
    ](inout self, owned task: Coroutine[type]) -> TaskGroupTask[type]:
        self.counter += 1
        let task_group_txt = task.get_ctx[TaskGroupContext]()
        task_group_txt.store(
            TaskGroupContext {
                callback: Self._task_complete,
                task_group: Pointer[TaskGroup].address_of(self),
            }
        )
        _async_execute(task, self.rt)
        return task ^

    @staticmethod
    fn await_body_impl(
        hdl: __mlir_type.`!kgen.pointer<i8>`, inout task_group: TaskGroup
    ):
        _async_and_then(hdl, Pointer[Chain].address_of(task_group.chain))
        task_group._task_complete()

    @always_inline
    fn __await__(inout self):
        let cur_hdl = __mlir_op.`pop.coroutine.opaque_handle`()

        __mlir_region await_body():
            Self.await_body_impl(cur_hdl, self)
            __mlir_op.`pop.coroutine.await.end`()

        __mlir_op.`pop.coroutine.await`[_region = "await_body".value]()

    fn wait(inout self):
        self._task_complete()
        _async_wait(Pointer[Chain].address_of(self.chain))


# ===----------------------------------------------------------------------===#
# OutputChainPtr and OwningOutputChainPtr
# ===----------------------------------------------------------------------===#


@register_passable
struct OutputChainPtr:
    """A pointer to a C++ heap/stack/closure allocated OutputChain, which is
    used by the Modular C++ runtime to coordinate execution with Mojo kernels.

    Pure CPU kernels which accept an OutputChainPtr argument are expected to
    call mark_ready(), mark_error(), or fork() before returning.

    CPU kernels which launch non-CPU kernels (eg a CUDA kernel) must accept
    an OuptputChainPtr, and must either call mark_error() before returning,
    or use a device-specific mechanism to coordinate execution back to the
    C++ runtime.
    """

    # Actually a KGEN::OutputChain*
    alias ptr_type = DTypePointer[DType.invalid]
    var ptr: Self.ptr_type

    @always_inline
    fn __init__() -> OutputChainPtr:
        """Casts a raw null OutputChainPtr."""
        return OutputChainPtr {ptr: Self.ptr_type()}

    @always_inline
    fn __init__(ptr: Self.ptr_type) -> OutputChainPtr:
        """Casts a raw pointer to our OutputChainPtr."""
        return OutputChainPtr {ptr: ptr}

    @always_inline("nodebug")
    fn __copyinit__(self) -> Self:
        return Self {ptr: self.ptr}

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        return self.ptr != Self.ptr_type()

    @always_inline
    fn fork(self) -> OwningOutputChainPtr:
        """Returns a pointer to a fresh heap-allocated LLCL::OutputChain
        containing a 'fork' of this.
        """
        return external_call[
            "KGEN_CompilerRT_LLCL_OutputChainPtr_CreateFork",
            OwningOutputChainPtr,
        ](self.ptr)

    @always_inline
    fn get_runtime(self) -> Runtime:
        """Returns the runtime managing the output chain."""
        return external_call[
            "KGEN_CompilerRT_LLCL_OutputChainPtr_GetRuntime", Runtime.ptr_type
        ](self.ptr)

    @always_inline
    fn mark_ready(self):
        """Marks the output chain as being ready.
        The underlying LLCL::OutputChain is not moved.
        """
        external_call[
            "KGEN_CompilerRT_LLCL_OutputChainPtr_MarkReady", NoneType
        ](self.ptr)

    @always_inline
    fn mark_error[
        single_thread_blocking_override: Bool
    ](self, message: StringLiteral):
        """Marks the output chain as having an error with a message.
        The underlying LLCL::OutputChain is not moved.
        """

        @parameter
        if not single_thread_blocking_override:
            self.mark_error(message)

    @always_inline
    fn mark_error[single_thread_blocking_override: Bool](self, err: Error):
        """Marks the output chain as having an error.
        The underlying LLCL::OutputChain is not moved.
        """

        @parameter
        if not single_thread_blocking_override:
            self.mark_error(err)

    @always_inline
    fn mark_error(self, message: StringLiteral):
        """Marks the output chain as having an error with a message.
        The underlying LLCL::OutputChain is not moved.
        """
        let strref = StringRef(message)
        external_call[
            "KGEN_CompilerRT_LLCL_OutputChainPtr_MarkError", NoneType
        ](
            self.ptr,
            strref.data,
            strref.length,
        )

    @always_inline
    fn mark_error(self, err: Error):
        """Marks the output chain as having an error.
        The underlying LLCL::OutputChain is not moved.
        """
        let str = err.__str__()
        let strref = str._strref_dangerous()
        external_call[
            "KGEN_CompilerRT_LLCL_OutputChainPtr_MarkError", NoneType
        ](
            self.ptr,
            strref.data,
            strref.length,
        )
        str._strref_keepalive()

    @always_inline
    fn trace[level: TraceLevel](self, label: StringRef):
        """If enabled, begin a time profile entry with label which will end
        when this chain is completed, either by mark_ready() or mark_error().
        """

        @parameter
        if is_mojo_profiling_disabled[level]():
            return
        external_call["KGEN_CompilerRT_LLCL_OutputChainPtr_Trace", NoneType](
            self.ptr,
            label.data,
            label.length.value,
            DTypePointer[DType.uint8].get_null(),
            0,
        )

    @always_inline
    fn trace[level: TraceLevel](self, label: StringRef, detail: StringRef):
        """If enabled, begin a time profile entry with label and detail which
        will end when this chain is completed, either by mark_ready() or
        mark_error().
        """

        @parameter
        if is_mojo_profiling_disabled[level]():
            return
        external_call["KGEN_CompilerRT_LLCL_OutputChainPtr_Trace", NoneType](
            self.ptr,
            label.data,
            label.length.value,
            detail.data,
            detail.length.value,
        )

    @always_inline
    fn trace[
        level: TraceLevel, detail_fn: fn () capturing -> String
    ](self, label: StringRef):
        """If enabled, begin a time profile entry with label and detail which
        will end when this chain is completed, either by mark_ready() or
        mark_error().
        """

        @parameter
        if is_mojo_profiling_disabled[level]():
            return

        let str = detail_fn()
        self.trace[level](label, str._strref_dangerous())
        str._strref_keepalive()

    @always_inline
    fn wait(self):
        """Returns only when the underlying LLCL::OutputChain is emplaced
        or set to an error. May execute arbitrary tasks while waiting.
        """
        external_call["KGEN_CompilerRT_LLCL_OutputChainPtr_Await", NoneType](
            self.ptr
        )

    @always_inline
    fn get_cuda_stream(self) -> Stream[is_borrowed=True]:
        """Return the CUstream to use for launching CUDA kernels from the
        CPU kernel 'shim'. These CPU kernels should never call mark_ready()."""
        return external_call[
            "KGEN_CompilerRT_LLCL_OutputChainPtr_GetCUDAStream",
            _StreamImpl,
        ](self.ptr)


@register_passable
struct OwningOutputChainPtr:
    """As for OutputChainPtr, but will destroy the underlying LLCL::OutputChain
    when goes out of scope.
    """

    # Actually LLCL::OutputChain*
    alias ptr_type = DTypePointer[DType.invalid]
    var ptr: Self.ptr_type

    @always_inline
    fn __init__() -> OwningOutputChainPtr:
        """Creates a null OwningOutputChainPtr."""
        return OwningOutputChainPtr {ptr: Self.ptr_type()}

    @always_inline
    fn __init__(ptr: Self.ptr_type) -> OwningOutputChainPtr:
        """Casts a raw pointer to our OwningOutputChainPtr."""
        return OwningOutputChainPtr {ptr: ptr}

    @always_inline
    fn __init__(rt: Runtime) -> OwningOutputChainPtr:
        """Returns a pointer to a heap-allocated empty LLCL::OutputChain.
        The LLCL::OutputChain will have an empty location and an unemplaced
        AsyncValueRef<Chain>.
        """
        let ptr = external_call[
            "KGEN_CompilerRT_LLCL_OutputChainPtr_CreateEmpty", Self.ptr_type
        ](rt.ptr)
        return OwningOutputChainPtr {ptr: ptr}

    @always_inline("nodebug")
    fn __del__(owned self):
        """Destroys the LLCL::OutputChain."""
        external_call["KGEN_CompilerRT_LLCL_OutputChainPtr_Destroy", NoneType](
            self.ptr
        )

    @always_inline
    fn borrow(self) -> OutputChainPtr:
        """Returns non-owning pointer to same LLCL::OutputChain."""
        return OutputChainPtr(self.ptr)

    @always_inline
    fn wait(self):
        """Returns only when the underlying LLCL::OutputChain is emplaced
        or set to an error. May execute arbitrary tasks while waiting.
        """
        external_call["KGEN_CompilerRT_LLCL_OutputChainPtr_Await", NoneType](
            self.ptr
        )

    @always_inline
    fn task_is_done(self):
        """Indicates the caller's task is done for the purposes of task overhang
        detection. Only needed for tasks which signal their completion by some
        mechanism other than mark_ready() or mark_error(). Is a no-op unless
        task overhang detection is enabled in the build.
        """
        external_call[
            "KGEN_CompilerRT_LLCL_OutputChainPtr_TaskIsDone", NoneType
        ](self.ptr)

    @always_inline
    fn assert_ready(self):
        """Asserts that the underlying LLCL::OutputChain is ready.

        FOR USE IN TESTS ONLY.
        """
        external_call[
            "KGEN_CompilerRT_LLCL_OutputChainPtr_AssertReady", NoneType
        ](self.ptr)


# ===----------------------------------------------------------------------===#
# AsyncTaskGroup and AsyncTaskGroupPtr
# ===----------------------------------------------------------------------===#


struct AsyncTaskGroupContext:
    alias tg_callback_fn_type = fn (inout AsyncTaskGroup) -> None

    var callback: Self.tg_callback_fn_type
    var async_task_group_ptr: Pointer[AsyncTaskGroup]

    fn __init__(
        inout self,
        callback: Self.tg_callback_fn_type,
        async_task_group_ptr: Pointer[AsyncTaskGroup],
    ):
        self.callback = callback
        self.async_task_group_ptr = async_task_group_ptr


struct CoroutineList[type: AnyType]:
    var data: Pointer[Coroutine[type]]
    var size: Int

    fn __init__(inout self, num_work_items: Int):
        self.data = Pointer[Coroutine[type]].alloc(num_work_items)
        self.size = 0

    fn add(inout self, owned hdl: Coroutine[type]):
        __get_address_as_uninit_lvalue(self.data.offset(self.size).address) = (
            hdl ^
        )
        self.size += 1

    fn __len__(self) -> Int:
        return self.size

    fn destroy(inout self):
        for i in range(self.size):
            _ = __get_address_as_owned_value(self.data.offset(i).address)
        self.data.free()


struct AsyncTaskGroup:
    """The target of an AsyncTaskGroupPtr. Holds exactly num_work_items
    Coroutines representing tasks. When all tasks are complete out_chain
    is marked as ready and this object is deleted.
    """

    # Number of tasks still in flight.
    var counter: Atomic[DType.index]
    # Output chain to mark_ready/mark_error when last task completed.
    # This will be 'forked' on construction to guarantee the correct lifetime.
    var out_chain: OwningOutputChainPtr
    # Vector holding co-routines.
    var coroutines: CoroutineList[NoneType]
    # Bool to indicate whether we use the monolithic Queue for scheduling
    # this AsyncTaskGroup or thread local queues.
    var _use_global_queue: Bool

    @always_inline
    fn __init__(inout self, num_work_items: Int, out_chain: OutputChainPtr):
        self.counter = num_work_items
        self.out_chain = out_chain.fork()
        self.coroutines = num_work_items
        # We choose to keep tasks not perfectly parallelizable to the
        # monolithic queue for better load balancing. Only push tasks which
        # occupy all the cores to taskId affinitized queues.
        self._use_global_queue = (
            num_work_items != out_chain.get_runtime().parallelism_level()
        )

    # This destroy's self when all the references are gone.
    @always_inline
    fn destroy(inout self):
        self.coroutines.destroy()

        # Replace the out_chain owned by this value with a null one, so the old
        # value is destroyed.
        self.out_chain = OwningOutputChainPtr()
        let self_ptr = Pointer[AsyncTaskGroup].address_of(self)
        self_ptr.free()

    @always_inline
    fn _counter_decr(inout self) -> Int:
        let prev: Int = self.counter.fetch_sub(1).value
        return prev - 1

    fn _task_complete(inout self):
        # Indicate the current task is done for the purpose of task overhang
        # detection. Is a no-op unless task overhang detection is enabled in
        # the build.
        @parameter
        if is_defined["MODULAR_PARANOID"]():
            self.out_chain.task_is_done()
        if self._counter_decr() == 0:
            self.out_chain.borrow().mark_ready()
            self.destroy()

    # TODO(#11915): Allow failure of coroutine to propagate error back to out_chain.
    fn add_task(inout self, owned coroutine: Coroutine[NoneType]):
        let ctx_ptr = coroutine.get_ctx[AsyncTaskGroupContext]()
        let self_ptr = Pointer[AsyncTaskGroup].address_of(self)
        __get_address_as_uninit_lvalue(ctx_ptr.address) = AsyncTaskGroupContext(
            Self._task_complete, self_ptr
        )
        let task_id = self.coroutines.__len__()
        # Take a copy of the handle reference, then move the coroutine onto the
        # list. Do this before scheduling the coroutine on the taskqueue.
        let hdl = coroutine._handle
        self.coroutines.add(coroutine ^)
        external_call[
            "KGEN_CompilerRT_LLCL_OutputChainPtr_ExecuteAsTask", NoneType
        ](
            self.out_chain.ptr,
            _coro_resume_fn,
            hdl,
            task_id,
            self._use_global_queue,
        )


struct AsyncTaskGroupPtr:
    """Holds a pointer to a dynamically allocated AsyncTaskGroup. Exactly
    num_work_items Coroutines representing tasks may be added. When all such
    tasks are complete the given out_chain is marked as ready, and the
    AsyncTaskGroup deletes itself. This type is only intended to be used
    locally in order to add tasks, and should never be stored.
    """

    var ptr: Pointer[AsyncTaskGroup]

    @always_inline
    fn __init__(inout self, num_work_items: Int, out_chain: OutputChainPtr):
        self.ptr = Pointer[AsyncTaskGroup].alloc(1)
        __get_address_as_uninit_lvalue(self.ptr.address) = AsyncTaskGroup(
            num_work_items, out_chain
        )

    @always_inline
    fn add_task(inout self, owned coroutine: Coroutine[NoneType]):
        __get_address_as_lvalue(self.ptr.address).add_task(coroutine ^)

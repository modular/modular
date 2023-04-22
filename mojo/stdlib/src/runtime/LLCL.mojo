# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the low level concurrency library."""

from Atomic import Atomic
from Coroutine import Coroutine, _get_coro_resume_fn
from DType import DType
from Pointer import Pointer, DTypePointer
from String import StringRef
from Range import range
from String import String
from Tracing import TraceLevel, is_mojo_profiling_disabled

# ===----------------------------------------------------------------------===#
# num_cores
# ===----------------------------------------------------------------------===#


@always_inline
fn num_cores() -> Int:
    """Returns the number of cores on the system.

    Returns:
        Int: The number of cores on the system
    """
    return __mlir_op.`pop.external_call`[
        func : __mlir_attr.`@KGEN_CompilerRT_CoreCount`,
        _type : __mlir_type.index,
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


# FIXME(traits): This shouldn't be a register_passable type but we need this
# until we have traits for proper parametric types.
@register_passable("trivial")
struct AsyncContext:
    """This struct models the coroutine context contained in every coroutine
    instance. The struct consists of a unary callback function that accepts a
    pointer argument. It is invoked with the second struct field, which is an
    opaque pointer. This struct is essentially a completion callback closure
    that is invokved by a coroutine when it completes and its results are made
    available.

    In async execution, a task's completion callback is to set its async token
    to available.
    """

    alias callback_fn_type = __mlir_type[`(`, Chain, `) -> `, NoneType]

    var callback: callback_fn_type
    var chain: Chain

    @staticmethod
    fn get_chain(ctx: Pointer[AsyncContext]) -> Pointer[Chain]:
        return __get_lvalue_as_address(
            __get_address_as_lvalue(ctx.address).chain
        )

    @staticmethod
    fn complete(ch: Chain):
        var chMem = ch
        _async_complete(Pointer[Chain].address_of(chMem))


# ===----------------------------------------------------------------------===#
# LLCL C Shims
# ===----------------------------------------------------------------------===#


fn _init_llcl_chain(rt: Runtime, chain: Pointer[Chain]):
    __mlir_op.`pop.external_call`[
        _type:None,
        func : __mlir_attr.`@KGEN_CompilerRT_LLCL_InitializeChain`,
    ](rt.ptr, chain.address)


fn _del_llcl_chain(chain: Pointer[Chain]):
    __mlir_op.`pop.external_call`[
        _type:None,
        func : __mlir_attr.`@KGEN_CompilerRT_LLCL_DestroyChain`,
    ](chain.address)


fn _async_and_then(hdl: __mlir_type.`!pop.pointer<i8>`, chain: Pointer[Chain]):
    __mlir_op.`pop.external_call`[
        _type:None, func : __mlir_attr.`@KGEN_CompilerRT_LLCL_AndThen`
    ](_get_coro_resume_fn(), chain.address, hdl)


fn _async_execute[type: AnyType](handle: Coroutine[type], rt: Runtime):
    __mlir_op.`pop.external_call`[
        _type:None, func : __mlir_attr.`@KGEN_CompilerRT_LLCL_Execute`
    ](_get_coro_resume_fn(), handle._handle, rt.ptr)


fn _async_wait(chain: Pointer[Chain]):
    __mlir_op.`pop.external_call`[
        _type:None, func : __mlir_attr.`@KGEN_CompilerRT_LLCL_Wait`
    ](chain.address)


fn _async_complete(chain: Pointer[Chain]):
    __mlir_op.`pop.external_call`[
        _type:None, func : __mlir_attr.`@KGEN_CompilerRT_LLCL_Complete`
    ](chain.address)


# ===----------------------------------------------------------------------===#
# Runtime
# ===----------------------------------------------------------------------===#

# FIXME(traits): This shouldn't be a register_passable type but we need this
# until we have traits for proper parametric types.
@register_passable
struct Runtime:
    alias ptr_type = DTypePointer[DType.invalid.value]
    var ptr: ptr_type

    # TODO: Probably don't want the runtime to be implicitly copyable.
    @always_inline("nodebug")
    fn __copyinit__(self) -> Self:
        return Self {ptr: self.ptr}

    fn __init__() -> Runtime:
        """Construct an LLCL Runtime with the same number of threads as
        processor cores.
        """
        return Runtime(num_cores())

    fn __init__(numThreads: Int) -> Runtime:
        """Construct an LLCL Runtime with the specified number of threads."""
        return __mlir_op.`pop.external_call`[
            _type:ptr_type,
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_CreateRuntime`,
        ](numThreads)

    fn __init__(numThreads: Int, profileFilename: StringRef) -> Runtime:
        """Construct an LLCL Runtime with the specified number of threads
        that writes tracing events to profileFilename.
        """
        return __mlir_op.`pop.external_call`[
            _type:ptr_type,
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile`,
        ](
            numThreads,
            profileFilename.data.address,
            profileFilename.length.value,
        )

    fn __init__(ptr: ptr_type) -> Runtime:
        return Runtime {ptr: ptr}

    fn __del__(self):
        """Destroys the LLCL Runtime. Note that this must be explicitly called
        when the Runtime goes out of scope.
        """
        __mlir_op.`pop.external_call`[
            _type:None,
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_DestroyRuntime`,
        ](self.ptr)

    fn parallelism_level(self) -> Int:
        """Gets the parallism level of the Runtime."""
        return __mlir_op.`pop.external_call`[
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_ParallelismLevel`,
            _type : __mlir_type.`!pop.scalar<si32>`,
        ](self.ptr)

    fn create_task[
        type: AnyType
    ](self, owned handle: Coroutine[type]) -> Task[type]:
        """Run the coroutine as a task on the LLCL Runtime."""
        let ctx = handle.get_ctx[AsyncContext]()
        _init_llcl_chain(self, AsyncContext.get_chain(ctx))
        let callbackPtr: Pointer[
            AsyncContext.callback_fn_type
        ] = __mlir_op.`lit.struct.gep`[
            _type : Pointer[AsyncContext.callback_fn_type].pointer_type,
            field : __mlir_attr.`"callback"`,
        ](
            ctx.address
        )
        callbackPtr.store(
            __mlir_op.`kgen.addressof`[
                _type : AsyncContext.callback_fn_type,
                callee : AsyncContext.complete,
                paramDecls : __mlir_attr.`#kgen<param.decls[]>`,
            ]()
        )
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

    fn __init__(self&, owned handle: Coroutine[type]):
        self.handle = handle ^

    fn get(self) -> type:
        """Get the task's result value."""
        return self.handle.get()

    fn __del___(owned self):
        """Destroy the memory associated with a task. This must be manually
        called when a task goes out of scope.
        """
        let ctx: Pointer[AsyncContext] = self.handle.get_ctx[AsyncContext]()
        let chainPtr: Pointer[Chain] = AsyncContext.get_chain(ctx)
        _del_llcl_chain(chainPtr)
        # FIXME(#13073): We should be able to write `_ = self.handle` instead.
        self.handle._keep()

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

        __mlir_op.`pop.coroutine.await`[_region : "await_body".value]()
        return self.get()

    fn wait(self) -> type:
        """Block the current thread until the future value becomes available."""
        _async_wait(AsyncContext.get_chain(self.handle.get_ctx[AsyncContext]()))
        return self.get()


# ===----------------------------------------------------------------------===#
# TaskGroup
# ===----------------------------------------------------------------------===#

# FIXME(traits): This shouldn't be a register_passable type but we need this
# until we have traits for proper parametric types.
@register_passable("trivial")
struct TaskGroupContext:
    alias tg_callback_fn_type = __mlir_type[
        `(!pop.pointer<`, TaskGroup, `>) -> `, NoneType
    ]

    var callback: tg_callback_fn_type
    var task_group: Pointer[TaskGroup]


struct TaskGroupTask[type: AnyType]:
    """A task that belongs to a taskgroup. This object retains ownership of the
    underlying coroutine handle, which can be used to query the results of the
    task once the taskground completes.
    """

    var handle: Coroutine[type]

    fn __init__(self&, owned handle: Coroutine[type]):
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

    fn __init__(self&, rt: Runtime):
        var chain = Chain()
        _init_llcl_chain(rt, Pointer[Chain].address_of(chain))
        self.counter = 1
        self.chain = chain
        self.rt = rt

    fn __del___(owned self):
        _del_llcl_chain(Pointer[Chain].address_of(self.chain))

    @always_inline
    fn _counter_decr(self&) -> Int:
        let prev: Int = self.counter.isub(1).value
        return prev - 1

    fn _task_complete(self&):
        if self._counter_decr() == 0:
            _async_complete(Pointer[Chain].address_of(self.chain))

    @staticmethod
    fn _get_complete_callback() -> __mlir_type[
        `(!pop.pointer<`, TaskGroup, `>) -> `, NoneType
    ]:
        return __mlir_op.`kgen.addressof`[
            _type : __mlir_type[
                `(!pop.pointer<`, TaskGroup, `>) -> `, NoneType
            ],
            callee:_task_complete,
            paramDecls : __mlir_attr.`#kgen<param.decls[]>`,
        ]()

    fn create_task[
        type: AnyType
    ](self&, owned task: Coroutine[type]) -> TaskGroupTask[type]:
        self.counter += 1
        let task_group_txt = task.get_ctx[TaskGroupContext]()
        task_group_txt.store(
            TaskGroupContext {
                callback: _get_complete_callback(),
                task_group: Pointer[TaskGroup].address_of(self),
            }
        )
        _async_execute(task, self.rt)
        return task ^

    @staticmethod
    fn await_body_impl(
        hdl: __mlir_type.`!pop.pointer<i8>`, task_group&: TaskGroup
    ):
        _async_and_then(hdl, Pointer[Chain].address_of(task_group.chain))
        task_group._task_complete()

    @always_inline
    fn __await__(self&):
        let cur_hdl = __mlir_op.`pop.coroutine.opaque_handle`()

        __mlir_region await_body():
            await_body_impl(cur_hdl, self)

        __mlir_op.`pop.coroutine.await`[_region : "await_body".value]()

    fn wait(self&):
        self._task_complete()
        _async_wait(Pointer[Chain].address_of(self.chain))


# ===----------------------------------------------------------------------===#
# OutputChainPtr and OwningOutputChainPtr
# ===----------------------------------------------------------------------===#


@register_passable
struct OutputChainPtr:
    """A pointer to a C++ heap/stack/closure allocated OutputChain, ie a pair
    of an AsyncValueRef<Chain> and an EncodedLocation. Kernels which accept
    an OutputChainPtr argument are expected to call mark_ready(), mark_error(),
    or move.
    """

    # Actually LLCL::OutputChain*
    alias ptr_type = DTypePointer[DType.invalid.value]
    var ptr: ptr_type

    @always_inline
    fn __init__() -> OutputChainPtr:
        """Casts a raw null OutputChainPtr."""
        return OutputChainPtr {ptr: ptr_type()}

    @always_inline
    fn __init__(ptr: ptr_type) -> OutputChainPtr:
        """Casts a raw pointer to our OutputChainPtr."""
        return OutputChainPtr {ptr: ptr}

    @always_inline("nodebug")
    fn __copyinit__(self) -> Self:
        return Self {ptr: self.ptr}

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        return self.ptr != ptr_type()

    @always_inline
    fn fork(self) -> OwningOutputChainPtr:
        """Returns a pointer to a fresh heap-allocated LLCL::OutputChain
        containing a 'fork' of this.
        """
        return __mlir_op.`pop.external_call`[
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_OutputChainPtr_CreateFork`,
            _type:OwningOutputChainPtr,
        ](self.ptr)

    @always_inline
    fn get_runtime(self) -> Runtime:
        """Returns the runtime managing the output chain."""
        return __mlir_op.`pop.external_call`[
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_OutputChainPtr_GetRuntime`,
            _type : Runtime.ptr_type,
        ](self.ptr)

    @always_inline
    fn mark_ready(self):
        """Marks the output chain as being ready.
        The underlying LLCL::OutputChain is not moved.
        """
        __mlir_op.`pop.external_call`[
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_OutputChainPtr_MarkReady`,
            _type:None,
        ](self.ptr)

    @always_inline
    fn mark_error(self, message: StringRef):
        """Marks the output chain as having an error with message.
        The underlying LLCL::OutputChain is not moved.
        """
        __mlir_op.`pop.external_call`[
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_OutputChainPtr_MarkError`,
            _type:None,
        ](
            self.ptr,
            message.data.address,
            message.length.value,
        )

    fn wait(self):
        """Returns only when the underlying LLCL::OutputChain is emplaced
        or set to an error. May execute arbitrary tasks while waiting.

        FOR USE IN TEST CODE ONLY. Kernels should never await.
        """
        __mlir_op.`pop.external_call`[
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_OutputChainPtr_Await`,
            _type:None,
        ](self.ptr)

    @always_inline
    fn trace[level: TraceLevel](self, label: StringRef):
        """If enabled, begin a time profile entry with label which will end
        when this chain is completed, either by mark_ready() or mark_error().
        """

        @parameter
        if is_mojo_profiling_disabled[level]():
            return
        __mlir_op.`pop.external_call`[
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_OutputChainPtr_Trace`,
            _type:None,
        ](
            self.ptr,
            label.data.address,
            label.length.value,
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
        __mlir_op.`pop.external_call`[
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_OutputChainPtr_TraceDetailed`,
            _type:None,
        ](
            self.ptr,
            label.data.address,
            label.length.value,
            detail.data.address,
            detail.length.value,
        )

    @always_inline
    fn trace_detail[
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


@register_passable
struct OwningOutputChainPtr:
    """As for OutputChainPtr, but will destroy the underlying LLCL::OutputChain
    when goes out of scope.
    """

    # Actually LLCL::OutputChain*
    alias ptr_type = DTypePointer[DType.invalid.value]
    var ptr: ptr_type

    @always_inline
    fn __init__() -> OwningOutputChainPtr:
        """Creates a null OwningOutputChainPtr."""
        return OwningOutputChainPtr {ptr: ptr_type()}

    @always_inline
    fn __init__(ptr: ptr_type) -> OwningOutputChainPtr:
        """Casts a raw pointer to our OwningOutputChainPtr."""
        return OwningOutputChainPtr {ptr: ptr}

    @always_inline
    fn __init__(rt: Runtime) -> OwningOutputChainPtr:
        """Returns a pointer to a heap-allocated empty LLCL::OutputChain.
        The LLCL::OutputChain will have an empty location and an unemplaced
        AsyncValueRef<Chain>.
        """
        let ptr = __mlir_op.`pop.external_call`[
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_OutputChainPtr_CreateEmpty`,
            _type:ptr_type,
        ](rt.ptr)
        return OwningOutputChainPtr {ptr: ptr}

    @always_inline("nodebug")
    fn __del___(owned self):
        """Destroys the LLCL::OutputChain."""
        __mlir_op.`pop.external_call`[
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_OutputChainPtr_Destroy`,
            _type:None,
        ](self.ptr)

    @always_inline
    fn borrow(self) -> OutputChainPtr:
        """Returns non-owning pointer to same LLCL::OutputChain."""
        return OutputChainPtr(self.ptr)

    fn wait(self):
        """Returns only when the underlying LLCL::OutputChain is emplaced
        or set to an error. May execute arbitrary tasks while waiting.

        FOR USE IN TEST CODE ONLY. Kernels should never await.
        """
        __mlir_op.`pop.external_call`[
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_OutputChainPtr_Await`,
            _type:None,
        ](self.ptr)


# ===----------------------------------------------------------------------===#
# AsyncTaskGroup and AsyncTaskGroupPtr
# ===----------------------------------------------------------------------===#


struct AsyncTaskGroupContext:
    alias tg_callback_fn_type = __mlir_type[
        `(!pop.pointer<`, AsyncTaskGroup, `>) -> `, NoneType
    ]

    var callback: tg_callback_fn_type
    var async_task_group_ptr: Pointer[AsyncTaskGroup]

    fn __init__(
        self&,
        callback: tg_callback_fn_type,
        async_task_group_ptr: Pointer[AsyncTaskGroup],
    ):
        self.callback = callback
        self.async_task_group_ptr = async_task_group_ptr


struct CoroutineList[type: AnyType]:
    var data: Pointer[Coroutine[type]]
    var size: Int

    fn __init__(self&, num_work_items: Int):
        self.data = Pointer[Coroutine[type]].alloc(num_work_items)
        self.size = 0

    fn add(self&, owned hdl: Coroutine[type]):
        __get_address_as_uninit_lvalue(self.data.offset(self.size).address) = (
            hdl ^
        )
        self.size += 1

    fn __len__(self) -> Int:
        return self.size

    fn destroy(self&):
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

    @always_inline
    fn __init__(self&, num_work_items: Int, out_chain: OutputChainPtr):
        self.counter = num_work_items
        self.out_chain = out_chain.fork()
        self.coroutines = num_work_items

    # This destroy's self when all the references are gone.
    @always_inline
    fn destroy(self&):
        self.coroutines.destroy()

        # Replace the out_chain owned by this value with a null one, so the old
        # value is destroyed.
        self.out_chain = OwningOutputChainPtr()
        let self_ptr = Pointer[AsyncTaskGroup].address_of(self)
        self_ptr.free()

    @always_inline
    fn _counter_decr(self&) -> Int:
        let prev: Int = self.counter.isub(1).value
        return prev - 1

    fn _task_complete(self&):
        if self._counter_decr() == 0:
            self.out_chain.borrow().mark_ready()
            self.destroy()

    @staticmethod
    fn _get_complete_callback() -> __mlir_type[
        `(!pop.pointer<`, AsyncTaskGroup, `>) -> `, NoneType
    ]:
        return __mlir_op.`kgen.addressof`[
            _type : __mlir_type[
                `(!pop.pointer<`, AsyncTaskGroup, `>) -> `, NoneType
            ],
            callee:_task_complete,
            paramDecls : __mlir_attr.`#kgen<param.decls[]>`,
        ]()

    # TODO(#11915): Allow failure of coroutine to propagate error back to out_chain.
    fn add_task(self&, owned coroutine: Coroutine[NoneType]):
        let ctx_ptr = coroutine.get_ctx[AsyncTaskGroupContext]()
        let self_ptr = Pointer[AsyncTaskGroup].address_of(self)
        __get_address_as_uninit_lvalue(ctx_ptr.address) = AsyncTaskGroupContext(
            _get_complete_callback(), self_ptr
        )
        let task_id = self.coroutines.__len__()
        # Take a copy of the handle reference, then move the coroutine onto the
        # list. Do this before scheduling the coroutine on the taskqueue.
        let hdl = coroutine._handle
        self.coroutines.add(coroutine ^)
        __mlir_op.`pop.external_call`[
            _type:None,
            func : __mlir_attr.`@KGEN_CompilerRT_LLCL_OutputChainPtr_ExecuteAsTask`,
        ](
            self.out_chain.ptr,
            _get_coro_resume_fn(),
            hdl,
            task_id,
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
    fn __init__(self&, num_work_items: Int, out_chain: OutputChainPtr):
        self.ptr = Pointer[AsyncTaskGroup].alloc(1)
        __get_address_as_uninit_lvalue(self.ptr.address) = AsyncTaskGroup(
            num_work_items, out_chain
        )

    @always_inline
    fn add_task(self&, owned coroutine: Coroutine[NoneType]):
        __get_address_as_lvalue(self.ptr.address).add_task(coroutine ^)

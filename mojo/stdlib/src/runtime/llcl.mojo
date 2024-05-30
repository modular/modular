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
from gpu.host import Stream, CUDADeviceStream

from builtin.coroutine import AnyCoroutine, _coro_resume_fn, _suspend_async
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
        self.storage = Pointer[Int]()

    fn __bool__(self) -> Bool:
        return self.storage != Pointer[Int]()


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

    alias callback_fn_type = fn (UnsafePointer[NoneType], Chain) -> None

    var callback: Self.callback_fn_type
    var chain: Chain

    @staticmethod
    fn get_chain(ctx: UnsafePointer[AsyncContext]) -> UnsafePointer[Chain]:
        return UnsafePointer.address_of(ctx[].chain)

    @staticmethod
    fn complete(hdl: UnsafePointer[NoneType], ch: Chain):
        var tmp = ch
        _async_complete(UnsafePointer[Chain].address_of(tmp))


# ===----------------------------------------------------------------------===#
# LLCL C Shims
# ===----------------------------------------------------------------------===#


fn _init_llcl_chain(rt: Runtime, chain: UnsafePointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_InitializeChain", NoneType](
        rt.ptr, chain.address
    )


fn _del_llcl_chain(chain: UnsafePointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_DestroyChain", NoneType](chain.address)


fn _async_and_then(hdl: AnyCoroutine, chain: UnsafePointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_AndThen", NoneType](
        _coro_resume_fn, chain.address, hdl
    )


fn _async_execute[
    type: AnyTrivialRegType
](handle: AnyCoroutine, rt: Runtime, desired_worker_id: Int,):
    external_call["KGEN_CompilerRT_LLCL_Execute", NoneType](
        _coro_resume_fn, handle, rt.ptr, desired_worker_id
    )


fn _async_wait(chain: UnsafePointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_Wait", NoneType](chain.address)


fn _async_complete(chain: UnsafePointer[Chain]):
    external_call["KGEN_CompilerRT_LLCL_Complete", NoneType](chain.address)


fn _async_wait_timeout(chain: UnsafePointer[Chain], timeout: Int) -> Bool:
    return external_call["KGEN_CompilerRT_LLCL_Wait_Timeout", Bool](
        chain.address, timeout
    )


struct ChainPromise:
    var chain: Chain

    fn __init__(inout self, rt: Runtime):
        self.chain = Chain()
        _init_llcl_chain(rt, UnsafePointer.address_of(self.chain))

    fn __init__(inout self, owned chain: Chain):
        self.chain = chain

    fn __del__(owned self):
        if self.chain:
            _del_llcl_chain(UnsafePointer.address_of(self.chain))

    @always_inline
    fn __await__(self):
        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            _async_and_then(cur_hdl, UnsafePointer.address_of(self.chain))

        _suspend_async[await_body]()

    fn wait(self):
        _async_wait(UnsafePointer.address_of(self.chain))


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


@register_passable("trivial")
struct Runtime:
    alias ptr_type = Pointer[NoneType]
    var ptr: Self.ptr_type
    var owning: Bool

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

    fn create_task(
        self,
        owned handle: Coroutine[*_],
        desired_worker_id: Int = -1,
    ) -> Task[handle.type, handle.lifetime]:
        """Run the coroutine as a task on the LLCL Runtime."""
        var ctx: UnsafePointer[AsyncContext] = handle._get_ctx[
            AsyncContext
        ]().address
        _init_llcl_chain(self, AsyncContext.get_chain(ctx))
        ctx[].callback = AsyncContext.complete
        _async_execute[handle.type](handle._handle, self, desired_worker_id)
        return Task[handle.type, handle.lifetime](handle^)

    fn run(self, owned handle: Coroutine[*_]) -> handle.type:
        var t = self.create_task(handle^)
        var result = t.wait()
        return result


# ===----------------------------------------------------------------------===#
# Task
# ===----------------------------------------------------------------------===#


struct Task[
    is_mut: Bool, //,
    type: AnyTrivialRegType,
    lifetime: AnyLifetime[is_mut].type,
]:
    var handle: Coroutine[type, lifetime]

    fn __init__(inout self, owned handle: Coroutine[type, lifetime]):
        self.handle = handle^

    fn get(self) -> type:
        """Get the task's result value."""
        return self.handle.get()

    fn __del__(owned self):
        """Destroy the memory associated with a task. This must be manually
        called when a task goes out of scope.
        """
        var ctx: UnsafePointer[AsyncContext] = self.handle._get_ctx[
            AsyncContext
        ]().address
        var chainPtr: UnsafePointer[Chain] = AsyncContext.get_chain(ctx)
        _del_llcl_chain(chainPtr)
        _ = self.handle^

    @always_inline
    fn __await__(self) -> type:
        """Suspend the current async function until the task completes and its
        result becomes available. This function must be force inlined into the
        calling async function.
        """

        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            _async_and_then(
                cur_hdl,
                AsyncContext.get_chain(
                    self.handle._get_ctx[AsyncContext]().address
                ),
            )

        _suspend_async[await_body]()
        return self.get()

    fn wait(self) -> type:
        """Block the current thread until the future value becomes available."""
        _async_wait(
            AsyncContext.get_chain(self.handle._get_ctx[AsyncContext]().address)
        )
        return self.get()


# ===----------------------------------------------------------------------===#
# TaskGroup
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct TaskGroupContext[is_mut: Bool, //, lifetime: AnyLifetime[is_mut].type]:
    alias tg_callback_fn_type = fn (
        Pointer[NoneType], inout TaskGroup[lifetime]
    ) -> None

    var callback: Self.tg_callback_fn_type
    var task_group: UnsafePointer[TaskGroup[lifetime]]


@register_passable
struct _TaskGroupBox(CollectionElement):
    """This struct is a type-erased owning box for an opaque coroutine."""

    var handle: AnyCoroutine

    fn __init__[
        type: AnyTrivialRegType
    ](inout self, owned coro: Coroutine[type]):
        var handle = coro._handle
        __mlir_op.`lit.ownership.mark_destroyed`(Reference(coro).value)
        self.handle = handle

    fn get[type: AnyTrivialRegType](self) -> type:
        return __mlir_op.`co.get_results`[_type=type](self.handle)

    fn __del__(owned self):
        __mlir_op.`co.destroy`(self.handle)

    # FIXME(MSTDL-573): `List` requires copyability. Just crash here because it
    # should never get called.
    fn __copyinit__(inout self, existing: Self):
        abort("_TaskGroupBox.__copyinit__ should never get called")
        while True:
            pass


@register_passable
struct TaskGroupTask[type: AnyTrivialRegType, lifetime: MutableLifetime]:
    """A task that belongs to a TaskGroup. This object retains ownership of the
    underlying coroutine handle, which can be used to query the results of the
    task once the taskgroup completes.
    """

    var handle_ref: Reference[_TaskGroupBox, True, lifetime]

    fn __init__(
        inout self, handle_ref: Reference[_TaskGroupBox, True, lifetime]
    ):
        self.handle_ref = handle_ref

    fn get(self) -> type:
        """Get the task's result value. This should only be called once the
        task result is available. I.e., when the taskgroup completes.

        Returns:
            The task's result value.
        """
        return self.handle_ref[].get[type]()


struct TaskGroup[is_mut: Bool, //, lifetime: AnyLifetime[is_mut].type]:
    var counter: Atomic[DType.index]
    var chain: Chain
    var rt: Runtime
    var tasks: List[_TaskGroupBox]

    fn __init__(inout self, rt: Runtime):
        var chain = Chain()
        _init_llcl_chain(rt, UnsafePointer[Chain].address_of(chain))
        self.counter = 1
        self.chain = chain
        self.rt = rt
        self.tasks = List[_TaskGroupBox](capacity=16)

    fn __del__(owned self):
        _del_llcl_chain(UnsafePointer[Chain].address_of(self.chain))

    @always_inline
    fn _counter_decr(inout self) -> Int:
        var prev: Int = self.counter.fetch_sub(1).value
        return prev - 1

    @staticmethod
    fn _task_complete_callback(
        hdl: Pointer[NoneType], inout tg: TaskGroup[lifetime]
    ):
        tg._task_complete()

    fn _task_complete(inout self):
        if self._counter_decr() == 0:
            _async_complete(UnsafePointer[Chain].address_of(self.chain))

    fn create_task(
        inout self,
        owned task: Coroutine[*_],
        desired_worker_id: Int = -1,
    ) -> TaskGroupTask[task.type, __lifetime_of(self)]:
        # TODO(MOCO-771): Enforce that `task.lifetime` is a subset of
        # `Self.lifetime`.
        self.counter += 1
        LegacyPointer(
            task._get_ctx[TaskGroupContext[lifetime]]().address
        ).store(
            TaskGroupContext[lifetime] {
                callback: Self._task_complete_callback,
                task_group: UnsafePointer[Self].address_of(self),
            }
        )
        _async_execute[task.type](task._handle, self.rt, desired_worker_id)
        self.tasks.append(_TaskGroupBox(task^))
        return self.tasks.__get_ref(-1)

    @staticmethod
    fn await_body_impl(hdl: AnyCoroutine, inout task_group: Self):
        _async_and_then(hdl, UnsafePointer[Chain].address_of(task_group.chain))
        task_group._task_complete()

    @always_inline
    fn __await__(inout self):
        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            Self.await_body_impl(cur_hdl, self)

        _suspend_async[await_body]()

    fn wait(inout self):
        self._task_complete()
        _async_wait(UnsafePointer[Chain].address_of(self.chain))


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
        return MojoCallContextPtr {ptr: DTypePointer[DType.invalid]()}

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
    fn get_stream(self) -> Stream:
        """Get the cuda stream."""
        var stream = external_call[
            "KGEN_CompilerRT_LLCL_MojoCallContext_GetCUStream", self.ptr_type
        ](
            self.ptr,
        )
        if not stream:
            abort("CUDA stream was not passed to MojoCallContext")
        return Stream(stream)

    @always_inline
    fn get_cuda_device(self) -> CUDADeviceStream:
        """Get the device context passed in."""
        var stream = self.get_stream()
        return CUDADeviceStream(stream)

    @always_inline
    fn set_to_error(self, err: Error):
        """Indicates to the C++ runtime that the kernel has failed."""
        var str = err.__str__()
        var strref = str._strref_dangerous()
        external_call[
            "KGEN_CompilerRT_LLCL_MojoCallContext_SetToError", NoneType
        ](self.ptr, strref.data, strref.length)
        str._strref_keepalive()

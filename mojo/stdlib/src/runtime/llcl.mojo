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

from builtin.coroutine import AnyCoroutine, _coro_resume_fn, _suspend_async
from gpu.host import Context as CudaContext
from gpu.host import CudaInstance, DeviceContext, Stream
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

    alias callback_fn_type = fn (Chain) -> None

    var callback: Self.callback_fn_type
    var chain: Chain

    @staticmethod
    fn get_chain(ctx: UnsafePointer[AsyncContext]) -> UnsafePointer[Chain]:
        return UnsafePointer.address_of(ctx[].chain)

    @staticmethod
    fn complete(ch: Chain):
        var tmp = ch
        _async_complete(UnsafePointer[Chain].address_of(tmp))
        _ = tmp


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
    type: AnyType
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

    @__named_result(task)
    fn create_task(
        self,
        owned handle: Coroutine[*_],
        desired_worker_id: Int = -1,
    ) -> Task[handle.type, handle.lifetimes]:
        """Run the coroutine as a task on the LLCL Runtime."""
        var ctx = handle._get_ctx[AsyncContext]()
        _init_llcl_chain(self, AsyncContext.get_chain(ctx))
        ctx[].callback = AsyncContext.complete
        task.__init__(handle^)
        _async_execute[handle.type](
            task._handle._handle, self, desired_worker_id
        )

    @always_inline
    @__named_result(out)
    fn run(self, owned handle: Coroutine[*_]) -> handle.type:
        var ctx = handle._get_ctx[AsyncContext]()
        _init_llcl_chain(self, AsyncContext.get_chain(ctx))
        ctx[].callback = AsyncContext.complete
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(out))
        handle._set_result_slot(UnsafePointer.address_of(out))
        _async_execute[handle.type](handle._handle, self, -1)
        _async_wait(AsyncContext.get_chain(ctx))
        _del_llcl_chain(AsyncContext.get_chain(ctx))
        _ = handle^

    @always_inline
    @__named_result(out)
    fn run(self, owned handle: RaisingCoroutine[*_]) raises -> handle.type:
        var ctx = handle._get_ctx[AsyncContext]()
        _init_llcl_chain(self, AsyncContext.get_chain(ctx))
        ctx[].callback = AsyncContext.complete
        handle._set_result_slot(
            __mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(out)),
            __mlir_op.`lit.ref.to_pointer`(
                __get_mvalue_as_litref(__get_nearest_error_slot())
            ),
        )
        _async_execute[handle.type](handle._handle, self, -1)
        _async_wait(AsyncContext.get_chain(ctx))
        _del_llcl_chain(AsyncContext.get_chain(ctx))
        if __mlir_op.`co.get_results`[_type = __mlir_type.i1](handle._handle):
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(__get_nearest_error_slot())
            )
            __mlir_op.`lit.raise`()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(out))
        _ = handle^


# ===----------------------------------------------------------------------===#
# Task
# ===----------------------------------------------------------------------===#


struct Task[type: AnyType, lifetimes: LifetimeSet]:
    var _handle: Coroutine[type, lifetimes]
    var _result: type

    fn __init__(inout self, owned handle: Coroutine[type, lifetimes]):
        self._handle = handle^
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._result)
        )
        self._handle._set_result_slot(UnsafePointer.address_of(self._result))

    fn get(self) -> ref [__lifetime_of(self)] type:
        """Get the task's result value. Calling this on an incomplete task is
        undefined behaviour."""
        return self._result

    fn __del__(owned self):
        """Destroy the memory associated with a task. This must be manually
        called when a task goes out of scope.
        """
        var ctx = self._handle._get_ctx[AsyncContext]()
        var chainPtr: UnsafePointer[Chain] = AsyncContext.get_chain(ctx)
        _del_llcl_chain(chainPtr)
        _ = self._handle^

    @always_inline
    fn __await__(self) -> ref [__lifetime_of(self)] type:
        """Suspend the current async function until the task completes and its
        result becomes available. This function must be force inlined into the
        calling async function.
        """

        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            _async_and_then(
                cur_hdl,
                AsyncContext.get_chain(self._handle._get_ctx[AsyncContext]()),
            )

        _suspend_async[await_body]()
        return self.get()

    fn wait(self) -> ref [__lifetime_of(self)] type:
        """Block the current thread until the future value becomes available."""
        _async_wait(
            AsyncContext.get_chain(self._handle._get_ctx[AsyncContext]())
        )
        return self.get()


# ===----------------------------------------------------------------------===#
# TaskGroup
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct TaskGroupContext[lifetimes: LifetimeSet]:
    alias tg_callback_fn_type = fn (inout TaskGroup[lifetimes]) -> None

    var callback: Self.tg_callback_fn_type
    var task_group: UnsafePointer[TaskGroup[lifetimes]]


@register_passable
struct _TaskGroupBox(CollectionElement):
    """This struct is a type-erased owning box for an opaque coroutine."""

    var handle: AnyCoroutine

    fn __init__[type: AnyType](inout self, owned coro: Coroutine[type]):
        var handle = coro._handle
        __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(coro))
        self.handle = handle

    fn __del__(owned self):
        __mlir_op.`co.destroy`(self.handle)

    # FIXME(MSTDL-573): `List` requires copyability. Just crash here because it
    # should never get called.
    fn __copyinit__(inout self, existing: Self):
        abort("_TaskGroupBox.__copyinit__ should never get called")
        while True:
            pass


struct TaskGroup[lifetimes: LifetimeSet]:
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
    fn _task_complete_callback(inout tg: TaskGroup[lifetimes]):
        tg._task_complete()

    fn _task_complete(inout self):
        if self._counter_decr() == 0:
            _async_complete(UnsafePointer[Chain].address_of(self.chain))

    fn create_task(
        inout self,
        owned task: Coroutine[NoneType],
        desired_worker_id: Int = -1,
    ):
        # TODO(MOCO-771): Enforce that `task.lifetimes` is a subset of
        # `Self.lifetimes`.
        self.counter += 1
        task._get_ctx[TaskGroupContext[lifetimes]]()[] = TaskGroupContext[
            lifetimes
        ] {
            callback: Self._task_complete_callback,
            task_group: UnsafePointer[Self].address_of(self),
        }
        _async_execute[task.type](task._handle, self.rt, desired_worker_id)
        self.tasks.append(_TaskGroupBox(task^))

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

    # Actually a MojoCallContext*
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
    fn get_cuda_device(self) -> DeviceContext:
        """Get the device context passed in."""
        var stream = self.get_stream()
        var ptr = external_call[
            "KGEN_CompilerRT_LLCL_MojoCallContext_GetCudaDevice",
            UnsafePointer[Tuple[CudaContext, CudaInstance]],
        ](
            self.ptr,
        )
        return DeviceContext(ptr[][1], ptr[][0], stream)

    @always_inline
    fn set_to_error(self, err: Error):
        """Indicates to the C++ runtime that the kernel has failed."""
        var str = err.__str__()
        var strref = str._strref_dangerous()
        external_call[
            "KGEN_CompilerRT_LLCL_MojoCallContext_SetToError", NoneType
        ](self.ptr, strref.data, strref.length)
        str._strref_keepalive()

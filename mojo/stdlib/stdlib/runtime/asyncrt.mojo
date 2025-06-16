# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""This module implements the low level concurrency library."""

from os import PathLike, abort
from os.atomic import Atomic
from sys import external_call
from sys.info import num_physical_cores
from sys.param_env import is_defined

from builtin.coroutine import AnyCoroutine, _coro_resume_fn, _suspend_async
from gpu.host import DeviceContext

from utils import StaticTuple

from .tracing import TraceLevel

# ===-----------------------------------------------------------------------===#
# _AsyncContext
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct _Chain(Boolable, Defaultable):
    """A proxy for the C++ runtime's AsyncValueRef<_Chain> type."""

    # Actually an AsyncValueRef<_Chain>, which is just an AsyncValue*
    var storage: UnsafePointer[Int]

    fn __init__(out self):
        self.storage = UnsafePointer[Int]()

    fn __bool__(self) -> Bool:
        return self.storage != UnsafePointer[Int]()


@register_passable("trivial")
struct _AsyncContext:
    """This struct models the coroutine context contained in every coroutine
    instance. The struct consists of a unary callback function that accepts a
    pointer argument. It is invoked with the second struct field, which is an
    opaque pointer. This struct is essentially a completion callback closure
    that is invoked by a coroutine when it completes and its results are made
    available.

    In async execution, a task's completion callback is to set its async token
    to available.
    """

    alias callback_fn_type = fn (_Chain) -> None

    var callback: Self.callback_fn_type
    var chain: _Chain

    @staticmethod
    fn get_chain(ctx: UnsafePointer[_AsyncContext]) -> UnsafePointer[_Chain]:
        return UnsafePointer(to=ctx[].chain)

    @staticmethod
    fn complete(ch: _Chain):
        var tmp = ch
        _async_complete(UnsafePointer(to=tmp))


# ===-----------------------------------------------------------------------===#
# AsyncRT C Shims
# ===-----------------------------------------------------------------------===#


fn _init_asyncrt_chain(chain: UnsafePointer[_Chain]):
    external_call["KGEN_CompilerRT_AsyncRT_InitializeChain", NoneType](
        chain.address
    )


fn _del_asyncrt_chain(chain: UnsafePointer[_Chain]):
    external_call["KGEN_CompilerRT_AsyncRT_DestroyChain", NoneType](
        chain.address
    )


fn _async_and_then(hdl: AnyCoroutine, chain: UnsafePointer[_Chain]):
    external_call["KGEN_CompilerRT_AsyncRT_AndThen", NoneType](
        _coro_resume_fn, chain.address, hdl
    )


fn _async_execute[type: AnyType](handle: AnyCoroutine, desired_worker_id: Int):
    external_call["KGEN_CompilerRT_AsyncRT_Execute", NoneType](
        _coro_resume_fn, handle, desired_worker_id
    )


fn _async_wait(chain: UnsafePointer[_Chain]):
    external_call["KGEN_CompilerRT_AsyncRT_Wait", NoneType](chain.address)


fn _async_complete(chain: UnsafePointer[_Chain]):
    external_call["KGEN_CompilerRT_AsyncRT_Complete", NoneType](chain.address)


fn _async_wait_timeout(chain: UnsafePointer[_Chain], timeout: Int) -> Bool:
    return external_call["KGEN_CompilerRT_AsyncRT_Wait_Timeout", Bool](
        chain.address, timeout
    )


# ===-----------------------------------------------------------------------===#
# Global Runtime
# ===-----------------------------------------------------------------------===#


@always_inline
fn parallelism_level() -> Int:
    """Gets the parallelism level of the Runtime.

    Returns:
        The number of worker threads available in the async runtime.
    """
    return Int(
        external_call[
            "KGEN_CompilerRT_AsyncRT_ParallelismLevel",
            Int32,
        ]()
    )


fn create_task(
    owned handle: Coroutine[*_], out task: Task[handle.type, handle.origins]
):
    """Run the coroutine as a task on the AsyncRT Runtime.

    This function creates a task from a coroutine and schedules it for execution
    on the async runtime. The task will execute asynchronously without blocking
    the current execution context.

    Args:
        handle: The coroutine to execute as a task. Ownership is transferred.

    Returns:
        The `task` output parameter is initialized with the created task.
    """
    var ctx = handle._get_ctx[_AsyncContext]()
    _init_asyncrt_chain(_AsyncContext.get_chain(ctx))
    ctx[].callback = _AsyncContext.complete
    task = Task(handle^)
    _async_execute[handle.type](task._handle._handle, desired_worker_id=-1)


@always_inline
fn _run(owned handle: Coroutine[*_], out result: handle.type):
    """Executes a coroutine and waits for its completion.
    This function runs the given coroutine on the async runtime and blocks until
    it completes. The result of the coroutine is stored in the output parameter.
    Args:
        handle: The coroutine to execute. Ownership is transferred.
    Returns:
        The `result` output parameter is initialized with the coroutine's result.
    """
    var ctx = handle._get_ctx[_AsyncContext]()
    _init_asyncrt_chain(_AsyncContext.get_chain(ctx))
    ctx[].callback = _AsyncContext.complete
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(result))
    handle._set_result_slot(UnsafePointer(to=result))
    _async_execute[handle.type](handle._handle, -1)
    _async_wait(_AsyncContext.get_chain(ctx))
    _del_asyncrt_chain(_AsyncContext.get_chain(ctx))
    handle^.force_destroy()


# ===-----------------------------------------------------------------------===#
# Task
# ===-----------------------------------------------------------------------===#


struct Task[type: AnyType, origins: OriginSet]:
    """Represents an asynchronous task that will produce a value of the specified type.

    A Task encapsulates a coroutine that is executing asynchronously and will eventually
    produce a result. Tasks can be awaited in async functions or waited on in synchronous code.

    Parameters:
        type: The type of value that this task will produce when completed.
        origins: The set of origins for the coroutine wrapped by this task.
    """

    var _handle: Coroutine[type, origins]
    """The underlying coroutine that executes the task."""

    var _result: type
    """Storage for the result value produced by the task."""

    @implicit
    fn __init__(out self, owned handle: Coroutine[type, origins]):
        """Initialize a task with a coroutine.

        Takes ownership of the provided coroutine and sets up the task to receive
        its result when completed.

        Args:
            handle: The coroutine to execute as a task. Ownership is transferred.
        """
        self._handle = handle^
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._result)
        )
        self._handle._set_result_slot(UnsafePointer(to=self._result))

    fn get(self) -> ref [self._result] type:
        """Get the task's result value. Calling this on an incomplete task is
        undefined behavior.

        Returns:
            A reference to the result value produced by the task.
        """
        return self._result

    fn __del__(owned self):
        """Destroy the memory associated with a task. This must be manually
        called when a task goes out of scope.
        """
        var ctx = self._handle._get_ctx[_AsyncContext]()
        _del_asyncrt_chain(_AsyncContext.get_chain(ctx))
        self._handle^.force_destroy()

    @always_inline
    fn __await__(self) -> ref [self.get()] type:
        """Suspend the current async function until the task completes and its
        result becomes available. This function must be force inlined into the
        calling async function.

        This method enables the use of the 'await' keyword with Task objects in
        async functions.

        Returns:
            A reference to the result value produced by the task.
        """

        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            _async_and_then(
                cur_hdl,
                _AsyncContext.get_chain(self._handle._get_ctx[_AsyncContext]()),
            )

        _suspend_async[await_body]()
        return self.get()

    fn wait(self) -> ref [self.get()] type:
        """Block the current thread until the future value becomes available.

        This method is used in synchronous code to wait for an asynchronous task
        to complete. Unlike `__await__`, this method does not suspend the current
        coroutine but instead blocks the entire thread.

        Returns:
            A reference to the result value produced by the task.
        """
        _async_wait(
            _AsyncContext.get_chain(self._handle._get_ctx[_AsyncContext]())
        )
        return self.get()


# ===-----------------------------------------------------------------------===#
# TaskGroup
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable("trivial")
struct TaskGroupContext(Copyable, Movable):
    """Context structure for task group operations.

    This structure holds a callback function and a pointer to a TaskGroup,
    allowing asynchronous operations to interact with their parent TaskGroup
    when they complete.
    """

    alias tg_callback_fn_type = fn (mut TaskGroup) -> None
    """Type definition for callback functions that operate on TaskGroups."""

    var callback: Self.tg_callback_fn_type
    """Callback function to be invoked on the TaskGroup when an operation completes."""

    var task_group: UnsafePointer[TaskGroup]
    """Pointer to the TaskGroup that owns or is associated with this context."""


@register_passable
struct _TaskGroupBox(Copyable, Movable):
    """This struct is a type-erased owning box for an opaque coroutine."""

    var handle: AnyCoroutine

    fn __init__[type: AnyType](out self, owned coro: Coroutine[type]):
        var handle = coro._handle
        __disable_del coro
        self.handle = handle

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    fn __del__(owned self):
        __mlir_op.`co.destroy`(self.handle)

    # FIXME(MSTDL-573): `List` requires copyability. Just crash here because it
    # should never get called.
    fn __copyinit__(out self, existing: Self):
        self = abort[Self]("_TaskGroupBox.__copyinit__ should never get called")

    # FIXME(MSTDL-573): `List` requires copyability. Just crash here because it
    # should never get called.
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return abort[Self]("_TaskGroupBox.copy should never get called")


struct TaskGroup(Defaultable):
    """A group of tasks that can be executed concurrently.

    TaskGroup manages a collection of coroutines that can be executed in parallel.
    It provides mechanisms to create, track, and wait for the completion of tasks.
    """

    var counter: Atomic[DType.index]
    """Atomic counter tracking the number of active tasks in the group."""

    var chain: _Chain
    """Chain used for asynchronous completion notification."""

    var tasks: List[_TaskGroupBox]
    """Collection of tasks managed by this TaskGroup."""

    fn __init__(out self):
        """Initialize a new TaskGroup with an empty task list and initialized chain.
        """
        var chain = _Chain()
        _init_asyncrt_chain(UnsafePointer[_Chain](to=chain))
        self.counter = Atomic[DType.index](1)
        self.chain = chain
        self.tasks = List[_TaskGroupBox](capacity=16)

    fn __del__(owned self):
        """Clean up resources associated with the TaskGroup."""
        _del_asyncrt_chain(UnsafePointer[_Chain](to=self.chain))

    @always_inline
    fn _counter_decr(mut self) -> Int:
        var prev: Int = self.counter.fetch_sub(1).value
        return prev - 1

    @staticmethod
    fn _task_complete_callback(mut tg: TaskGroup):
        tg._task_complete()

    fn _task_complete(mut self):
        if self._counter_decr() == 0:
            _async_complete(UnsafePointer[_Chain](to=self.chain))

    fn create_task(
        mut self,
        # FIXME(MSTDL-722): Avoid accessing ._mlir_type here, use `NoneType`.
        owned task: Coroutine[NoneType._mlir_type],
    ):
        """Add a new task to the TaskGroup for execution.

        Args:
            task: The coroutine to be executed as a task.
        """
        self._create_task(task^, desired_worker_id=-1)

    # Deprecated, use create_task() instead
    # Only sync_parallelize() uses this to pass desired_worker_id
    fn _create_task(
        mut self,
        # FIXME(MSTDL-722): Avoid accessing ._mlir_type here, use `NoneType`.
        owned task: Coroutine[NoneType._mlir_type],
        desired_worker_id: Int = -1,
    ):
        # TODO(MOCO-771): Enforce that `task.origins` is a subset of
        # `Self.origins`.
        self.counter += 1
        task._get_ctx[TaskGroupContext]()[] = TaskGroupContext(
            Self._task_complete_callback,
            UnsafePointer[Self](to=self),
        )
        _async_execute[task.type](task._handle, desired_worker_id)
        self.tasks.append(_TaskGroupBox(task^))

    @staticmethod
    fn await_body_impl(hdl: AnyCoroutine, mut task_group: Self):
        """Implementation of the await functionality for TaskGroup.

        Args:
            hdl: The coroutine handle to be awaited.
            task_group: The TaskGroup to be awaited.
        """
        _async_and_then(hdl, UnsafePointer[_Chain](to=task_group.chain))
        task_group._task_complete()

    @always_inline
    fn __await__(mut self):
        """Make TaskGroup awaitable in async contexts.

        This allows using 'await task_group' syntax in async functions.
        """

        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            Self.await_body_impl(cur_hdl, self)

        _suspend_async[await_body]()

    fn wait[origins: OriginSet = __origin_of()](mut self):
        """Wait for all tasks in the `TaskGroup` to complete.

        This is a blocking call that returns only when all tasks have finished.

        Parameters:
            origins: The origin set for the wait operation.
        """
        self._task_complete()
        _async_wait(UnsafePointer[_Chain](to=self.chain))


# ===-----------------------------------------------------------------------===#
# DeviceContext
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct DeviceContextPtr(Defaultable):
    """Exposes a pointer to a C++ DeviceContext to Mojo.

    Note: When initializing a `DeviceContext` from a pointer, the refcount is not
    incremented. This is considered safe because `get_device_context()`
    is only used within kernels and the `DeviceContext` lifetime is managed
    by the graph compiler.
    """

    var _handle: UnsafePointer[NoneType]
    """The underlying pointer to the C++ `DeviceContext`."""

    @always_inline
    fn __init__(out self):
        """Initialize an empty `DeviceContextPtr` with a null pointer.

        This creates a `DeviceContextPtr` that doesn't point to any device context.
        """
        self._handle = UnsafePointer[NoneType]()

    @implicit
    fn __init__(out self, handle: UnsafePointer[NoneType]):
        """Initialize a `DeviceContextPtr` from a raw pointer.

        Args:
            handle: A raw pointer to a C++ `DeviceContext`.
        """
        self._handle = handle

    @implicit
    fn __init__(out self, device: DeviceContext):
        """Initialize a DeviceContextPtr from a `DeviceContext`.

        This constructor allows implicit conversion from `DeviceContext` to `DeviceContextPtr`.

        Args:
            device: The `DeviceContext` to wrap in this pointer.
        """
        self._handle = rebind[UnsafePointer[NoneType]](device._handle)

    fn __getitem__(self) -> DeviceContext:
        """Dereference the pointer to get the `DeviceContext`.

        Returns:
            The `DeviceContext` that this pointer points to.
        """
        return DeviceContext(self._handle)

    fn get_device_context(self) -> DeviceContext:
        """Get the `DeviceContext` that this pointer points to.

        This is an alias for the dereference operator.

        Returns:
            The `DeviceContext` that this pointer points to.
        """
        return self[]


@register_passable("trivial")
struct DeviceContextPtrList[size: Int](Sized):
    """A fixed-size collection of `DeviceContextPtr` objects.

    This struct provides a lightweight, register-passable container for a fixed number
    of `DeviceContextPtr` objects, with array-like access semantics.

    Parameters:
        size: The fixed number of `DeviceContextPtr` objects in the collection.
    """

    var ptrs: StaticTuple[DeviceContextPtr, size]
    """The underlying storage for the device context pointers."""

    @always_inline
    fn __init__(out self, ptrs: StaticTuple[DeviceContextPtr, size]):
        """Initialize with a StaticTuple of `DeviceContextPtr` objects.

        Args:
            ptrs: A StaticTuple containing the `DeviceContextPtr` objects to store.
        """
        self.ptrs = ptrs

    fn __getitem__[index: Int](self) -> DeviceContext:
        """Access a `DeviceContext` at a compile-time known index.

        Parameters:
            index: A compile-time integer index.

        Returns:
            The `DeviceContext` at the specified index.
        """
        return self.ptrs[index][]

    fn __getitem__[I: Indexer, //](self, idx: I) -> DeviceContext:
        """Access a `DeviceContext` using a runtime index value.

        Parameters:
            I: A type that conforms to the `Indexer` trait.

        Args:
            idx: A runtime index value that conforms to the Indexer trait.

        Returns:
            The `DeviceContext` at the specified index.
        """
        return self.ptrs[idx][]

    fn __len__(self) -> Int:
        """Get the number of `DeviceContextPtr` objects in the collection.

        Returns:
            The size of the collection as specified by the size parameter.
        """
        return size

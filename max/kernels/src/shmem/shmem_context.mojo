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

from algorithm import parallelize
from collections.optional import OptionalReg
from os import abort, getenv, setenv
from sys import (
    CompilationTarget,
    argv,
    has_nvidia_gpu_accelerator,
    is_amd_gpu,
    is_nvidia_gpu,
    size_of,
)
from sys.ffi import c_int, c_size_t, external_call

from gpu.host import (
    ConstantMemoryMapping,
    DeviceAttribute,
    DeviceContext,
    DeviceEvent,
    DeviceFunction,
    DeviceStream,
    Dim,
    FuncAttribute,
    HostBuffer,
    LaunchAttribute,
)
from gpu.host.device_context import (
    _ConstCharPtr,
    _checked,
    _DeviceContextPtr,
    _DumpPath,
)
from gpu.host.launch_attribute import LaunchAttributeID, LaunchAttributeValue

from ._mpi import (
    MPI_Comm_rank,
    MPI_Comm_size,
    MPI_Finalize,
    MPI_Init,
    get_mpi_comm_world,
)
from .shmem_api import (
    SHMEM_TEAM_NODE,
    shmem_barrier_all_on_stream,
    shmem_finalize,
    shmem_init,
    shmem_init_thread,
    shmem_module_init,
    shmem_team_t,
)


fn shmem_launch[func: fn (ctx: SHMEMContext) raises]() raises:
    """Takes a function defining the entirety of the SHMEM program on a single
    PE (Processing Element).

    Parameters:
        func: The function to run once per attached GPU per node.

    ```mojo
    def simple_shift(ctx: SHMEMContext):
        var destination = ctx.enqueue_create_buffer[DType.int32](1)

        ctx.enqueue_function[simple_shift_kernel](
            destination.unsafe_ptr(), grid_dim=1, block_dim=1
        )

        ctx.barrier_all()

        var msg = Int32(0)
        destination.enqueue_copy_to(UnsafePointer(to=msg))

        ctx.synchronize()

        var mype = shmem_my_pe()
        print("PE:", mype, "received message:", msg)
        assert_equal(msg, (mype + 1) % shmem_n_pes())

    def main():
        shmem_launch[simple_shift]()
    ```

    This initializes SHMEM and runs the program in parallel across each attached
    GPU, taking care of initialization and cleanup logic.

    If running through `mpirun` or similar launchers via multi-node, this also
    handles that initialization logic.

    Any unhandled exceptions will abort with the device id and error message of
    the exception.
    ```
    """
    # Enable running across multiple nodes with `mpirun`
    var _argv = argv()
    var argc = len(_argv)
    MPI_Init(argc, _argv)

    # Get the amount of devices attached to this node
    var npes_node = DeviceContext.number_of_devices()

    # Enable any exceptions inside the closure passed to abort with the original
    # error and device ID in the message, as `parallelize` can't run on raising
    # functions.
    fn shmem_error_wrapper(device_id_node: Int) capturing:
        try:
            var ctx = DeviceContext(device_id=device_id_node)
            with SHMEMContext(ctx) as shmem_ctx:
                func(shmem_ctx)
        except e:
            abort(
                String(
                    "SHMEM failure on local device id: ",
                    device_id_node,
                    ": ",
                    e,
                )
            )

    # Same number of tasks as worker threads
    parallelize[shmem_error_wrapper](npes_node, npes_node)

    # Cleanup MPI resources
    MPI_Finalize()


struct SHMEMContext(ImplicitlyCopyable, Movable):
    """Usable as a context manager to run kernels on a GPU with SHMEM support,
    on exit it will finalize SHMEM and clean up resources.

    Example:

    ```mojo
    from shmem import SHMEMContext

    with SHMEMContext() as ctx:
        ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
    ```
    """

    var _ctx: DeviceContext
    var _main_stream: DeviceStream
    var _priority_stream: DeviceStream
    var _begin_event: DeviceEvent
    var _end_event: DeviceEvent
    var _multiprocessor_count: Int
    var _cooperative: Bool
    var _thread_per_gpu: Bool

    fn __init__(out self, team: shmem_team_t = SHMEM_TEAM_NODE) raises:
        """Initializes a device context with SHMEM support.

        This constructor sets up MPI, initializes SHMEM, and creates a device
        context for the current PE's assigned GPU device.

        Warning: if you're not using this as a context manager, you must call
        `SHMEMContext.finalize()` manually.

        Raises:
            If initialization fails.
        """
        constrained[
            has_nvidia_gpu_accelerator(),
            "SHMEMContext is currently only available on NVIDIA GPUs",
        ]()
        shmem_init()
        var mype = shmem_team_my_pe(team)
        self._ctx = DeviceContext(device_id=Int(mype))
        # Store main stream to avoid retrieving it in each collective launch.
        self._main_stream = self._ctx.stream()

        # Set up priority stream and events to be reused across collective launches
        var priority = self._ctx.stream_priority_range().greatest
        self._priority_stream = self._ctx.create_stream(
            priority=priority, blocking=False
        )
        self._begin_event = self._ctx.create_event()
        self._end_event = self._ctx.create_event()

        # Store attributes to avoid retrieving them in each collective launch.
        self._multiprocessor_count = self._ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        # TODO(MSTDL-1761): add ability to query AMD cooperative launch
        # capability with: hipLaunchAttributeCooperative and create function
        # that works across NVIDIA/AMD.
        self._cooperative = Bool(
            self._ctx.get_attribute(DeviceAttribute.COOPERATIVE_LAUNCH)
        )
        self._thread_per_gpu = False

    fn __init__(out self, ctx: DeviceContext) raises:
        """Initializes a device context with SHMEM support, using one thread
        per GPU.

        This constructor expects that MPI has already been initialized in the
        main thread, it then initializes SHMEM, and creates a device context for
        the associated PE on this node.

        Warning: if you're not using this as a context manager, you must call
        `SHMEMContext.finalize()` manually.

        Raises:
            If initialization fails.
        """
        constrained[
            has_nvidia_gpu_accelerator(),
            "SHMEMContext is currently only available on NVIDIA GPUs",
        ]()

        shmem_init_thread(ctx)
        self._ctx = ctx
        # Store main stream to avoid retrieving it in each collective launch.
        self._main_stream = self._ctx.stream()

        # Set up priority stream and events to be reused across collective launches
        var priority = self._ctx.stream_priority_range().greatest
        self._priority_stream = self._ctx.create_stream(
            priority=priority, blocking=False
        )
        self._begin_event = self._ctx.create_event()
        self._end_event = self._ctx.create_event()

        # Store attributes to avoid retrieving them in each collective launch.
        self._multiprocessor_count = self._ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        # TODO(MSTDL-1761): add ability to query AMD cooperative launch
        # capability with: hipLaunchAttributeCooperative and create function
        # that works across NVIDIA/AMD.
        self._cooperative = Bool(
            self._ctx.get_attribute(DeviceAttribute.COOPERATIVE_LAUNCH)
        )
        self._thread_per_gpu = True

    fn __enter__(var self) -> Self:
        """Context manager entry method.

        Returns:
            Self for use in with statements.
        """
        return self^

    fn __del__(deinit self):
        """Context manager exit method.

        Automatically finalizes SHMEM when exiting the context.
        """
        try:
            self.finalize()
        except e:
            abort(String(e))

    fn finalize(mut self) raises:
        """Finalizes the SHMEM runtime environment.

        Cleans up SHMEM and MPI resources.
        """
        shmem_finalize()
        if not self._thread_per_gpu:
            MPI_Finalize()

    fn barrier_all(self) raises:
        """Performs a barrier synchronization across all PEs.

        All PEs must call this function before any PE can proceed past the
        barrier.

        Raises:
            If the barrier operation fails.
        """
        shmem_barrier_all_on_stream(self._main_stream)

    fn enqueue_create_buffer[
        dtype: DType
    ](self, size: Int) raises -> SHMEMBuffer[dtype]:
        """Creates a SHMEM buffer that can be accessed by all PEs.

        Parameters:
            dtype: The data type of elements in the buffer.

        Args:
            size: Number of elements in the buffer.

        Returns:
            A SHMEMBuffer instance for the allocated memory.

        Raises:
            String: If buffer creation fails.
        """
        return SHMEMBuffer[dtype](self._ctx, size)

    fn enqueue_create_host_buffer[
        dtype: DType
    ](self, size: Int) raises -> HostBuffer[dtype]:
        """Enqueues the creation of a HostBuffer.

        This function allocates memory on the host that is accessible by the device.
        The memory is page-locked (pinned) for efficient data transfer between host and device.

        Pinned memory is guaranteed to remain resident in the host's RAM, not be
        paged/swapped out to disk. Memory allocated normally (for example, using
        [`UnsafePointer.alloc()`](/mojo/stdlib/memory/unsafe_ptr/UnsafePointer#alloc))
        is pageable—individual pages of memory can be moved to secondary storage
        (disk/SSD) when main memory fills up.

        Using pinned memory allows devices to make fast transfers
        between host memory and device memory, because they can use direct
        memory access (DMA) to transfer data without relying on the CPU.

        Allocating too much pinned memory can cause performance issues, since it
        reduces the amount of memory available for other processes.

        Parameters:
            dtype: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            A `HostBuffer` object that wraps the allocated host memory.

        Raises:
            If memory allocation fails or if the device context is invalid.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        with DeviceContext() as ctx:
            # Allocate host memory accessible by the device
            var host_buffer = ctx.enqueue_create_host_buffer[DType.float32](1024)

            # Use the host buffer for device operations
            # ...
        ```
        """
        var host_ptr = UnsafePointer[Scalar[dtype]].alloc(size)
        return HostBuffer[dtype](self._ctx, host_ptr, size, owning=True)

    @always_inline
    @parameter
    fn enqueue_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *Ts: AnyType,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            func_type: The dtype of the function to launch.
            func: The function to launch.
            Ts: The dtypes of the arguments being passed to the function.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from gpu.host import DeviceContext

        fn kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        with DeviceContext() as ctx:
            var compile_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```
        """
        var gpu_kernel = self._ctx.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=func_attribute)

        shmem_module_init(gpu_kernel)

        self._ctx._enqueue_function_unchecked(
            gpu_kernel,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

        shmem_module_finalize(gpu_kernel)

    @always_inline
    @parameter
    fn enqueue_function_collective[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *Ts: AnyType,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            func_type: The dtype of the function to launch.
            func: The function to launch.
            Ts: The dtypes of the arguments being passed to the function.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from gpu.host import DeviceContext

        fn kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        with DeviceContext() as ctx:
            var compile_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```
        """
        var gpu_kernel = self._ctx.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=func_attribute)
        shmem_module_init(gpu_kernel)

        var block_size = block_dim[0] * block_dim[1] * block_dim[2]
        var shared_mem_bytes_val = (
            shared_mem_bytes.value() if shared_mem_bytes else 0
        )
        var max_blocks_sm = (
            gpu_kernel.occupancy_max_active_blocks_per_multiprocessor(
                block_size, shared_mem_bytes_val
            )
        )
        var grid_size = -1
        var launch_failed = True

        var grid_x = grid_dim[0]
        var grid_y = grid_dim[1]
        var grid_z = grid_dim[2]
        if grid_x == 0 and grid_y == 0 and grid_z == 0:
            grid_size = 0
        elif grid_x != 0 and grid_y != 0 and grid_z != 0:
            grid_size = grid_x * grid_y * grid_z

        if grid_size == 0:
            if max_blocks_sm == 0:
                launch_failed = False
            grid_x = max_blocks_sm * self._multiprocessor_count
            grid_y = 1
            grid_z = 1
        elif grid_size > 0:
            if (
                max_blocks_sm > 0
                and grid_size <= max_blocks_sm * self._multiprocessor_count
            ):
                launch_failed = False

        if launch_failed:
            raise Error(
                "One or more GPUs cannot collectively launch the kernel"
            )

        # Mark point in main stream and wait for it to complete in priority stream
        self._main_stream.record_event(self._begin_event)
        self._priority_stream.enqueue_wait_for(self._begin_event)

        if self._cooperative:
            attributes.append(
                LaunchAttribute(
                    id=LaunchAttributeID.COOPERATIVE,
                    value=LaunchAttributeValue(True),
                )
            )
        else:
            print(
                "Warning: cooperative launch not supported on at least one PE;"
                " GPU-side synchronization may cause hang"
            )
        self._priority_stream._enqueue_function_unchecked(
            gpu_kernel,
            args,
            grid_dim=Dim(grid_x, grid_y, grid_z),
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )
        # Mark point in priority stream and wait for it to complete in main stream
        self._priority_stream.record_event(self._end_event)
        self._main_stream.enqueue_wait_for(self._end_event)
        shmem_module_finalize(gpu_kernel)

    @always_inline
    fn synchronize(self) raises:
        """Blocks until all asynchronous calls on the stream associated with
        this device context have completed.

        Raises:
            If synchronization fails.
        """
        # const char * AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx)
        self._ctx.synchronize()

    @always_inline
    fn get_device_context(self) -> DeviceContext:
        """Returns the device context associated with this SHMEMContext.

        Returns:
            The device context associated with this SHMEMContext.
        """
        return self._ctx

    @staticmethod
    @always_inline
    fn number_of_devices(
        *, api: String = String(DeviceContext.default_device_info.api)
    ) -> Int:
        """Returns the number of devices available that support the specified API.

        This function queries the system for available devices that support the
        requested API (such as CUDA or HIP). It's useful for determining how many
        accelerators are available before allocating resources or distributing work.

        Args:
            api: Requested device API (for example, "cuda" or "hip"). Defaults
                to the device API specified by current target accelerator.

        Returns:
            The number of available devices supporting the specified API.

        """
        return DeviceContext.number_of_devices(api=api)

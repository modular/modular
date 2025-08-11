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


from collections.optional import OptionalReg
from gpu.host import (
    DeviceContext,
    ConstantMemoryMapping,
    DeviceFunction,
    LaunchAttribute,
    FuncAttribute,
    Dim,
)
from gpu.host._nvidia_cuda import CUDA, CUDA_MODULE, CUstream, CUcontext
from gpu.host.device_context import _DumpPath
from os import getenv, setenv
from sys import (
    CompilationTarget,
    is_amd_gpu,
    has_nvidia_gpu_accelerator,
    sizeof,
    argv,
)
from sys.ffi import c_int, external_call

from ._mpi import (
    get_mpi_comm_world,
    MPI_Init,
    MPI_Comm_rank,
    MPI_Comm_size,
    MPI_Finalize,
)
from ._nvshmem_host import (
    NVSHMEMX_INIT_WITH_MPI_COMM,
    NVSHMEMX_TEAM_NODE,
    NVSHMEMXInitAttr,
    nvshmemx_init_attr,
    nvshmemid_init_status,
    nvshmemx_init_status,
    nvshmem_malloc,
    nvshmem_calloc,
    nvshmemx_cumodule_init,
    nvshmemx_barrier_all_on_stream,
    nvshmemx_hostlib_finalize,
    nvshmem_team_my_pe,
    nvshmem_free,
)

alias SHMEM_TEAM_MODE: Int = NVSHMEMX_TEAM_NODE


fn shmem_init() raises:
    """Initializes the SHMEM (Shared Memory) runtime environment.

    This function sets up the MPI communication environment and initializes
    SHMEM for GPU-to-GPU communication. It must be called before any other
    SHMEM operations.

    Raises:
        If SHMEM initialization fails.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        var _argv = argv()
        var argc = len(_argv)
        var mpi_status = MPI_Init(argc, _argv)

        # Get MPI rank and size
        var rank = c_int(0)
        var mpi_comm = get_mpi_comm_world()

        _ = MPI_Comm_rank(mpi_comm, UnsafePointer(to=rank))
        # Set CUDA device early - needed for CUDA-related NVSHMEM initialization
        _ = DeviceContext(device_id=Int(rank)).set_as_current()

        # Initialize NVSHMEM with MPI
        var attr = NVSHMEMXInitAttr(UnsafePointer(to=mpi_comm))
        _ = nvshmemx_init_attr(
            NVSHMEMX_INIT_WITH_MPI_COMM, UnsafePointer(to=attr)
        )

        # Check initialization status
        var internal_status = nvshmemid_init_status()
        var public_status = nvshmemx_init_status()
        if not internal_status == 2 or not public_status == 2:
            raise String("failed to initialize shmem")
    else:
        return CompilationTarget.unsupported_target_error[
            operation="shmem_init"
        ]()


fn shmem_malloc[dtype: DType](size: UInt) -> UnsafePointer[Scalar[dtype]]:
    """Allocates memory in the symmetric heap that is accessible by all PEs
    (Processing Elements).

    Parameters:
        dtype: The data type of elements to allocate memory for.

    Args:
        size: Number of elements to allocate.

    Returns:
        UnsafePointer to the allocated memory.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        return nvshmem_malloc[dtype](sizeof[dtype]() * size)
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_malloc"]()
        return UnsafePointer[Scalar[dtype]]()


fn shmem_calloc[
    dtype: DType
](count: UInt, size: UInt = sizeof[dtype]()) -> UnsafePointer[Scalar[dtype]]:
    """Allocates and zero-initializes memory in the symmetric heap.

    Parameters:
        dtype: The data type of elements to allocate memory for.

    Args:
        count: Number of elements to allocate.
        size: Size in bytes of each element (defaults to sizeof[dtype]()).

    Returns:
        UnsafePointer to the zero-initialized allocated memory.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        return nvshmem_calloc[dtype](count, size)
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_calloc"]()
        return {}


fn shmem_module_init(device_function: DeviceFunction) raises:
    """Initializes device state for SHMEM operations on a compiled function.

    This must be called for each device function that will use SHMEM operations
    before the function is launched.

    Args:
        device_function: The compiled device function to initialize with NVSHMEM.

    Raises:
        String: If module initialization fails.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        var func = CUDA_MODULE(device_function)
        _ = nvshmemx_cumodule_init(func)
    else:
        CompilationTarget.unsupported_target_error[
            operation="shmem_cumodule_init",
        ]()


fn shmem_barrier_all_on_stream(ctx: DeviceContext) raises:
    """Performs a barrier synchronization across all PEs.

    All PEs must call this function before any PE can proceed past the barrier.

    Args:
        ctx: The device context whose stream to perform the barrier on.

    Raises:
        If the barrier operation fails.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        var cuda_stream = CUDA(ctx.stream())
        nvshmemx_barrier_all_on_stream(cuda_stream)
    else:
        return CompilationTarget.unsupported_target_error[
            operation="shmem_barrier_all_on_stream",
        ]()


fn shmem_barrier_all_on_stream(cuda_stream: CUstream) raises:
    """Performs a barrier synchronization across all PEs.

    All PEs must call this function before any PE can proceed past the barrier.

    Args:
        cuda_stream: The device context whose stream to perform the barrier on.

    Raises:
        If the barrier operation fails.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        nvshmemx_barrier_all_on_stream(cuda_stream)
    else:
        return CompilationTarget.unsupported_target_error[
            operation="shmem_barrier_all_on_stream",
        ]()


fn shmem_free[dtype: DType](ptr: UnsafePointer[Scalar[dtype]]):
    """Frees memory that was allocated in the symmetric heap.

    Parameters:
        dtype: The data type of the memory being freed.

    Args:
        ptr: Pointer to the memory to free.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        nvshmem_free(ptr)
    else:
        return CompilationTarget.unsupported_target_error[
            operation="shmem_free",
        ]()


fn shmem_finalize():
    """Finalizes the SHMEM runtime environment.

    This function cleans up NVSHMEM resources and finalizes MPI. Should be
    called before program termination.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        nvshmemx_hostlib_finalize()
        _ = MPI_Finalize()
    else:
        return CompilationTarget.unsupported_target_error[
            operation="shmem_finalize",
        ]()


fn shmem_team_my_pe(team: Int) -> Int:
    """Returns the PE number of the calling PE within the specified team.

    Args:
        team: The team identifier (e.g., SHMEM_TEAM_MODE for node team).

    Returns:
        The PE number within the team, or 0 on unsupported targets.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        return Int(nvshmem_team_my_pe(c_int(team)))
    else:
        CompilationTarget.unsupported_target_error[
            operation="shmem_team_my_pe",
        ]()
        return 0


struct DeviceContextSHMEM:
    """Usable as a context manager to run kernels on a GPU with SHMEM support,
    on exit it will finalize SHMEM and clean up resources.

    Example:

    ```mojo
    from shmem import DeviceContextSHMEM

    with DeviceContextSHMEM() as ctx:
        ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
    ```
    """

    var _ctx: DeviceContext

    fn __init__(out self) raises:
        """Initializes a device context with SHMEM support.

        This constructor sets up MPI, initializes NVSHMEM, and creates a device
        context for the current PE's assigned GPU device.

        Raises:
            If initialization fails.
        """
        var _argv = argv()
        var argc = len(_argv)
        var mpi_status = MPI_Init(argc, _argv)

        # Get MPI rank and size
        var rank = c_int(0)
        var mpi_comm = get_mpi_comm_world()

        _ = MPI_Comm_rank(mpi_comm, UnsafePointer(to=rank))
        self._ctx = DeviceContext(device_id=Int(rank))
        self._ctx.set_as_current()

        # Initialize NVSHMEM with MPI
        var attr = NVSHMEMXInitAttr(UnsafePointer(to=mpi_comm))
        _ = nvshmemx_init_attr(
            NVSHMEMX_INIT_WITH_MPI_COMM, UnsafePointer(to=attr)
        )

        # Check initialization status
        var internal_status = nvshmemid_init_status()
        var public_status = nvshmemx_init_status()
        if not internal_status == 2 or not public_status == 2:
            raise String("failed to initialize shmem")

    fn team_my_pe(self, mode: Int = SHMEM_TEAM_MODE) -> Int:
        """Returns the PE number of this PE within the specified team.

        Args:
            mode: The team mode (defaults to SHMEM_TEAM_MODE for node team).

        Returns:
            The PE number within the team.
        """
        return shmem_team_my_pe(mode)

    fn n_pes(self) -> Int:
        """Returns the total number of PEs in the job.

        Returns:
            The total number of processing elements.
        """
        return Int(shmem_n_pes())

    fn my_pe(self) -> Int:
        """Returns the PE number of the calling PE.

        Returns:
            The PE number (process rank) of this processing element.
        """
        return Int(shmem_my_pe())

    fn __enter__(self) -> Self:
        """Context manager entry method.

        Returns:
            Self for use in with statements.
        """
        return self

    fn __exit__(mut self):
        """Context manager exit method.

        Automatically finalizes SHMEM when exiting the context.
        """
        self.finalize()

    fn finalize(mut self):
        """Finalizes the SHMEM runtime environment.

        Cleans up NVSHMEM and MPI resources.
        """
        shmem_finalize()

    fn __copyinit__(out self, other: Self):
        """Copy constructor for DeviceContextSHMEM.

        Args:
            other: The instance to copy from.
        """
        self._ctx = other._ctx

    fn barrier_all(self) raises:
        """Performs a barrier synchronization across all PEs.

        All PEs must call this function before any PE can proceed past the
        barrier.

        Raises:
            If the barrier operation fails.
        """
        shmem_barrier_all_on_stream(self._ctx)

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

    @always_inline
    fn synchronize(self) raises:
        """Blocks until all asynchronous calls on the stream associated with
        this device context have completed.

        Raises:
            If synchronization fails.
        """
        # const char * AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx)
        self._ctx.synchronize()

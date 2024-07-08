# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the result type."""

# ===----------------------------------------------------------------------===#
# Result
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Result(Stringable, EqualityComparable):
    var code: Int32

    alias SUCCESS = Self(0)
    """The API call returned with no errors. In the case of query calls, this
    also means that the operation being queried is complete (see
    ::cuEventQuery() and ::cuStreamQuery()).
    """

    alias INVALID_VALUE = Self(1)
    """This indicates that one or more of the parameters passed to the API call
    is not within an acceptable range of values.
    """

    alias OUT_OF_MEMORY = Self(2)
    """The API call failed because it was unable to allocate enough memory to
    perform the requested operation.
    """

    alias NOT_INITIALIZED = Self(3)
    """This indicates that the CUDA driver has not been initialized with
    ::cuInit() or that initialization has failed.
    """

    alias DEINITIALIZED = Self(4)
    """This indicates that the CUDA driver is in the process of shutting down.
    """

    alias PROFILER_DISABLED = Self(5)
    """This indicates profiler is not initialized for this run. This can
    happen when the application is running with external profiling tools
    like visual profiler.
    """

    alias PROFILER_NOT_INITIALIZED = Self(6)
    """This error return is deprecated as of CUDA 5.0. It is no longer an error
    to attempt to enable/disable the profiling via ::cuProfilerStart or
    ::cuProfilerStop without initialization.
    """

    alias PROFILER_ALREADY_STARTED = Self(7)
    """This error return is deprecated as of CUDA 5.0. It is no longer an error
    to call cuProfilerStart() when profiling is already enabled.
    """

    alias PROFILER_ALREADY_STOPPED = Self(8)
    """This error return is deprecated as of CUDA 5.0. It is no longer an error
    to call cuProfilerStop() when profiling is already disabled.
    """

    alias STUB_LIBRARY = Self(34)
    """This indicates that the CUDA driver that the application has loaded is a
    stub library. Applications that run with the stub rather than a real
    driver loaded will result in CUDA API returning this error.
    """

    alias DEVICE_UNAVAILABLE = Self(46)
    """This indicates that requested CUDA device is unavailable at the current
    time. Devices are often unavailable due to use of
    ::CU_COMPUTEMODE_EXCLUSIVE_PROCESS or ::CU_COMPUTEMODE_PROHIBITED.
    """

    alias NO_DEVICE = Self(100)
    """This indicates that no CUDA-capable devices were detected by the
    installed CUDA driver.
    """

    alias INVALID_DEVICE = Self(101)
    """This indicates that the device ordinal supplied by the user does not
    correspond to a valid CUDA device or that the action requested is
    invalid for the specified device.
    """

    alias DEVICE_NOT_LICENSED = Self(102)
    """This error indicates that the Grid license is not applied.
    """

    alias INVALID_IMAGE = Self(200)
    """This indicates that the device kernel image is invalid. This can also
    indicate an invalid CUDA module.
    """

    alias INVALID_CONTEXT = Self(201)
    """This most frequently indicates that there is no context bound to the
    current thread. This can also be returned if the context passed to an
    API call is not a valid handle (such as a context that has had
    ::cuCtxDestroy() invoked on it). This can also be returned if a user
    mixes different API versions (i.e. 3010 context with 3020 API calls).
    See ::cuCtxGetApiVersion() for more details.
    """

    alias CONTEXT_ALREADY_CURRENT = Self(202)
    """This indicated that the context being supplied as a parameter to the
    API call was already the active context.
    [[depricated]]
    This error return is deprecated as of CUDA 3.2. It is no longer an
    error to attempt to push the active context via ::cuCtxPushCurrent().
    """

    alias MAP_FAILED = Self(205)
    """This indicates that a map or register operation has failed.
    """

    alias UNMAP_FAILED = Self(206)
    """This indicates that an unmap or unregister operation has failed.
    """

    alias ARRAY_IS_MAPPED = Self(207)
    """This indicates that the specified array is currently mapped and thus
    cannot be destroyed.
    """

    alias ALREADY_MAPPED = Self(208)
    """This indicates that the resource is already mapped.
    """

    alias NO_BINARY_FOR_GPU = Self(209)
    """This indicates that there is no kernel image available that is suitable
    for the device. This can occur when a user specifies code generation
    options for a particular CUDA source file that do not include the
    corresponding device configuration.
    """

    alias ALREADY_ACQUIRED = Self(210)
    """This indicates that a resource has already been acquired.
    """

    alias NOT_MAPPED = Self(211)
    """This indicates that a resource is not mapped.
    """

    alias NOT_MAPPED_AS_ARRAY = Self(212)
    """This indicates that a mapped resource is not available for access as an
    array.
    """

    alias NOT_MAPPED_AS_POINTER = Self(213)
    """This indicates that a mapped resource is not available for access as a
    pointer.
    """

    alias ECC_UNCORRECTABLE = Self(214)
    """This indicates that an uncorrectable ECC error was detected during
    execution.
    """

    alias UNSUPPORTED_LIMIT = Self(215)
    """This indicates that the ::CUlimit passed to the API call is not
    supported by the active device.
    """

    alias CONTEXT_ALREADY_IN_USE = Self(216)
    """This indicates that the ::CUcontext passed to the API call can
    only be bound to a single CPU thread at a time but is already
    bound to a CPU thread.
    """

    alias PEER_ACCESS_UNSUPPORTED = Self(217)
    """This indicates that peer access is not supported across the given
    devices.
    """

    alias INVALID_PTX = Self(218)
    """This indicates that a PTX JIT compilation failed.
    """

    alias INVALID_GRAPHICS_CONTEXT = Self(219)
    """This indicates an error with OpenGL or DirectX context.
    """

    alias NVLINK_UNCORRECTABLE = Self(220)
    """This indicates that an uncorrectable NVLink error was detected during the
    execution.
    """

    alias JIT_COMPILER_NOT_FOUND = Self(221)
    """This indicates that the PTX JIT compiler library was not found.
    """

    alias UNSUPPORTED_PTX_VERSION = Self(222)
    """This indicates that the provided PTX was compiled with an unsupported
    toolchain.
    """

    alias JIT_COMPILATION_DISABLED = Self(223)
    """This indicates that the PTX JIT compilation was disabled.
    """

    alias UNSUPPORTED_EXEC_AFFINITY = Self(224)
    """This indicates that the ::CUexecAffinityType passed to the API call is
    not supported by the active device.
    """

    alias UNSUPPORTED_DEVSIDE_SYNC = Self(225)
    """This indicates that the code to be compiled by the PTX JIT contains
    unsupported call to cudaDeviceSynchronize.
    """

    alias INVALID_SOURCE = Self(300)
    """This indicates that the device kernel source is invalid. This includes
    compilation/linker errors encountered in device code or user error.
    """

    alias FILE_NOT_FOUND = Self(301)
    """This indicates that the file specified was not found.
    """

    alias SHARED_OBJECT_SYMBOL_NOT_FOUND = Self(302)
    """This indicates that a link to a shared object failed to resolve.
    """

    alias SHARED_OBJECT_INIT_FAILED = Self(303)
    """This indicates that initialization of a shared object failed.
    """

    alias OPERATING_SYSTEM = Self(304)
    """This indicates that an OS call failed.
    """

    alias INVALID_HANDLE = Self(400)
    """This indicates that a resource handle passed to the API call was not
    valid. Resource handles are opaque types like ::CUstream and ::CUevent.
    """

    alias ILLEGAL_STATE = Self(401)
    """This indicates that a resource required by the API call is not in a
    valid state to perform the requested operation.
    """

    alias NOT_FOUND = Self(500)
    """This indicates that a named symbol was not found. Examples of symbols
    are global/constant variable names, driver function names, texture names,
    and surface names.
    """

    alias NOT_READY = Self(600)
    """This indicates that asynchronous operations issued previously have not
    completed yet. This result is not actually an error, but must be indicated
    differently than ::SUCCESS (which indicates completion). Calls that
    may return this value include ::cuEventQuery() and ::cuStreamQuery().
    """

    alias ILLEGAL_ADDRESS = Self(700)
    """While executing a kernel, the device encountered a
    load or store instruction on an invalid memory address.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias LAUNCH_OUT_OF_RESOURCES = Self(701)
    """This indicates that a launch did not occur because it did not have
    appropriate resources. This error usually indicates that the user has
    attempted to pass too many arguments to the device kernel, or the
    kernel launch specifies too many threads for the kernel's register
    count. Passing arguments of the wrong size (i.e. a 64-bit pointer
    when a 32-bit int is expected) is equivalent to passing too many
    arguments and can also result in this error.
    """

    alias LAUNCH_TIMEOUT = Self(702)
    """This indicates that the device kernel took too long to execute. This can
    only occur if timeouts are enabled - see the device attribute
    ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias LAUNCH_INCOMPATIBLE_TEXTURING = Self(703)
    """This error indicates a kernel launch that uses an incompatible texturing
    mode.
    """

    alias PEER_ACCESS_ALREADY_ENABLED = Self(704)
    """This error indicates that a call to ::cuCtxEnablePeerAccess() is
    trying to re-enable peer access to a context which has already
    had peer access to it enabled.
    """

    alias PEER_ACCESS_NOT_ENABLED = Self(705)
    """This error indicates that ::cuCtxDisablePeerAccess() is
    trying to disable peer access which has not been enabled yet
    via ::cuCtxEnablePeerAccess().
    """

    alias PRIMARY_CONTEXT_ACTIVE = Self(708)
    """This error indicates that the primary context for the specified device
    has already been initialized.
    """

    alias CONTEXT_IS_DESTROYED = Self(709)
    """This error indicates that the context current to the calling thread
    has been destroyed using ::cuCtxDestroy, or is a primary context which
    has not yet been initialized.
    """

    alias ASSERT = Self(710)
    """A device-side assert triggered during kernel execution. The context
    cannot be used anymore, and must be destroyed. All existing device
    memory allocations from this context are invalid and must be
    reconstructed if the program is to continue using CUDA.
    """

    alias TOO_MANY_PEERS = Self(711)
    """This error indicates that the hardware resources required to enable
    peer access have been exhausted for one or more of the devices
    passed to ::cuCtxEnablePeerAccess().
    """

    alias HOST_MEMORY_ALREADY_REGISTERED = Self(712)
    """This error indicates that the memory range passed to ::cuMemHostRegister
    has already been registered.
    """

    alias HOST_MEMORY_NOT_REGISTERED = Self(713)
    """This error indicates that the pointer passed to ::cuMemHostUnregister()
    does not correspond to any currently registered memory region.
    """

    alias HARDWARE_STACK_ERROR = Self(714)
    """While executing a kernel, the device encountered a stack error.
    This can be due to stack corruption or exceeding the stack size limit.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias ILLEGAL_INSTRUCTION = Self(715)
    """While executing a kernel, the device encountered an illegal instruction.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias MISALIGNED_ADDRESS = Self(716)
    """While executing a kernel, the device encountered a load or store
    instruction on a memory address which is not aligned.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias INVALID_ADDRESS_SPACE = Self(717)
    """While executing a kernel, the device encountered an instruction
    which can only operate on memory locations in certain address spaces
    (global, shared, or local), but was supplied a memory address not belonging
    to an allowed address space.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias INVALID_PC = Self(718)
    """While executing a kernel, the device program counter wrapped its address
    space. This leaves the process in an inconsistent state and any further CUDA
    work will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias LAUNCH_FAILED = Self(719)
    """An exception occurred on the device while executing a kernel. Common
    causes include dereferencing an invalid device pointer and accessing
    out of bounds shared memory. Less common cases can be system specific - more
    information about these cases can be found in the system specific user guide.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias COOPERATIVE_LAUNCH_TOO_LARGE = Self(720)
    """This error indicates that the number of blocks launched per grid for a
    kernel that was launched via either ::cuLaunchCooperativeKernel or
    ::cuLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks
    as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor or
    ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of
    multiprocessors as specified by the device attribute
    ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
    """

    alias NOT_PERMITTED = Self(800)
    """This error indicates that the attempted operation is not permitted.
    """

    alias NOT_SUPPORTED = Self(801)
    """This error indicates that the attempted operation is not supported
    on the current system or device.
    """

    alias SYSTEM_NOT_READY = Self(802)
    """This error indicates that the system is not yet ready to start any CUDA
    work.  To continue using CUDA, verify the system configuration is in a
    valid state and all required driver daemons are actively running.
    More information about this error can be found in the system specific
    user guide.
    """

    alias SYSTEM_DRIVER_MISMATCH = Self(803)
    """This error indicates that there is a mismatch between the versions of
    the display driver and the CUDA driver. Refer to the compatibility
    documentation for supported versions.
    """

    alias COMPAT_NOT_SUPPORTED_ON_DEVICE = Self(804)
    """This error indicates that the system was upgraded to run with forward
    compatibility ut the visible hardware detected by CUDA does not support this
    configuration. Refer to the compatibility documentation for the supported
    hardware matrix or ensure that only supported hardware is visible during
    initialization via the CUDA_VISIBLE_DEVICES environment variable.
    """

    alias MPS_CONNECTION_FAILED = Self(805)
    """This error indicates that the MPS client failed to connect to the MPS
    control daemon or the MPS server.
    """

    alias MPS_RPC_FAILURE = Self(806)
    """This error indicates that the remote procedural call between the MPS
    server and the MPS client failed.
    """

    alias MPS_SERVER_NOT_READY = Self(807)
    """This error indicates that the MPS server is not ready to accept new MPS
    client requests. This error can be returned when the MPS server is in the
    process of recovering from a fatal failure.
    """

    alias MPS_MAX_CLIENTS_REACHED = Self(808)
    """This error indicates that the hardware resources required to create MPS
    client have been exhausted.
    """

    alias MPS_MAX_CONNECTIONS_REACHED = Self(809)
    """This error indicates the the hardware resources required to support
    device connections have been exhausted.
    """

    alias MPS_CLIENT_TERMINATED = Self(810)
    """This error indicates that the MPS client has been terminated by the
    server. To continue using CUDA, the process must be terminated and
    relaunched.
    """

    alias CDP_NOT_SUPPORTED = Self(811)
    """This error indicates that the module is using CUDA Dynamic Parallelism,
    but the current configuration, like MPS, does not support it.
    """

    alias CDP_VERSION_MISMATCH = Self(812)
    """This error indicates that a module contains an unsupported interaction
    between different versions of CUDA Dynamic Parallelism.
    """

    alias STREAM_CAPTURE_UNSUPPORTED = Self(900)
    """This error indicates that the operation is not permitted when
    the stream is capturing.
    """

    alias STREAM_CAPTURE_INVALIDATED = Self(901)
    """This error indicates that the current capture sequence on the stream
    has been invalidated due to a previous error.
    """

    alias STREAM_CAPTURE_MERGE = Self(902)
    """This error indicates that the operation would have resulted in a merge
    of two independent capture sequences.
    """

    alias STREAM_CAPTURE_UNMATCHED = Self(903)
    """This error indicates that the capture was not initiated in this stream.
    """

    alias STREAM_CAPTURE_UNJOINED = Self(904)
    """This error indicates that the capture sequence contains a fork that was
    not joined to the primary stream.
    """

    alias STREAM_CAPTURE_ISOLATION = Self(905)
    """This error indicates that a dependency would have been created which
    crosses the capture sequence boundary. Only implicit in-stream ordering
    dependencies are allowed to cross the boundary.
    """

    alias STREAM_CAPTURE_IMPLICIT = Self(906)
    """This error indicates a disallowed implicit dependency on a current
    capture sequence from cudaStreamLegacy.
    """

    alias CAPTURED_EVENT = Self(907)
    """This error indicates that the operation is not permitted on an event
    which was last recorded in a capturing stream.
    """

    alias STREAM_CAPTURE_WRONG_THREAD = Self(908)
    """A stream capture sequence not initiated with the
    ::CU_STREAM_CAPTURE_MODE_RELAXED argument to ::cuStreamBeginCapture was
    passed to ::cuStreamEndCapture in a different thread.
    """

    alias TIMEOUT = Self(909)
    """This error indicates that the timeout specified for the wait operation
    has lapsed.
    """

    alias GRAPH_EXEC_UPDATE_FAILURE = Self(910)
    """This error indicates that the graph update was not performed because it
    included changes which violated constraints specific to instantiated graph
    update.
    """

    alias EXTERNAL_DEVICE = Self(911)
    """This indicates that an async error has occurred in a device outside of
    CUDA. If CUDA was waiting for an external device's signal before consuming
    shared data, the external device signaled an error indicating that the data
    is not valid for consumption. This leaves the process in an inconsistent
    state and any further CUDA work will return the same error. To continue
    using CUDA, the process must be terminated and relaunched.
    """

    alias INVALID_CLUSTER_SIZE = Self(912)
    """Indicates a kernel launch error due to cluster misconfiguration.
    """

    alias UNKNOWN = Self(999)
    """This indicates that an unknown internal error has occurred.
    """

    @always_inline("nodebug")
    fn __init__(inout self, code: Int32):
        self.code = code

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        if self == Result.SUCCESS:
            return "SUCCESS"
        elif self == Result.INVALID_VALUE:
            return "INVALID_VALUE"
        elif self == Result.OUT_OF_MEMORY:
            return "OUT_OF_MEMORY"
        elif self == Result.NOT_INITIALIZED:
            return "NOT_INITIALIZED"
        elif self == Result.DEINITIALIZED:
            return "DEINITIALIZED"
        elif self == Result.PROFILER_DISABLED:
            return "PROFILER_DISABLED"
        elif self == Result.PROFILER_NOT_INITIALIZED:
            return "PROFILER_NOT_INITIALIZED"
        elif self == Result.PROFILER_ALREADY_STARTED:
            return "PROFILER_ALREADY_STARTED"
        elif self == Result.PROFILER_ALREADY_STOPPED:
            return "PROFILER_ALREADY_STOPPED"
        elif self == Result.STUB_LIBRARY:
            return "STUB_LIBRARY"
        elif self == Result.DEVICE_UNAVAILABLE:
            return "DEVICE_UNAVAILABLE"
        elif self == Result.NO_DEVICE:
            return "NO_DEVICE"
        elif self == Result.INVALID_DEVICE:
            return "INVALID_DEVICE"
        elif self == Result.DEVICE_NOT_LICENSED:
            return "DEVICE_NOT_LICENSED"
        elif self == Result.INVALID_IMAGE:
            return "INVALID_IMAGE"
        elif self == Result.INVALID_CONTEXT:
            return "INVALID_CONTEXT"
        elif self == Result.CONTEXT_ALREADY_CURRENT:
            return "CONTEXT_ALREADY_CURRENT"
        elif self == Result.MAP_FAILED:
            return "MAP_FAILED"
        elif self == Result.UNMAP_FAILED:
            return "UNMAP_FAILED"
        elif self == Result.ARRAY_IS_MAPPED:
            return "ARRAY_IS_MAPPED"
        elif self == Result.ALREADY_MAPPED:
            return "ALREADY_MAPPED"
        elif self == Result.NO_BINARY_FOR_GPU:
            return "NO_BINARY_FOR_GPU"
        elif self == Result.ALREADY_ACQUIRED:
            return "ALREADY_ACQUIRED"
        elif self == Result.NOT_MAPPED:
            return "NOT_MAPPED"
        elif self == Result.NOT_MAPPED_AS_ARRAY:
            return "NOT_MAPPED_AS_ARRAY"
        elif self == Result.NOT_MAPPED_AS_POINTER:
            return "NOT_MAPPED_AS_POINTER"
        elif self == Result.ECC_UNCORRECTABLE:
            return "ECC_UNCORRECTABLE"
        elif self == Result.UNSUPPORTED_LIMIT:
            return "UNSUPPORTED_LIMIT"
        elif self == Result.CONTEXT_ALREADY_IN_USE:
            return "CONTEXT_ALREADY_IN_USE"
        elif self == Result.PEER_ACCESS_UNSUPPORTED:
            return "PEER_ACCESS_UNSUPPORTED"
        elif self == Result.INVALID_PTX:
            return "INVALID_PTX"
        elif self == Result.INVALID_GRAPHICS_CONTEXT:
            return "INVALID_GRAPHICS_CONTEXT"
        elif self == Result.NVLINK_UNCORRECTABLE:
            return "NVLINK_UNCORRECTABLE"
        elif self == Result.JIT_COMPILER_NOT_FOUND:
            return "JIT_COMPILER_NOT_FOUND"
        elif self == Result.UNSUPPORTED_PTX_VERSION:
            return "UNSUPPORTED_PTX_VERSION"
        elif self == Result.JIT_COMPILATION_DISABLED:
            return "JIT_COMPILATION_DISABLED"
        elif self == Result.UNSUPPORTED_EXEC_AFFINITY:
            return "UNSUPPORTED_EXEC_AFFINITY"
        elif self == Result.UNSUPPORTED_DEVSIDE_SYNC:
            return "UNSUPPORTED_DEVSIDE_SYNC"
        elif self == Result.INVALID_SOURCE:
            return "INVALID_SOURCE"
        elif self == Result.FILE_NOT_FOUND:
            return "FILE_NOT_FOUND"
        elif self == Result.SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return "SHARED_OBJECT_SYMBOL_NOT_FOUND"
        elif self == Result.SHARED_OBJECT_INIT_FAILED:
            return "SHARED_OBJECT_INIT_FAILED"
        elif self == Result.OPERATING_SYSTEM:
            return "OPERATING_SYSTEM"
        elif self == Result.INVALID_HANDLE:
            return "INVALID_HANDLE"
        elif self == Result.ILLEGAL_STATE:
            return "ILLEGAL_STATE"
        elif self == Result.NOT_FOUND:
            return "NOT_FOUND"
        elif self == Result.NOT_READY:
            return "NOT_READY"
        elif self == Result.ILLEGAL_ADDRESS:
            return "ILLEGAL_ADDRESS"
        elif self == Result.LAUNCH_OUT_OF_RESOURCES:
            return "LAUNCH_OUT_OF_RESOURCES"
        elif self == Result.LAUNCH_TIMEOUT:
            return "LAUNCH_TIMEOUT"
        else:
            return self._str_shard()

    # Shard the long if-elif-else chain in __str__ to work around #23478.
    @no_inline
    fn _str_shard(self) -> String:
        if self == Result.LAUNCH_INCOMPATIBLE_TEXTURING:
            return "LAUNCH_INCOMPATIBLE_TEXTURING"
        elif self == Result.PEER_ACCESS_ALREADY_ENABLED:
            return "PEER_ACCESS_ALREADY_ENABLED"
        elif self == Result.PEER_ACCESS_NOT_ENABLED:
            return "PEER_ACCESS_NOT_ENABLED"
        elif self == Result.PRIMARY_CONTEXT_ACTIVE:
            return "PRIMARY_CONTEXT_ACTIVE"
        elif self == Result.CONTEXT_IS_DESTROYED:
            return "CONTEXT_IS_DESTROYED"
        elif self == Result.ASSERT:
            return "ASSERT"
        elif self == Result.TOO_MANY_PEERS:
            return "TOO_MANY_PEERS"
        elif self == Result.HOST_MEMORY_ALREADY_REGISTERED:
            return "HOST_MEMORY_ALREADY_REGISTERED"
        elif self == Result.HOST_MEMORY_NOT_REGISTERED:
            return "HOST_MEMORY_NOT_REGISTERED"
        elif self == Result.HARDWARE_STACK_ERROR:
            return "HARDWARE_STACK_ERROR"
        elif self == Result.ILLEGAL_INSTRUCTION:
            return "ILLEGAL_INSTRUCTION"
        elif self == Result.MISALIGNED_ADDRESS:
            return "MISALIGNED_ADDRESS"
        elif self == Result.INVALID_ADDRESS_SPACE:
            return "INVALID_ADDRESS_SPACE"
        elif self == Result.INVALID_PC:
            return "INVALID_PC"
        elif self == Result.LAUNCH_FAILED:
            return "LAUNCH_FAILED"
        elif self == Result.COOPERATIVE_LAUNCH_TOO_LARGE:
            return "COOPERATIVE_LAUNCH_TOO_LARGE"
        elif self == Result.NOT_PERMITTED:
            return "NOT_PERMITTED"
        elif self == Result.NOT_SUPPORTED:
            return "NOT_SUPPORTED"
        elif self == Result.SYSTEM_NOT_READY:
            return "SYSTEM_NOT_READY"
        elif self == Result.SYSTEM_DRIVER_MISMATCH:
            return "SYSTEM_DRIVER_MISMATCH"
        elif self == Result.COMPAT_NOT_SUPPORTED_ON_DEVICE:
            return "COMPAT_NOT_SUPPORTED_ON_DEVICE"
        elif self == Result.MPS_CONNECTION_FAILED:
            return "MPS_CONNECTION_FAILED"
        elif self == Result.MPS_RPC_FAILURE:
            return "MPS_RPC_FAILURE"
        elif self == Result.MPS_SERVER_NOT_READY:
            return "MPS_SERVER_NOT_READY"
        elif self == Result.MPS_MAX_CLIENTS_REACHED:
            return "MPS_MAX_CLIENTS_REACHED"
        elif self == Result.MPS_MAX_CONNECTIONS_REACHED:
            return "MPS_MAX_CONNECTIONS_REACHED"
        elif self == Result.MPS_CLIENT_TERMINATED:
            return "MPS_CLIENT_TERMINATED"
        elif self == Result.CDP_NOT_SUPPORTED:
            return "CDP_NOT_SUPPORTED"
        elif self == Result.CDP_VERSION_MISMATCH:
            return "CDP_VERSION_MISMATCH"
        elif self == Result.STREAM_CAPTURE_UNSUPPORTED:
            return "STREAM_CAPTURE_UNSUPPORTED"
        elif self == Result.STREAM_CAPTURE_INVALIDATED:
            return "STREAM_CAPTURE_INVALIDATED"
        elif self == Result.STREAM_CAPTURE_MERGE:
            return "STREAM_CAPTURE_MERGE"
        elif self == Result.STREAM_CAPTURE_UNMATCHED:
            return "STREAM_CAPTURE_UNMATCHED"
        elif self == Result.STREAM_CAPTURE_UNJOINED:
            return "STREAM_CAPTURE_UNJOINED"
        elif self == Result.STREAM_CAPTURE_ISOLATION:
            return "STREAM_CAPTURE_ISOLATION"
        elif self == Result.STREAM_CAPTURE_IMPLICIT:
            return "STREAM_CAPTURE_IMPLICIT"
        elif self == Result.CAPTURED_EVENT:
            return "CAPTURED_EVENT"
        elif self == Result.STREAM_CAPTURE_WRONG_THREAD:
            return "STREAM_CAPTURE_WRONG_THREAD"
        elif self == Result.TIMEOUT:
            return "TIMEOUT"
        elif self == Result.GRAPH_EXEC_UPDATE_FAILURE:
            return "GRAPH_EXEC_UPDATE_FAILURE"
        elif self == Result.EXTERNAL_DEVICE:
            return "EXTERNAL_DEVICE"
        elif self == Result.INVALID_CLUSTER_SIZE:
            return "INVALID_CLUSTER_SIZE"
        elif self == Result.UNKNOWN:
            return "UNKNOWN"
        else:
            return "<UNKNOWN>"

    fn __repr__(self) -> String:
        return self.__str__()

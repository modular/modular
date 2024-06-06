# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements device operations."""

from sys.ffi import DLHandle

from memory.unsafe import DTypePointer, Pointer

from utils import StringRef

from ._utils import _check_error, _human_memory
from .cuda_instance import *
from .dim import Dim

# ===----------------------------------------------------------------------===#
# Device Information
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct DeviceAttribute:
    var _value: Int32

    alias MAX_THREADS_PER_BLOCK = DeviceAttribute(1)
    """Maximum number of threads per block
    """

    alias MAX_BLOCK_DIM_X = DeviceAttribute(2)
    """Maximum block dimension X
    """

    alias MAX_BLOCK_DIM_Y = DeviceAttribute(3)
    """Maximum block dimension Y
    """

    alias MAX_BLOCK_DIM_Z = DeviceAttribute(4)
    """Maximum block dimension Z
    """

    alias MAX_GRID_DIM_X = DeviceAttribute(5)
    """Maximum grid dimension X
    """

    alias MAX_GRID_DIM_Y = DeviceAttribute(6)
    """Maximum grid dimension Y
    """

    alias MAX_GRID_DIM_Z = DeviceAttribute(7)
    """Maximum grid dimension Z
    """

    alias MAX_SHARED_MEMORY_PER_BLOCK = DeviceAttribute(8)
    """Maximum shared memory available per block in bytes
    """

    alias SHARED_MEMORY_PER_BLOCK = DeviceAttribute(8)
    """Deprecated, use alias MAX_SHARED_MEMORY_PER_BLOCK
    """

    alias TOTAL_CONSTANT_MEMORY = DeviceAttribute(9)
    """Memory available on device for __constant__ variables in a CUDA C kernel
    in bytes
    """

    alias WARP_SIZE = DeviceAttribute(10)
    """Warp size in threads
    """

    alias MAX_PITCH = DeviceAttribute(11)
    """Maximum pitch in bytes allowed by memory copies
    """

    alias MAX_REGISTERS_PER_BLOCK = DeviceAttribute(12)
    """Maximum number of 32-bit registers available per block
    """

    alias REGISTERS_PER_BLOCK = DeviceAttribute(12)
    """Deprecated, use alias MAX_REGISTERS_PER_BLOCK
    """

    alias CLOCK_RATE = DeviceAttribute(13)
    """Typical clock frequency in kilohertz
    """

    alias TEXTURE_ALIGNMENT = DeviceAttribute(14)
    """Alignment requirement for textures
    """

    alias GPU_OVERLAP = DeviceAttribute(15)
    """Device can possibly copy memory and execute a kernel concurrently.
    Deprecated. Use instead alias ASYNC_ENGINE_COUNT.)
    """

    alias MULTIPROCESSOR_COUNT = DeviceAttribute(16)
    """Number of multiprocessors on device
    """

    alias KERNEL_EXEC_TIMEOUT = DeviceAttribute(17)
    """Specifies whether there is a run time limit on kernels
    """

    alias INTEGRATED = DeviceAttribute(18)
    """Device is integrated with host memory
    """

    alias CAN_MAP_HOST_MEMORY = DeviceAttribute(19)
    """Device can map host memory into CUDA address space
    """

    alias COMPUTE_MODE = DeviceAttribute(20)
    """Compute mode (See ::CUcomputemode for details))
    """

    alias MAXIMUM_TEXTURE1D_WIDTH = DeviceAttribute(21)
    """Maximum 1D texture width
    """

    alias MAXIMUM_TEXTURE2D_WIDTH = DeviceAttribute(22)
    """Maximum 2D texture width
    """

    alias MAXIMUM_TEXTURE2D_HEIGHT = DeviceAttribute(23)
    """Maximum 2D texture height
    """

    alias MAXIMUM_TEXTURE3D_WIDTH = DeviceAttribute(24)
    """Maximum 3D texture width
    """

    alias MAXIMUM_TEXTURE3D_HEIGHT = DeviceAttribute(25)
    """Maximum 3D texture height
    """

    alias MAXIMUM_TEXTURE3D_DEPTH = DeviceAttribute(26)
    """Maximum 3D texture depth
    """

    alias MAXIMUM_TEXTURE2D_LAYERED_WIDTH = DeviceAttribute(27)
    """Maximum 2D layered texture width
    """

    alias MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = DeviceAttribute(28)
    """Maximum 2D layered texture height
    """

    alias MAXIMUM_TEXTURE2D_LAYERED_LAYERS = DeviceAttribute(29)
    """Maximum layers in a 2D layered texture
    """

    alias MAXIMUM_TEXTURE2D_ARRAY_WIDTH = DeviceAttribute(27)
    """Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_WIDTH
    """

    alias MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = DeviceAttribute(28)
    """Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
    """

    alias MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = DeviceAttribute(29)
    """Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_LAYERS
    """

    alias SURFACE_ALIGNMENT = DeviceAttribute(30)
    """Alignment requirement for surfaces
    """

    alias CONCURRENT_KERNELS = DeviceAttribute(31)
    """Device can possibly execute multiple kernels concurrently
    """

    alias ECC_ENABLED = DeviceAttribute(32)
    """Device has ECC support enabled
    """

    alias PCI_BUS_ID = DeviceAttribute(33)
    """PCI bus ID of the device
    """

    alias PCI_DEVICE_ID = DeviceAttribute(34)
    """PCI device ID of the device
    """

    alias TCC_DRIVER = DeviceAttribute(35)
    """Device is using TCC driver model
    """

    alias MEMORY_CLOCK_RATE = DeviceAttribute(36)
    """Peak memory clock frequency in kilohertz
    """

    alias GLOBAL_MEMORY_BUS_WIDTH = DeviceAttribute(37)
    """Global memory bus width in bits
    """

    alias L2_CACHE_SIZE = DeviceAttribute(38)
    """Size of L2 cache in bytes
    """

    alias MAX_THREADS_PER_MULTIPROCESSOR = DeviceAttribute(39)
    """Maximum resident threads per multiprocessor
    """

    alias ASYNC_ENGINE_COUNT = DeviceAttribute(40)
    """Number of asynchronous engines
    """

    alias UNIFIED_ADDRESSING = DeviceAttribute(41)
    """Device shares a unified address space with the host
    """

    alias MAXIMUM_TEXTURE1D_LAYERED_WIDTH = DeviceAttribute(42)
    """Maximum 1D layered texture width
    """

    alias MAXIMUM_TEXTURE1D_LAYERED_LAYERS = DeviceAttribute(43)
    """Maximum layers in a 1D layered texture
    """

    alias CAN_TEX2D_GATHER = DeviceAttribute(44)
    """Deprecated, do not use.)
    """

    alias MAXIMUM_TEXTURE2D_GATHER_WIDTH = DeviceAttribute(45)
    """Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
    """

    alias MAXIMUM_TEXTURE2D_GATHER_HEIGHT = DeviceAttribute(46)
    """Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
    """

    alias MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = DeviceAttribute(47)
    """Alternate maximum 3D texture width
    """

    alias MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = DeviceAttribute(48)
    """Alternate maximum 3D texture height
    """

    alias MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = DeviceAttribute(49)
    """Alternate maximum 3D texture depth
    """

    alias PCI_DOMAIN_ID = DeviceAttribute(50)
    """PCI domain ID of the device
    """

    alias TEXTURE_PITCH_ALIGNMENT = DeviceAttribute(51)
    """Pitch alignment requirement for textures
    """

    alias MAXIMUM_TEXTURECUBEMAP_WIDTH = DeviceAttribute(52)
    """Maximum cubemap texture width/height
    """

    alias MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = DeviceAttribute(53)
    """Maximum cubemap layered texture width/height
    """

    alias MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = DeviceAttribute(54)
    """Maximum layers in a cubemap layered texture
    """

    alias MAXIMUM_SURFACE1D_WIDTH = DeviceAttribute(55)
    """Maximum 1D surface width
    """

    alias MAXIMUM_SURFACE2D_WIDTH = DeviceAttribute(56)
    """Maximum 2D surface width
    """

    alias MAXIMUM_SURFACE2D_HEIGHT = DeviceAttribute(57)
    """Maximum 2D surface height
    """

    alias MAXIMUM_SURFACE3D_WIDTH = DeviceAttribute(58)
    """Maximum 3D surface width
    """

    alias MAXIMUM_SURFACE3D_HEIGHT = DeviceAttribute(59)
    """Maximum 3D surface height
    """

    alias MAXIMUM_SURFACE3D_DEPTH = DeviceAttribute(60)
    """Maximum 3D surface depth
    """

    alias MAXIMUM_SURFACE1D_LAYERED_WIDTH = DeviceAttribute(61)
    """Maximum 1D layered surface width
    """

    alias MAXIMUM_SURFACE1D_LAYERED_LAYERS = DeviceAttribute(62)
    """Maximum layers in a 1D layered surface
    """

    alias MAXIMUM_SURFACE2D_LAYERED_WIDTH = DeviceAttribute(63)
    """Maximum 2D layered surface width
    """

    alias MAXIMUM_SURFACE2D_LAYERED_HEIGHT = DeviceAttribute(64)
    """Maximum 2D layered surface height
    """

    alias MAXIMUM_SURFACE2D_LAYERED_LAYERS = DeviceAttribute(65)
    """Maximum layers in a 2D layered surface
    """

    alias MAXIMUM_SURFACECUBEMAP_WIDTH = DeviceAttribute(66)
    """Maximum cubemap surface width
    """

    alias MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = DeviceAttribute(67)
    """Maximum cubemap layered surface width
    """

    alias MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = DeviceAttribute(68)
    """Maximum layers in a cubemap layered surface
    """

    alias MAXIMUM_TEXTURE1D_LINEAR_WIDTH = DeviceAttribute(69)
    """Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or
    cuDeviceGetTexture1DLinearMaxWidth() instead.)
    """

    alias MAXIMUM_TEXTURE2D_LINEAR_WIDTH = DeviceAttribute(70)
    """Maximum 2D linear texture width
    """

    alias MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = DeviceAttribute(71)
    """Maximum 2D linear texture height
    """

    alias MAXIMUM_TEXTURE2D_LINEAR_PITCH = DeviceAttribute(72)
    """Maximum 2D linear texture pitch in bytes
    """

    alias MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = DeviceAttribute(73)
    """Maximum mipmapped 2D texture width
    """

    alias MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = DeviceAttribute(74)
    """Maximum mipmapped 2D texture height
    """

    alias COMPUTE_CAPABILITY_MAJOR = DeviceAttribute(75)
    """Major compute capability version number
    """

    alias COMPUTE_CAPABILITY_MINOR = DeviceAttribute(76)
    """Minor compute capability version number
    """

    alias MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = DeviceAttribute(77)
    """Maximum mipmapped 1D texture width
    """

    alias STREAM_PRIORITIES_SUPPORTED = DeviceAttribute(78)
    """Device supports stream priorities
    """

    alias GLOBAL_L1_CACHE_SUPPORTED = DeviceAttribute(79)
    """Device supports caching globals in L1
    """

    alias LOCAL_L1_CACHE_SUPPORTED = DeviceAttribute(80)
    """Device supports caching locals in L1
    """

    alias MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = DeviceAttribute(81)
    """Maximum shared memory available per multiprocessor in bytes
    """

    alias MAX_REGISTERS_PER_MULTIPROCESSOR = DeviceAttribute(82)
    """Maximum number of 32-bit registers available per multiprocessor
    """

    alias MANAGED_MEMORY = DeviceAttribute(83)
    """Device can allocate managed memory on this system
    """

    alias MULTI_GPU_BOARD = DeviceAttribute(84)
    """Device is on a multi-GPU board
    """

    alias MULTI_GPU_BOARD_GROUP_ID = DeviceAttribute(85)
    """Unique id for a group of devices on the same multi-GPU board
    """

    alias HOST_NATIVE_ATOMIC_SUPPORTED = DeviceAttribute(86)
    """Link between the device and the host supports native atomic operations
    (this is a placeholder attribute, and is not supported on any current
    hardware).
    """

    alias SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = DeviceAttribute(87)
    """Ratio of single precision performance (in floating-point operations per
    second) to double precision performance.
    """

    alias PAGEABLE_MEMORY_ACCESS = DeviceAttribute(88)
    """Device supports coherently accessing pageable memory without calling
    cudaHostRegister on it.
    """

    alias CONCURRENT_MANAGED_ACCESS = DeviceAttribute(89)
    """Device can coherently access managed memory concurrently with the CPU
    """

    alias COMPUTE_PREEMPTION_SUPPORTED = DeviceAttribute(90)
    """Device supports compute preemption.
    """

    alias CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = DeviceAttribute(91)
    """Device can access host registered memory at the same virtual address as
    the CPU
    """

    alias CAN_USE_STREAM_MEM_OPS_V1 = DeviceAttribute(92)
    """Deprecated, along with v1 MemOps API, ::cuStreamBatchMemOp and related
    APIs are supported.
    """

    alias CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = DeviceAttribute(93)
    """Deprecated, along with v1 MemOps API, 64-bit operations are supported in
    ::cuStreamBatchMemOp and related APIs.
    """

    alias CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = DeviceAttribute(94)
    """Deprecated, along with v1 MemOps API, ::CU_STREAM_WAIT_VALUE_NOR is
    supported.
    """

    alias COOPERATIVE_LAUNCH = DeviceAttribute(95)
    """Device supports launching cooperative kernels via
    ::cuLaunchCooperativeKernel
    """

    alias COOPERATIVE_MULTI_DEVICE_LAUNCH = DeviceAttribute(96)
    """Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated.)
    """

    alias MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = DeviceAttribute(97)
    """Maximum optin shared memory per block
    """

    alias CAN_FLUSH_REMOTE_WRITES = DeviceAttribute(98)
    """The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the
    ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device.
    See \ref CUDA_MEMOP for additional details.
    """

    alias HOST_REGISTER_SUPPORTED = DeviceAttribute(99)
    """Device supports host memory registration via ::cudaHostRegister.
    """

    alias PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = DeviceAttribute(100)
    """Device accesses pageable memory via the host's page tables.
    """

    alias DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = DeviceAttribute(101)
    """The host can directly access managed memory on the device without
    migration.
    """

    alias VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = DeviceAttribute(102)
    """Deprecated, Use alias VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
    """

    alias VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = DeviceAttribute(102)
    """Device supports virtual memory management APIs like
    ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs
    """

    alias HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = DeviceAttribute(103)
    """Device supports exporting memory to a posix file descriptor with
    ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    """

    alias HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = DeviceAttribute(104)
    """Device supports exporting memory to a Win32 NT handle with
    ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    """

    alias HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = DeviceAttribute(105)
    """Device supports exporting memory to a Win32 KMT handle with
    ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    """

    alias MAX_BLOCKS_PER_MULTIPROCESSOR = DeviceAttribute(106)
    """Maximum number of blocks per multiprocessor
    """

    alias GENERIC_COMPRESSION_SUPPORTED = DeviceAttribute(107)
    """Device supports compression of memory
    """

    alias MAX_PERSISTING_L2_CACHE_SIZE = DeviceAttribute(108)
    """Maximum L2 persisting lines capacity setting in bytes.
    """

    alias MAX_ACCESS_POLICY_WINDOW_SIZE = DeviceAttribute(109)
    """Maximum value of CUaccessPolicyWindow::num_bytes.
    """

    alias GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = DeviceAttribute(110)
    """Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate
    """

    alias RESERVED_SHARED_MEMORY_PER_BLOCK = DeviceAttribute(111)
    """Shared memory reserved by CUDA driver per block in bytes
    """

    alias SPARSE_CUDA_ARRAY_SUPPORTED = DeviceAttribute(112)
    """Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
    """

    alias READ_ONLY_HOST_REGISTER_SUPPORTED = DeviceAttribute(113)
    """Device supports using the ::cuMemHostRegister flag
    ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as
    read-only to the GPU
    """

    alias TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = DeviceAttribute(114)
    """External timeline semaphore interop is supported on the device
    """

    alias MEMORY_POOLS_SUPPORTED = DeviceAttribute(115)
    """Device supports using the ::cuMemAllocAsync and ::cuMemPool family of
    APIs
    """

    alias GPU_DIRECT_RDMA_SUPPORTED = DeviceAttribute(116)
    """Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages
    (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)
    """

    alias GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = DeviceAttribute(117)
    """The returned attribute shall be interpreted as a bitmask, where the
    individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions
    enum
    """

    alias GPU_DIRECT_RDMA_WRITES_ORDERING = DeviceAttribute(118)
    """GPUDirect RDMA writes to the device do not need to be flushed for
    consumers within the scope indicated by the returned attribute. See
    ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
    """

    alias MEMPOOL_SUPPORTED_HANDLE_TYPES = DeviceAttribute(119)
    """Handle types supported with mempool based IPC
    """

    alias CLUSTER_LAUNCH = DeviceAttribute(120)
    """Indicates device supports cluster launch
    """

    alias DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = DeviceAttribute(121)
    """Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
    """

    alias CAN_USE_64_BIT_STREAM_MEM_OPS = DeviceAttribute(122)
    """64-bit operations are supported in ::cuStreamBatchMemOp and related
    MemOp APIs.
    """

    alias CAN_USE_STREAM_WAIT_VALUE_NOR = DeviceAttribute(123)
    """::CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs.
    """

    alias DMA_BUF_SUPPORTED = DeviceAttribute(124)
    """Device supports buffer sharing with dma_buf mechanism.
    """

    alias IPC_EVENT_SUPPORTED = DeviceAttribute(125)
    """Device supports IPC Events.)
    """

    alias MEM_SYNC_DOMAIN_COUNT = DeviceAttribute(126)
    """Number of memory domains the device supports.
    """

    alias TENSOR_MAP_ACCESS_SUPPORTED = DeviceAttribute(127)
    """Device supports accessing memory using Tensor Map.
    """

    alias UNIFIED_FUNCTION_POINTERS = DeviceAttribute(129)
    """Device supports unified function pointers.
    """

    alias MULTICAST_SUPPORTED = DeviceAttribute(132)
    """Device supports switch multicast and reduction operations.
    """

    fn __init__(inout self, value: Int32):
        self._value = value


fn device_count() raises -> Int:
    """
    Returns the number of devices with compute capability greater than or equal
    to 2.0 that are available for execution.
    """

    var cuDeviceGetCount = cuDeviceGetCount.load()
    var res: Int32 = 0
    _check_error(cuDeviceGetCount(Pointer.address_of(res)))
    return int(res)


struct Device(StringableRaising):
    var id: Int32
    var cuda_dll: Optional[CudaDLL]

    fn __init__(inout self, id: Int = 0):
        self.id = id
        self.cuda_dll = None

    fn __init__(inout self, cuda_instance: CudaInstance, id: Int = 0):
        self.id = id
        self.cuda_dll = cuda_instance.cuda_dll

    fn __str__(self) raises -> String:
        var res = "name: " + self._name() + "\n"
        res += String("memory: ") + _human_memory(self._total_memory()) + "\n"
        res += (
            String("compute_capability: ")
            + str(self._query(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR))
            + "."
            + str(self._query(DeviceAttribute.COMPUTE_CAPABILITY_MINOR))
            + "\n"
        )
        res += (
            String("clock_rate: ")
            + str(self._query(DeviceAttribute.CLOCK_RATE))
            + "\n"
        )
        res += (
            String("warp_size: ")
            + str(self._query(DeviceAttribute.WARP_SIZE))
            + "\n"
        )
        res += (
            String("max_threads_per_block: ")
            + str(self._query(DeviceAttribute.MAX_THREADS_PER_BLOCK))
            + "\n"
        )
        res += (
            String("max_shared_memory: ")
            + _human_memory(
                self._query(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK)
            )
            + "\n"
        )
        res += (
            String("max_block: ")
            + Dim(
                self._query(DeviceAttribute.MAX_BLOCK_DIM_X),
                self._query(DeviceAttribute.MAX_BLOCK_DIM_Y),
                self._query(DeviceAttribute.MAX_BLOCK_DIM_Z),
            ).__str__()
            + "\n"
        )
        res += (
            String("max_grid: ")
            + Dim(
                self._query(DeviceAttribute.MAX_GRID_DIM_X),
                self._query(DeviceAttribute.MAX_GRID_DIM_Y),
                self._query(DeviceAttribute.MAX_GRID_DIM_Z),
            ).__str__()
            + "\n"
        )
        res += (
            String("SM count: ")
            + str(self._query(DeviceAttribute.MULTIPROCESSOR_COUNT))
            + "\n"
        )

        return res

    fn _name(self) raises -> String:
        """Get an identifier string for the device."""

        alias buffer_size = 256
        var buffer = UnsafePointer[C_char]._from_dtype_ptr(
            stack_allocation[buffer_size, C_char.type]()
        )

        var cuDeviceGetName = self.cuda_dll.value().cuDeviceGetName if self.cuda_dll else cuDeviceGetName.load()
        _ = cuDeviceGetName(buffer, Int32(buffer_size), self.id)

        return StringRef(buffer)

    fn _total_memory(self) raises -> Int:
        """Returns the total amount of memory on the device."""

        var cuDeviceTotalMem = self.cuda_dll.value().cuDeviceTotalMem if self.cuda_dll else cuDeviceTotalMem.load()
        var res: Int = 0
        _check_error(cuDeviceTotalMem(Pointer.address_of(res), self.id))
        return res

    fn _query(self, attr: DeviceAttribute) raises -> Int:
        """Returns information about a particular device attribute."""

        var cuDeviceGetAttribute = self.cuda_dll.value().cuDeviceGetAttribute if self.cuda_dll else cuDeviceGetAttribute.load()
        var res: Int32 = 0
        _check_error(
            cuDeviceGetAttribute(Pointer.address_of(res), attr, self.id)
        )
        return int(res)

    fn multiprocessor_count(self) raises -> Int:
        """Returns the number of multiprocessors on this device."""
        return self._query(DeviceAttribute.MULTIPROCESSOR_COUNT)

    fn max_registers_per_block(self) raises -> Int:
        """Returns the maximum number of 32-bit registers available per block.
        """
        return self._query(DeviceAttribute.MAX_REGISTERS_PER_BLOCK)

    fn max_threads_per_sm(self) raises -> Int:
        """Returns the maximum resident threads per multiprocessor."""
        return self._query(DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR)

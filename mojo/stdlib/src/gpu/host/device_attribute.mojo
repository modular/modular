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
"""This module defines GPU device attributes that can be queried from CUDA-compatible devices.

The module provides the `DeviceAttribute` struct which encapsulates the various device
properties and capabilities that can be queried through the CUDA driver API. Each attribute
is represented as a constant with a corresponding integer value that maps to the CUDA
driver's attribute enumeration.

These attributes allow applications to query specific hardware capabilities and limitations
of GPU devices, such as maximum thread counts, memory sizes, compute capabilities, and
supported features.
"""


@value
@register_passable("trivial")
struct DeviceAttribute:
    """
    Represents CUDA device attributes that can be queried from a GPU device.

    This struct encapsulates the various device properties and capabilities that can be
    queried through the CUDA driver API. Each attribute is represented as a constant
    with a corresponding integer value that maps to the CUDA driver's attribute enum.
    """

    var _value: Int32
    """The integer value representing the specific device attribute."""

    alias MAX_THREADS_PER_BLOCK = Self(1)
    """Maximum number of threads per block
    """

    alias MAX_BLOCK_DIM_X = Self(2)
    """Maximum block dimension X
    """

    alias MAX_BLOCK_DIM_Y = Self(3)
    """Maximum block dimension Y
    """

    alias MAX_BLOCK_DIM_Z = Self(4)
    """Maximum block dimension Z
    """

    alias MAX_GRID_DIM_X = Self(5)
    """Maximum grid dimension X
    """

    alias MAX_GRID_DIM_Y = Self(6)
    """Maximum grid dimension Y
    """

    alias MAX_GRID_DIM_Z = Self(7)
    """Maximum grid dimension Z
    """

    alias MAX_SHARED_MEMORY_PER_BLOCK = Self(8)
    """Maximum shared memory available per block in bytes
    """

    alias SHARED_MEMORY_PER_BLOCK = Self(8)
    """Deprecated, use alias MAX_SHARED_MEMORY_PER_BLOCK
    """

    alias TOTAL_CONSTANT_MEMORY = Self(9)
    """Memory available on device for __constant__ variables in a CUDA C kernel
    in bytes
    """

    alias WARP_SIZE = Self(10)
    """Warp size in threads
    """

    alias MAX_PITCH = Self(11)
    """Maximum pitch in bytes allowed by memory copies
    """

    alias MAX_REGISTERS_PER_BLOCK = Self(12)
    """Maximum number of 32-bit registers available per block
    """

    alias REGISTERS_PER_BLOCK = Self(12)
    """Deprecated, use alias MAX_REGISTERS_PER_BLOCK
    """

    alias CLOCK_RATE = Self(13)
    """Typical clock frequency in kilohertz
    """

    alias TEXTURE_ALIGNMENT = Self(14)
    """Alignment requirement for textures
    """

    alias GPU_OVERLAP = Self(15)
    """Device can possibly copy memory and execute a kernel concurrently.
    Deprecated. Use instead alias ASYNC_ENGINE_COUNT.)
    """

    alias MULTIPROCESSOR_COUNT = Self(16)
    """Number of multiprocessors on device
    """

    alias KERNEL_EXEC_TIMEOUT = Self(17)
    """Specifies whether there is a run time limit on kernels
    """

    alias INTEGRATED = Self(18)
    """Device is integrated with host memory
    """

    alias CAN_MAP_HOST_MEMORY = Self(19)
    """Device can map host memory into CUDA address space
    """

    alias COMPUTE_MODE = Self(20)
    """Compute mode (See ::CUcomputemode for details))
    """

    alias MAXIMUM_TEXTURE1D_WIDTH = Self(21)
    """Maximum 1D texture width
    """

    alias MAXIMUM_TEXTURE2D_WIDTH = Self(22)
    """Maximum 2D texture width
    """

    alias MAXIMUM_TEXTURE2D_HEIGHT = Self(23)
    """Maximum 2D texture height
    """

    alias MAXIMUM_TEXTURE3D_WIDTH = Self(24)
    """Maximum 3D texture width
    """

    alias MAXIMUM_TEXTURE3D_HEIGHT = Self(25)
    """Maximum 3D texture height
    """

    alias MAXIMUM_TEXTURE3D_DEPTH = Self(26)
    """Maximum 3D texture depth
    """

    alias MAXIMUM_TEXTURE2D_LAYERED_WIDTH = Self(27)
    """Maximum 2D layered texture width
    """

    alias MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = Self(28)
    """Maximum 2D layered texture height
    """

    alias MAXIMUM_TEXTURE2D_LAYERED_LAYERS = Self(29)
    """Maximum layers in a 2D layered texture
    """

    alias MAXIMUM_TEXTURE2D_ARRAY_WIDTH = Self(27)
    """Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_WIDTH
    """

    alias MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = Self(28)
    """Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
    """

    alias MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = Self(29)
    """Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_LAYERS
    """

    alias SURFACE_ALIGNMENT = Self(30)
    """Alignment requirement for surfaces
    """

    alias CONCURRENT_KERNELS = Self(31)
    """Device can possibly execute multiple kernels concurrently
    """

    alias ECC_ENABLED = Self(32)
    """Device has ECC support enabled
    """

    alias PCI_BUS_ID = Self(33)
    """PCI bus ID of the device
    """

    alias PCI_DEVICE_ID = Self(34)
    """PCI device ID of the device
    """

    alias TCC_DRIVER = Self(35)
    """Device is using TCC driver model
    """

    alias MEMORY_CLOCK_RATE = Self(36)
    """Peak memory clock frequency in kilohertz
    """

    alias GLOBAL_MEMORY_BUS_WIDTH = Self(37)
    """Global memory bus width in bits
    """

    alias L2_CACHE_SIZE = Self(38)
    """Size of L2 cache in bytes
    """

    alias MAX_THREADS_PER_MULTIPROCESSOR = Self(39)
    """Maximum resident threads per multiprocessor
    """

    alias ASYNC_ENGINE_COUNT = Self(40)
    """Number of asynchronous engines
    """

    alias UNIFIED_ADDRESSING = Self(41)
    """Device shares a unified address space with the host
    """

    alias MAXIMUM_TEXTURE1D_LAYERED_WIDTH = Self(42)
    """Maximum 1D layered texture width
    """

    alias MAXIMUM_TEXTURE1D_LAYERED_LAYERS = Self(43)
    """Maximum layers in a 1D layered texture
    """

    alias CAN_TEX2D_GATHER = Self(44)
    """Deprecated, do not use.)
    """

    alias MAXIMUM_TEXTURE2D_GATHER_WIDTH = Self(45)
    """Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
    """

    alias MAXIMUM_TEXTURE2D_GATHER_HEIGHT = Self(46)
    """Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
    """

    alias MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = Self(47)
    """Alternate maximum 3D texture width
    """

    alias MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = Self(48)
    """Alternate maximum 3D texture height
    """

    alias MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = Self(49)
    """Alternate maximum 3D texture depth
    """

    alias PCI_DOMAIN_ID = Self(50)
    """PCI domain ID of the device
    """

    alias TEXTURE_PITCH_ALIGNMENT = Self(51)
    """Pitch alignment requirement for textures
    """

    alias MAXIMUM_TEXTURECUBEMAP_WIDTH = Self(52)
    """Maximum cubemap texture width/height
    """

    alias MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = Self(53)
    """Maximum cubemap layered texture width/height
    """

    alias MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = Self(54)
    """Maximum layers in a cubemap layered texture
    """

    alias MAXIMUM_SURFACE1D_WIDTH = Self(55)
    """Maximum 1D surface width
    """

    alias MAXIMUM_SURFACE2D_WIDTH = Self(56)
    """Maximum 2D surface width
    """

    alias MAXIMUM_SURFACE2D_HEIGHT = Self(57)
    """Maximum 2D surface height
    """

    alias MAXIMUM_SURFACE3D_WIDTH = Self(58)
    """Maximum 3D surface width
    """

    alias MAXIMUM_SURFACE3D_HEIGHT = Self(59)
    """Maximum 3D surface height
    """

    alias MAXIMUM_SURFACE3D_DEPTH = Self(60)
    """Maximum 3D surface depth
    """

    alias MAXIMUM_SURFACE1D_LAYERED_WIDTH = Self(61)
    """Maximum 1D layered surface width
    """

    alias MAXIMUM_SURFACE1D_LAYERED_LAYERS = Self(62)
    """Maximum layers in a 1D layered surface
    """

    alias MAXIMUM_SURFACE2D_LAYERED_WIDTH = Self(63)
    """Maximum 2D layered surface width
    """

    alias MAXIMUM_SURFACE2D_LAYERED_HEIGHT = Self(64)
    """Maximum 2D layered surface height
    """

    alias MAXIMUM_SURFACE2D_LAYERED_LAYERS = Self(65)
    """Maximum layers in a 2D layered surface
    """

    alias MAXIMUM_SURFACECUBEMAP_WIDTH = Self(66)
    """Maximum cubemap surface width
    """

    alias MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = Self(67)
    """Maximum cubemap layered surface width
    """

    alias MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = Self(68)
    """Maximum layers in a cubemap layered surface
    """

    alias MAXIMUM_TEXTURE1D_LINEAR_WIDTH = Self(69)
    """Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or
    cuDeviceGetTexture1DLinearMaxWidth() instead.)
    """

    alias MAXIMUM_TEXTURE2D_LINEAR_WIDTH = Self(70)
    """Maximum 2D linear texture width
    """

    alias MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = Self(71)
    """Maximum 2D linear texture height
    """

    alias MAXIMUM_TEXTURE2D_LINEAR_PITCH = Self(72)
    """Maximum 2D linear texture pitch in bytes
    """

    alias MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = Self(73)
    """Maximum mipmapped 2D texture width
    """

    alias MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = Self(74)
    """Maximum mipmapped 2D texture height
    """

    alias COMPUTE_CAPABILITY_MAJOR = Self(75)
    """Major compute capability version number
    """

    alias COMPUTE_CAPABILITY_MINOR = Self(76)
    """Minor compute capability version number
    """

    alias MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = Self(77)
    """Maximum mipmapped 1D texture width
    """

    alias STREAM_PRIORITIES_SUPPORTED = Self(78)
    """Device supports stream priorities
    """

    alias GLOBAL_L1_CACHE_SUPPORTED = Self(79)
    """Device supports caching globals in L1
    """

    alias LOCAL_L1_CACHE_SUPPORTED = Self(80)
    """Device supports caching locals in L1
    """

    alias MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = Self(81)
    """Maximum shared memory available per multiprocessor in bytes
    """

    alias MAX_REGISTERS_PER_MULTIPROCESSOR = Self(82)
    """Maximum number of 32-bit registers available per multiprocessor
    """

    alias MANAGED_MEMORY = Self(83)
    """Device can allocate managed memory on this system
    """

    alias MULTI_GPU_BOARD = Self(84)
    """Device is on a multi-GPU board
    """

    alias MULTI_GPU_BOARD_GROUP_ID = Self(85)
    """Unique id for a group of devices on the same multi-GPU board
    """

    alias HOST_NATIVE_ATOMIC_SUPPORTED = Self(86)
    """Link between the device and the host supports native atomic operations
    (this is a placeholder attribute, and is not supported on any current
    hardware).
    """

    alias SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = Self(87)
    """Ratio of single precision performance (in floating-point operations per
    second) to double precision performance.
    """

    alias PAGEABLE_MEMORY_ACCESS = Self(88)
    """Device supports coherently accessing pageable memory without calling
    cudaHostRegister on it.
    """

    alias CONCURRENT_MANAGED_ACCESS = Self(89)
    """Device can coherently access managed memory concurrently with the CPU
    """

    alias COMPUTE_PREEMPTION_SUPPORTED = Self(90)
    """Device supports compute preemption.
    """

    alias CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = Self(91)
    """Device can access host registered memory at the same virtual address as
    the CPU
    """

    alias CAN_USE_STREAM_MEM_OPS_V1 = Self(92)
    """Deprecated, along with v1 MemOps API, ::cuStreamBatchMemOp and related
    APIs are supported.
    """

    alias CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = Self(93)
    """Deprecated, along with v1 MemOps API, 64-bit operations are supported in
    ::cuStreamBatchMemOp and related APIs.
    """

    alias CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = Self(94)
    """Deprecated, along with v1 MemOps API, ::CU_STREAM_WAIT_VALUE_NOR is
    supported.
    """

    alias COOPERATIVE_LAUNCH = Self(95)
    """Device supports launching cooperative kernels via
    ::cuLaunchCooperativeKernel
    """

    alias COOPERATIVE_MULTI_DEVICE_LAUNCH = Self(96)
    """Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated.)
    """

    alias MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = Self(97)
    """Maximum optin shared memory per block
    """

    alias CAN_FLUSH_REMOTE_WRITES = Self(98)
    """The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the
    ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device.
    See \ref CUDA_MEMOP for additional details.
    """

    alias HOST_REGISTER_SUPPORTED = Self(99)
    """Device supports host memory registration via ::cudaHostRegister.
    """

    alias PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = Self(100)
    """Device accesses pageable memory via the host's page tables.
    """

    alias DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = Self(101)
    """The host can directly access managed memory on the device without
    migration.
    """

    alias VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = Self(102)
    """Deprecated, Use alias VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
    """

    alias VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = Self(102)
    """Device supports virtual memory management APIs like
    ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs
    """

    alias HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = Self(103)
    """Device supports exporting memory to a posix file descriptor with
    ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    """

    alias HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = Self(104)
    """Device supports exporting memory to a Win32 NT handle with
    ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    """

    alias HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = Self(105)
    """Device supports exporting memory to a Win32 KMT handle with
    ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    """

    alias MAX_BLOCKS_PER_MULTIPROCESSOR = Self(106)
    """Maximum number of blocks per multiprocessor
    """

    alias GENERIC_COMPRESSION_SUPPORTED = Self(107)
    """Device supports compression of memory
    """

    alias MAX_PERSISTING_L2_CACHE_SIZE = Self(108)
    """Maximum L2 persisting lines capacity setting in bytes.
    """

    alias MAX_ACCESS_POLICY_WINDOW_SIZE = Self(109)
    """Maximum value of CUaccessPolicyWindow::num_bytes.
    """

    alias GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = Self(110)
    """Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate
    """

    alias RESERVED_SHARED_MEMORY_PER_BLOCK = Self(111)
    """Shared memory reserved by CUDA driver per block in bytes
    """

    alias SPARSE_CUDA_ARRAY_SUPPORTED = Self(112)
    """Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
    """

    alias READ_ONLY_HOST_REGISTER_SUPPORTED = Self(113)
    """Device supports using the ::cuMemHostRegister flag
    ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as
    read-only to the GPU
    """

    alias TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = Self(114)
    """External timeline semaphore interop is supported on the device
    """

    alias MEMORY_POOLS_SUPPORTED = Self(115)
    """Device supports using the ::cuMemAllocAsync and ::cuMemPool family of
    APIs
    """

    alias GPU_DIRECT_RDMA_SUPPORTED = Self(116)
    """Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages
    (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)
    """

    alias GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = Self(117)
    """The returned attribute shall be interpreted as a bitmask, where the
    individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions
    enum
    """

    alias GPU_DIRECT_RDMA_WRITES_ORDERING = Self(118)
    """GPUDirect RDMA writes to the device do not need to be flushed for
    consumers within the scope indicated by the returned attribute. See
    ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
    """

    alias MEMPOOL_SUPPORTED_HANDLE_TYPES = Self(119)
    """Handle types supported with mempool based IPC
    """

    alias CLUSTER_LAUNCH = Self(120)
    """Indicates device supports cluster launch
    """

    alias DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = Self(121)
    """Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
    """

    alias CAN_USE_64_BIT_STREAM_MEM_OPS = Self(122)
    """64-bit operations are supported in ::cuStreamBatchMemOp and related
    MemOp APIs.
    """

    alias CAN_USE_STREAM_WAIT_VALUE_NOR = Self(123)
    """::CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs.
    """

    alias DMA_BUF_SUPPORTED = Self(124)
    """Device supports buffer sharing with dma_buf mechanism.
    """

    alias IPC_EVENT_SUPPORTED = Self(125)
    """Device supports IPC Events.)
    """

    alias MEM_SYNC_DOMAIN_COUNT = Self(126)
    """Number of memory domains the device supports.
    """

    alias TENSOR_MAP_ACCESS_SUPPORTED = Self(127)
    """Device supports accessing memory using Tensor Map.
    """

    alias UNIFIED_FUNCTION_POINTERS = Self(129)
    """Device supports unified function pointers.
    """

    alias MULTICAST_SUPPORTED = Self(132)
    """Device supports switch multicast and reduction operations.
    """

    @implicit
    fn __init__(out self, value: Int32):
        """
        Initialize a DeviceAttribute with the given integer value.

        Args:
            value: The integer value representing a specific device attribute.

        This constructor allows implicit conversion from Int32 to DeviceAttribute,
        making it easier to use integer constants with this type.
        """
        self._value = value

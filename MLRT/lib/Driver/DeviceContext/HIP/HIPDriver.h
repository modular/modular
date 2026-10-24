//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file declares the HIP driver interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDRIVER_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDRIVER_H

#include "Support/ErrorOr.h"
#include "Support/Preprocessor.h"
#include "Support/max_license/Tracy/Tracy.h"

#include <stddef.h>
#include <stdint.h>

namespace M::Driver {

// clang-format off

using hipCtx_t = struct ihipCtx_t *;
using hipDevice_t = int;
using hipDeviceptr_t = void *;
using hipFunction_t = struct ihipModuleSymbol_t *;
using hipModule_t = struct ihipModule_t *;
using hipStream_t = struct ihipStream_t *;
using hipEvent_t = struct ihipEvent_t *;
using hipMemPool_t = struct ihipMemPoolHandle_t *;
using hipGraph_t = struct ihipGraph_t *;
using hipGraphExec_t = struct hipGraphExec *;
using hipGraphNode_t = struct hipGraphNode *;
using hipHostFn_t = void(void *);

// COPIED FROM hip_runtime_api.h

enum class hipStreamCaptureMode {
  hipStreamCaptureModeGlobal = 0,
  hipStreamCaptureModeThreadLocal,
  hipStreamCaptureModeRelaxed
};

enum class hipStreamCaptureStatus {
  hipStreamCaptureStatusNone = 0,    ///< Stream is not capturing
  hipStreamCaptureStatusActive,      ///< Stream is actively capturing
  hipStreamCaptureStatusInvalidated  ///< Stream is part of a capture sequence that has been
                                     ///< invalidated, but not terminated
};

enum class hipGraphInstantiateFlags {
  hipGraphInstantiateFlagAutoFreeOnLaunch =
      1,  ///< Automatically free memory allocated in a graph before relaunching.
  hipGraphInstantiateFlagUpload = 2,  ///< Automatically upload the graph after instantiation.
  hipGraphInstantiateFlagDeviceLaunch =
      4,  ///< Instantiate the graph to be launched from the device.
  hipGraphInstantiateFlagUseNodePriority =
      8,  ///< Run the graph using the per-node priority attributes rather than the priority of the
          ///< stream it is launched into.
};

enum class hipGraphDebugDotFlags {
  hipGraphDebugDotFlagsVerbose =
      1 << 0, /**< Output all debug data as if every debug flag is enabled */
  hipGraphDebugDotFlagsKernelNodeParams = 1 << 2, /**< Adds hipKernelNodeParams to output */
  hipGraphDebugDotFlagsMemcpyNodeParams = 1 << 3, /**< Adds hipMemcpy3DParams to output */
  hipGraphDebugDotFlagsMemsetNodeParams = 1 << 4, /**< Adds hipMemsetParams to output */
  hipGraphDebugDotFlagsHostNodeParams = 1 << 5,   /**< Adds hipHostNodeParams to output */
  hipGraphDebugDotFlagsEventNodeParams =
      1 << 6, /**< Adds hipEvent_t handle from record and wait nodes to output */
  hipGraphDebugDotFlagsExtSemasSignalNodeParams =
      1 << 7, /**< Adds hipExternalSemaphoreSignalNodeParams values to output */
  hipGraphDebugDotFlagsExtSemasWaitNodeParams =
      1 << 8, /**< Adds hipExternalSemaphoreWaitNodeParams to output */
  hipGraphDebugDotFlagsKernelNodeAttributes =
      1 << 9, /**< Adds hipKernelNodeAttrID values to output */
  hipGraphDebugDotFlagsHandles =
      1 << 10 /**< Adds node handles and every kernel function handle to output */
};

// Memory pool access types (mirroring CUDA's CUmemAccess* types).
enum class hipMemAccessFlags {
  hipMemAccessFlagsProtNone = 0x0,
  hipMemAccessFlagsProtRead = 0x1,
  hipMemAccessFlagsProtReadWrite = 0x3,
};

enum class hipMemLocationType {
  hipMemLocationTypeInvalid = 0x0,
  hipMemLocationTypeDevice = 0x1,
};

using hipMemLocation = struct hipMemLocation_st {
  hipMemLocationType type;
  int id;
};

using hipMemAccessDesc = struct hipMemAccessDesc_st {
  hipMemLocation location;
  hipMemAccessFlags flags;
};

enum class hipMemcpySrcAccessOrder {
  hipMemcpySrcAccessOrderInvalid = 0,
  hipMemcpySrcAccessOrderStream = 1,
  hipMemcpySrcAccessOrderDuringApiCall = 2,
};

using hipMemcpyAttributes = struct hipMemcpyAttributes_st {
  hipMemcpySrcAccessOrder srcAccessOrder; 
  hipMemLocation srcLocHint;
  hipMemLocation dstLocHint;
  unsigned int flags; 
};

// Virtual Memory Management types (mirroring CUDA's CUmem* VMM types). Used by
// the defragmenting allocator; see
// MLRT/lib/Driver/DeviceContext/MemoryManager.h. Layouts must match
// hip_runtime_api.h exactly since the driver is loaded via dlsym.
using hipMemGenericAllocationHandle_t = struct ihipMemGenericAllocationHandle *;

enum class hipMemAllocationType {
  hipMemAllocationTypeInvalid = 0x0,
  hipMemAllocationTypePinned = 0x1,
  hipMemAllocationTypeMax = 0x7FFFFFFF,
};

enum class hipMemAllocationHandleType {
  hipMemHandleTypeNone = 0x0,
  hipMemHandleTypePosixFileDescriptor = 0x1,
  hipMemHandleTypeWin32 = 0x2,
  hipMemHandleTypeWin32Kmt = 0x4,
};

enum class hipMemAllocationGranularity_flags {
  hipMemAllocationGranularityMinimum = 0x0,
  hipMemAllocationGranularityRecommended = 0x1,
};

using hipMemAllocationProp = struct hipMemAllocationProp_st {
  hipMemAllocationType type;
  hipMemAllocationHandleType requestedHandleType;
  hipMemLocation location;
  void *win32HandleMetaData;
  struct {
    unsigned char compressionType;
    unsigned char gpuDirectRDMACapable;
    unsigned short usage;
  } allocFlags;
};

/// Minimal dim3 definition for graph node parameter structs.
/// Matches the layout of the HIP/CUDA runtime dim3 type.
struct hipDim3 {
  unsigned int x = 1;
  unsigned int y = 1;
  unsigned int z = 1;
};

/// Kernel node parameters for hipGraphAddKernelNode.
/// Layout matches hipKernelNodeParams from hip_runtime_api.h.
using hipKernelNodeParams = struct hipKernelNodeParams_t {
  hipDim3 blockDim;        ///< Block dimensions (threads per block)
  void **extra;            ///< Extra launch options (typically nullptr)
  void *func;              ///< Kernel function pointer
  hipDim3 gridDim;         ///< Grid dimensions (blocks per grid)
  void **kernelParams;     ///< Array of pointers to kernel arguments
  unsigned int sharedMemBytes; ///< Dynamic shared-memory size in bytes
};

/// Memcpy direction for hipGraphAddMemcpyNode1D.
/// Layout matches hipMemcpyKind from hip_runtime_api.h.
enum class hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4,
};

/// Allocation flags for hipHostMalloc.
enum class hipHostMallocFlags : unsigned int {
  hipHostMallocDefault = 0,
  hipHostMallocPortable = 0x1,
  hipHostMallocMapped = 0x2,
  hipHostMallocWriteCombined = 0x4,
};

inline hipHostMallocFlags operator|(hipHostMallocFlags a, hipHostMallocFlags b)
{
    return static_cast<hipHostMallocFlags>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

/// Memset node parameters for hipGraphAddMemsetNode.
/// Layout matches hipMemsetParams from hip_runtime_api.h.
using hipMemsetParams = struct hipMemsetParams_t {
  void *dst;
  size_t elementSize;
  size_t height;
  size_t pitch;
  unsigned int value;
  size_t width;
};

/// Flags for hipStreamWaitValue.
enum class hipStreamWaitValueFlags : unsigned int {
  hipStreamWaitValueGte = 0,
  hipStreamWaitValueEq = 1,
  hipStreamWaitValueAnd = 2,
  hipStreamWaitValueNor = 3,
};

/// Batch memory operation kinds.
/// Layout matches hipStreamBatchMemOpType from hip_runtime_api.h.
enum class hipStreamBatchMemOpType : unsigned int {
  hipStreamMemOpWaitValue32 = 0x1,
  hipStreamMemOpWriteValue32 = 0x2,
  hipStreamMemOpWaitValue64 = 0x4,
  hipStreamMemOpWriteValue64 = 0x5,
};

/// A single batch memory operation. Layout matches the
/// hipStreamBatchMemOpParams union from hip_runtime_api.h; only the wait-value
/// member is modelled, and `pad` keeps the size in step with the runtime's
/// union so an array of these can be handed to the driver.
union hipStreamBatchMemOpParams {
  struct {
    hipStreamBatchMemOpType operation;
    hipDeviceptr_t address;
    uint64_t value64;
    unsigned int flags;
    hipDeviceptr_t alias; ///< Unused by the AMD backend.
  } waitValue;
  uint64_t pad[6];
};

/// Node parameters for hipGraphAddBatchMemOpNode.
/// Layout matches hipBatchMemOpNodeParams from hip_runtime_api.h.
/// The driver does not deep-copy `paramArray`: it dereferences the array at
/// graph launch, so the pointee must outlive every hipGraphExec_t built from
/// the graph this node was added to.
using hipBatchMemOpNodeParams = struct hipBatchMemOpNodeParams_t {
  hipCtx_t ctx;
  unsigned int count;
  hipStreamBatchMemOpParams *paramArray;
  unsigned int flags;
};

/// Flags for hipStreamUpdateCaptureDependencies.
enum class hipStreamUpdateCaptureDependenciesFlags : unsigned int {
  hipStreamAddCaptureDependencies = 0,
  hipStreamSetCaptureDependencies = 1,
};

// HSA Runtime types (for getting device architecture info).
using hsaStatus_t = uint32_t;
using hsaAgent_t = uint64_t;
using hsaIsa_t = uint64_t;

// COPIED FROM hsa.h

enum class hsaAgentInfo_t {
  /**
   * Agent name. The type of this attribute is a NUL-terminated char[64]. The
   * name must be at most 63 characters long (not including the NUL terminator)
   * and all array elements not used for the name must be NUL.
   */
  HSA_AGENT_INFO_NAME = 0,
  /**
   * Name of vendor. The type of this attribute is a NUL-terminated char[64].
   * The name must be at most 63 characters long (not including the NUL
   * terminator) and all array elements not used for the name must be NUL.
   */
  HSA_AGENT_INFO_VENDOR_NAME = 1,
  /**
   * Agent capability. The type of this attribute is ::hsa_agent_feature_t.
   */
  HSA_AGENT_INFO_FEATURE = 2,
  /**
   * @deprecated Query ::HSA_ISA_INFO_MACHINE_MODELS for a given instruction set
   * architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Machine model supported by the agent. The type of this attribute is
   * ::hsa_machine_model_t.
   */
  HSA_AGENT_INFO_MACHINE_MODEL = 3,
  /**
   * @deprecated Query ::HSA_ISA_INFO_PROFILES for a given instruction set
   * architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Profile supported by the agent. The type of this attribute is
   * ::hsa_profile_t.
   */
  HSA_AGENT_INFO_PROFILE = 4,
  /**
   * @deprecated Query ::HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES for a given
   * instruction set architecture supported by the agent instead.  If more than
   * one ISA is supported by the agent, the returned value corresponds to the
   * first ISA enumerated by ::hsa_agent_iterate_isas.
   *
   * Default floating-point rounding mode. The type of this attribute is
   * ::hsa_default_float_rounding_mode_t, but the value
   * ::HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT is not allowed.
   */
  HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 5,
  /**
   * @deprecated Query ::HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES
   * for a given instruction set architecture supported by the agent instead.  If
   * more than one ISA is supported by the agent, the returned value corresponds
   * to the first ISA enumerated by ::hsa_agent_iterate_isas.
   *
   * A bit-mask of ::hsa_default_float_rounding_mode_t values, representing the
   * default floating-point rounding modes supported by the agent in the Base
   * profile. The type of this attribute is uint32_t. The default floating-point
   * rounding mode (::HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE) bit must not
   * be set.
   */
  HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 23,
  /**
   * @deprecated Query ::HSA_ISA_INFO_FAST_F16_OPERATION for a given instruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Flag indicating that the f16 HSAIL operation is at least as fast as the
   * f32 operation in the current agent. The value of this attribute is
   * undefined if the agent is not a kernel agent. The type of this
   * attribute is bool.
   */
  HSA_AGENT_INFO_FAST_F16_OPERATION = 24,
  /**
   * @deprecated Query ::HSA_WAVEFRONT_INFO_SIZE for a given wavefront and
   * instruction set architecture supported by the agent instead.  If more than
   * one ISA is supported by the agent, the returned value corresponds to the
   * first ISA enumerated by ::hsa_agent_iterate_isas and the first wavefront
   * enumerated by ::hsa_isa_iterate_wavefronts for that ISA.
   *
   * Number of work-items in a wavefront. Must be a power of 2 in the range
   * [1,256]. The value of this attribute is undefined if the agent is not
   * a kernel agent. The type of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_WAVEFRONT_SIZE = 6,
  /**
   * @deprecated Query ::HSA_ISA_INFO_WORKGROUP_MAX_DIM for a given instruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum number of work-items of each dimension of a work-group.  Each
   * maximum must be greater than 0. No maximum can exceed the value of
   * ::HSA_AGENT_INFO_WORKGROUP_MAX_SIZE. The value of this attribute is
   * undefined if the agent is not a kernel agent. The type of this
   * attribute is uint16_t[3].
   */
  HSA_AGENT_INFO_WORKGROUP_MAX_DIM = 7,
  /**
   * @deprecated Query ::HSA_ISA_INFO_WORKGROUP_MAX_SIZE for a given instruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum total number of work-items in a work-group. The value of this
   * attribute is undefined if the agent is not a kernel agent. The type
   * of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_WORKGROUP_MAX_SIZE = 8,
  /**
   * @deprecated Query ::HSA_ISA_INFO_GRID_MAX_DIM for a given instruction set
   * architecture supported by the agent instead.
   *
   * Maximum number of work-items of each dimension of a grid. Each maximum must
   * be greater than 0, and must not be smaller than the corresponding value in
   * ::HSA_AGENT_INFO_WORKGROUP_MAX_DIM. No maximum can exceed the value of
   * ::HSA_AGENT_INFO_GRID_MAX_SIZE. The value of this attribute is undefined
   * if the agent is not a kernel agent. The type of this attribute is
   * ::hsa_dim3_t.
   */
  HSA_AGENT_INFO_GRID_MAX_DIM = 9,
  /**
   * @deprecated Query ::HSA_ISA_INFO_GRID_MAX_SIZE for a given instruction set
   * architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum total number of work-items in a grid. The value of this attribute
   * is undefined if the agent is not a kernel agent. The type of this
   * attribute is uint32_t.
   */
  HSA_AGENT_INFO_GRID_MAX_SIZE = 10,
  /**
   * @deprecated Query ::HSA_ISA_INFO_FBARRIER_MAX_SIZE for a given instruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum number of fbarriers per work-group. Must be at least 32. The value
   * of this attribute is undefined if the agent is not a kernel agent. The
   * type of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_FBARRIER_MAX_SIZE = 11,
  /**
   * @deprecated The maximum number of queues is not statically determined.
   *
   * Maximum number of queues that can be active (created but not destroyed) at
   * one time in the agent. The type of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_QUEUES_MAX = 12,
  /**
   * Minimum number of packets that a queue created in the agent
   * can hold. Must be a power of 2 greater than 0. Must not exceed
   * the value of ::HSA_AGENT_INFO_QUEUE_MAX_SIZE. The type of this
   * attribute is uint32_t.
   */
  HSA_AGENT_INFO_QUEUE_MIN_SIZE = 13,
  /**
   * Maximum number of packets that a queue created in the agent can
   * hold. Must be a power of 2 greater than 0. The type of this attribute
   * is uint32_t.
   */
  HSA_AGENT_INFO_QUEUE_MAX_SIZE = 14,
  /**
   * Type of a queue created in the agent. The type of this attribute is
   * ::hsa_queue_type32_t.
   */
  HSA_AGENT_INFO_QUEUE_TYPE = 15,
  /**
   * @deprecated NUMA information is not exposed anywhere else in the API.
   *
   * Identifier of the NUMA node associated with the agent. The type of this
   * attribute is uint32_t.
   */
  HSA_AGENT_INFO_NODE = 16,
  /**
   * Type of hardware device associated with the agent. The type of this
   * attribute is ::hsaDeviceType_t.
   */
  HSA_AGENT_INFO_DEVICE = 17,
  /**
   * @deprecated Query ::hsa_agent_iterate_caches to retrieve information about
   * the caches present in a given agent.
   *
   * Array of data cache sizes (L1..L4). Each size is expressed in bytes. A size
   * of 0 for a particular level indicates that there is no cache information
   * for that level. The type of this attribute is uint32_t[4].
   */
  HSA_AGENT_INFO_CACHE_SIZE = 18,
  /**
   * @deprecated An agent may support multiple instruction set
   * architectures. See ::hsa_agent_iterate_isas.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Instruction set architecture of the agent. The type of this attribute
   * is ::hsaIsa_t.
   */
  HSA_AGENT_INFO_ISA = 19,
  /**
   * Bit-mask indicating which extensions are supported by the agent. An
   * extension with an ID of @p i is supported if the bit at position @p i is
   * set. The type of this attribute is uint8_t[128].
   */
  HSA_AGENT_INFO_EXTENSIONS = 20,
  /**
   * Major version of the HSA runtime specification supported by the
   * agent. The type of this attribute is uint16_t.
   */
  HSA_AGENT_INFO_VERSION_MAJOR = 21,
  /**
   * Minor version of the HSA runtime specification supported by the
   * agent. The type of this attribute is uint16_t.
   */
  HSA_AGENT_INFO_VERSION_MINOR = 22,
  /**
   * This enum does not have a fixed underlying type, thus in C++ post D2338:
   * If the enumeration type does not have a fixed underlying type, the value is
   * unchanged if the original value is within the range of the enumeration
   * values (9.7.1 [dcl.enum]), and otherwise, the behavior is
   * undefined.
   * Thus increase the range of this enum to encompass vendor extensions.
   */
  HSA_AGENT_INFO_LAST = INT32_MAX,

  /**
   * PCIe domain for the agent.  Pairs with HSA_AMD_AGENT_INFO_BDFID
   * to give the full physical location of the Agent.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_DOMAIN = 0xA00F,

  /**
   * Agent BDF_ID, named LocationID in thunk. The type of this attribute is
   * uint32_t.
   */
  HSA_AMD_AGENT_INFO_BDFID = 0xA006,

};

constexpr hsaStatus_t HSA_STATUS_SUCCESS = 0x0;
constexpr hsaStatus_t HSA_STATUS_INFO_BREAK = 0x1;

enum class hsaIsaInfo_t {
  /**
   * The length of the ISA name in bytes, not including the NUL terminator. The
   * type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_NAME_LENGTH = 0,
  /**
   * Human-readable description.  The type of this attribute is character array
   * with the length equal to the value of ::HSA_ISA_INFO_NAME_LENGTH attribute.
   */
  HSA_ISA_INFO_NAME = 1,
  /**
   * @deprecated
   *
   * Number of call conventions supported by the instruction set architecture.
   * Must be greater than zero. The type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_CALL_CONVENTION_COUNT = 2,
  /**
   * @deprecated
   *
   * Number of work-items in a wavefront for a given call convention. Must be a
   * power of 2 in the range [1,256]. The type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE = 3,
  /**
   * @deprecated
   *
   * Number of wavefronts per compute unit for a given call convention. In
   * practice, other factors (for example, the amount of group memory used by a
   * work-group) may further limit the number of wavefronts per compute
   * unit. The type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT = 4,
  /**
   * Machine models supported by the instruction set architecture. The type of
   * this attribute is a bool[2]. If the ISA supports the small machine model,
   * the element at index ::HSA_MACHINE_MODEL_SMALL is true. If the ISA supports
   * the large model, the element at index ::HSA_MACHINE_MODEL_LARGE is true.
   */
  HSA_ISA_INFO_MACHINE_MODELS = 5,
  /**
   * Profiles supported by the instruction set architecture. The type of this
   * attribute is a bool[2]. If the ISA supports the base profile, the element
   * at index ::HSA_PROFILE_BASE is true. If the ISA supports the full profile,
   * the element at index ::HSA_PROFILE_FULL is true.
   */
  HSA_ISA_INFO_PROFILES = 6,
  /**
   * Default floating-point rounding modes supported by the instruction set
   * architecture. The type of this attribute is a bool[3]. The value at a given
   * index is true if the corresponding rounding mode in
   * ::hsa_default_float_rounding_mode_t is supported. At least one default mode
   * has to be supported.
   *
   * If the default mode is supported, then
   * ::HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES must report that
   * both the zero and the near roundings modes are supported.
   */
  HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES = 7,
  /**
   * Default floating-point rounding modes supported by the instruction set
   * architecture in the Base profile. The type of this attribute is a
   * bool[3]. The value at a given index is true if the corresponding rounding
   * mode in ::hsa_default_float_rounding_mode_t is supported. The value at
   * index HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT must be false.  At least one
   * of the values at indexes ::HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO or
   * HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR must be true.
   */
  HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 8,
  /**
   * Flag indicating that the f16 HSAIL operation is at least as fast as the
   * f32 operation in the instruction set architecture. The type of this
   * attribute is bool.
   */
  HSA_ISA_INFO_FAST_F16_OPERATION = 9,
  /**
   * Maximum number of work-items of each dimension of a work-group.  Each
   * maximum must be greater than 0. No maximum can exceed the value of
   * ::HSA_ISA_INFO_WORKGROUP_MAX_SIZE. The type of this attribute is
   * uint16_t[3].
   */
  HSA_ISA_INFO_WORKGROUP_MAX_DIM = 12,
  /**
   * Maximum total number of work-items in a work-group. The type
   * of this attribute is uint32_t.
   */
  HSA_ISA_INFO_WORKGROUP_MAX_SIZE = 13,
  /**
   * Maximum number of work-items of each dimension of a grid. Each maximum must
   * be greater than 0, and must not be smaller than the corresponding value in
   * ::HSA_ISA_INFO_WORKGROUP_MAX_DIM. No maximum can exceed the value of
   * ::HSA_ISA_INFO_GRID_MAX_SIZE. The type of this attribute is
   * ::hsa_dim3_t.
   */
  HSA_ISA_INFO_GRID_MAX_DIM = 14,
  /**
   * Maximum total number of work-items in a grid. The type of this
   * attribute is uint64_t.
   */
  HSA_ISA_INFO_GRID_MAX_SIZE = 16,
  /**
   * Maximum number of fbarriers per work-group. Must be at least 32. The
   * type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_FBARRIER_MAX_SIZE = 17
};

// COPIED FROM hip_runtime_api.h

enum class hipDeviceAttribute_t {
    hipDeviceAttributeCudaCompatibleBegin = 0,

    hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin, ///< Whether ECC support is enabled.
    hipDeviceAttributeAccessPolicyMaxWindowSize,        ///< Cuda only. The maximum size of the window policy in bytes.
    hipDeviceAttributeAsyncEngineCount,                 ///< Asynchronous engines number.
    hipDeviceAttributeCanMapHostMemory,                 ///< Whether host memory can be mapped into device address space
    hipDeviceAttributeCanUseHostPointerForRegisteredMem,///< Device can access host registered memory
                                                        ///< at the same virtual address as the CPU
    hipDeviceAttributeClockRate,                        ///< Peak clock frequency in kilohertz.
    hipDeviceAttributeComputeMode,                      ///< Compute mode that device is currently in.
    hipDeviceAttributeComputePreemptionSupported,       ///< Device supports Compute Preemption.
    hipDeviceAttributeConcurrentKernels,                ///< Device can possibly execute multiple kernels concurrently.
    hipDeviceAttributeConcurrentManagedAccess,          ///< Device can coherently access managed memory concurrently with the CPU
    hipDeviceAttributeCooperativeLaunch,                ///< Support cooperative launch
    hipDeviceAttributeCooperativeMultiDeviceLaunch,     ///< Support cooperative launch on multiple devices
    hipDeviceAttributeDeviceOverlap,                    ///< Device can concurrently copy memory and execute a kernel.
                                                        ///< Deprecated. Use instead asyncEngineCount.
    hipDeviceAttributeDirectManagedMemAccessFromHost,   ///< Host can directly access managed memory on
                                                        ///< the device without migration
    hipDeviceAttributeGlobalL1CacheSupported,           ///< Device supports caching globals in L1
    hipDeviceAttributeHostNativeAtomicSupported,        ///< Link between the device and the host supports native atomic operations
    hipDeviceAttributeIntegrated,                       ///< Device is integrated GPU
    hipDeviceAttributeIsMultiGpuBoard,                  ///< Multiple GPU devices.
    hipDeviceAttributeKernelExecTimeout,                ///< Run time limit for kernels executed on the device
    hipDeviceAttributeL2CacheSize,                      ///< Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
    hipDeviceAttributeLocalL1CacheSupported,            ///< caching locals in L1 is supported
    hipDeviceAttributeLuid,                             ///< 8-byte locally unique identifier in 8 bytes. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeLuidDeviceNodeMask,               ///< Luid device node mask. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeComputeCapabilityMajor,           ///< Major compute capability version number.
    hipDeviceAttributeManagedMemory,                    ///< Device supports allocating managed memory on this system
    hipDeviceAttributeMaxBlocksPerMultiProcessor,       ///< Max block size per multiprocessor
    hipDeviceAttributeMaxBlockDimX,                     ///< Max block size in width.
    hipDeviceAttributeMaxBlockDimY,                     ///< Max block size in height.
    hipDeviceAttributeMaxBlockDimZ,                     ///< Max block size in depth.
    hipDeviceAttributeMaxGridDimX,                      ///< Max grid size  in width.
    hipDeviceAttributeMaxGridDimY,                      ///< Max grid size  in height.
    hipDeviceAttributeMaxGridDimZ,                      ///< Max grid size  in depth.
    hipDeviceAttributeMaxSurface1D,                     ///< Maximum size of 1D surface.
    hipDeviceAttributeMaxSurface1DLayered,              ///< Cuda only. Maximum dimensions of 1D layered surface.
    hipDeviceAttributeMaxSurface2D,                     ///< Maximum dimension (width, height) of 2D surface.
    hipDeviceAttributeMaxSurface2DLayered,              ///< Cuda only. Maximum dimensions of 2D layered surface.
    hipDeviceAttributeMaxSurface3D,                     ///< Maximum dimension (width, height, depth) of 3D surface.
    hipDeviceAttributeMaxSurfaceCubemap,                ///< Cuda only. Maximum dimensions of Cubemap surface.
    hipDeviceAttributeMaxSurfaceCubemapLayered,         ///< Cuda only. Maximum dimension of Cubemap layered surface.
    hipDeviceAttributeMaxTexture1DWidth,                ///< Maximum size of 1D texture.
    hipDeviceAttributeMaxTexture1DLayered,              ///< Maximum dimensions of 1D layered texture.
    hipDeviceAttributeMaxTexture1DLinear,               ///< Maximum number of elements allocatable in a 1D linear texture.
                                                        ///< Use cudaDeviceGetTexture1DLinearMaxWidth() instead on Cuda.
    hipDeviceAttributeMaxTexture1DMipmap,               ///< Maximum size of 1D mipmapped texture.
    hipDeviceAttributeMaxTexture2DWidth,                ///< Maximum dimension width of 2D texture.
    hipDeviceAttributeMaxTexture2DHeight,               ///< Maximum dimension height of 2D texture.
    hipDeviceAttributeMaxTexture2DGather,               ///< Maximum dimensions of 2D texture if gather operations  performed.
    hipDeviceAttributeMaxTexture2DLayered,              ///< Maximum dimensions of 2D layered texture.
    hipDeviceAttributeMaxTexture2DLinear,               ///< Maximum dimensions (width, height, pitch) of 2D textures bound to pitched memory.
    hipDeviceAttributeMaxTexture2DMipmap,               ///< Maximum dimensions of 2D mipmapped texture.
    hipDeviceAttributeMaxTexture3DWidth,                ///< Maximum dimension width of 3D texture.
    hipDeviceAttributeMaxTexture3DHeight,               ///< Maximum dimension height of 3D texture.
    hipDeviceAttributeMaxTexture3DDepth,                ///< Maximum dimension depth of 3D texture.
    hipDeviceAttributeMaxTexture3DAlt,                  ///< Maximum dimensions of alternate 3D texture.
    hipDeviceAttributeMaxTextureCubemap,                ///< Maximum dimensions of Cubemap texture
    hipDeviceAttributeMaxTextureCubemapLayered,         ///< Maximum dimensions of Cubemap layered texture.
    hipDeviceAttributeMaxThreadsDim,                    ///< Maximum dimension of a block
    hipDeviceAttributeMaxThreadsPerBlock,               ///< Maximum number of threads per block.
    hipDeviceAttributeMaxThreadsPerMultiProcessor,      ///< Maximum resident threads per multiprocessor.
    hipDeviceAttributeMaxPitch,                         ///< Maximum pitch in bytes allowed by memory copies
    hipDeviceAttributeMemoryBusWidth,                   ///< Global memory bus width in bits.
    hipDeviceAttributeMemoryClockRate,                  ///< Peak memory clock frequency in kilohertz.
    hipDeviceAttributeComputeCapabilityMinor,           ///< Minor compute capability version number.
    hipDeviceAttributeMultiGpuBoardGroupID,             ///< Unique ID of device group on the same multi-GPU board
    hipDeviceAttributeMultiprocessorCount,              ///< Number of multiprocessors on the device.
    hipDeviceAttributeUnused1,                          ///< Previously hipDeviceAttributeName
    hipDeviceAttributePageableMemoryAccess,             ///< Device supports coherently accessing pageable memory
                                                        ///< without calling hipHostRegister on it
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables, ///< Device accesses pageable memory via the host's page tables
    hipDeviceAttributePciBusId,                         ///< PCI Bus ID.
    hipDeviceAttributePciDeviceId,                      ///< PCI Device ID.
    hipDeviceAttributePciDomainID,                      ///< PCI Domain ID.
    hipDeviceAttributePersistingL2CacheMaxSize,         ///< Maximum l2 persisting lines capacity in bytes
    hipDeviceAttributeMaxRegistersPerBlock,             ///< 32-bit registers available to a thread block. This number is shared
                                                        ///< by all thread blocks simultaneously resident on a multiprocessor.
    hipDeviceAttributeMaxRegistersPerMultiprocessor,    ///< 32-bit registers available per block.
    hipDeviceAttributeReservedSharedMemPerBlock,        ///< Shared memory reserved by CUDA driver per block.
    hipDeviceAttributeMaxSharedMemoryPerBlock,          ///< Maximum shared memory available per block in bytes.
    hipDeviceAttributeSharedMemPerBlockOptin,           ///< Maximum shared memory per block usable by special opt in.
    hipDeviceAttributeSharedMemPerMultiprocessor,       ///< Shared memory available per multiprocessor.
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio, ///< Cuda only. Performance ratio of single precision to double precision.
    hipDeviceAttributeStreamPrioritiesSupported,        ///< Whether to support stream priorities.
    hipDeviceAttributeSurfaceAlignment,                 ///< Alignment requirement for surfaces
    hipDeviceAttributeTccDriver,                        ///< Cuda only. Whether device is a Tesla device using TCC driver
    hipDeviceAttributeTextureAlignment,                 ///< Alignment requirement for textures
    hipDeviceAttributeTexturePitchAlignment,            ///< Pitch alignment requirement for 2D texture references bound to pitched memory;
    hipDeviceAttributeTotalConstantMemory,              ///< Constant memory size in bytes.
    hipDeviceAttributeTotalGlobalMem,                   ///< Global memory available on devicice.
    hipDeviceAttributeUnifiedAddressing,                ///< Cuda only. An unified address space shared with the host.
    hipDeviceAttributeUnused2,                          ///< Previously hipDeviceAttributeUuid
    hipDeviceAttributeWarpSize,                         ///< Warp size in threads.
    hipDeviceAttributeMemoryPoolsSupported,             ///< Device supports HIP Stream Ordered Memory Allocator
    hipDeviceAttributeVirtualMemoryManagementSupported, ///< Device supports HIP virtual memory management
    hipDeviceAttributeHostRegisterSupported,            ///< Can device support host memory registration via hipHostRegister
    hipDeviceAttributeMemoryPoolSupportedHandleTypes,   ///< Supported handle mask for HIP Stream Ordered Memory Allocator

    hipDeviceAttributeCudaCompatibleEnd = 9999,
    hipDeviceAttributeAmdSpecificBegin = 10000,

    hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin,  ///< Frequency in khz of the timer used by the device-side "clock*"
    hipDeviceAttributeUnused3,                                  ///< Previously hipDeviceAttributeArch
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,         ///< Maximum Shared Memory PerMultiprocessor.
    hipDeviceAttributeUnused4,                                  ///< Previously hipDeviceAttributeGcnArch
    hipDeviceAttributeUnused5,                                  ///< Previously hipDeviceAttributeGcnArchName
    hipDeviceAttributeHdpMemFlushCntl,                          ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpRegFlushCntl,                          ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,      ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched functions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,   ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched grid dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,  ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched block dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem, ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched shared memories
    hipDeviceAttributeIsLargeBar,                               ///< Whether it is LargeBar
    hipDeviceAttributeAsicRevision,                             ///< Revision of the GPU in this device
    hipDeviceAttributeCanUseStreamWaitValue,                    ///< '1' if Device supports hipStreamWaitValue32() and
                                                                ///< hipStreamWaitValue64(), '0' otherwise.
    hipDeviceAttributeImageSupport,                             ///< '1' if Device supports image, '0' otherwise.
    hipDeviceAttributePhysicalMultiProcessorCount,              ///< All available physical compute
                                                                ///< units for the device
    hipDeviceAttributeFineGrainSupport,                         ///< '1' if Device supports fine grain, '0' otherwise
    hipDeviceAttributeWallClockRate,                            ///< Constant frequency of wall clock in kilohertz.

    hipDeviceAttributeAmdSpecificEnd = 19999,
    hipDeviceAttributeVendorSpecificBegin = 20000,
    // Extended attributes for vendors
};

 /**
 * hipMemoryType (for pointer attributes)
 *
 * @note hipMemoryType enum values are combination of cudaMemoryType and cuMemoryType and AMD specific enum values.
 *
 */
enum class hipMemoryType {
    hipMemoryTypeUnregistered = 0,  ///< Unregistered memory
    hipMemoryTypeHost         = 1,  ///< Memory is physically located on host
    hipMemoryTypeDevice       = 2,  ///< Memory is physically located on device. (see deviceId for
                                    ///< specific device)
    hipMemoryTypeManaged      = 3,  ///< Managed memory, automatically managed by the unified
                                    ///< memory system
                                    ///< place holder for new values.
    hipMemoryTypeArray        = 10, ///< Array memory, physically located on device. (see deviceId for
                                    ///< specific device)
    hipMemoryTypeUnified      = 11  ///< unified address space

};

/**
 * Pointer attributes
 */
using hipPointerAttribute_t = struct hipPointerAttribute_t {
    enum hipMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    unsigned allocationFlags; /* flags specified when memory was allocated*/
    /* peers? */
};

// clang-format on

enum class hipJitOption {
  JIT_MAX_REGISTERS = 0,
  JIT_THREADS_PER_BLOCK = 1,
  JIT_OPTIMIZATION_LEVEL = 7,
  JIT_GENERATE_DEBUG_INFO = 11,
  JIT_GENERATE_LINE_INFO = 13,
  JIT_CACHE_MODE = 14,
};

enum class hipFunction_attribute {
  FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
};

enum class hipFuncCache_t {
  FUNC_CACHE_PREFER_NONE = 0,
  FUNC_CACHE_PREFER_SHARED = 1,
  FUNC_CACHE_PREFER_L1 = 2,
  FUNC_CACHE_PREFER_EQUAL = 3,
};

class HIPDriver {
public:
  enum Result {
    SUCCESS = 0,
    NOT_READY = 600,
  };

#define HIP_FUNC(name, ret, ...)                                               \
  ret name(EXPAND_PARAMS(__VA_ARGS__)) const {                                 \
    TRACY_ZONE_SCOPED_NC(#name, TRACY_COLOR_RED);                              \
    return name##_(EXPAND_ARGS(__VA_ARGS__));                                  \
  }
#include "HIPDriver.inc"
#undef HIP_OPT_FUNC
#undef HIP_FUNC

  bool hasMemcpyBatchAsync() const { return hipMemcpyBatchAsync_ != nullptr; }

  /// Whether this driver can record a value-wait into a graph being captured.
  /// hipGraphAddBatchMemOpNode only appeared in ROCm 6.4, and the capture-info
  /// entry points it has to be paired with are equally optional here so that an
  /// older runtime degrades to a diagnosable error rather than failing to load.
  bool hasBatchMemOpGraphNode() const {
    return hipGraphAddBatchMemOpNode_ != nullptr &&
           hipStreamGetCaptureInfo_v2_ != nullptr &&
           hipStreamUpdateCaptureDependencies_ != nullptr &&
           hipCtxGetCurrent_ != nullptr;
  }

  // Declared by hand: HIP_FUNC cannot take a zero-argument entry point, since
  // an empty argument list still expands to one parameter.
  Result hipGetLastError() const { return hipGetLastError_(); }

  // HSA Runtime methods for getting device architecture info
  hsaStatus_t HsaInit() const {
    return hsa_init_ ? hsa_init_() : 1; // Return error if not available
  }

  hsaStatus_t HsaShutDown() const {
    return hsa_shut_down_ ? hsa_shut_down_() : 1;
  }

  hsaStatus_t HsaIterateAgents(hsaStatus_t (*callback)(hsaAgent_t agent,
                                                       void *data),
                               void *data) const {
    return hsa_iterate_agents_ ? hsa_iterate_agents_(callback, data) : 1;
  }

  hsaStatus_t HsaAgentGetInfo(hsaAgent_t agent, hsaAgentInfo_t attribute,
                              void *value) const {
    return hsa_agent_get_info_ ? hsa_agent_get_info_(agent, attribute, value)
                               : 1;
  }

  hsaStatus_t HsaAgentIterateIsas(hsaAgent_t agent,
                                  hsaStatus_t (*callback)(hsaIsa_t isa,
                                                          void *data),
                                  void *data) const {
    return hsa_agent_iterate_isas_
               ? hsa_agent_iterate_isas_(agent, callback, data)
               : 1;
  }

  hsaStatus_t HsaIsaGetInfo(hsaIsa_t isa, hsaIsaInfo_t attribute,
                            void *value) const {
    return hsa_isa_get_info_alt_ ? hsa_isa_get_info_alt_(isa, attribute, value)
                                 : 1;
  }

  static ErrorOr<const HIPDriver &> get();

  // HIPDriver object lifetime is tied to a static unique_ptr.
  HIPDriver() = default;
  ~HIPDriver() = default;

private:
  // Shared library name for the HIP driver.
  static const char lib_name[];

#define HIP_FUNC(name, ret, ...)                                               \
  using t##name = ret(__VA_ARGS__);                                            \
  t##name *name##_;
#include "HIPDriver.inc"
#undef HIP_FUNC

  using thipGetLastError = Result();
  thipGetLastError *hipGetLastError_;

  // HSA Runtime function pointers (optional - may not be available).
  using thsa_init = hsaStatus_t();
  thsa_init *hsa_init_;

  using thsa_shut_down = hsaStatus_t();
  thsa_shut_down *hsa_shut_down_;

  using thsa_iterate_agents = hsaStatus_t(
      hsaStatus_t (*callback)(hsaAgent_t agent, void *data), void *data);
  thsa_iterate_agents *hsa_iterate_agents_;

  using thsa_agent_get_info = hsaStatus_t(hsaAgent_t agent,
                                          hsaAgentInfo_t attribute,
                                          void *value);
  thsa_agent_get_info *hsa_agent_get_info_;

  using thsa_agent_iterate_isas = hsaStatus_t(
      hsaAgent_t agent, hsaStatus_t (*callback)(hsaIsa_t isa, void *data),
      void *data);
  thsa_agent_iterate_isas *hsa_agent_iterate_isas_;

  using thsa_isa_get_info_alt = hsaStatus_t(hsaIsa_t isa,
                                            hsaIsaInfo_t attribute,
                                            void *value);
  thsa_isa_get_info_alt *hsa_isa_get_info_alt_;

  static ErrorOr<std::unique_ptr<HIPDriver>> allocAndInitDriver();
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDRIVER_H

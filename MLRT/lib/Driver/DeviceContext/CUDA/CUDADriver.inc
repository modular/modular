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
// This file lists CUDA driver functions along with their types. This is used
// by the CUDADriver which defines CUDA_FUNC macros and then include this to
// define wrappers, types, and so on, for all the CUDA APIs easily.
//
// To add a new CUDA APIs to the driver class, simply add an entry to this file
// in the format:
//
//     CUDA_FUNC(<API name>, <return type>, <parameter type>...)
//
// Note: For parameters it should be the type only, no parameter names.
//
// Note: If a new API has more than 15 parameters, the macros in
// Support/Preprocessor.h will need to be updated.
//
//===----------------------------------------------------------------------===//

// No macro guards since this is meant to be included multiple times, but make
// sure CUDA_FUNC is defined.
#ifndef CUDA_FUNC
#error "CUDADriver.inc included without defining CUDA_FUNC"
#endif

// Optional entries: CUDA_OPT_FUNC stores nullptr on dlsym miss instead of
// returning an error. Used for APIs not present on every supported driver
// version (e.g. cuMemcpyBatchAsync introduced in CUDA 12.8). If undefined by
// the includer, fall back to the required CUDA_FUNC to keep old includers
// working.
#ifndef CUDA_OPT_FUNC
#define CUDA_OPT_FUNC CUDA_FUNC
#define CUDA_OPT_FUNC_DEFINED_LOCALLY
#endif

CUDA_FUNC(cuInit, Result, unsigned int)

CUDA_FUNC(cuGetErrorName, Result, Result, const char **)
CUDA_FUNC(cuGetErrorString, Result, Result, const char **)

CUDA_FUNC(cuDriverGetVersion, Result, int *)

CUDA_FUNC(cuDeviceGet, Result, CUdevice *, int)
CUDA_FUNC(cuDeviceGetMemPool, Result, CUmemoryPool *, CUdevice)
CUDA_FUNC(cuDeviceGetAttribute, Result, int *, CUdevice_attribute, CUdevice)
CUDA_FUNC(cuDeviceGetCount, Result, int *)
CUDA_FUNC(cuDeviceGetName, Result, char *, int, CUdevice)
CUDA_FUNC(cuDeviceCanAccessPeer, Result, int *, CUdevice, CUdevice)
CUDA_FUNC(cuDeviceGetP2PAttribute, Result, int *, CUdevice_P2PAttribute,
          CUdevice, CUdevice)

CUDA_FUNC(cuDevicePrimaryCtxRetain, Result, CUcontext *, CUdevice)
CUDA_FUNC(cuDevicePrimaryCtxRelease_v2, Result, CUdevice)

CUDA_FUNC(cuCtxCreate_v2, Result, CUcontext *, unsigned int, CUdevice)
CUDA_FUNC(cuCtxDestroy, Result, CUcontext)
CUDA_FUNC(cuCtxSetCurrent, Result, CUcontext)
CUDA_FUNC(cuCtxGetCurrent, Result, CUcontext *)
CUDA_FUNC(cuCtxPushCurrent, Result, CUcontext)
CUDA_FUNC(cuCtxPopCurrent, Result, CUcontext *)
CUDA_FUNC(cuCtxEnablePeerAccess, Result, CUcontext, unsigned int)
CUDA_FUNC(cuCtxGetStreamPriorityRange, Result, int *, int *)

CUDA_FUNC(cuStreamCreate, Result, CUstream *, unsigned int)
CUDA_FUNC(cuStreamCreateWithPriority, Result, CUstream *, unsigned int, int)
CUDA_FUNC(cuStreamDestroy, Result, CUstream)
CUDA_FUNC(cuStreamSynchronize, Result, CUstream)
CUDA_FUNC(cuStreamWaitEvent, Result, CUstream, CUevent, unsigned int)
CUDA_FUNC(cuStreamWaitValue64_v2, Result, CUstream, CUdeviceptr,
          unsigned long long, unsigned int)
CUDA_FUNC(cuStreamBeginCapture, Result, CUstream, CUstreamCaptureMode)
CUDA_FUNC(cuStreamEndCapture, Result, CUstream, CUgraph *)
CUDA_FUNC(cuStreamIsCapturing, Result, CUstream, CUstreamCaptureStatus *)
// The 6-parameter capture-introspection form (capture id, capturing graph,
// and the current leaf-node set); the versioned driver symbol is _v2.
CUDA_FUNC(cuStreamGetCaptureInfo_v2, Result, CUstream,
          CUstreamCaptureStatus *, unsigned long long *, CUgraph *,
          const CUgraphNode **, size_t *)

CUDA_FUNC(cuGraphCreate, Result, CUgraph *, unsigned int)
CUDA_FUNC(cuGraphAddKernelNode, Result, CUgraphNode *, CUgraph,
          const CUgraphNode *, size_t, const CUDA_KERNEL_NODE_PARAMS_v2 *)
CUDA_FUNC(cuGraphAddMemcpyNode, Result, CUgraphNode *, CUgraph,
          const CUgraphNode *, size_t, const CUDA_MEMCPY3D_v2 *, CUcontext)
CUDA_FUNC(cuGraphAddMemsetNode, Result, CUgraphNode *, CUgraph,
          const CUgraphNode *, size_t, const CUDA_MEMSET_NODE_PARAMS *,
          CUcontext)
CUDA_FUNC(cuGraphAddEmptyNode, Result, CUgraphNode *, CUgraph,
          const CUgraphNode *, size_t)
// The final CUgraph argument (the child) is copied into the parent graph at
// node-creation time; the caller still owns and must destroy it.
CUDA_FUNC(cuGraphAddChildGraphNode, Result, CUgraphNode *, CUgraph,
          const CUgraphNode *, size_t, CUgraph)
// cuGraphAddDependencies_v2: introduced in CUDA 12.3, the variant that takes
// per-edge data. Loaded optionally so the driver still initializes on older
// hosts; device graphs then add full edges only.
CUDA_OPT_FUNC(cuGraphAddDependencies_v2, Result, CUgraph, const CUgraphNode *,
              const CUgraphNode *, const CUgraphEdgeData *, size_t)
// Third parameter is const CUkernelNodeAttrValue * in the driver — a 64-byte
// union passed here as the CUlaunchAttribute::value bytes.
CUDA_FUNC(cuGraphKernelNodeSetAttribute, Result, CUgraphNode,
          CUlaunchAttributeID, const void *)

CUDA_FUNC(cuGraphInstantiate, Result, CUgraphExec *, CUgraph, CUgraphNode *,
          char *, size_t)
CUDA_FUNC(cuGraphInstantiateWithFlags, Result, CUgraphExec *, CUgraph,
          unsigned long long)
CUDA_FUNC(cuGraphInstantiateWithParams, Result, CUgraphExec *, CUgraph,
          CUDA_GRAPH_INSTANTIATE_PARAMS *)
CUDA_FUNC(cuGraphDestroy, Result, CUgraph)
CUDA_FUNC(cuGraphExecDestroy, Result, CUgraphExec)
CUDA_FUNC(cuGraphLaunch, Result, CUgraphExec, CUstream)
CUDA_FUNC(cuGraphDebugDotPrint, Result, CUgraph, const char *, unsigned int)
CUDA_FUNC(cuDeviceGetGraphMemAttribute, Result, CUdevice, CUgraphMem_attribute,
          void *)

CUDA_FUNC(cuMemAllocHost_v2, Result, void **, size_t)
CUDA_FUNC(cuMemHostAlloc, Result, void **, size_t, unsigned int)
CUDA_FUNC(cuMemHostGetDevicePointer_v2, Result, CUdeviceptr *, void *,
          unsigned int)
CUDA_FUNC(cuMemHostRegister_v2, Result, void *, size_t, unsigned int)
CUDA_FUNC(cuMemHostUnregister, Result, void *)
CUDA_FUNC(cuMemFreeHost, Result, void *)
CUDA_FUNC(cuMemAllocAsync, Result, CUdeviceptr *, size_t, CUstream)
CUDA_FUNC(cuMemFreeAsync, Result, CUdeviceptr, CUstream)
CUDA_FUNC(cuMemAlloc_v2, Result, CUdeviceptr *, size_t)
CUDA_FUNC(cuMemFree_v2, Result, CUdeviceptr)
CUDA_FUNC(cuMemGetAddressRange_v2, Result, CUdeviceptr *, size_t *, CUdeviceptr)
CUDA_FUNC(cuMemGetHandleForAddressRange, Result, void *, CUdeviceptr, size_t,
          CUmemRangeHandleType, unsigned long long)

CUDA_FUNC(cuMemPoolSetAccess, Result, CUmemoryPool, const CUmemAccessDesc *,
          size_t)
CUDA_FUNC(cuMemPoolTrimTo, Result, CUmemoryPool, size_t)

CUDA_FUNC(cuMemcpyHtoDAsync_v2, Result, CUdeviceptr, const void *, size_t,
          CUstream)
CUDA_FUNC(cuMemcpyDtoHAsync_v2, Result, void *, CUdeviceptr, size_t, CUstream)
CUDA_FUNC(cuMemcpyDtoDAsync_v2, Result, CUdeviceptr, CUdeviceptr, size_t,
          CUstream)
CUDA_FUNC(cuMemcpyPeerAsync, Result, CUdeviceptr, CUcontext, CUdeviceptr,
          CUcontext, size_t, CUstream)
CUDA_FUNC(cuMemcpyHtoD_v2, Result, CUdeviceptr, const void *, size_t)
CUDA_FUNC(cuMemcpyDtoH_v2, Result, void *, CUdeviceptr, size_t)
CUDA_FUNC(cuMemcpyDtoD_v2, Result, CUdeviceptr, CUdeviceptr, size_t)
CUDA_FUNC(cuMemcpyPeer, Result, CUdeviceptr, CUcontext, CUdeviceptr, CUcontext,
          size_t)

CUDA_FUNC(cuMemsetD8Async, Result, CUdeviceptr, unsigned char, size_t, CUstream)
CUDA_FUNC(cuMemsetD16Async, Result, CUdeviceptr, unsigned short, size_t,
          CUstream)
CUDA_FUNC(cuMemsetD32Async, Result, CUdeviceptr, unsigned int, size_t, CUstream)
CUDA_FUNC(cuMemsetD2D32Async, Result, CUdeviceptr, size_t, unsigned int, size_t,
          size_t, CUstream)
CUDA_FUNC(cuMemsetD8_v2, Result, CUdeviceptr, unsigned char, size_t)
CUDA_FUNC(cuMemsetD16_v2, Result, CUdeviceptr, unsigned short, size_t)
CUDA_FUNC(cuMemsetD32_v2, Result, CUdeviceptr, unsigned int, size_t)
CUDA_FUNC(cuMemsetD2D32_v2, Result, CUdeviceptr, unsigned int, unsigned int,
          size_t, size_t)

CUDA_FUNC(cuModuleLoadDataEx, Result, CUmodule *, const void *, unsigned int,
          CUjit_option *, void **)
CUDA_FUNC(cuModuleGetGlobal_v2, Result, CUdeviceptr *, size_t *, CUmodule,
          const char *)
CUDA_FUNC(cuModuleUnload, Result, CUmodule)
CUDA_FUNC(cuModuleGetFunction, Result, CUfunction *, CUmodule, const char *)
CUDA_FUNC(cuFuncGetAttribute, Result, int *, CUfunction_attribute, CUfunction)
CUDA_FUNC(cuFuncSetAttribute, Result, CUfunction, CUfunction_attribute, int)
CUDA_FUNC(cuLaunchHostFunc, Result, CUstream, CUhostFn, void *)
CUDA_FUNC(cuLaunchKernel, Result, CUfunction, unsigned int, unsigned int,
          unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
          CUstream, void **, void **)
CUDA_FUNC(cuLaunchKernelEx, Result, const CUlaunchConfig *, CUfunction, void **,
          void **)

CUDA_FUNC(cuEventCreate, Result, CUevent *, unsigned int)
CUDA_FUNC(cuEventDestroy, Result, CUevent)
CUDA_FUNC(cuEventRecord, Result, CUevent, CUstream)
CUDA_FUNC(cuEventQuery, Result, CUevent)
CUDA_FUNC(cuEventSynchronize, Result, CUevent)
CUDA_FUNC(cuEventElapsedTime, Result, float *, CUevent, CUevent)

CUDA_FUNC(cuMemGetInfo_v2, Result, size_t *, size_t *)

CUDA_FUNC(cuMemGetAllocationGranularity, Result, size_t *,
          const CUmemAllocationProp *, CUmemAllocationGranularity_flags)
CUDA_FUNC(cuMemCreate, Result, CUmemGenericAllocationHandle *, size_t,
          const CUmemAllocationProp *, unsigned long long)
CUDA_FUNC(cuMemExportToShareableHandle, Result, void *,
          CUmemGenericAllocationHandle, CUmemAllocationHandleType,
          unsigned long long)
CUDA_FUNC(cuMemRelease, Result, CUmemGenericAllocationHandle)
CUDA_FUNC(cuMemAddressReserve, Result, CUdeviceptr *, size_t, size_t,
          CUdeviceptr, unsigned long long)
CUDA_FUNC(cuMemAddressFree, Result, CUdeviceptr, size_t)
CUDA_FUNC(cuMemMap, Result, CUdeviceptr, size_t, size_t,
          CUmemGenericAllocationHandle, unsigned long long)
CUDA_FUNC(cuMemUnmap, Result, CUdeviceptr, size_t)
CUDA_FUNC(cuMemSetAccess, Result, CUdeviceptr, size_t, const CUmemAccessDesc *,
          size_t)

CUDA_FUNC(cuMulticastCreate, Result, CUmemGenericAllocationHandle *,
          const CUmulticastObjectProp *)
CUDA_FUNC(cuMulticastGetGranularity, Result, size_t *,
          const CUmulticastObjectProp *, CUmulticastGranularity_flags)
CUDA_FUNC(cuMulticastAddDevice, Result, CUmemGenericAllocationHandle, CUdevice)
CUDA_FUNC(cuMulticastBindMem, Result, CUmemGenericAllocationHandle, size_t,
          CUmemGenericAllocationHandle, size_t, size_t, unsigned long long)
CUDA_FUNC(cuMulticastUnbind, Result, CUmemGenericAllocationHandle, CUdevice,
          size_t, size_t)

CUDA_FUNC(cuTensorMapEncodeTiled, Result, void *, CUtensorMapDataType, uint32_t,
          void *, const uint64_t *, const uint64_t *, const uint32_t *,
          const uint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
          CUtensorMapL2promotion, CUtensorMapFloatOOBfill)
CUDA_FUNC(cuTensorMapEncodeIm2col, Result, void *, CUtensorMapDataType,
          uint32_t, void *, const uint64_t *, const uint64_t *, const int *,
          const int *, uint32_t, uint32_t, const uint32_t *,
          CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
          CUtensorMapFloatOOBfill)
CUDA_FUNC(cuPointerGetAttributes, Result, unsigned int, CUpointer_attribute *,
          void **, CUdeviceptr)
CUDA_FUNC(cuPointerSetAttribute, Result, void *, CUpointer_attribute,
          CUdeviceptr)

CUDA_FUNC(cuOccupancyMaxActiveBlocksPerMultiprocessor, Result, int *,
          CUfunction, int, size_t)

CUDA_FUNC(cuFuncGetParamInfo, Result, CUfunction, size_t, size_t *, size_t *)

// cuMemcpyBatchAsync: introduced in CUDA 12.8. Loaded optionally so the
// driver still initializes on CUDA < 12.8 hosts (CUDADeviceContext::
// memBatchCopy then falls back to the scalar per-pair loop).
// Signature (CUDA 12.8): dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs,
// failIdx (out: index of the failing copy), stream. The failIdx out-param sits
// before the stream -- omitting it shifts the stream argument and yields
// CUDA_ERROR_INVALID_VALUE.
CUDA_OPT_FUNC(cuMemcpyBatchAsync, Result, void **, const void **, size_t *,
              size_t, CUmemcpyAttributes *, size_t *, size_t, size_t *,
              CUstream)

#ifdef CUDA_OPT_FUNC_DEFINED_LOCALLY
#undef CUDA_OPT_FUNC
#undef CUDA_OPT_FUNC_DEFINED_LOCALLY
#endif

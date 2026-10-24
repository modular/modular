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
// This file lists CUDA driver functions by name. The type of each function
// pointer is derived from the official cuda.h header via decltype(&::name).
//
// To add a new CUDA API to the driver class, simply add:
//
//     CUDA_FN(<API name>)
//
// Always use the BARE function name (e.g. cuCtxDestroy, not cuCtxDestroy_v2).
// cuda.h #define-redirects bare names to the correct versioned variant, and
// the two-level stringify in CUDALoader.cpp (CUDA_XSTR) ensures dlsym also
// receives the expanded versioned symbol name.
//
//===----------------------------------------------------------------------===//

// No macro guards since this is meant to be included multiple times, but make
// sure CUDA_FN is defined.
#ifndef CUDA_FN
#error "CUDALoader.inc included without defining CUDA_FN"
#endif

CUDA_FN(cuInit)

CUDA_FN(cuGetErrorName)
CUDA_FN(cuGetErrorString)

CUDA_FN(cuDriverGetVersion)

CUDA_FN(cuDeviceGet)
CUDA_FN(cuDeviceGetMemPool)
CUDA_FN(cuDeviceGetAttribute)
CUDA_FN(cuDeviceGetCount)
CUDA_FN(cuDeviceGetName)
CUDA_FN(cuDeviceCanAccessPeer)
CUDA_FN(cuDeviceGetP2PAttribute)

CUDA_FN(cuDevicePrimaryCtxRetain)
CUDA_FN(cuDevicePrimaryCtxRelease)

CUDA_FN(cuCtxCreate)
CUDA_FN(cuCtxDestroy)
CUDA_FN(cuCtxSetCurrent)
CUDA_FN(cuCtxGetCurrent)
CUDA_FN(cuCtxPushCurrent)
CUDA_FN(cuCtxPopCurrent)
CUDA_FN(cuCtxEnablePeerAccess)
CUDA_FN(cuCtxGetDevice_v2)
CUDA_FN(cuCtxGetStreamPriorityRange)

CUDA_FN(cuStreamCreate)
CUDA_FN(cuStreamCreateWithPriority)
CUDA_FN(cuStreamDestroy)
CUDA_FN(cuStreamSynchronize)
CUDA_FN(cuStreamWaitValue64)
CUDA_FN(cuStreamWaitEvent)
CUDA_FN(cuStreamBeginCapture)
CUDA_FN(cuStreamEndCapture)
CUDA_FN(cuStreamIsCapturing)

CUDA_FN(cuGraphInstantiateWithFlags)
CUDA_FN(cuGraphInstantiateWithParams)
CUDA_FN(cuGraphDestroy)
CUDA_FN(cuGraphExecDestroy)
CUDA_FN(cuGraphLaunch)
CUDA_FN(cuGraphDebugDotPrint)

CUDA_FN(cuMemAllocHost)
CUDA_FN(cuMemFreeHost)
CUDA_FN(cuMemAllocAsync)
CUDA_FN(cuMemFreeAsync)
CUDA_FN(cuMemAlloc)
CUDA_FN(cuMemFree)

CUDA_FN(cuMemPoolSetAccess)
CUDA_FN(cuMemPoolTrimTo)

CUDA_FN(cuMemcpyHtoDAsync)
CUDA_FN(cuMemcpyDtoHAsync)
CUDA_FN(cuMemcpyDtoDAsync)
CUDA_FN(cuMemcpyPeerAsync)
CUDA_FN(cuMemcpyHtoD)
CUDA_FN(cuMemcpyDtoH)
CUDA_FN(cuMemcpyDtoD)
CUDA_FN(cuMemcpyPeer)

CUDA_FN(cuMemsetD8Async)
CUDA_FN(cuMemsetD16Async)
CUDA_FN(cuMemsetD32Async)
CUDA_FN(cuMemsetD2D32Async)
CUDA_FN(cuMemsetD8)
CUDA_FN(cuMemsetD16)
CUDA_FN(cuMemsetD32)
CUDA_FN(cuMemsetD2D32)

CUDA_FN(cuModuleLoadDataEx)
CUDA_FN(cuModuleGetGlobal)
CUDA_FN(cuModuleUnload)
CUDA_FN(cuModuleGetFunction)
CUDA_FN(cuFuncGetAttribute)
CUDA_FN(cuFuncSetAttribute)
CUDA_FN(cuLaunchHostFunc)
CUDA_FN(cuLaunchKernel)
CUDA_FN(cuLaunchKernelEx)

CUDA_FN(cuEventCreate)
CUDA_FN(cuEventDestroy)
CUDA_FN(cuEventRecord)
CUDA_FN(cuEventQuery)
CUDA_FN(cuEventSynchronize)
CUDA_FN(cuEventElapsedTime)

CUDA_FN(cuMemGetInfo)

CUDA_FN(cuMemGetAllocationGranularity)
CUDA_FN(cuMemCreate)
CUDA_FN(cuMemRelease)
CUDA_FN(cuMemAddressReserve)
CUDA_FN(cuMemAddressFree)
CUDA_FN(cuMemMap)
CUDA_FN(cuMemUnmap)
CUDA_FN(cuMemSetAccess)

CUDA_FN(cuMulticastCreate)
CUDA_FN(cuMulticastGetGranularity)
CUDA_FN(cuMulticastAddDevice)
CUDA_FN(cuMulticastBindMem)
CUDA_FN(cuMulticastUnbind)

CUDA_FN(cuTensorMapEncodeTiled)
CUDA_FN(cuTensorMapEncodeIm2col)
CUDA_FN(cuPointerGetAttributes)

CUDA_FN(cuOccupancyMaxActiveBlocksPerMultiprocessor)

CUDA_FN(cuFuncGetParamInfo)

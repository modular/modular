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

#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/NVML.h"
#include "ScopedEnvVar.h"
#include "TestUtils.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Process.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace M::Driver;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

std::string formatCudaResult(const CUDADriver &driver,
                             CUDADriver::Result result) {
  const char *name = nullptr;
  const char *message = nullptr;
  (void)driver.cuGetErrorName(result, &name);
  (void)driver.cuGetErrorString(result, &message);

  std::string formatted = name ? name : "unknown";
  if (message) {
    formatted += " (";
    formatted += message;
    formatted += ")";
  }
  return formatted;
}

static const char *vecAddPtx = R"PTX(
// From max/kernels/test/gpu/vec_add.ptx

.version 6.3
.target sm_75
.address_size 64

	// .globl	vec_add

.visible .entry vec_add(
	.param .b64 vec_add_param_0,
	.param .b64 vec_add_param_1,
	.param .b64 vec_add_param_2,
	.param .b64 vec_add_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<28>;
	.reg .f32 	%f<4>;
	.reg .b64 	%rd<12>;

	mov.u32 	%r1, %tid.x;
	cvt.s64.s32 	%rd6, %r1;
	ld.param.u64 	%rd8, [vec_add_param_3];
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %ctaid.x;
	mul.wide.s32 	%rd9, %r2, %r3;
	add.s64 	%rd10, %rd9, %rd6;
	setp.ge.s64 	%p1, %rd10, %rd8;
	@%p1 bra 	$L__BB0_2;
	ld.param.u64 	%rd4, [vec_add_param_0];
	ld.param.u64 	%rd5, [vec_add_param_1];
	ld.param.u64 	%rd7, [vec_add_param_2];
	shl.b64 	%rd11, %rd10, 2;
	add.s64 	%rd1, %rd4, %rd11;
	add.s64 	%rd2, %rd7, %rd11;
	add.s64 	%rd3, %rd5, %rd11;
	ld.u8 	%r4, [%rd3];
	ld.u8 	%r5, [%rd3+1];
	shl.b32 	%r6, %r5, 8;
	or.b32  	%r7, %r6, %r4;
	ld.u8 	%r8, [%rd3+2];
	shl.b32 	%r9, %r8, 16;
	ld.u8 	%r10, [%rd3+3];
	shl.b32 	%r11, %r10, 24;
	or.b32  	%r12, %r11, %r9;
	or.b32  	%r13, %r12, %r7;
	mov.b32 	%f1, %r13;
	ld.u8 	%r14, [%rd2];
	ld.u8 	%r15, [%rd2+1];
	shl.b32 	%r16, %r15, 8;
	or.b32  	%r17, %r16, %r14;
	ld.u8 	%r18, [%rd2+2];
	shl.b32 	%r19, %r18, 16;
	ld.u8 	%r20, [%rd2+3];
	shl.b32 	%r21, %r20, 24;
	or.b32  	%r22, %r21, %r19;
	or.b32  	%r23, %r22, %r17;
	mov.b32 	%f2, %r23;
	add.rn.f32 	%f3, %f1, %f2;
	mov.b32 	%r24, %f3;
	st.u8 	[%rd1], %r24;
	shr.u32 	%r25, %r24, 16;
	st.u8 	[%rd1+2], %r25;
	shr.u32 	%r26, %r24, 24;
	st.u8 	[%rd1+3], %r26;
	shr.u32 	%r27, %r24, 8;
	st.u8 	[%rd1+1], %r27;
$L__BB0_2:
	ret;
}
)PTX";

// One PTX module exposing two distinct entry points. `kernel_a` writes 1.0 to
// every element; `kernel_b` writes 2.0. Generated with `nvcc -arch=sm_75 -ptx`
// from two trivial kernels.
static const char *twoKernelPtx = R"PTX(
.version 8.8
.target sm_75
.address_size 64

.visible .entry kernel_a(
	.param .u64 kernel_a_param_0,
	.param .u64 kernel_a_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<9>;

	ld.param.u64 	%rd2, [kernel_a_param_0];
	ld.param.u64 	%rd3, [kernel_a_param_1];
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mul.wide.u32 	%rd4, %r1, %r2;
	mov.u32 	%r3, %tid.x;
	cvt.u64.u32 	%rd5, %r3;
	add.s64 	%rd1, %rd4, %rd5;
	setp.ge.s64 	%p1, %rd1, %rd3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd6, %rd2;
	shl.b64 	%rd7, %rd1, 2;
	add.s64 	%rd8, %rd6, %rd7;
	mov.u32 	%r4, 1065353216;
	st.global.u32 	[%rd8], %r4;

$L__BB0_2:
	ret;
}

.visible .entry kernel_b(
	.param .u64 kernel_b_param_0,
	.param .u64 kernel_b_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<9>;

	ld.param.u64 	%rd2, [kernel_b_param_0];
	ld.param.u64 	%rd3, [kernel_b_param_1];
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mul.wide.u32 	%rd4, %r1, %r2;
	mov.u32 	%r3, %tid.x;
	cvt.u64.u32 	%rd5, %r3;
	add.s64 	%rd1, %rd4, %rd5;
	setp.ge.s64 	%p1, %rd1, %rd3;
	@%p1 bra 	$L__BB1_2;

	cvta.to.global.u64 	%rd6, %rd2;
	shl.b64 	%rd7, %rd1, 2;
	add.s64 	%rd8, %rd6, %rd7;
	mov.u32 	%r4, 1073741824;
	st.global.u32 	[%rd8], %r4;

$L__BB1_2:
	ret;
}
)PTX";

class CUDADeviceContextTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // Print information about the device to make it easier to debug the
    // environment.
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      // Have the tests deal with device errors.
      return;

    std::cerr << "##### CUDADeviceContextTest running on "
              << Device::numberOfDevices(Device::gpuAPI) << " x "
              << (*device)->name() << "\n";
  }
};

TEST_F(CUDADeviceContextTest, DeviceExec) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));
  ASSERT_TRUE(ctx);

  // Check that the API and driver versions are available.
  // The versions may vary depending on the machine, so we don't check the
  // values.
  EXPECT_NO_ERR(ctx->device().getApiVersion());
  EXPECT_NO_ERR(ctx->device().getDriverVersion());

  // Check reference count.
  EXPECT_EQ(ctx->getNumReferences(), 1);

  {
    std::cerr << "Compiling vecAddPtx...\n";
    ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunction,
                  ctx->loadFunction("vec_add_module", "vec_add", vecAddPtx));

    // One reference count from ctx and one from deviceFunction.
    EXPECT_EQ(ctx->getNumReferences(), 2);
    EXPECT_EQ(deviceFunction->getNumReferences(), 1);

    constexpr size_t length = 1024;
    constexpr size_t elemSize = sizeof(float);

    float *in0Host = reinterpret_cast<float *>(malloc(length * elemSize));
    float *in1Host = reinterpret_cast<float *>(malloc(length * elemSize));
    float *outHost = reinterpret_cast<float *>(malloc(length * elemSize));

    for (size_t i = 0; i < length; ++i) {
      in0Host[i] = static_cast<float>(i);
      in1Host[i] = 2.0;
      outHost[i] =
          static_cast<float>(i) * 2.0; // Write a value that is not expected.
    }

    ASSIGN_NO_ERR(DeviceBufferRef, in0Device,
                  ctx->enqueueCreateBuffer(length * elemSize));
    EXPECT_TRUE(in0Device);

    ASSIGN_NO_ERR(DeviceBufferRef, in1Device,
                  ctx->enqueueCreateBuffer(length * elemSize));
    EXPECT_TRUE(in1Device);

    ASSIGN_NO_ERR(DeviceBufferRef, outDevice,
                  ctx->enqueueCreateBuffer(length * elemSize));
    EXPECT_TRUE(outDevice);

    EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*in0Device, in0Host));
    EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*in1Device, in1Host));
    // This copy is not strictly necessary, but fill outDevice with known bad
    // values.
    EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*outDevice, outHost));

    constexpr uint32_t blockDim = 32;

    void *outPtr = outDevice->ptr();
    void *in0Ptr = in0Device->ptr();
    void *in1Ptr = in1Device->ptr();
    void *args[] = {&outPtr, &in0Ptr, &in1Ptr, (void *)&length};
    size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                         sizeof(length)};

    EXPECT_NO_ERR(ctx->enqueueFunctionExecDirect(
        *deviceFunction, length / blockDim, 1, 1, blockDim, 1, 1, 0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args, sizeof(args) / sizeof(void *),
        argSizes));

    EXPECT_NO_ERR(ctx->enqueueCopyFromDevice(outHost, *outDevice));

    EXPECT_NO_ERR(ctx->synchronize());

    for (size_t i = 0; i < length; ++i) {
      EXPECT_EQ(outHost[i], static_cast<float>(i) + 2.0);
    }

    // Check cache hit.
    {
      ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunction2,
                    ctx->loadFunction("vec_add_module", "vec_add", vecAddPtx));
      EXPECT_TRUE(deviceFunction2);

      // A cache hit will point to the same backing store for the function (e.g.
      // name).
      EXPECT_EQ(&deviceFunction->name(), &deviceFunction2->name());
      // Even with a cache hit, we do get different DeviceFunction objects.
      EXPECT_EQ(deviceFunction->getNumReferences(), 1);
      EXPECT_EQ(deviceFunction2->getNumReferences(), 1);
    }

    free(outHost);
    free(in1Host);
    free(in0Host);
  }

  // Check that cached functions do not keep reference count on context. After
  // ensuring that all pending operations have terminated.
  EXPECT_NO_ERR(ctx->synchronize());
  EXPECT_EQ(ctx->getNumReferences(), 1);
}

// Loading two distinct entry points from the same PTX/cubin blob must return
// two distinct functions. Before the fix the cache key omitted the function
// name (and, with no module name, keyed on the blob bytes alone), so the second
// load collided with the first and silently returned `kernel_a`. The empty
// module name exercises that blob-keyed path.
TEST_F(CUDADeviceContextTest, LoadFunctionDistinctEntryPointsSameBlob) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));
  ASSERT_TRUE(ctx);

  // Load both kernels from the same blob with an empty module name.
  ASSIGN_NO_ERR(DeviceFunctionRef, funcA,
                ctx->loadFunction(/*moduleName=*/"", "kernel_a", twoKernelPtx));
  ASSIGN_NO_ERR(DeviceFunctionRef, funcB,
                ctx->loadFunction(/*moduleName=*/"", "kernel_b", twoKernelPtx));

  // The two handles must resolve to their requested entry points. Pre-fix,
  // funcB would alias funcA and report "kernel_a" here.
  EXPECT_EQ(funcA->name(), "kernel_a");
  EXPECT_EQ(funcB->name(), "kernel_b");

  // Execute both and confirm they write their distinct constants (1.0 vs 2.0).
  constexpr int64_t length = 256;
  constexpr uint32_t blockDim = 32;
  constexpr size_t bytes = length * sizeof(float);

  auto runAndCheck = [&](DeviceFunctionRef &func, float expected) {
    ASSIGN_NO_ERR(DeviceBufferRef, outDevice, ctx->enqueueCreateBuffer(bytes));
    ASSERT_TRUE(outDevice);

    std::vector<float> outHost(length, -1.0f);
    EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*outDevice, outHost.data()));

    void *outPtr = outDevice->ptr();
    void *args[] = {&outPtr, (void *)&length};
    size_t argSizes[] = {sizeof(outPtr), sizeof(length)};
    EXPECT_NO_ERR(ctx->enqueueFunctionExecDirect(
        *func, length / blockDim, 1, 1, blockDim, 1, 1, 0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args, sizeof(args) / sizeof(void *),
        argSizes));
    EXPECT_NO_ERR(ctx->enqueueCopyFromDevice(outHost.data(), *outDevice));
    EXPECT_NO_ERR(ctx->synchronize());

    for (int64_t i = 0; i < length; ++i)
      EXPECT_EQ(outHost[i], expected) << "at index " << i;
  };

  runAndCheck(funcA, 1.0f);
  runAndCheck(funcB, 2.0f);
}

TEST_F(CUDADeviceContextTest, WrappedCuMemGetAddressRangeV2Works) {
  constexpr size_t byteSize = 64 * 1024;

  auto driverOrError = CUDADriver::get();
  ASSERT_FALSE(driverOrError.isError()) << driverOrError.getError();
  const auto &driver = *driverOrError;

  CUdevice device = 0;
  ASSERT_EQ(driver.cuDeviceGet(&device, 0), CUDADriver::SUCCESS);

  CUcontext primaryContext = nullptr;
  ASSERT_EQ(driver.cuDevicePrimaryCtxRetain(&primaryContext, device),
            CUDADriver::SUCCESS);
  auto releasePrimaryContext = llvm::scope_exit(
      [&]() { (void)driver.cuDevicePrimaryCtxRelease_v2(device); });

  CUcontext previousContext = nullptr;
  ASSERT_EQ(driver.cuCtxGetCurrent(&previousContext), CUDADriver::SUCCESS);
  auto restoreContext = llvm::scope_exit(
      [&]() { (void)driver.cuCtxSetCurrent(previousContext); });

  ASSERT_EQ(driver.cuCtxSetCurrent(primaryContext), CUDADriver::SUCCESS);

  CUdeviceptr ptr = nullptr;
  ASSERT_EQ(driver.cuMemAlloc_v2(&ptr, byteSize), CUDADriver::SUCCESS);
  auto freePtr = llvm::scope_exit([&]() { (void)driver.cuMemFree_v2(ptr); });

  CUdeviceptr basePtr = nullptr;
  size_t totalSize = 0;
  auto result = driver.cuMemGetAddressRange_v2(&basePtr, &totalSize, ptr);
  ASSERT_EQ(result, CUDADriver::SUCCESS) << formatCudaResult(driver, result);

  EXPECT_EQ(basePtr, ptr);
  EXPECT_EQ(totalSize, byteSize);
}

TEST_F(CUDADeviceContextTest, MultiGPUKernels) {
  const char *p2pCopyPtx = R"PTX(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-32688072
// Cuda compilation tools, release 12.1, V12.1.105
// Based on NVVM 7.0.1
//
// Compiled by nvcc -ptx
// Simple copy kernel
// __global__ void p2pMemcpyKernel(float* dst, float* src, size_t numElements) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < numElements) {
//         dst[idx] = src[idx];
//     }
// }

.version 8.1
.target sm_52
.address_size 64

	// .globl	_Z15p2pMemcpyKernelPfS_m

.visible .entry _Z15p2pMemcpyKernelPfS_m(
	.param .u64 _Z15p2pMemcpyKernelPfS_m_param_0,
	.param .u64 _Z15p2pMemcpyKernelPfS_m_param_1,
	.param .u64 _Z15p2pMemcpyKernelPfS_m_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [_Z15p2pMemcpyKernelPfS_m_param_0];
	ld.param.u64 	%rd3, [_Z15p2pMemcpyKernelPfS_m_param_1];
	ld.param.u64 	%rd4, [_Z15p2pMemcpyKernelPfS_m_param_2];
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd4;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd5, %rd3;
	shl.b64 	%rd6, %rd1, 2;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.f32 	%f1, [%rd7];
	cvta.to.global.u64 	%rd8, %rd2;
	add.s64 	%rd9, %rd8, %rd6;
	st.global.f32 	[%rd9], %f1;

$L__BB0_2:
	ret;

}
)PTX";

  int numDevices = Device::numberOfDevices(Device::cudaAPI);
  std::cerr << "Number of CUDA devices: " << numDevices << "\n";

  ASSIGN_NO_ERR(DeviceContextRef, srcCtx,
                DeviceContext::create(Device::cudaAPI, 0));
  ASSERT_TRUE(srcCtx);

  for (int curDeviceId = -1; curDeviceId < numDevices; ++curDeviceId) {
    DeviceContextRef dstCtx;
    if (curDeviceId < 0) {
      std::cerr << "Copying within the same device context.\n";
      dstCtx = DeviceContextRef::copy(srcCtx.getPointer());
    } else {
      std::cerr << "Copying from context on GPU[0] to context on GPU["
                << curDeviceId << "].\n";

      ErrorOr<DeviceContextRef> errCtx =
          DeviceContext::create(Device::cudaAPI, curDeviceId);
      ASSERT_FALSE(errCtx) << errCtx.getError() << "\n";
      dstCtx = std::move(errCtx.get());
    }
    ASSERT_TRUE(dstCtx);

    if (srcCtx->device().id() != dstCtx->device().id()) {
      ASSIGN_NO_ERR(bool, accessible,
                    srcCtx->device().canAccess(dstCtx->device()));
      ASSERT_TRUE(accessible);
      EXPECT_NO_ERR(srcCtx->device().enablePeerAccess(dstCtx->device()));
      EXPECT_NO_ERR(dstCtx->device().enablePeerAccess(srcCtx->device()));
    }

    const size_t numElems = 256;
    const size_t allocLen = 4 * numElems;

    char *checkData = new char[allocLen];
    // Initialize checkData with a pattern specific for this run.
    for (size_t i = 0; i < allocLen; ++i) {
      checkData[i] = curDeviceId + i;
    }

    ASSIGN_NO_ERR(DeviceBufferRef, srcPtr, srcCtx->createHostBuffer(allocLen));
    EXPECT_TRUE(srcPtr);

    ASSIGN_NO_ERR(DeviceBufferRef, srcBuf,
                  srcCtx->enqueueCreateBuffer(allocLen));
    EXPECT_NO_ERR(srcCtx->synchronize());
    EXPECT_TRUE(srcBuf);

    ASSIGN_NO_ERR(DeviceBufferRef, dstBuf,
                  dstCtx->enqueueCreateBuffer(allocLen));
    EXPECT_NO_ERR(dstCtx->synchronize());
    EXPECT_TRUE(dstBuf);

    char *dstPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(dstPtr, nullptr);
    // Initialize with known bad values.
    for (size_t i = 0; i < allocLen; ++i) {
      // Use a different factor for i than the checkData, so that we can't ever
      // overlap.
      dstPtr[i] = -static_cast<char>((7 + 2 * i) & 127);
    }

    // Copy between different buffers.
    EXPECT_NO_ERR(srcPtr->copyFrom(checkData));
    EXPECT_NO_ERR(srcCtx->enqueueCopyToDevice(*srcBuf, srcPtr->ptr()));

    ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunction,
                  srcCtx->loadFunction("my_module", "_Z15p2pMemcpyKernelPfS_m",
                                       p2pCopyPtx));
    EXPECT_TRUE(deviceFunction);

    constexpr uint32_t blockDim = 256;

    void *dstBufPtr = dstBuf->ptr();
    void *srcBufPtr = srcBuf->ptr();
    void *args[] = {&dstBufPtr, &srcBufPtr, (void *)&numElems};
    size_t argSizes[] = {sizeof(dstBufPtr), sizeof(srcBufPtr),
                         sizeof(numElems)};

    EXPECT_NO_ERR(srcCtx->enqueueFunctionExecDirect(
        *deviceFunction, numElems / blockDim, 1, 1, blockDim, 1, 1, 0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args, sizeof(args) / sizeof(void *),
        argSizes));

    EXPECT_NO_ERR(srcCtx->synchronize());

    EXPECT_NO_ERR(dstCtx->enqueueCopyFromDevice(dstPtr, *dstBuf));

    // Ensure that all copy operations have finished.
    EXPECT_NO_ERR(dstCtx->synchronize());
    EXPECT_NO_ERR(srcCtx->synchronize());

    EXPECT_EQ(memcmp(dstPtr, checkData, allocLen), 0);

    delete[] checkData;
    free(dstPtr);
  }
}

// TODO(PLATENG-312): Re-enable after fixing CUDA_ERROR_ILLEGAL_ADDRESS on B200.
TEST_F(CUDADeviceContextTest, DISABLED_MultiMapMemory) {
  const char *copyPtx = R"PTX(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-32688072
// Cuda compilation tools, release 12.1, V12.1.105
// Based on NVVM 7.0.1
//
// Compiled by nvcc -ptx
// Simple copy kernel
// __global__ void p2pMemcpyKernel(float* dst, float* src, size_t numElements) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < numElements) {
//         dst[idx] = src[idx];
//     }
// }

.version 8.1
.target sm_52
.address_size 64

	// .globl	_Z15p2pMemcpyKernelPfS_m

.visible .entry _Z15p2pMemcpyKernelPfS_m(
	.param .u64 _Z15p2pMemcpyKernelPfS_m_param_0,
	.param .u64 _Z15p2pMemcpyKernelPfS_m_param_1,
	.param .u64 _Z15p2pMemcpyKernelPfS_m_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [_Z15p2pMemcpyKernelPfS_m_param_0];
	ld.param.u64 	%rd3, [_Z15p2pMemcpyKernelPfS_m_param_1];
	ld.param.u64 	%rd4, [_Z15p2pMemcpyKernelPfS_m_param_2];
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd4;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd5, %rd3;
	shl.b64 	%rd6, %rd1, 2;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.f32 	%f1, [%rd7];
	cvta.to.global.u64 	%rd8, %rd2;
	add.s64 	%rd9, %rd8, %rd6;
	st.global.f32 	[%rd9], %f1;

$L__BB0_2:
	ret;

}
)PTX";

  const char *splatPtx = R"PTX(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-32688072
// Cuda compilation tools, release 12.1, V12.1.105
// Based on NVVM 7.0.1
//
// Compiled by nvcc -ptx
// Simple copy kernel
// __global__ void p2pMemcpyKernel(float* dst, float* src, size_t numElements) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < numElements) {
//         dst[idx] = src[idx];
//     }
// }

.version 8.1
.target sm_90
.address_size 64

	// .globl	_Z15p2pMemSplatKernelPfS_m

.visible .entry _Z15p2pMemSplatKernelPfS_m(
	.param .u64 _Z15p2pMemSplatKernelPfS_m_param_0,
	.param .u64 _Z15p2pMemSplatKernelPfS_m_param_1,
	.param .u64 _Z15p2pMemSplatKernelPfS_m_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [_Z15p2pMemSplatKernelPfS_m_param_0];
	ld.param.u64 	%rd3, [_Z15p2pMemSplatKernelPfS_m_param_1];
	ld.param.u64 	%rd4, [_Z15p2pMemSplatKernelPfS_m_param_2];
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd4;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd5, %rd3;
	shl.b64 	%rd6, %rd1, 2;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.f32 	%f1, [%rd7];
	cvta.to.global.u64 	%rd8, %rd2;
	add.s64 	%rd9, %rd8, %rd6;
	multimem.st.global.f32 	[%rd9], %f1;

$L__BB0_2:
	ret;

}
)PTX";

  const char *mapPtx = R"PTX(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-32688072
// Cuda compilation tools, release 12.1, V12.1.105
// Based on NVVM 7.0.1
//
// Compiled by nvcc -ptx
// Simple copy kernel
// __global__ void p2pMemcpyKernel(float* dst, float* src, size_t numElements) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < numElements) {
//         dst[idx] = src[idx];
//     }
// }

.version 8.1
.target sm_90
.address_size 64

	// .globl	_Z15p2pMemMapKernelPfS_m

.visible .entry _Z15p2pMemMapKernelPfS_m(
	.param .u64 _Z15p2pMemMapKernelPfS_m_param_0,
	.param .u64 _Z15p2pMemMapKernelPfS_m_param_1,
	.param .u64 _Z15p2pMemMapKernelPfS_m_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [_Z15p2pMemMapKernelPfS_m_param_0];
	ld.param.u64 	%rd3, [_Z15p2pMemMapKernelPfS_m_param_1];
	ld.param.u64 	%rd4, [_Z15p2pMemMapKernelPfS_m_param_2];
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd4;
	@%p1 bra 	$L__BB0_2;

  // Did not change anything: fence.proxy.alias;
	cvta.to.global.u64 	%rd5, %rd3;
	shl.b64 	%rd6, %rd1, 2;
	add.s64 	%rd7, %rd5, %rd6;
	multimem.ld_reduce.global.sys.add.f32 	%f1, [%rd7];
	cvta.to.global.u64 	%rd8, %rd2;
	add.s64 	%rd9, %rd8, %rd6;
	st.global.f32 	[%rd9], %f1;

$L__BB0_2:
	ret;

}
)PTX";

  int numDevices = Device::numberOfDevices(Device::cudaAPI);
  std::cerr << "Number of CUDA devices: " << numDevices << "\n";
  ASSERT_GT(numDevices, 0);

  std::vector<DeviceContextRef> contexts;
  for (int i = 0; i < numDevices; ++i) {
    ErrorOr<DeviceContextRef> errCtx =
        DeviceContext::create(Device::cudaAPI, i);
    ASSERT_FALSE(errCtx) << errCtx.getError() << "\n";
    DeviceContextRef ctx = std::move(errCtx.get());
    ASSERT_TRUE(ctx);
    contexts.push_back(std::move(ctx));
  }

  DeviceContext &ctx0 = *contexts[0];

  ASSIGN_NO_ERR(
      DeviceFunctionRef, splatFun,
      ctx0.loadFunction("my_module", "_Z15p2pMemSplatKernelPfS_m", splatPtx));

  ASSIGN_NO_ERR(
      DeviceFunctionRef, mapFun,
      ctx0.loadFunction("my_module", "_Z15p2pMemMapKernelPfS_m", mapPtx));

  constexpr uint32_t blockDim = 16;
  const size_t numElems = 256;
  const size_t allocLen = 4 * numElems;

  ASSIGN_NO_ERR(bool, multicastSupported, ctx0.device().supportsMulticast());
  if (!multicastSupported) {
    ErrorOr<DeviceMulticastBufferRef> err =
        DeviceMulticastBuffer::allocate(contexts, allocLen);
    EXPECT_TRUE(err.isError());
    EXPECT_THAT(err.getError(),
                testing::HasSubstr("Multicast buffer not supported"));
    return;
  }

  if (numDevices < 2) {
    ErrorOr<DeviceMulticastBufferRef> err =
        DeviceMulticastBuffer::allocate(contexts, allocLen);
    EXPECT_TRUE(err.isError());
    EXPECT_THAT(err.getError(),
                testing::HasSubstr("At least two devices necessary"));
    return;
  }

  ASSIGN_NO_ERR(DeviceMulticastBufferRef, multiBuf,
                DeviceMulticastBuffer::allocate(contexts, allocLen));
  {
    std::cerr << "Using unicast buffers on each device.\n";
    for (int i = 0; i < numDevices; ++i) {
      DeviceContext &ctx = *contexts[i];
      ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx.createHostBuffer(allocLen));
      float *hostPtr = static_cast<float *>(hostBuf->ptr());
      EXPECT_NE(hostPtr, nullptr);
      for (size_t j = 0; j < numElems; ++j) {
        hostPtr[j] = j + (3.0 * (ctx.device().id() + 1));
      }
      ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx.enqueueCreateBuffer(allocLen));
      EXPECT_NO_ERR(hostBuf->copyTo(*devBuf));
      ASSIGN_NO_ERR(DeviceBufferRef, mapBuf, multiBuf->unicastBufferFor(ctx));

      ASSIGN_NO_ERR(
          DeviceFunctionRef, copyFun,
          ctx.loadFunction("my_module", "_Z15p2pMemcpyKernelPfS_m", copyPtx));

      void *mapPtr = mapBuf->ptr();
      void *devPtr = devBuf->ptr();
      void *copyArgs[] = {&mapPtr, &devPtr, (void *)&numElems};
      size_t copyArgSizes[] = {sizeof(mapPtr), sizeof(devPtr),
                               sizeof(numElems)};

      EXPECT_NO_ERR(ctx.enqueueFunctionExecDirect(
          *copyFun, numElems / blockDim, 1, 1, blockDim, 1, 1, 0,
          /*attrs=*/nullptr, /*numAttrs=*/0, copyArgs,
          sizeof(copyArgs) / sizeof(void *), copyArgSizes));

      bzero(hostPtr, allocLen);
      EXPECT_NO_ERR(mapBuf->copyTo(*hostBuf));

      EXPECT_NO_ERR(ctx.synchronize());
      for (size_t j = 0; j < numElems; ++j) {
        if (j < 5) {
          std::cerr << "unicast buf[" << j << "/device(" << ctx.device().id()
                    << ")]: " << hostPtr[j] << "\n";
        }
        EXPECT_EQ(hostPtr[j], j + (3.0 * (ctx.device().id() + 1)));
      }
      std::cerr << "------------\n";
    }
  }

  // Setup multimem based tests.
  ASSIGN_NO_ERR(DeviceBufferRef, mapBuf, multiBuf->multicastBuffer(ctx0));
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx0.enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx0.createHostBuffer(allocLen));
  float *hostPtr = static_cast<float *>(hostBuf->ptr());
  {
    // No synchronization with producing streams on other devices necessary
    // since the verification code above did synchronization.
    std::cerr << "Using multicast to map from all devices.\n";
    void *devPtr = devBuf->ptr();
    void *mapPtr = mapBuf->ptr();
    void *mapArgs[] = {&devPtr, &mapPtr, (void *)&numElems};
    size_t mapArgSizes[] = {sizeof(devPtr), sizeof(mapPtr), sizeof(numElems)};

    EXPECT_NO_ERR(ctx0.enqueueFunctionExecDirect(
        *mapFun, numElems / blockDim, 1, 1, blockDim, 1, 1, 0,
        /*attrs=*/nullptr, /*numAttrs=*/0, mapArgs,
        sizeof(mapArgs) / sizeof(void *), mapArgSizes));

    bzero(hostPtr, allocLen); // Reset the memory used to validate.
    EXPECT_NO_ERR(devBuf->copyTo(*hostBuf));
    EXPECT_NO_ERR(ctx0.synchronize());
    for (size_t j = 0; j < numElems; ++j) {
      if (j < 5) {
        std::cerr << "buf[" << j << "]: " << hostPtr[j] << "\n";
      }
      // TODO(iposva): Figure out why multimem reduction instructions do not
      // appear to work.
    }
    std::cerr << "------------\n";
  }

  {
    std::cerr << "Using multicast to splat across all devices.\n";
    for (size_t j = 0; j < numElems; ++j) {
      hostPtr[j] = j + (5.0 * j);
    }
    EXPECT_NO_ERR(hostBuf->copyTo(*devBuf));

    void *splatMapPtr = mapBuf->ptr();
    void *splatDevPtr = devBuf->ptr();
    void *splatArgs[] = {&splatMapPtr, &splatDevPtr, (void *)&numElems};
    size_t splatArgSizes[] = {sizeof(splatMapPtr), sizeof(splatDevPtr),
                              sizeof(numElems)};

    EXPECT_NO_ERR(ctx0.enqueueFunctionExecDirect(
        *splatFun, numElems / blockDim, 1, 1, blockDim, 1, 1, 0,
        /*attrs=*/nullptr, /*numAttrs=*/0, splatArgs,
        sizeof(splatArgs) / sizeof(void *), splatArgSizes));

    EXPECT_NO_ERR(ctx0.synchronize());
    for (int i = 0; i < numDevices; ++i) {
      DeviceContext &ctx = *contexts[i];
      bzero(hostPtr, allocLen); // Reset the memory used to validate.
      ASSIGN_NO_ERR(DeviceBufferRef, uniBuf, multiBuf->unicastBufferFor(ctx));
      EXPECT_NO_ERR(uniBuf->copyTo(*hostBuf));
      EXPECT_NO_ERR(ctx.synchronize());
      for (size_t j = 0; j < numElems; ++j) {
        if (j < 5) {
          std::cerr << "unicast buf[" << j << "/device(" << ctx.device().id()
                    << ")]: " << hostPtr[j] << "\n";
        }
        EXPECT_EQ(hostPtr[j], j + (5.0 * j));
      }
      std::cerr << "------------\n";
    }
  }
}

// A context per CUDA device when the node can back a symmetric pool, else
// empty so the caller skips. The pool registry is process-wide and create-once,
// so each SymmetricPool test is its own binary (see BUILD.bazel) and starts
// with the registry empty.
static std::vector<DeviceContextRef> symmetricPoolContexts() {
  int numDevices = Device::numberOfDevices(Device::cudaAPI);
  std::vector<DeviceContextRef> contexts;
  if (numDevices < 2)
    return contexts;
  for (int i = 0; i < numDevices; ++i) {
    ErrorOr<DeviceContextRef> errCtx =
        DeviceContext::create(Device::cudaAPI, i);
    if (errCtx.isError())
      return {};
    contexts.push_back(errCtx.takeValue());
  }
  ErrorOr<bool> multicastSupported = contexts[0]->device().supportsMulticast();
  if (multicastSupported.isError() || !multicastSupported.get())
    return {};
  return contexts;
}

TEST_F(CUDADeviceContextTest, SymmetricPoolCreateOnceAndLookup) {
  std::vector<DeviceContextRef> contexts = symmetricPoolContexts();
  if (contexts.empty())
    GTEST_SKIP() << "symmetric pool needs >= 2 multicast-capable devices";
  const int numDevices = static_cast<int>(contexts.size());

  // Lookup never creates.
  EXPECT_FALSE(SymmetricPool::lookup(contexts));

  const size_t requestedBytes = 4 * 1024;
  ASSIGN_NO_ERR(DeviceMulticastBufferRef, pool,
                SymmetricPool::create(contexts, requestedBytes));
  ASSERT_TRUE(pool);
  EXPECT_EQ(pool->byteSize(), requestedBytes);

  // A replaced pool would leave stale pointers in captured device graphs.
  ErrorOr<DeviceMulticastBufferRef> second =
      SymmetricPool::create(contexts, requestedBytes);
  EXPECT_TRUE(second.isError());

  // Lookup is stable, and independent of the order contexts are listed in.
  DeviceMulticastBufferRef found = SymmetricPool::lookup(contexts);
  ASSERT_TRUE(found);
  EXPECT_EQ(found.getPointer(), pool.getPointer());

  std::vector<DeviceContextRef> reordered;
  for (auto it = contexts.rbegin(); it != contexts.rend(); ++it)
    reordered.push_back(it->copy());
  DeviceMulticastBufferRef foundReordered = SymmetricPool::lookup(reordered);
  ASSERT_TRUE(foundReordered);
  EXPECT_EQ(foundReordered.getPointer(), pool.getPointer());

  // A subset of the device set is a different key, not a partial match.
  if (numDevices > 2) {
    std::vector<DeviceContextRef> subset;
    subset.push_back(contexts[0].copy());
    subset.push_back(contexts[1].copy());
    EXPECT_FALSE(SymmetricPool::lookup(subset));
  }

  // Each device's unicast view is writable memory holding its own data.
  const size_t numElems = requestedBytes / sizeof(float);
  for (int i = 0; i < numDevices; ++i) {
    DeviceContext &ctx = *contexts[i];
    ASSIGN_NO_ERR(DeviceBufferRef, hostBuf,
                  ctx.createHostBuffer(requestedBytes));
    float *hostPtr = static_cast<float *>(hostBuf->ptr());
    for (size_t j = 0; j < numElems; ++j)
      hostPtr[j] = static_cast<float>(j + 3 * (ctx.device().id() + 1));

    ASSIGN_NO_ERR(DeviceBufferRef, uniBuf, pool->unicastBufferFor(ctx));
    EXPECT_NO_ERR(hostBuf->copyTo(*uniBuf));
    bzero(hostPtr, requestedBytes);
    EXPECT_NO_ERR(uniBuf->copyTo(*hostBuf));
    EXPECT_NO_ERR(ctx.synchronize());
    for (size_t j = 0; j < numElems; ++j) {
      ASSERT_EQ(hostPtr[j],
                static_cast<float>(j + 3 * (ctx.device().id() + 1)));
    }
  }
}

// Free-memory comparisons tolerate the driver's own lazy allocations.
static constexpr size_t kFreeMemoryTolerance = 128 * 1024 * 1024;

// A failed create is a fallback in serving, so it must leave the registry
// empty and give back everything it reserved, including whatever a later
// create needs.
TEST_F(CUDADeviceContextTest, SymmetricPoolFailedCreateLeavesNothingBehind) {
  std::vector<DeviceContextRef> contexts = symmetricPoolContexts();
  if (contexts.empty())
    GTEST_SKIP() << "symmetric pool needs >= 2 multicast-capable devices";
  const Device &dev0 = contexts[0]->device();

  ASSIGN_NO_ERR(Device::MemInfo, before, dev0.getMemoryInfo());
  // More than the device holds, so the per-device physical allocation cannot
  // succeed whatever else does.
  ErrorOr<DeviceMulticastBufferRef> failed =
      SymmetricPool::create(contexts, before.total);
  ASSERT_TRUE(failed.isError());
  EXPECT_FALSE(SymmetricPool::lookup(contexts));

  ASSIGN_NO_ERR(Device::MemInfo, after, dev0.getMemoryInfo());
  EXPECT_GE(after.free + kFreeMemoryTolerance, before.free)
      << "failed create leaked " << (before.free - after.free) << " bytes";

  ASSIGN_NO_ERR(DeviceMulticastBufferRef, pool,
                SymmetricPool::create(contexts, 4 * 1024));
  ASSERT_TRUE(pool);
  EXPECT_TRUE(SymmetricPool::lookup(contexts));
}

TEST_F(CUDADeviceContextTest, NVMLGetPciInfo) {
  // Get NVML instance
  ErrorOr<const NVML &> nvmlOr = NVML::get();
  ASSERT_FALSE(nvmlOr.isError()) << nvmlOr.getError();
  const NVML &nvml = *nvmlOr;

  // Get device handle for GPU 0
  nvmlDevice_t device;
  ASSERT_EQ(nvml.nvmlDeviceGetHandleByIndex(0, &device), NVML::SUCCESS);

  // Retrieve PCI info
  nvmlPciInfo_t pciInfo;
  ASSERT_EQ(nvml.nvmlDeviceGetPciInfo_v3(device, &pciInfo), NVML::SUCCESS);

  // Validate PCI info fields
  EXPECT_GT(strlen(pciInfo.busId), 0u);
  EXPECT_LE(pciInfo.bus, 0xffu);
  EXPECT_LE(pciInfo.device, 31u);
}

TEST_F(CUDADeviceContextTest, CUDA13DriverVersion) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, 0));
  ASSIGN_NO_ERR(int, apiVersion, ctx->device().getApiVersion());
  EXPECT_GE(apiVersion, 13000)
      << "Expected CUDA 13+ API version after libnvptxcompiler upgrade";
}

// ===== GPU Healthcheck Tests ===== //

TEST_F(CUDADeviceContextTest, HealthcheckSucceedsOnHealthyGPU) {
  // On a healthy GPU, the healthcheck should pass.
  auto deviceOr = CUDADevice::get(0);
  ASSERT_FALSE(deviceOr.isError()) << deviceOr.getError();
  EXPECT_NO_ERR(deviceOr->runHealthcheck());
}

TEST_F(CUDADeviceContextTest, HealthcheckDisabledByEnvVar) {
  ScopedEnvVar disableHealthcheck("MODULAR_DEVICE_CONTEXT_DISABLE_HEALTHCHECK",
                                  "true");

  auto deviceOr = CUDADevice::get(0);
  ASSERT_FALSE(deviceOr.isError()) << deviceOr.getError();
  EXPECT_NO_ERR(deviceOr->runHealthcheck());
}

TEST_F(CUDADeviceContextTest, NVMLThrottleReasonsQueryable) {
  // Verify the new NVML throttle reasons API is callable.
  ErrorOr<const NVML &> nvmlOr = NVML::get();
  ASSERT_FALSE(nvmlOr.isError()) << nvmlOr.getError();
  const NVML &nvml = *nvmlOr;

  nvmlDevice_t device;
  ASSERT_EQ(nvml.nvmlDeviceGetHandleByIndex(0, &device), NVML::SUCCESS);

  unsigned long long throttleReasons = 0;
  NVML::Result result =
      nvml.nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttleReasons);
  // Should succeed or return NOT_SUPPORTED on some hardware.
  EXPECT_TRUE(result == NVML::SUCCESS || result == NVML::NOT_SUPPORTED)
      << "Unexpected NVML result: " << result;
}

TEST_F(CUDADeviceContextTest, NVMLEccErrorsQueryable) {
  // Verify the new NVML ECC errors API is callable.
  ErrorOr<const NVML &> nvmlOr = NVML::get();
  ASSERT_FALSE(nvmlOr.isError()) << nvmlOr.getError();
  const NVML &nvml = *nvmlOr;

  nvmlDevice_t device;
  ASSERT_EQ(nvml.nvmlDeviceGetHandleByIndex(0, &device), NVML::SUCCESS);

  unsigned long long eccErrors = 0;
  NVML::Result result = nvml.nvmlDeviceGetTotalEccErrors(
      device, NVML_MEMORY_ERROR_TYPE_UNCORRECTABLE, NVML_VOLATILE_ECC,
      &eccErrors);
  // Should succeed or return NOT_SUPPORTED (e.g., consumer GPUs without ECC).
  EXPECT_TRUE(result == NVML::SUCCESS || result == NVML::NOT_SUPPORTED)
      << "Unexpected NVML result: " << result;
}

TEST_F(CUDADeviceContextTest, NVMLComputeProcessesQueryable) {
  // Verify the new NVML compute processes API is callable.
  ErrorOr<const NVML &> nvmlOr = NVML::get();
  ASSERT_FALSE(nvmlOr.isError()) << nvmlOr.getError();
  const NVML &nvml = *nvmlOr;

  nvmlDevice_t device;
  ASSERT_EQ(nvml.nvmlDeviceGetHandleByIndex(0, &device), NVML::SUCCESS);

  // Query with nullptr buffer to just get the count.
  unsigned int processCount = 0;
  NVML::Result result = nvml.nvmlDeviceGetComputeRunningProcesses_v3(
      device, &processCount, nullptr);
  // SUCCESS means 0 processes, any other non-error result means processes
  // exist. We just verify the call doesn't crash.
  (void)result;
  (void)processCount;
}

// When NVML refuses the parent-device memory query — as it does in a container
// granted only a MIG compute instance — CUDADevice::get() takes the device's
// memory from CUDA instead. On non-MIG hardware both sources describe the same
// device, so their totals must agree; this pins the argument order and units
// the fallback depends on.
TEST_F(CUDADeviceContextTest, CUDAMemoryInfoAgreesWithNVML) {
  auto deviceOr = CUDADevice::get(0);
  ASSERT_FALSE(deviceOr.isError()) << deviceOr.getError();
  ASSIGN_NO_ERR(Device::MemInfo, cudaMem, deviceOr->getMemoryInfo());

  EXPECT_GT(cudaMem.total, 0u);
  EXPECT_LE(cudaMem.free, cudaMem.total);

  ErrorOr<const NVML &> nvmlOr = NVML::get();
  ASSERT_FALSE(nvmlOr.isError()) << nvmlOr.getError();

  nvmlDevice_t nvmlDevice;
  ASSERT_EQ(nvmlOr->nvmlDeviceGetHandleByIndex(0, &nvmlDevice), NVML::SUCCESS);

  nvmlMemory_t nvmlMem;
  if (nvmlOr->nvmlDeviceGetMemoryInfo(nvmlDevice, &nvmlMem) != NVML::SUCCESS)
    GTEST_SKIP() << "NVML cannot report memory for this device";

  // CUDA reports memory usable by CUDA, which is a little below the installed
  // framebuffer NVML reports, so compare within a tolerance rather than
  // exactly. The comparison presumes a device NVML and CUDA describe alike,
  // which a MIG slice is not: there NVML answers for the parent GPU and the
  // two totals differ by an order of magnitude.
  EXPECT_GT(cudaMem.total, nvmlMem.total * 9 / 10)
      << "CUDA total " << cudaMem.total << " is far below NVML total "
      << nvmlMem.total
      << ": either cuMemGetInfo's out-parameters are swapped, or this is a "
         "MIG device and the test should skip instead";
  EXPECT_LE(cudaMem.total, nvmlMem.total);
}

// Whichever source the count came from, every id in [0, count) must be one
// cuDeviceGet accepts, because that is how the count is subsequently spent.
// NVML supplies it when no device has MIG enabled and CUDA when one does;
// without MIG the two agree, which is what this asserts.
TEST_F(CUDADeviceContextTest, DeviceCountMatchesCUDAEnumeration) {
  ErrorOr<const CUDADriver &> driverOr = CUDADriver::get();
  ASSERT_FALSE(driverOr.isError()) << driverOr.getError();

  int cudaCount = 0;
  ASSERT_EQ(driverOr->cuDeviceGetCount(&cudaCount), CUDADriver::SUCCESS);

  ASSIGN_NO_ERR(int, count, CUDADevice::numberOfGPUs());
  EXPECT_GT(count, 0);
  // CUDA_VISIBLE_DEVICES narrowed after cuInit only ever lowers the count.
  if (llvm::sys::Process::GetEnv("CUDA_VISIBLE_DEVICES"))
    EXPECT_LE(count, cudaCount);
  else
    EXPECT_EQ(count, cudaCount);

  for (int id = 0; id < count; ++id) {
    CUdevice cuDevice;
    EXPECT_EQ(driverOr->cuDeviceGet(&cuDevice, id), CUDADriver::SUCCESS)
        << "device " << id << " is counted but CUDA will not resolve it";
  }
}

// Each device's NVML handle is resolved by PCI location rather than by reusing
// the device id as an NVML index, because the two enumerations are independent
// — under MIG several CUDA devices share one physical GPU. On non-MIG hardware
// the two agree, so the handle a device carries must describe the PCI location
// CUDA reports for it.
TEST_F(CUDADeviceContextTest, PciBusIdMatchesCUDAsViewOfTheDevice) {
  ErrorOr<const CUDADriver &> driverOr = CUDADriver::get();
  ASSERT_FALSE(driverOr.isError()) << driverOr.getError();

  ASSIGN_NO_ERR(int, count, CUDADevice::numberOfGPUs());
  for (int id = 0; id < count; ++id) {
    ErrorOr<CUDADevice &> deviceOr = CUDADevice::get(id);
    ASSERT_FALSE(deviceOr.isError()) << deviceOr.getError();
    ASSIGN_NO_ERR(std::string, busId, deviceOr->getPciBusId());

    int domain = 0, bus = 0, slot = 0;
    ASSERT_EQ(driverOr->cuDeviceGetAttribute(&domain,
                                             CUdevice_attribute::PCI_DOMAIN_ID,
                                             deviceOr->cudaDevice()),
              CUDADriver::SUCCESS);
    ASSERT_EQ(driverOr->cuDeviceGetAttribute(
                  &bus, CUdevice_attribute::PCI_BUS_ID, deviceOr->cudaDevice()),
              CUDADriver::SUCCESS);
    ASSERT_EQ(driverOr->cuDeviceGetAttribute(&slot,
                                             CUdevice_attribute::PCI_DEVICE_ID,
                                             deviceOr->cudaDevice()),
              CUDADriver::SUCCESS);

    char expected[32];
    snprintf(expected, sizeof(expected), "%08X:%02X:%02X.0", domain, bus, slot);
    EXPECT_EQ(busId, std::string(expected))
        << "device " << id << " carries an NVML handle for a different GPU";
  }
}

// CUDA does not override maxSingleAllocationSize, so the default Device
// implementation returns getMemoryInfo().total.
TEST_F(CUDADeviceContextTest, MaxSingleAllocationSize_DefaultReturnsTotal) {
  ASSIGN_NO_ERR(DeviceRef, device, Device::create(Device::gpuAPI));
  ASSIGN_NO_ERR(Device::MemInfo, info, device->getMemoryInfo());
  ASSIGN_NO_ERR(size_t, maxAlloc, device->maxSingleAllocationSize());
  EXPECT_EQ(maxAlloc, info.total);
}

// Create/destroy cycles must recycle cached events, not grow the cache.
TEST_F(CUDADeviceContextTest, EventCreateReusesCachedEvent) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));

  ASSIGN_NO_ERR(DeviceEventRef, first, ctx->eventCreate(false));
  CUevent handle = CUDADeviceEvent::cast(*first).cudaEvent();
  first = DeviceEventRef(); // The destructor pushes the handle to the cache.

  ASSIGN_NO_ERR(DeviceEventRef, second, ctx->eventCreate(false));
  EXPECT_EQ(CUDADeviceEvent::cast(*second).cudaEvent(), handle);
  second = DeviceEventRef();

  // Timing events bypass the cache and must not consume the cached handle.
  ASSIGN_NO_ERR(DeviceEventRef, timed, ctx->eventCreate(true));
  EXPECT_NE(CUDADeviceEvent::cast(*timed).cudaEvent(), handle);
  ASSIGN_NO_ERR(DeviceEventRef, third, ctx->eventCreate(false));
  EXPECT_EQ(CUDADeviceEvent::cast(*third).cudaEvent(), handle);
}

} // namespace

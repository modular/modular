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

#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"

#include "llvm/ADT/Sequence.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>

using namespace M::Driver;
using M::ErrorOr;
using M::ErrorOrSuccess;
using M::success;

namespace {

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
	add.f32 	%f3, %f2, %f1;
	st.f32 	[%rd1], %f3;
$L__BB0_2:
	ret;
}
)PTX";

class GraphCaptureMultiDeviceTest : public ::testing::Test {
protected:
  void SetUp() override {
    runtime_ = M::AsyncRT::getOrCreateCPUDevice(
        M::AsyncRT::CPUDeviceSource::Test,
        M::AsyncRT::CPUDeviceOptions().withMainWillNotDonate());
  }
  M::AsyncRT::CPUDeviceRef runtime_;
};

TEST_F(GraphCaptureMultiDeviceTest, MultiDeviceGraphSupport) {
  int numDevices = Device::numberOfDevices(Device::cudaAPI);
  ASSERT_GE(numDevices, 2);

  llvm::SmallVector<DeviceContextRef> contexts;
  llvm::SmallVector<DeviceStreamRef> streams;

  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceContextRef, ctx,
                  DeviceContext::create(Device::cudaAPI, /*id=*/id));
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

    contexts.push_back(std::move(ctx));
    streams.push_back(std::move(stream));
  }

  llvm::SmallVector<DeviceFunctionRef> deviceFuncs;

  // Setup each device function.
  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(
        DeviceFunctionRef, deviceFunc,
        contexts[id]->loadFunction("vec_add_module", "vec_add", vecAddPtx));
    deviceFuncs.push_back(std::move(deviceFunc));
  }

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);

  llvm::SmallVector<
      std::tuple<DeviceBufferRef, DeviceBufferRef, DeviceBufferRef>>
      kernelArgs;

  // Setup the input/output buffers for each device function.
  for (auto id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));

    kernelArgs.push_back(
        {std::move(bufOut), std::move(bufIn0), std::move(bufIn1)});
  }

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  // Capture an execution of the kernel on each device.
  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, streams, [&]() -> ErrorOrSuccess {
        for (auto id : llvm::seq(numDevices)) {
          void *outPtr = std::get<0>(kernelArgs[id])->ptr();
          void *bufIn0 = std::get<1>(kernelArgs[id])->ptr();
          void *bufIn1 = std::get<2>(kernelArgs[id])->ptr();

          void *args[] = {(void *)&outPtr, (void *)&bufIn0, (void *)&bufIn1,
                          (void *)&length};

          auto err = streams[id]->enqueueFunctionExecDirect(
              *deviceFuncs[id],
              /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
              /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
              /*sharedMemBytes=*/0,
              /*attrs=*/nullptr, /*numAttrs=*/0, args);

          if (err.isError())
            return err.takeError();
        }

        return success();
      });

  // Successfully ran the capture process.
  EXPECT_FALSE(captured.isError());

  // One graph was produced.
  EXPECT_EQ(captured->numGraphs(), numDevices);

  // The graph can be replayed.
  EXPECT_NO_ERR(captured->replay());
  EXPECT_NO_ERR(captured->replay());
}

// Test: debugVerifyReplay succeeds when multi-device execution matches captured
// graphs.
TEST_F(GraphCaptureMultiDeviceTest, DebugVerifyReplayMultiDeviceSuccess) {
  int numDevices = Device::numberOfDevices(Device::cudaAPI);
  ASSERT_GE(numDevices, 2);

  llvm::SmallVector<DeviceContextRef> contexts;
  llvm::SmallVector<DeviceStreamRef> streams;

  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceContextRef, ctx,
                  DeviceContext::create(Device::cudaAPI, /*id=*/id));
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

    contexts.push_back(std::move(ctx));
    streams.push_back(std::move(stream));
  }

  llvm::SmallVector<DeviceFunctionRef> deviceFuncs;

  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(
        DeviceFunctionRef, deviceFunc,
        contexts[id]->loadFunction("vec_add_module", "vec_add", vecAddPtx));
    deviceFuncs.push_back(std::move(deviceFunc));
  }

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);

  llvm::SmallVector<
      std::tuple<DeviceBufferRef, DeviceBufferRef, DeviceBufferRef>>
      kernelArgs;

  for (auto id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));

    kernelArgs.push_back(
        {std::move(bufOut), std::move(bufIn0), std::move(bufIn1)});
  }

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  auto execute = [&]() -> ErrorOrSuccess {
    for (auto id : llvm::seq(numDevices)) {
      void *outPtr = std::get<0>(kernelArgs[id])->ptr();
      void *bufIn0 = std::get<1>(kernelArgs[id])->ptr();
      void *bufIn1 = std::get<2>(kernelArgs[id])->ptr();

      void *args[] = {(void *)&outPtr, (void *)&bufIn0, (void *)&bufIn1,
                      (void *)&length};

      auto err = streams[id]->enqueueFunctionExecDirect(
          *deviceFuncs[id],
          /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
          /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args);

      if (err.isError())
        return err.takeError();
    }
    return success();
  };

  // Capture execution on all devices
  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, streams, execute);
  EXPECT_FALSE(captured.isError());
  EXPECT_EQ(captured->numGraphs(), numDevices);

  // Verify that the same execution matches all captured graphs
  EXPECT_NO_ERR(captured->debugVerifyReplay(execute));
}

// Test: debugVerifyReplay detects mismatch on one device in multi-device setup.
TEST_F(GraphCaptureMultiDeviceTest,
       DebugVerifyReplayMultiDeviceDetectsMismatch) {
  int numDevices = Device::numberOfDevices(Device::cudaAPI);
  ASSERT_GE(numDevices, 2);

  llvm::SmallVector<DeviceContextRef> contexts;
  llvm::SmallVector<DeviceStreamRef> streams;

  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceContextRef, ctx,
                  DeviceContext::create(Device::cudaAPI, /*id=*/id));
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

    contexts.push_back(std::move(ctx));
    streams.push_back(std::move(stream));
  }

  llvm::SmallVector<DeviceFunctionRef> deviceFuncs;

  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(
        DeviceFunctionRef, deviceFunc,
        contexts[id]->loadFunction("vec_add_module", "vec_add", vecAddPtx));
    deviceFuncs.push_back(std::move(deviceFunc));
  }

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);

  llvm::SmallVector<
      std::tuple<DeviceBufferRef, DeviceBufferRef, DeviceBufferRef>>
      kernelArgs;

  for (auto id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));

    kernelArgs.push_back(
        {std::move(bufOut), std::move(bufIn0), std::move(bufIn1)});
  }

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  // Capture: execute once on each device
  auto captureExecute = [&]() -> ErrorOrSuccess {
    for (auto id : llvm::seq(numDevices)) {
      void *outPtr = std::get<0>(kernelArgs[id])->ptr();
      void *bufIn0 = std::get<1>(kernelArgs[id])->ptr();
      void *bufIn1 = std::get<2>(kernelArgs[id])->ptr();

      void *args[] = {(void *)&outPtr, (void *)&bufIn0, (void *)&bufIn1,
                      (void *)&length};

      auto err = streams[id]->enqueueFunctionExecDirect(
          *deviceFuncs[id],
          /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
          /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args);

      if (err.isError())
        return err.takeError();
    }
    return success();
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, streams, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Verify: execute TWICE on the first device (creates mismatch on device 0)
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    for (auto id : llvm::seq(numDevices)) {
      void *outPtr = std::get<0>(kernelArgs[id])->ptr();
      void *bufIn0 = std::get<1>(kernelArgs[id])->ptr();
      void *bufIn1 = std::get<2>(kernelArgs[id])->ptr();

      void *args[] = {(void *)&outPtr, (void *)&bufIn0, (void *)&bufIn1,
                      (void *)&length};

      auto err = streams[id]->enqueueFunctionExecDirect(
          *deviceFuncs[id],
          /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
          /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args);

      if (err.isError())
        return err.takeError();

      // Extra launch on first device only (creates mismatch)
      if (id == 0) {
        auto err2 = streams[id]->enqueueFunctionExecDirect(
            *deviceFuncs[id],
            /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
            /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
            /*sharedMemBytes=*/0,
            /*attrs=*/nullptr, /*numAttrs=*/0, args);
        if (err2.isError())
          return err2.takeError();
      }
    }
    return success();
  };

  // Should detect mismatch on device 0
  EXPECT_ERR(captured->debugVerifyReplay(verifyExecute));
}

// Test: debugVerifyReplay detects different grid dimensions on one device.
TEST_F(GraphCaptureMultiDeviceTest,
       DebugVerifyReplayMultiDeviceDetectsDifferentGridDim) {
  int numDevices = Device::numberOfDevices(Device::cudaAPI);
  ASSERT_GE(numDevices, 2);

  llvm::SmallVector<DeviceContextRef> contexts;
  llvm::SmallVector<DeviceStreamRef> streams;

  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceContextRef, ctx,
                  DeviceContext::create(Device::cudaAPI, /*id=*/id));
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

    contexts.push_back(std::move(ctx));
    streams.push_back(std::move(stream));
  }

  llvm::SmallVector<DeviceFunctionRef> deviceFuncs;

  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(
        DeviceFunctionRef, deviceFunc,
        contexts[id]->loadFunction("vec_add_module", "vec_add", vecAddPtx));
    deviceFuncs.push_back(std::move(deviceFunc));
  }

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);

  llvm::SmallVector<
      std::tuple<DeviceBufferRef, DeviceBufferRef, DeviceBufferRef>>
      kernelArgs;

  for (auto id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));

    kernelArgs.push_back(
        {std::move(bufOut), std::move(bufIn0), std::move(bufIn1)});
  }

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  // Capture: standard grid dimensions on all devices
  auto captureExecute = [&]() -> ErrorOrSuccess {
    for (auto id : llvm::seq(numDevices)) {
      void *outPtr = std::get<0>(kernelArgs[id])->ptr();
      void *bufIn0 = std::get<1>(kernelArgs[id])->ptr();
      void *bufIn1 = std::get<2>(kernelArgs[id])->ptr();

      void *args[] = {(void *)&outPtr, (void *)&bufIn0, (void *)&bufIn1,
                      (void *)&length};

      auto err = streams[id]->enqueueFunctionExecDirect(
          *deviceFuncs[id],
          /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
          /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args);

      if (err.isError())
        return err.takeError();
    }
    return success();
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, streams, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Verify: different grid dimension on the last device
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    for (auto id : llvm::seq(numDevices)) {
      void *outPtr = std::get<0>(kernelArgs[id])->ptr();
      void *bufIn0 = std::get<1>(kernelArgs[id])->ptr();
      void *bufIn1 = std::get<2>(kernelArgs[id])->ptr();

      void *args[] = {(void *)&outPtr, (void *)&bufIn0, (void *)&bufIn1,
                      (void *)&length};

      // Use different grid dimensions on last device
      uint32_t useGridDim = (id == numDevices - 1) ? gridDim * 2 : gridDim;

      auto err = streams[id]->enqueueFunctionExecDirect(
          *deviceFuncs[id],
          /*gridX=*/useGridDim, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
          /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args);

      if (err.isError())
        return err.takeError();
    }
    return success();
  };

  // Should detect parameter mismatch on last device
  EXPECT_ERR(captured->debugVerifyReplay(verifyExecute));
}

// Test: debugVerifyReplay detects when execution is missing on one device.
TEST_F(GraphCaptureMultiDeviceTest,
       DebugVerifyReplayMultiDeviceMissingExecution) {
  int numDevices = Device::numberOfDevices(Device::cudaAPI);
  ASSERT_GE(numDevices, 2);

  llvm::SmallVector<DeviceContextRef> contexts;
  llvm::SmallVector<DeviceStreamRef> streams;

  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceContextRef, ctx,
                  DeviceContext::create(Device::cudaAPI, /*id=*/id));
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

    contexts.push_back(std::move(ctx));
    streams.push_back(std::move(stream));
  }

  llvm::SmallVector<DeviceFunctionRef> deviceFuncs;

  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(
        DeviceFunctionRef, deviceFunc,
        contexts[id]->loadFunction("vec_add_module", "vec_add", vecAddPtx));
    deviceFuncs.push_back(std::move(deviceFunc));
  }

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);

  llvm::SmallVector<
      std::tuple<DeviceBufferRef, DeviceBufferRef, DeviceBufferRef>>
      kernelArgs;

  for (auto id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));

    kernelArgs.push_back(
        {std::move(bufOut), std::move(bufIn0), std::move(bufIn1)});
  }

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  // Capture: execute on all devices
  auto captureExecute = [&]() -> ErrorOrSuccess {
    for (auto id : llvm::seq(numDevices)) {
      void *outPtr = std::get<0>(kernelArgs[id])->ptr();
      void *bufIn0 = std::get<1>(kernelArgs[id])->ptr();
      void *bufIn1 = std::get<2>(kernelArgs[id])->ptr();

      void *args[] = {(void *)&outPtr, (void *)&bufIn0, (void *)&bufIn1,
                      (void *)&length};

      auto err = streams[id]->enqueueFunctionExecDirect(
          *deviceFuncs[id],
          /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
          /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args);

      if (err.isError())
        return err.takeError();
    }
    return success();
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, streams, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Verify: skip execution on the last device (trace will be empty)
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    for (auto id : llvm::seq(numDevices - 1)) { // Skip last device
      void *outPtr = std::get<0>(kernelArgs[id])->ptr();
      void *bufIn0 = std::get<1>(kernelArgs[id])->ptr();
      void *bufIn1 = std::get<2>(kernelArgs[id])->ptr();

      void *args[] = {(void *)&outPtr, (void *)&bufIn0, (void *)&bufIn1,
                      (void *)&length};

      auto err = streams[id]->enqueueFunctionExecDirect(
          *deviceFuncs[id],
          /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
          /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args);

      if (err.isError())
        return err.takeError();
    }
    return success();
  };

  // Should detect missing execution on last device
  EXPECT_ERR(captured->debugVerifyReplay(verifyExecute));
}

// Test: debugVerifyReplay handles reordered execution on multiple devices.
// Simulates AsyncRT non-deterministic scheduling across multiple GPUs.
TEST_F(GraphCaptureMultiDeviceTest,
       DebugVerifyReplayHandlesReorderedExecution) {
  int numDevices = Device::numberOfDevices(Device::cudaAPI);
  ASSERT_GE(numDevices, 2);

  llvm::SmallVector<DeviceContextRef> contexts;
  llvm::SmallVector<DeviceStreamRef> streams;

  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceContextRef, ctx,
                  DeviceContext::create(Device::cudaAPI, /*id=*/id));
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

    contexts.push_back(std::move(ctx));
    streams.push_back(std::move(stream));
  }

  llvm::SmallVector<DeviceFunctionRef> deviceFuncs;

  for (int id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(
        DeviceFunctionRef, deviceFunc,
        contexts[id]->loadFunction("vec_add_module", "vec_add", vecAddPtx));
    deviceFuncs.push_back(std::move(deviceFunc));
  }

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);

  llvm::SmallVector<
      std::tuple<DeviceBufferRef, DeviceBufferRef, DeviceBufferRef>>
      kernelArgs;

  for (auto id : llvm::seq(numDevices)) {
    ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));
    ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                  contexts[id]->enqueueCreateBuffer(length * elemSize));

    kernelArgs.push_back(
        {std::move(bufOut), std::move(bufIn0), std::move(bufIn1)});
  }

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  // Capture: Each device launches two kernels (A with full grid, B with half
  // grid)
  auto captureExecute = [&]() -> ErrorOrSuccess {
    for (auto id : llvm::seq(numDevices)) {
      void *outPtr = std::get<0>(kernelArgs[id])->ptr();
      void *bufIn0 = std::get<1>(kernelArgs[id])->ptr();
      void *bufIn1 = std::get<2>(kernelArgs[id])->ptr();

      void *args[] = {(void *)&outPtr, (void *)&bufIn0, (void *)&bufIn1,
                      (void *)&length};

      // Kernel A: full grid
      auto errA = streams[id]->enqueueFunctionExecDirect(
          *deviceFuncs[id],
          /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
          /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args);

      if (errA.isError())
        return errA.takeError();

      // Kernel B: half grid (different hash from A)
      auto errB = streams[id]->enqueueFunctionExecDirect(
          *deviceFuncs[id],
          /*gridX=*/gridDim / 2, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
          /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args);

      if (errB.isError())
        return errB.takeError();
    }

    return success();
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, streams, captureExecute);
  EXPECT_FALSE(captured.isError());
  EXPECT_EQ(captured->numGraphs(), numDevices);

  // Verify: Each device launches kernels in reverse order (B then A)
  // This simulates AsyncRT non-deterministic scheduling
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    for (auto id : llvm::seq(numDevices)) {
      void *outPtr = std::get<0>(kernelArgs[id])->ptr();
      void *bufIn0 = std::get<1>(kernelArgs[id])->ptr();
      void *bufIn1 = std::get<2>(kernelArgs[id])->ptr();

      void *args[] = {(void *)&outPtr, (void *)&bufIn0, (void *)&bufIn1,
                      (void *)&length};

      // Kernel B first: half grid
      auto errB = streams[id]->enqueueFunctionExecDirect(
          *deviceFuncs[id],
          /*gridX=*/gridDim / 2, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
          /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args);

      if (errB.isError())
        return errB.takeError();

      // Kernel A second: full grid
      auto errA = streams[id]->enqueueFunctionExecDirect(
          *deviceFuncs[id],
          /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
          /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args);

      if (errA.isError())
        return errA.takeError();
    }

    return success();
  };

  // Should succeed: traces are permutations (same kernels, different order on
  // each device)
  EXPECT_NO_ERR(captured->debugVerifyReplay(verifyExecute));
}

} // namespace

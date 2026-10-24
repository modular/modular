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
#include "DeviceGraphTestUtils.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "ScopedEnvVar.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include "llvm/ADT/Sequence.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

using namespace M::Driver;
using M::ErrorOr;
using M::ErrorOrSuccess;
using M::success;

namespace {

static const char *smallScalarPtx = R"PTX(
// Kernel that takes sub-pointer-sized scalar parameters to test that
// parameters < 8 bytes ARE included in trace hash.

.version 6.3
.target sm_75
.address_size 64

.visible .entry small_scalar_kernel(
	.param .b64 ptr,           // pointer (8 bytes - excluded from hash)
	.param .u32 scalar32,      // 32-bit scalar (4 bytes - HASHED)
	.param .u32 grid_size      // 32-bit scalar (4 bytes - HASHED)
)
{
	ret;
}
)PTX";

static const char *mixedArgsPtx = R"PTX(
// Kernel that takes pointer-sized scalar arguments to test that
// parameters >= 8 bytes are conservatively excluded from hash.

.version 6.3
.target sm_75
.address_size 64

.visible .entry mixed_args(
	.param .b64 mixed_args_param_0,  // pointer (8 bytes)
	.param .b64 mixed_args_param_1,  // scalar (8 bytes - NOT hashed)
	.param .b64 mixed_args_param_2,  // pointer (8 bytes)
	.param .b64 mixed_args_param_3   // scalar (8 bytes - NOT hashed)
)
{
	ret;
}
)PTX";

static bool supportsLaunchTraceVerification(const DeviceContext &ctx) {
  return llvm::is_contained({Device::cudaAPI, Device::hipAPI},
                            ctx.device().api());
}

static bool supportsKernelReflection(const DeviceContext &ctx) {
  return llvm::is_contained({Device::cudaAPI}, ctx.device().api());
}

class GraphCaptureTest : public ::testing::Test {
protected:
  void SetUp() override {
    runtime_ = M::AsyncRT::getOrCreateCPUDevice(
        M::AsyncRT::CPUDeviceSource::Test,
        M::AsyncRT::CPUDeviceOptions().withMainWillNotDonate());
  }
  M::AsyncRT::CPUDeviceRef runtime_;
};

TEST_F(GraphCaptureTest, BeginAndEndCapture) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_NO_ERR(stream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());
  EXPECT_TRUE(graph);
}

TEST_F(GraphCaptureTest, DoubleBeginCaptureErrors) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_NO_ERR(stream->beginCapture());
  EXPECT_ERR(stream->beginCapture());

  // Clean up
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());
}

TEST_F(GraphCaptureTest, EndCaptureWithoutBeginErrors) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_ERR(stream->endCapture());
}

TEST_F(GraphCaptureTest, CaptureAndReplayEmpty) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_NO_ERR(stream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());

  // Instantiate before replaying (endCapture only ends stream capture).
  EXPECT_NO_ERR(graph->instantiate());

  // Replay the captured graph
  EXPECT_NO_ERR(graph->replay());
}

TEST_F(GraphCaptureTest, ReplayMultipleTimes) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_NO_ERR(stream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());

  // Instantiate before replaying (endCapture only ends stream capture).
  EXPECT_NO_ERR(graph->instantiate());

  // Replay multiple times
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(graph->replay());
}

TEST_F(GraphCaptureTest, CPUDeviceReturnsNotSupported) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

  EXPECT_ERR(stream->beginCapture());
  EXPECT_ERR(stream->endCapture());
}

TEST_F(GraphCaptureTest, ReplayOnCPUStreamErrors) {
  ASSIGN_NO_ERR(DeviceContextRef, gpuCtx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, gpuStream, gpuCtx->createStream());
  ASSIGN_NO_ERR(DeviceContextRef, cpuCtx,
                DeviceContext::create(Device::cpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, cpuStream, cpuCtx->stream());

  EXPECT_NO_ERR(gpuStream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, gpuStream->endCapture());
  graph->setStream(DeviceStreamRef::copy(cpuStream.getPointer()));
  EXPECT_ERR(graph->replay());
}

TEST_F(GraphCaptureTest, CaptureProducesGraphs) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc, loadVecAdd(*ctx));

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, [&]() -> ErrorOrSuccess {
        return stream->enqueueFunctionExecDirect(
            *deviceFunc,
            /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
            /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
            /*sharedMemBytes=*/0,
            /*attrs=*/nullptr, /*numAttrs=*/0, args);
      });

  // Successfully ran the capture process.
  EXPECT_FALSE(captured.isError());

  // One graph was produced.
  EXPECT_EQ(captured->numGraphs(), 1);

  // The graph can be replayed.
  EXPECT_NO_ERR(captured->replay());
  EXPECT_NO_ERR(captured->replay());
}

// Test: Capture using context->stream() (default stream pattern used by
// device graph capture in the engine API).
TEST_F(GraphCaptureTest, CaptureOnDefaultStream) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  // Get the default stream.
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

  EXPECT_NO_ERR(stream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());
  EXPECT_TRUE(graph);

  // Instantiate before replaying (endCapture only ends stream capture).
  EXPECT_NO_ERR(graph->instantiate());

  // Replay on the same stream
  EXPECT_NO_ERR(graph->replay());
}

// Test: debugVerifyReplay succeeds when execution matches captured graph.
TEST_F(GraphCaptureTest, DebugVerifyReplaySuccess) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc, loadVecAdd(*ctx));

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  auto execute = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Capture the execution
  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, execute);
  EXPECT_FALSE(captured.isError());

  // Verify that replaying the same execution matches the captured graph
  EXPECT_NO_ERR(captured->debugVerifyReplay(execute));
}

// Test: debugVerifyReplay detects when execution differs from captured graph.
TEST_F(GraphCaptureTest, DebugVerifyReplayDetectsMismatch) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is supported for CUDA and HIP";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc, loadVecAdd(*ctx));

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  // Capture with one kernel launch
  auto captureExecute = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Verify with TWO kernel launches (should mismatch)
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    auto err1 = stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
    if (err1.isError())
      return err1;

    // Second launch (creates a mismatch)
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Should detect the mismatch
  EXPECT_ERR(captured->debugVerifyReplay(verifyExecute));
}

// Test: debugVerifyReplay detects different grid dimensions.
TEST_F(GraphCaptureTest, DebugVerifyReplayDetectsDifferentGridDim) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is supported for CUDA and HIP";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc, loadVecAdd(*ctx));

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  // Capture with one grid dimension
  auto captureExecute = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Verify with different grid dimension
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim * 2, /*gridY=*/1, /*gridZ=*/1, // Different grid!
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Should detect the mismatch in launch parameters
  EXPECT_ERR(captured->debugVerifyReplay(verifyExecute));
}

// Test: debugVerifyReplay handles execution failure gracefully.
TEST_F(GraphCaptureTest, DebugVerifyReplayWithExecutionFailure) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  // Capture an empty graph
  auto captureExecute = [&]() -> ErrorOrSuccess { return success(); };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Verify with an execution that fails
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    return M::Error("Simulated execution failure");
  };

  // Should propagate the execution error
  EXPECT_ERR(captured->debugVerifyReplay(verifyExecute));
}

// Test: debugVerifyReplay handles empty graph (no operations).
TEST_F(GraphCaptureTest, DebugVerifyReplayEmptyGraph) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  // Capture and verify empty execution
  auto emptyExecute = [&]() -> ErrorOrSuccess { return success(); };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, emptyExecute);
  EXPECT_FALSE(captured.isError());

  // Verify empty execution should succeed
  EXPECT_NO_ERR(captured->debugVerifyReplay(emptyExecute));
}

// Test: debugVerifyReplay handles different execution order (simulates AsyncRT
// non-determinism). Verifies that traces are checked as permutations, not exact
// order matches.
TEST_F(GraphCaptureTest, DebugVerifyReplayHandlesReorderedExecution) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is supported for CUDA and HIP";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc, loadVecAdd(*ctx));

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut1,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut2,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *out1Ptr = bufOut1->ptr();
  void *out2Ptr = bufOut2->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  // Capture: Execute kernel A with gridDim, then kernel B with gridDim/2
  // Use different grid dimensions so kernels have different hashes
  auto captureExecute = [&]() -> ErrorOrSuccess {
    // Kernel A: bufOut1 = bufIn0 + bufIn1 (full grid)
    void *argsA[] = {(void *)&out1Ptr, (void *)&in0Ptr, (void *)&in1Ptr,
                     (void *)&length};
    auto errA = stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, argsA);
    if (errA.isError())
      return errA;

    // Kernel B: bufOut2 = bufIn0 + bufIn1 (half grid)
    void *argsB[] = {(void *)&out2Ptr, (void *)&in0Ptr, (void *)&in1Ptr,
                     (void *)&length};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim / 2, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, argsB);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Verify: Execute kernel B then kernel A (different order)
  // This simulates AsyncRT non-deterministic scheduling
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    // Kernel B first (half grid)
    void *argsB[] = {(void *)&out2Ptr, (void *)&in0Ptr, (void *)&in1Ptr,
                     (void *)&length};
    auto errB = stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim / 2, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, argsB);
    if (errB.isError())
      return errB;

    // Kernel A second (full grid)
    void *argsA[] = {(void *)&out1Ptr, (void *)&in0Ptr, (void *)&in1Ptr,
                     (void *)&length};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, argsA);
  };

  // Should succeed since traces are permutations of each other (same
  // operations, different order)
  EXPECT_NO_ERR(captured->debugVerifyReplay(verifyExecute));
}

// Test: Graph capture and replay works correctly with sync mode enabled.
TEST_F(GraphCaptureTest, CaptureWithSyncMode) {
  // Set MODULAR_DEVICE_CONTEXT_SYNC_MODE=true before creating context
  ScopedEnvVar env("MODULAR_DEVICE_CONTEXT_SYNC_MODE", "true");

  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  // Verify that sync mode is enabled
  EXPECT_TRUE(ctx->syncMode());

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc, loadVecAdd(*ctx));

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, [&]() -> ErrorOrSuccess {
        return stream->enqueueFunctionExecDirect(
            *deviceFunc,
            /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
            /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
            /*sharedMemBytes=*/0,
            /*attrs=*/nullptr, /*numAttrs=*/0, args);
      });

  // Successfully ran the capture process with sync mode enabled.
  EXPECT_FALSE(captured.isError());

  // One graph was produced.
  EXPECT_EQ(captured->numGraphs(), 1);

  // The graph can be replayed multiple times with sync mode enabled.
  EXPECT_NO_ERR(captured->replay());
  EXPECT_NO_ERR(captured->replay());
  EXPECT_NO_ERR(captured->replay());
}

// Test: Scalar parameter changes are detected in trace.
TEST_F(GraphCaptureTest, TraceDetectsScalarParameterChange) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is supported for CUDA and HIP";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc, loadVecAdd(*ctx));

  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(1024 * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(1024 * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(1024 * elemSize));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();

  const uint32_t blockDim = 256;

  // Capture with length=1024
  uint64_t captureLength = 1024;
  auto captureExecute = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                    (void *)&captureLength};
    const uint32_t gridDim = (captureLength + blockDim - 1) / blockDim;
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Replay with different length=2048 should detect mismatch
  uint64_t replayLength = 2048;
  auto replayExecute = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                    (void *)&replayLength};
    const uint32_t gridDim = (replayLength + blockDim - 1) / blockDim;
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Should detect the scalar parameter change
  EXPECT_ERR(captured->debugVerifyReplay(replayExecute));

  // Replay with same length=1024 should succeed
  EXPECT_NO_ERR(captured->debugVerifyReplay(captureExecute));
}

// Test: Pointer reallocation doesn't affect trace (existing behavior).
TEST_F(GraphCaptureTest, TraceStableAcrossBufferReallocation) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is supported for CUDA and HIP";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc, loadVecAdd(*ctx));

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);

  // Allocate first set of buffers
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut1,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn01,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn11,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *outPtr1 = bufOut1->ptr();
  void *in0Ptr1 = bufIn01->ptr();
  void *in1Ptr1 = bufIn11->ptr();
  void *args1[] = {(void *)&outPtr1, (void *)&in0Ptr1, (void *)&in1Ptr1,
                   (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  auto execute1 = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args1);
  };

  // Capture with first set of buffers
  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, execute1);
  EXPECT_FALSE(captured.isError());

  // Free first set of buffers and allocate new ones (likely at different
  // addresses)
  bufOut1.reset();
  bufIn01.reset();
  bufIn11.reset();

  ASSIGN_NO_ERR(DeviceBufferRef, bufOut2,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn02,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn12,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *outPtr2 = bufOut2->ptr();
  void *in0Ptr2 = bufIn02->ptr();
  void *in1Ptr2 = bufIn12->ptr();
  void *args2[] = {(void *)&outPtr2, (void *)&in0Ptr2, (void *)&in1Ptr2,
                   (void *)&length};

  auto execute2 = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args2);
  };

  // Execute with new buffers should match trace (pointer values excluded)
  EXPECT_NO_ERR(captured->debugVerifyReplay(execute2));
}

// Test: Mixed pointer and scalar parameters - changing scalar is detected.
TEST_F(GraphCaptureTest, DISABLED_TraceMixedPointerAndScalarParams) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is supported for CUDA and HIP";

  if (!supportsKernelReflection(*ctx))
    GTEST_SKIP()
        << "Scalar argument verification requires kernel reflection support";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc, loadVecAdd(*ctx));

  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(1024 * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(1024 * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(1024 * elemSize));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();

  // Capture with length=512
  uint64_t length1 = 512;
  auto execute1 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                    (void *)&length1};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/1, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, execute1);
  EXPECT_FALSE(captured.isError());

  // Replay with different scalar (length=1024) should mismatch
  uint64_t length2 = 1024;
  auto execute2 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                    (void *)&length2};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/1, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Changing the scalar parameter should trigger mismatch
  EXPECT_ERR(captured->debugVerifyReplay(execute2));
}

// Test: Sub-pointer-sized scalars ARE included in the trace hash.
// This verifies that the size-based detection correctly identifies and hashes
// parameters smaller than pointer size (e.g., uint32_t, float).
TEST_F(GraphCaptureTest, TraceDetectsSubPointerSizedScalarChanges) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is supported for CUDA and HIP";

  if (!supportsKernelReflection(*ctx))
    GTEST_SKIP()
        << "Scalar argument verification requires kernel reflection support";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  // Load kernel with mixed parameter sizes: (b64 pointer, u32, u32)
  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                ctx->loadFunction("small_scalar_module", "small_scalar_kernel",
                                  smallScalarPtx));

  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, buf,
                ctx->enqueueCreateBuffer(1024 * elemSize));

  void *bufPtr = buf->ptr();

  // Keep grid dimensions CONSTANT - we're only testing parameter value changes
  const uint32_t gridDim = 1;
  const uint32_t blockDim = 256;

  // Test 1: Capture with initial scalar values
  uint32_t scalar1 = 12345;
  uint32_t scalar2 = 67890;
  auto execute1 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&bufPtr, (void *)&scalar1, (void *)&scalar2};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc, gridDim, /*gridY=*/1, /*gridZ=*/1, blockDim,
        /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0, /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, execute1);
  EXPECT_FALSE(captured.isError());

  // Test 2: Change the first u32 scalar - should be DETECTED
  // Grid dimensions remain the same, ONLY the parameter value changes
  uint32_t scalar1Changed = 99999;
  auto execute2 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&bufPtr, (void *)&scalar1Changed, (void *)&scalar2};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc, gridDim, /*gridY=*/1, /*gridZ=*/1, blockDim,
        /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0, /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Should detect the change in the u32 parameter (4 bytes < 8 bytes)
  EXPECT_ERR(captured->debugVerifyReplay(execute2));

  // Test 3: Change the second u32 scalar - should also be DETECTED
  uint32_t scalar2Changed = 11111;
  auto execute3 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&bufPtr, (void *)&scalar1, (void *)&scalar2Changed};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc, gridDim, /*gridY=*/1, /*gridZ=*/1, blockDim,
        /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0, /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  EXPECT_ERR(captured->debugVerifyReplay(execute3));

  // Test 4: Replay with original values should succeed
  EXPECT_NO_ERR(captured->debugVerifyReplay(execute1));
}

// Test: Size-based scalar detection conservatively excludes pointer-sized
// scalars to avoid false positives. Demonstrates that uint64_t changes are
// NOT detected (trade-off for reliability).
TEST_F(GraphCaptureTest, TraceDistinguishesScalarsFromPointerBitPatterns) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is supported for CUDA and HIP";

  if (!supportsKernelReflection(*ctx))
    GTEST_SKIP()
        << "Scalar argument verification requires kernel reflection support";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  // Signature: (ptr, uint64_t, ptr, uint64_t)
  // All scalar parameters are 8 bytes (pointer-sized), so they are NOT hashed
  ASSIGN_NO_ERR(
      DeviceFunctionRef, deviceFunc,
      ctx->loadFunction("mixed_args_module", "mixed_args", mixedArgsPtx));

  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, buf1,
                ctx->enqueueCreateBuffer(1024 * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, buf2,
                ctx->enqueueCreateBuffer(1024 * elemSize));

  void *ptr1 = buf1->ptr();
  void *ptr2 = buf2->ptr();

  const uint32_t blockDim = 256;
  const uint32_t gridDim = 1;

  // Capture with initial scalar values
  uint64_t scalar1Value1 = 0x7fff00001000ULL;
  uint64_t scalar2Value1 = 0x7fff00002000ULL;
  auto execute1 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&ptr1, (void *)&scalar1Value1, (void *)&ptr2,
                    (void *)&scalar2Value1};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, execute1);
  EXPECT_FALSE(captured.isError());

  // Verify that changing pointer-sized (8-byte) scalar parameters is NOT
  // detected. This demonstrates the conservative approach: we exclude
  // pointer-sized parameters to avoid false positives from uint64_t values
  // that might accidentally look like valid CUDA memory addresses.
  uint64_t scalar1Value2 = scalar1Value1 + 0x1000;
  uint64_t scalar2Value2 = scalar2Value1 + 0x2000;
  auto execute2 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&ptr1, (void *)&scalar1Value2, (void *)&ptr2,
                    (void *)&scalar2Value2};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Should NOT detect the change in pointer-sized scalars (conservative)
  // This is the expected trade-off: we miss uint64_t changes but avoid false
  // positives from pointer-like bit patterns.
  EXPECT_NO_ERR(captured->debugVerifyReplay(execute2));

  // Replay with original scalar values should also succeed
  EXPECT_NO_ERR(captured->debugVerifyReplay(execute1));
}

// Test: Loop that captures graphs with unfreed allocations.
// This test is designed to detect memory leaks in the graph capture/replay
// system. It intentionally allocates memory during capture without freeing it,
// then replays the captured graph repeatedly.
TEST_F(GraphCaptureTest, LoopWithUnfreedAllocationNoReplay) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  const size_t allocSize = 100ULL * 1024 * 1024; // 100 MB per iteration
  const size_t maxIterations = 1024;

  for (auto _ : llvm::seq(maxIterations)) {
    DeviceBufferRef saved = {};

    // Capture a graph that allocates memory without freeing it
    ErrorOr<DeviceGraphCollection> captured =
        DeviceGraph::capture(*runtime_, {stream}, [&]() -> ErrorOrSuccess {
          // Allocate a buffer inside the capture
          DEFINE_OR_RETURN_ERROR(DeviceBufferRef, buf,
                                 ctx->enqueueCreateBuffer(allocSize));
          // Move the buffer to a location outside the capture.
          // This is equivalent to how outputs of a MO graph are handled, and
          // it will result in the buffer not being free'd within the captured
          // graph.
          saved = std::move(buf);

          return success();
        });

    EXPECT_FALSE(captured.isError());
    EXPECT_NO_ERR(ctx->synchronize());
  }
}

// Test: Loop that captures graphs with unfreed allocations.
// This test is designed to detect memory leaks in the graph capture/replay
// system. It intentionally allocates memory during capture without freeing it,
// then replays the captured graph repeatedly.
TEST_F(GraphCaptureTest, LoopWithUnfreedAllocationReplay) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  const size_t allocSize = 100ULL * 1024 * 1024; // 100 MB per iteration
  const size_t maxIterations = 1000;

  for (auto _ : llvm::seq(maxIterations)) {
    DeviceBufferRef saved = {};

    // Capture a graph that allocates memory without freeing it
    ErrorOr<DeviceGraphCollection> captured =
        DeviceGraph::capture(*runtime_, {stream}, [&]() -> ErrorOrSuccess {
          // Allocate a buffer inside the capture
          DEFINE_OR_RETURN_ERROR(DeviceBufferRef, buf,
                                 ctx->enqueueCreateBuffer(allocSize));
          // Move the buffer to a location outside the capture.
          // This is equivalent to how outputs of a MO graph are handled, and
          // it will result in the buffer not being free'd within the captured
          // graph.
          saved = std::move(buf);

          return success();
        });

    ASSERT_FALSE(captured.isError());

    // Replay the captured graph
    EXPECT_NO_ERR(captured->replay());
    EXPECT_NO_ERR(ctx->synchronize());
  }
}

// Test: Buffer allocated during graph capture comes from the graph free list,
// not the device stream, so no memory management node is encoded in the graph.
// We verify this by checking that the buffer pointer is stable across multiple
// replays (memory management nodes would reallocate on each replay).
TEST_F(GraphCaptureTest, GraphCaptureDoesNotEncodeAllocNodes) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  const size_t allocSize = 4 * 1024 * 1024; // 4 MB

  // Capture a graph that allocates a buffer inside the callback.
  DeviceBufferRef capturedBuf = {};
  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, [&]() -> ErrorOrSuccess {
        DEFINE_OR_RETURN_ERROR(DeviceBufferRef, buf,
                               ctx->enqueueCreateBuffer(allocSize));
        capturedBuf = std::move(buf);
        return success();
      });

  ASSERT_FALSE(captured.isError());
  ASSERT_TRUE(capturedBuf);

  // The buffer pointer must be non-null and the buffer must be valid.
  void *ptrAfterCapture = capturedBuf->ptr();
  EXPECT_NE(ptrAfterCapture, nullptr);
  EXPECT_EQ(capturedBuf->byteSize(), allocSize);

  // Replay the graph multiple times; the buffer pointer must remain stable.
  // If alloc nodes were encoded, CUDA would reallocate on each replay and
  // the pointer could change or the replay could fail.
  for (int i = 0; i < 3; ++i) {
    EXPECT_NO_ERR(captured->replay());
    EXPECT_NO_ERR(ctx->synchronize());
    // The captured buffer ref still points to the same memory.
    EXPECT_EQ(capturedBuf->ptr(), ptrAfterCapture);
  }
}

// Test: Event handlers enqueued during graph capture are processed after
// synchronize. This verifies that enqueueEventHandler() works correctly when
// called inside a graph capture: the event is recorded with
// recordedDuringCapture=true, so isReady() returns true immediately when
// polled, and the handler fires during synchronize().
TEST_F(GraphCaptureTest, EventHandlerCalledAfterCapture) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  // Use the context's default stream, which is also the stream used internally
  // by enqueueEventHandler. This ensures the event is recorded on the captured
  // stream and therefore has recordedDuringCapture=true.
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

  bool handlerCalled = false;

  // Perform graph capture. Inside the capture, enqueue an event handler that
  // records whether it was invoked.
  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, [&]() -> ErrorOrSuccess {
        return ctx->enqueueEventHandler(
            [&handlerCalled]() { handlerCalled = true; });
      });

  ASSERT_FALSE(captured.isError());

  // Handler is not called until ctx.pollPendingEvents is called.
  EXPECT_FALSE(handlerCalled);

  // Synchronize to process all pending event handlers.
  EXPECT_NO_ERR(ctx->synchronize());

  // The handler must have been called.
  EXPECT_TRUE(handlerCalled);
}

// A host-value wait enqueued inside a capture must be RECORDED into the
// graph, not executed eagerly against the flag's capture-time value. When a
// backend drops it, capture still succeeds and every replay runs ungated, so
// any in-graph consumer of host-produced data silently reads stale bytes
// instead of hanging or erroring.
TEST_F(GraphCaptureTest, WaitOnHostValueIsRecordedIntoCapturedGraph) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (ctx->device().api() != Device::cudaAPI &&
      ctx->device().api() != Device::hipAPI)
    GTEST_SKIP() << "wait-on-host-value is only implemented for CUDA and HIP";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());
  ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());

  // Pre-signal so a backend that (wrongly) evaluates the wait at capture time
  // finishes capture instead of hanging inside the capture callback.
  flag->signal(1);

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture(*runtime_, {stream}, [&]() -> ErrorOrSuccess {
        return stream->enqueueWaitOnHostValue(*flag, 1);
      });
  ASSERT_FALSE(captured.isError()) << std::string(captured.getError());

  // Clear the flag. A recorded wait node must now gate the replay until the
  // host signals again.
  flag->reset();
  ASSERT_EQ(flag->load(), 0u);

  std::atomic<bool> signalled{false};
  std::thread signaller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    signalled.store(true, std::memory_order_release);
    flag->signal(1);
  });

  EXPECT_NO_ERR(captured->replay());
  EXPECT_NO_ERR(stream->synchronize());

  EXPECT_TRUE(signalled.load(std::memory_order_acquire))
      << "replay drained without observing the host signal: the "
         "wait-on-host-value was not recorded as a graph node, so every "
         "replay of this graph runs ungated";
  signaller.join();
}

// Regression: a host-value wait captured into a graph must still gate that
// graph's downstream work on every replay.
//
// ROCm's stream capture does not record hipStreamWaitValue64. It reports
// success, adds no node, and executes the wait eagerly against the capturing
// stream instead, so the replayed graph ran its H2D with no gate and the
// consumer read stale host data (ENABLE-2950). Every other wait-on-host-value
// test in the tree enqueues the wait eagerly, where the primitive works fine,
// so none of them can see this. The graph here has the shape of the
// constrained-decoding bitmask gate: wait on the flag, then copy the buffer
// the host fills.
TEST_F(GraphCaptureTest, WaitOnHostValueGatesReplayedGraph) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (ctx->device().api() != Device::cudaAPI &&
      ctx->device().api() != Device::hipAPI)
    GTEST_SKIP() << "wait-on-host-value is a CUDA/HIP stream operation";

  // The H2D below is issued on the context's default stream, so the capture
  // has to run on that same stream for the copy to land in the graph.
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());
  ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());

  constexpr size_t numElems = 256;
  constexpr size_t byteSize = numElems * sizeof(uint32_t);
  constexpr uint32_t staleWord = 0xAAAAAAAAu;
  constexpr uint32_t freshWord = 0xBBBBBBBBu;

  ASSIGN_NO_ERR(DeviceBufferRef, pinnedSrc, ctx->createHostBuffer(byteSize));
  ASSIGN_NO_ERR(DeviceBufferRef, deviceDst, ctx->enqueueCreateBuffer(byteSize));
  auto *src = static_cast<uint32_t *>(pinnedSrc->ptr());

  auto stageStaleOnDevice = [&]() {
    std::fill_n(src, numElems, staleWord);
    EXPECT_NO_ERR(deviceDst->copyFrom(src));
    EXPECT_NO_ERR(stream->synchronize());
  };

  stageStaleOnDevice();

  flag->reset();
  EXPECT_NO_ERR(stream->beginCapture());
  EXPECT_NO_ERR(stream->enqueueWaitOnHostValue(*flag, 1));
  EXPECT_NO_ERR(deviceDst->copyFrom(src));
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());
  EXPECT_NO_ERR(graph->instantiate());

  // Retire anything the wait enqueued against the real queue while the stream
  // was capturing. Without this, such a leaked wait blocks the first replay and
  // the assertions below would pass for the wrong reason.
  flag->signal(1);
  EXPECT_NO_ERR(stream->synchronize());

  for (int iteration = 0; iteration < 2; ++iteration) {
    stageStaleOnDevice();
    flag->reset();

    std::atomic<bool> sourceRefreshed{false};
    std::thread producer([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      std::fill_n(src, numElems, freshWord);
      sourceRefreshed.store(true, std::memory_order_release);
      flag->signal(1);
    });

    EXPECT_NO_ERR(graph->replay());
    EXPECT_NO_ERR(stream->synchronize());
    // Sampled before the join, which would otherwise wait for the producer and
    // make this true no matter how the replay behaved.
    const bool refreshedBeforeReplayFinished =
        sourceRefreshed.load(std::memory_order_acquire);
    producer.join();

    EXPECT_TRUE(refreshedBeforeReplayFinished)
        << "iteration " << iteration
        << ": the replay completed before the host signalled the flag, so the "
           "captured wait did not gate it";

    std::vector<uint32_t> readback(numElems, 0);
    EXPECT_NO_ERR(deviceDst->copyTo(readback.data()));
    EXPECT_NO_ERR(stream->synchronize());
    EXPECT_EQ(readback.front(), freshWord)
        << "iteration " << iteration
        << ": the replayed graph copied stale host data, so the in-graph "
           "host-value wait did not gate the H2D";
    EXPECT_EQ(readback.back(), freshWord) << "iteration " << iteration;
  }
}

} // namespace

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

#include "HostMapping.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace M::Driver {

// Test fixture is needed to enable testing of private methods in DeviceContext.
class GPUDeviceContextTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // Print information about the device to make it easier to debug the
    // environment.
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      // Have the tests deal with device errors.
      return;

    std::cerr << "##### GPUDeviceContextTest running on "
              << Device::numberOfDevices(Device::gpuAPI) << " x "
              << (*device)->name() << "\n";
  }

  void SetUp() override {
    // No common setup needed.
  }
};

TEST_F(GPUDeviceContextTest, Smoke) {
  ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);

  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  std::string deviceName = ctx->device().name();
  if (ctx->device().api() == Device::cudaAPI) {
    EXPECT_NE(deviceName.find("NVIDIA"), std::string::npos) << deviceName;
    std::cerr << "Name of default CUDA device: " << deviceName << "\n";
  } else if (ctx->device().api() == Device::metalAPI) {
    EXPECT_NE(deviceName.find("Apple"), std::string::npos) << deviceName;
    std::cerr << "Name of default Metal device: " << deviceName << "\n";
  } else {
    EXPECT_NE(deviceName.find("AMD"), std::string::npos) << deviceName;
    std::cerr << "Name of default HIP device: " << deviceName << "\n";
  }
}

TEST_F(GPUDeviceContextTest, NumDevices) {
  int num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_GT(num, 0);
  EXPECT_LT(num, 16);
  std::cerr << "Number of GPU devices: " << num << "\n";

  // Create devices outside of allowed range.
  EXPECT_ERR_MSG(Device::create(Device::gpuAPI, -7), "must be positive");
  EXPECT_ERR_MSG(Device::create(Device::gpuAPI, num), "not available");
}

// Keyed off the model, not the api: an api-derived expectation would call every
// non-Apple part discrete, which Orin, GB10 and MI300A falsify.
static std::optional<bool> knownHostUnified(std::string_view name) {
  static constexpr std::pair<std::string_view, bool> kModels[] = {
      {"Apple M", true}, {"Orin", true},    {"GB10", true},
      {"MI300A", true},  {"GB200", false},  {"B200", false},
      {"H100", false},   {"MI300X", false}, {"MI355X", false},
  };
  // Longest match wins, so "GB200" is not shadowed by "B200".
  std::optional<bool> expected;
  size_t matched = 0;
  for (auto &[model, unified] : kModels)
    if (name.find(model) != std::string_view::npos && model.size() > matched) {
      expected = unified;
      matched = model.size();
    }
  return expected;
}

// `integrated` and `UNIFIED_ADDRESSING` differ only on discrete parts, so pin
// the value rather than smoke-test it.
TEST_F(GPUDeviceContextTest, IsHostUnifiedMatchesKnownTopology) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(bool, unified, ctx->device().isHostUnified());
  std::optional<bool> expected = knownHostUnified(ctx->device().name());
  ASSERT_TRUE(expected.has_value())
      << "unlisted device \"" << ctx->device().name()
      << "\": add it to kModels";
  EXPECT_EQ(unified, *expected);
}

TEST_F(GPUDeviceContextTest, ComputeCapability) {
  ASSIGN_NO_ERR(DeviceRef, dev, Device::create(Device::gpuAPI));

  ASSIGN_NO_ERR(int, cap, dev->computeCapability());
  if (dev->api() == Device::metalAPI) {
    // Metal returns 1,2,3,4 for compute capability (chip #)
    EXPECT_GE(cap, 0);
  } else {
    EXPECT_GE(cap, 75);
  }
  std::cerr << "Compute capability of GPU[0]: " << cap << "\n";
}

TEST_F(GPUDeviceContextTest, MAXCompatibility) {
  ASSIGN_NO_ERR(DeviceRef, dev, Device::create(Device::gpuAPI));
  EXPECT_NO_ERR(dev->isCompatible());
}

TEST_F(GPUDeviceContextTest, DeviceContextCreateCallsIsCompatible) {
  // DeviceContext::create() now calls isCompatible() before creating context.
  // On a compatible GPU, this should succeed.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  EXPECT_NE(ctx.getPointer(), nullptr);
}

TEST_F(GPUDeviceContextTest, DeviceContextGetOrCreateCallsIsCompatible) {
  // DeviceContext::getOrCreate() now also calls isCompatible().
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::getOrCreate(Device::gpuAPI));
  EXPECT_NE(ctx.getPointer(), nullptr);
}

TEST_F(GPUDeviceContextTest, DefaultContext) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx0, DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::getOrCreate(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, ctx2,
                DeviceContext::getOrCreate(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, ctx3, DeviceContext::create(Device::gpuAPI));
  EXPECT_NE(ctx0.getPointer(), ctx1.getPointer());
  EXPECT_NE(ctx0.getPointer(), ctx2.getPointer());
  EXPECT_NE(ctx0.getPointer(), ctx3.getPointer());
  EXPECT_EQ(ctx1.getPointer(), ctx2.getPointer());
  EXPECT_NE(ctx1.getPointer(), ctx3.getPointer());

  int num = Device::numberOfDevices(Device::gpuAPI);
  if (num > 1) {
    ASSIGN_NO_ERR(DeviceContextRef, ctx4,
                  DeviceContext::getOrCreate(Device::gpuAPI, 1));
    ASSIGN_NO_ERR(DeviceContextRef, ctx5,
                  DeviceContext::getOrCreate(Device::gpuAPI, 1));
    EXPECT_EQ(ctx4.getPointer(), ctx5.getPointer());
    EXPECT_NE(ctx1.getPointer(), ctx4.getPointer());
    EXPECT_NE(ctx0.getPointer(), ctx4.getPointer());
    EXPECT_NE(ctx3.getPointer(), ctx4.getPointer());
  }
}

TEST_F(GPUDeviceContextTest, DeviceContextScope) {

  // Create two different device contexts
  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::create(Device::gpuAPI, 0));

  // Get the initial driver context
  const Device &dev = ctx1->device();

  // Skip test for Metal
  if (dev.api() == Device::metalAPI) {
    GTEST_SKIP() << "Metal not supported";
  }

  ASSIGN_NO_ERR(void *, initialContext, dev.getCurrentDriverContext());

  {
    // Create a scope for the first context
    DeviceContextScope scope(*ctx1);
    EXPECT_FALSE(scope.err().isError()) << scope.err().getError();

    // Verify the context is now active
    ASSIGN_NO_ERR(void *, currentContext, dev.getCurrentDriverContext());
    EXPECT_EQ(currentContext, ctx1->driverContext());

    // Create a buffer to verify the context is working
    ASSIGN_NO_ERR(DeviceBufferRef, buf, ctx1->enqueueCreateBuffer(256));
    EXPECT_NO_ERR(ctx1->synchronize());
    EXPECT_NE(buf->ptr(), nullptr);

    // Create a nested scope with the same context (should be a no-op)
    {
      DeviceContextScope nestedScope(*ctx1);
      EXPECT_FALSE(nestedScope.err().isError());

      ASSIGN_NO_ERR(void *, nestedContext, dev.getCurrentDriverContext());
      EXPECT_EQ(nestedContext, currentContext);
    }

    // Verify the context is still active after nested scope ends
    ASSIGN_NO_ERR(void *, afterNestedContext, dev.getCurrentDriverContext());
    EXPECT_EQ(afterNestedContext, currentContext);
  }

  // After the scope ends, verify the context is restored to initial state
  ASSIGN_NO_ERR(void *, restoredContext, dev.getCurrentDriverContext());
  EXPECT_EQ(restoredContext, initialContext);

  // Test with multiple devices if available
  int numDevices = Device::numberOfDevices(Device::gpuAPI);
  if (numDevices > 1) {
    std::cerr << "Testing DeviceContextScope with multiple devices\n";

    ASSIGN_NO_ERR(DeviceContextRef, ctx2,
                  DeviceContext::create(Device::gpuAPI, 1));

    // Switch to the second context
    {
      DeviceContextScope scope2(*ctx2);
      EXPECT_FALSE(scope2.err().isError()) << scope2.err().getError();

      // The current context should be from the second device
      ASSIGN_NO_ERR(void *, ctx2Current, dev.getCurrentDriverContext());
      EXPECT_EQ(ctx2Current, ctx2->driverContext());

      // Create a buffer to verify the second context works
      ASSIGN_NO_ERR(DeviceBufferRef, buf2, ctx2->enqueueCreateBuffer(256));
      EXPECT_NO_ERR(ctx2->synchronize());
      EXPECT_NE(buf2->ptr(), nullptr);
    }

    // Verify it switches back to initial context
    ASSIGN_NO_ERR(void *, afterMultiContext, dev.getCurrentDriverContext());
    EXPECT_EQ(afterMultiContext, initialContext);
  }

  // Test error handling by creating an invalid device context scope
  // This part is tricky because we don't have a reliable way to create an
  // invalid context in a unit test, but we can check that the error propagation
  // works
  {
    DeviceContextScope invalidScope(*ctx1); // This should succeed
    EXPECT_FALSE(invalidScope.err().isError());

    // If we wanted to test failure, we'd need a way to create an invalid
    // context ErrorOrSuccess testErr = invalidScope.err();
    // EXPECT_TRUE(testErr.isError()); // This would check error propagation
  }
}

static void initWithData(void *dst, size_t dstLen, const void *checkData,
                         size_t checkDataLen) {
  char *cur = static_cast<char *>(dst);
  size_t remaining = dstLen;
  while (remaining > 0) {
    size_t toCopy = std::min(remaining, checkDataLen);
    memcpy(cur, checkData, toCopy);
    remaining -= toCopy;
    cur += toCopy;
  }
}

static void compareWithData(const void *dst, size_t dstLen,
                            const void *checkData, size_t checkDataLen) {
  const char *cur = static_cast<const char *>(dst);
  size_t remaining = dstLen;
  while (remaining > 0) {
    size_t toCheck = std::min(remaining, checkDataLen);
    EXPECT_EQ(memcmp(cur, checkData, toCheck), 0);
    remaining -= toCheck;
    cur += toCheck;
  }
}

TEST_F(GPUDeviceContextTest, DeviceBuffers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, 0));

  char checkData[] = {
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
      'E',
  };

  const int copyLengths[] = {0, 256, 1024 * 1024};
  const int numCopyLengths = 3;

  for (int i = 0; i < numCopyLengths; ++i) {
    int allocLen = copyLengths[i];

    ASSIGN_NO_ERR(DeviceBufferRef, emptyBuf, ctx->enqueueCreateBuffer(0));
    EXPECT_EQ(emptyBuf->ptr(), DeviceBuffer::emptyBufferPointer());
    EXPECT_EQ(emptyBuf->byteSize(), 0);

    std::cerr << "DeviceContext copies(len: " << allocLen << ")\n";

    ASSIGN_NO_ERR(DeviceBufferRef, aBuf, ctx->createHostBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, cBuf, ctx->enqueueCreateBuffer(allocLen));

    char *dPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(dPtr, nullptr);

    // Initialize with known good and bad values.
    initWithData(aBuf->ptr(), allocLen, checkData, sizeof(checkData));

    for (int i = 0; i < allocLen; ++i) {
      dPtr[i] = -static_cast<char>((7 + i) & 127);
    }

    // Copy between different buffers.
    EXPECT_NO_ERR(aBuf->copyTo(*bBuf));
    EXPECT_NO_ERR(bBuf->copyTo(*cBuf));
    EXPECT_NO_ERR(cBuf->copyTo(dPtr));

    // Ensure that all copy operations have finished.
    EXPECT_NO_ERR(ctx->synchronize());
    compareWithData(dPtr, allocLen, checkData, sizeof(checkData));

    free(dPtr);
  }
}

TEST_F(GPUDeviceContextTest, FailDeviceBuffers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  // FIXME(GEX-3391): Metal returns a different error message for freeing
  // untracked allocations, and cross-context copies produce wrong results.
  if (ctx->device().api() == Device::metalAPI) {
    GTEST_SKIP() << "Known Metal issue (GEX-3391)";
  }

  int allocLen = 256;

  std::cerr << "DeviceContext failing copies(len: " << allocLen << ")\n";

  char *aPtr = static_cast<char *>(malloc(allocLen));
  EXPECT_NE(aPtr, nullptr);

  // Create a source buffer that is too small.
  ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(allocLen - 32));
  ASSIGN_NO_ERR(DeviceBufferRef, cBuf, ctx->enqueueCreateBuffer(allocLen));

  // Initialize with known good and bad values.
  for (int i = 0; i < allocLen; ++i) {
    aPtr[i] = static_cast<char>(i & 127);
  }

  // Copy between different buffers.
  EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*bBuf, aPtr));
  EXPECT_ERR_MSG(ctx->enqueueCopyDeviceToDevice(*cBuf, *bBuf),
                 "not enough data in src");

  // Ensure that all copy operations have finished.
  EXPECT_NO_ERR(ctx->synchronize());

  // Create an *owned* DeviceBuffer which will be invalid when destructed.
  {
    DeviceBufferRef bad = ctx->createBufferOwning(aPtr, allocLen,
                                                  /* ownning = */ true);
  }
  // The bad-buffer free surfaces an error differently per backend:
  //   CUDA: cuMemFree_v2 is deferred via event handler; the handler fires
  //         during the first synchronize's event poll and sets a pending
  //         error; the second synchronize returns that error.
  //   HIP:  hipFree is called immediately and sets a pending error; the
  //         first synchronize returns it.
  // Call synchronize twice and assert that exactly one call returns an error
  // containing "failed".
  auto err1 = ctx->synchronize();
  auto err2 = ctx->synchronize();
  if (err1.isError()) {
    EXPECT_THAT(err1.getError(), testing::HasSubstr("failed"));
    EXPECT_FALSE(err2.isError()) << err2.getError();
  } else {
    EXPECT_TRUE(err2.isError());
    EXPECT_THAT(err2.getError(), testing::HasSubstr("failed"));
  }

  free(aPtr);
}

TEST_F(GPUDeviceContextTest, SubBuffer) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  int allocLen = 256;

  int subOffset = 0;
  int subAllocLen = 3;

  std::cerr << "DeviceContext sub buffer copies(len: " << allocLen << ")\n";

  char *aPtr = static_cast<char *>(malloc(allocLen));
  EXPECT_NE(aPtr, nullptr);

  ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, cBuf,
                bBuf->createSubBuffer(subOffset, subAllocLen * 4));

  char *dPtr = static_cast<char *>(malloc(allocLen));
  EXPECT_NE(dPtr, nullptr);

  float *ePtr = static_cast<float *>(malloc(subAllocLen * sizeof(float)));
  EXPECT_NE(ePtr, nullptr);

  // Initialize with known good and bad values.
  for (int i = 0; i < allocLen; ++i) {
    aPtr[i] = static_cast<char>(i & 127);
    dPtr[i] = -static_cast<char>((7 + i) & 127);
  }

  for (int i = 0; i < subAllocLen; ++i) {
    ePtr[i] = static_cast<float>(i) * (((i & 1) == 0) ? -1.0 : 1.0);
  }

  // Copy base pattern into base buffer.
  EXPECT_NO_ERR(bBuf->copyFrom(aPtr));
  // Copy overlay pattern into sub buffer.
  EXPECT_NO_ERR(cBuf->copyFrom(ePtr));
  // Copy base buffer back to host memory.
  EXPECT_NO_ERR(bBuf->copyTo(dPtr));

  EXPECT_NO_ERR(ctx->synchronize());
  uint32_t *dIntPtr = reinterpret_cast<uint32_t *>(dPtr);
  uint32_t expectedVals[] = {0x80000000, 0x3f800000, 0xc0000000};
  for (int i = 0; i < (subAllocLen); ++i) {
    uint32_t val = dIntPtr[i];
    std::cerr << "SubBuffer: d[" << i << "] = 0x" << std::hex << val << std::dec
              << "\n";
    EXPECT_EQ(val, expectedVals[i]);
  }
  for (int i = subAllocLen * 4; i < allocLen; ++i) {
    int val = dPtr[i];
    if (i < 20) {
      std::cerr << "Original: d[" << std::dec << i << "] = 0x" << std::hex
                << val << std::dec << "\n";
    }
    EXPECT_EQ(val, i & 127);
  }
  free(ePtr);
  free(dPtr);
  free(aPtr);
}
TEST_F(GPUDeviceContextTest, OverlappingCopies) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  int allocLen =
      127 * 1024 * 1024; // 127MB to fit into host memory manager chunks.
  ASSIGN_NO_ERR(DeviceBufferRef, aBuf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, refBuf, ctx->createHostBuffer(allocLen));

  void *aHostPtr = nullptr;
  {
    ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx->createHostBuffer(allocLen));
    aHostPtr = hostBuf->ptr();
    memset(aHostPtr, 12, allocLen);
    EXPECT_NO_ERR(hostBuf->copyTo(*aBuf));
    // hostBuf is released here, but should not be reallocated from the host
    // memory manager until the copy above has finished.
  }

  // Allocate another host buffer of the same size and overwrite the last byte.
  // If the above hostBuf was released too early this write will race with the
  // copy.
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx->createHostBuffer(allocLen));
  char *bHostPtr = reinterpret_cast<char *>(hostBuf->ptr());
  bHostPtr[allocLen - 1] = 15;
  EXPECT_NO_ERR(hostBuf->copyTo(*bBuf));

  // Verify that the original hostBuf was not overwritten.
  EXPECT_NO_ERR(ctx->synchronize());
  EXPECT_NO_ERR(aBuf->copyTo(*refBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  char *refPtr = reinterpret_cast<char *>(refBuf->ptr());
  for (int i = 0; i < allocLen; ++i) {
    EXPECT_EQ(refPtr[i], 12) << "failure at index: " << i << "\n";
  }
}

// Shared body for the multi-stream pinned-free regressions: a host/pinned
// buffer used as an async-copy staging buffer on a stream other than its owning
// context must not have its backing memory freed (and recycled) until that
// other stream's copy completes.
//
// Two contexts on the (default) device give two distinct streams. The pinned
// buffer is owned by `ownCtx` but copied on `copyCtx`; the copy therefore runs
// on a stream `ownCtx` never orders against in the buggy implementation. This
// mirrors OverlappingCopies' recycle-detection trick: if the free lands early,
// the same-size reallocation reuses the still-in-flight memory and clobbering
// it corrupts the copy's result.
//
// When `viaSubBuffer` is set the copy source is a sub-buffer aliasing the whole
// pinned block. The sub-buffer's destructor performs no free (the root owns the
// memory), so the used-context recording must be forwarded to the root for the
// free to stay correctly ordered.
static void runPinnedMultiStreamFree(bool viaSubBuffer) {
  ASSIGN_NO_ERR(DeviceContextRef, ownCtx,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, copyCtx,
                DeviceContext::create(Device::gpuAPI));
  ASSERT_NE(ownCtx.getPointer(), copyCtx.getPointer());

  // Large enough that the H2D copy stays in flight across the free below.
  const int allocLen = 127 * 1024 * 1024; // 127MB, fits host manager chunks.

  // Device buffer on copyCtx's stream that receives the pinned-buffer copy, and
  // a host buffer to read it back for verification.
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf,
                copyCtx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, refBuf, ownCtx->createHostBuffer(allocLen));

  {
    // Pinned/host staging buffer OWNED by ownCtx, but copied on copyCtx.
    ASSIGN_NO_ERR(DeviceBufferRef, pinned, ownCtx->createHostBuffer(allocLen));
    memset(pinned->ptr(), 12, allocLen);
    // copyTo routes through copyCtx (dst.context()), so the copy runs on
    // copyCtx's stream and records copyCtx as a used context on the buffer that
    // owns the memory -- `pinned` directly, or its root via a sub-buffer.
    if (viaSubBuffer) {
      ASSIGN_NO_ERR(DeviceBufferRef, sub, pinned->createSubBuffer(0, allocLen));
      EXPECT_NO_ERR(sub->copyTo(*devBuf));
    } else {
      EXPECT_NO_ERR(pinned->copyTo(*devBuf));
    }
    // `pinned` (and any sub-buffer) released here: its host free must be
    // ordered against copyCtx's in-flight copy, not just ownCtx's idle stream.
  }

  // Drain ownCtx's ready event handlers. In the buggy (owning-stream-only)
  // implementation this frees the pinned memory now -- ownCtx's stream is idle
  // -- even though copyCtx's copy is still running.
  EXPECT_NO_ERR(ownCtx->pollPendingEvents());

  // Reallocate the same size from ownCtx's host free list and clobber only the
  // final byte. If the pinned memory was recycled early (the bug), this single
  // instant write lands on the last byte the in-flight H2D copy reads, so it
  // reliably wins the race and corrupts devBuf (writing the whole buffer would
  // lose to the much faster DMA read; cf. OverlappingCopies).
  ASSIGN_NO_ERR(DeviceBufferRef, clobber, ownCtx->createHostBuffer(allocLen));
  reinterpret_cast<char *>(clobber->ptr())[allocLen - 1] = 99;

  // Let the copy finish and read devBuf back; the loop below verifies it kept
  // the original data.
  EXPECT_NO_ERR(copyCtx->synchronize());
  EXPECT_NO_ERR(devBuf->copyTo(*refBuf));
  EXPECT_NO_ERR(copyCtx->synchronize());
  EXPECT_NO_ERR(ownCtx->synchronize());

  const char *refPtr = reinterpret_cast<const char *>(refBuf->ptr());
  for (int i = 0; i < allocLen; ++i) {
    ASSERT_EQ(refPtr[i], 12)
        << "corruption at index " << i
        << ": pinned buffer freed before copyCtx's copy completed";
  }
}

// Mirror of `runPinnedMultiStreamFree` with the copy direction flipped: the
// host buffer is the DESTINATION, so the recording goes through
// `enqueueCopyDeviceToDevice`'s `dst.isHost()` branch, which the source-side
// twin never reaches.
static void runPinnedMultiStreamFreeAsDestination(bool viaSubBuffer) {
  ASSIGN_NO_ERR(DeviceContextRef, ownCtx,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, copyCtx,
                DeviceContext::create(Device::gpuAPI));
  ASSERT_NE(ownCtx.getPointer(), copyCtx.getPointer());

  // Large enough that the D2H copy stays in flight across the free below.
  const int allocLen = 127 * 1024 * 1024; // 127MB, fits host manager chunks.

  // Source on copyCtx's stream, primed with a known byte.
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf,
                copyCtx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, seed, copyCtx->createHostBuffer(allocLen));
  memset(seed->ptr(), 12, allocLen);
  EXPECT_NO_ERR(seed->copyTo(*devBuf));
  EXPECT_NO_ERR(copyCtx->synchronize());

  {
    // Pad OWNED by ownCtx, written by a copy that runs on copyCtx's stream.
    ASSIGN_NO_ERR(DeviceBufferRef, pad, ownCtx->createHostBuffer(allocLen));
    if (viaSubBuffer) {
      ASSIGN_NO_ERR(DeviceBufferRef, sub, pad->createSubBuffer(0, allocLen));
      EXPECT_NO_ERR(devBuf->copyTo(*sub, 0, allocLen));
    } else {
      EXPECT_NO_ERR(devBuf->copyTo(*pad, 0, allocLen));
    }
    // `pad` released here with the copy still in flight on copyCtx.
  }

  EXPECT_NO_ERR(ownCtx->pollPendingEvents());

  // Recycle the same size from ownCtx's host free list and mark only the final
  // byte. If the pad was freed early, this allocation IS the pad's memory and
  // the still-running copy writes 12 over the mark; marking one byte at the end
  // loses to no part of the DMA, which writes it last (cf. the src-side twin).
  ASSIGN_NO_ERR(DeviceBufferRef, recycled, ownCtx->createHostBuffer(allocLen));
  reinterpret_cast<char *>(recycled->ptr())[allocLen - 1] = 99;

  EXPECT_NO_ERR(copyCtx->synchronize());
  EXPECT_NO_ERR(ownCtx->synchronize());

  EXPECT_EQ(reinterpret_cast<const char *>(recycled->ptr())[allocLen - 1], 99)
      << "pad recycled before copyCtx's device-to-host copy completed";
}

// Regression: host buffer written by a foreign stream, freed while that copy is
// in flight.
TEST_F(GPUDeviceContextTest, PinnedDestinationMultiStreamFree) {
  runPinnedMultiStreamFreeAsDestination(/*viaSubBuffer=*/false);
}

// Regression: the same through a sub-buffer, the shape the Mach shim uses --
// the recording must forward to the root pad before the sub-buffer dies.
TEST_F(GPUDeviceContextTest, PinnedDestinationSubBufferMultiStreamFree) {
  runPinnedMultiStreamFreeAsDestination(/*viaSubBuffer=*/true);
}

// Regression: pinned buffer copied directly on a foreign stream.
TEST_F(GPUDeviceContextTest, PinnedBufferMultiStreamFree) {
  runPinnedMultiStreamFree(/*viaSubBuffer=*/false);
}

// Regression: pinned buffer copied on a foreign stream through a sub-buffer,
// whose used-context recording must forward to the root owner.
TEST_F(GPUDeviceContextTest, PinnedSubBufferMultiStreamFree) {
  runPinnedMultiStreamFree(/*viaSubBuffer=*/true);
}

TEST_F(GPUDeviceContextTest, WrapHostMemoryRoundTrip) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  const size_t byteSize = hostPageSize();
  HostMapping mapping(2 * byteSize);
  ASSERT_NE(mapping.data(), nullptr);

  const size_t count = byteSize / sizeof(uint32_t);
  uint32_t *src = reinterpret_cast<uint32_t *>(mapping.bytes());
  uint32_t *dst = reinterpret_cast<uint32_t *>(mapping.bytes() + byteSize);
  for (size_t i = 0; i < count; ++i)
    src[i] = static_cast<uint32_t>(i * 11 + 5);
  std::memset(dst, 0, byteSize);

  {
    ASSIGN_NO_ERR(DeviceBufferRef, wrappedSrc,
                  ctx->wrapHostMemory(src, byteSize));
    ASSIGN_NO_ERR(DeviceBufferRef, wrappedDst,
                  ctx->wrapHostMemory(dst, byteSize));
    EXPECT_EQ(wrappedSrc->byteSize(), byteSize);
    EXPECT_TRUE(wrappedSrc->isHost());
    // Only CUDA maps a registered range back at the host address.
    if (ctx->device().api() == Device::cudaAPI)
      EXPECT_EQ(wrappedSrc->ptr(), src);
    else
      EXPECT_NE(wrappedSrc->ptr(), src);

    ASSIGN_NO_ERR(DeviceBufferRef, deviceBuf,
                  ctx->enqueueCreateBuffer(byteSize));
    EXPECT_NO_ERR(wrappedSrc->copyTo(*deviceBuf));
    EXPECT_NO_ERR(deviceBuf->copyTo(*wrappedDst));
    EXPECT_NO_ERR(ctx->synchronize());

    for (size_t i = 0; i < count; ++i)
      ASSERT_EQ(dst[i], src[i]) << "mismatch at index " << i;
  }

  // Handlers are polled rather than pushed, so the release needs both calls.
  EXPECT_NO_ERR(ctx->synchronize());
  EXPECT_NO_ERR(ctx->pollPendingEvents());

  // Re-wrapping proves the drop unregistered: registering a still-registered
  // range fails.
  EXPECT_NO_ERR(ctx->wrapHostMemory(src, byteSize));
  EXPECT_NO_ERR(ctx->synchronize());
  EXPECT_NO_ERR(ctx->pollPendingEvents());

  // The pages survived, and are still the test's to unmap.
  std::memset(mapping.data(), 0x5A, mapping.byteSize());
  EXPECT_TRUE(std::all_of(mapping.bytes(), mapping.bytes() + mapping.byteSize(),
                          [](uint8_t b) { return b == 0x5A; }));
}

// MAX's host KV tier shape: one mapping, wrapped in independent chunks.
TEST_F(GPUDeviceContextTest, WrapHostMemoryChunksOfOneMapping) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  constexpr size_t kChunks = 4;
  const size_t chunkBytes = 2 * hostPageSize();
  HostMapping mapping(kChunks * chunkBytes);
  ASSERT_NE(mapping.data(), nullptr);

  std::vector<DeviceBufferRef> handles;
  for (size_t i = 0; i < kChunks; ++i) {
    ASSIGN_NO_ERR(
        DeviceBufferRef, chunk,
        ctx->wrapHostMemory(mapping.bytes() + i * chunkBytes, chunkBytes));
    handles.push_back(std::move(chunk));
  }

  // A copy straddling two registered chunks is what fails, so stay inside one.
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuf,
                ctx->enqueueCreateBuffer(chunkBytes));
  for (DeviceBufferRef &chunk : handles)
    EXPECT_NO_ERR(chunk->copyTo(*deviceBuf));
  EXPECT_NO_ERR(ctx->synchronize());

  handles.erase(handles.begin() + 1); // middle, then last, then the rest
  handles.pop_back();
  handles.clear();
  EXPECT_NO_ERR(ctx->synchronize());
  EXPECT_NO_ERR(ctx->pollPendingEvents());
}

// The multi-GPU KV-offload shape: the unregister runs from the last
// participating handler, so every context that touched the range must drain.
TEST_F(GPUDeviceContextTest, WrapHostMemoryReleaseNeedsEveryTouchingContext) {
  ASSIGN_NO_ERR(DeviceContextRef, owner, DeviceContext::create(Device::gpuAPI));
  // Only CUDA rejects re-registering a live range, so only there can a re-wrap
  // observe that the release has not run. HIP accepts it (measured on MI355)
  // and Metal mints a fresh buffer each time; MetalDeviceContextTest covers the
  // same claim through the buffer tracker.
  if (owner->device().api() != Device::cudaAPI)
    GTEST_SKIP() << "re-wrap rejection is CUDA-only; see comment";
  ASSIGN_NO_ERR(DeviceContextRef, foreign,
                DeviceContext::create(Device::gpuAPI));

  const size_t byteSize = hostPageSize();
  HostMapping mapping(byteSize);
  ASSERT_NE(mapping.data(), nullptr);

  {
    ASSIGN_NO_ERR(DeviceBufferRef, wrapped,
                  owner->wrapHostMemory(mapping.data(), byteSize));
    ASSIGN_NO_ERR(DeviceBufferRef, deviceBuf,
                  foreign->enqueueCreateBuffer(byteSize));
    EXPECT_NO_ERR(wrapped->copyTo(*deviceBuf));
    EXPECT_NO_ERR(foreign->synchronize());
  }

  // `foreign`'s handler still holds a reference, so the range stays registered.
  EXPECT_NO_ERR(owner->synchronize());
  EXPECT_NO_ERR(owner->pollPendingEvents());
  EXPECT_ERR(owner->wrapHostMemory(mapping.data(), byteSize));

  EXPECT_NO_ERR(foreign->synchronize());
  EXPECT_NO_ERR(foreign->pollPendingEvents());
  EXPECT_NO_ERR(owner->wrapHostMemory(mapping.data(), byteSize));

  // Leave the range unregistered before ~HostMapping unmaps it.
  EXPECT_NO_ERR(owner->synchronize());
  EXPECT_NO_ERR(owner->pollPendingEvents());
}

TEST_F(GPUDeviceContextTest, WrapHostMemoryRejectsBadRanges) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  HostMapping mapping(hostPageSize());
  ASSERT_NE(mapping.data(), nullptr);

  EXPECT_ERR_MSG(ctx->wrapHostMemory(nullptr, hostPageSize()),
                 "pointer is null");
  EXPECT_ERR_MSG(ctx->wrapHostMemory(mapping.data(), 0), "byteSize is zero");
}

TEST_F(GPUDeviceContextTest, RecycleBuffers) {
  ASSIGN_NO_ERR(DeviceContextRef, allocCtx,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, freeCtx,
                DeviceContext::create(Device::gpuAPI));

  constexpr size_t kb = 1024;
  constexpr size_t mb = 1024 * kb;
  constexpr size_t sizes[8] = {3 * kb,  2 * kb,   1 * mb,  511 * kb,
                               32 * kb, 128 * mb, 32 * mb, 78 * mb};
  // When run with MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY=true this will
  // move all free buffers from allocCtx to freeCtx unless we regularly flush
  // buffers from freeCtx to the parent.
  for (int i = 0; i < 1000; ++i) {
    size_t allocSize = sizes[i & 7];
    ASSIGN_NO_ERR(DeviceBufferRef, buf,
                  allocCtx->enqueueCreateBuffer(allocSize));
    ASSIGN_NO_ERR(DeviceEventRef, allocated, allocCtx->enqueueEvent());
    EXPECT_NO_ERR(freeCtx->enqueueWaitForEvent(*allocated));
    buf->setContext(*freeCtx);
    // this synchronize guarantees that the flush handler is called
    EXPECT_NO_ERR(freeCtx->synchronize());
    // buf goes out of scope here and is freed by freeCtx.
  }
}

TEST_F(GPUDeviceContextTest, SetMemory) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  int allocLen = 256;
  ASSIGN_NO_ERR(DeviceBufferRef, buf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx->createHostBuffer(allocLen));

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 123, 1));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint8_t *ptr = static_cast<uint8_t *>(hostBuf->ptr());
    for (int i = 0; i < allocLen; ++i) {
      EXPECT_EQ(ptr[i], 123);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 12345, 2));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint16_t *ptr = static_cast<uint16_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 2); ++i) {
      EXPECT_EQ(ptr[i], 12345);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 1234567890, 4));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint32_t *ptr = static_cast<uint32_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 4); ++i) {
      EXPECT_EQ(ptr[i], 1234567890);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 0x0101010101010101ul, 8));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint64_t *ptr = static_cast<uint64_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 8); ++i) {
      EXPECT_EQ(ptr[i], 0x0101010101010101ul);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 1234567890123456789, 8));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint64_t *ptr = static_cast<uint64_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 8); ++i) {
      EXPECT_EQ(ptr[i], 1234567890123456789);
    }
  }
}

TEST_F(GPUDeviceContextTest, Timers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  ASSIGN_NO_ERR(DeviceTimerRef, timer, ctx->startTimer());
  { // Run some memory copies within the timer.
    const size_t allocLen = 1024 * 1024;

    ASSIGN_NO_ERR(DeviceBufferRef, aBuf, ctx->createHostBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(256));

    for (int i = 0; i < 20; ++i) {
      EXPECT_NO_ERR(aBuf->copyTo(*bBuf));
    }
    EXPECT_NO_ERR(ctx->synchronize());
  }
  ASSIGN_NO_ERR(std::chrono::nanoseconds, nanos, ctx->stopTimer(*timer));
  int64_t elapsedNanos = nanos.count();

  std::cerr << "Timer measured: " << elapsedNanos << "ns\n";

  EXPECT_GT(elapsedNanos, 0);
  EXPECT_LT(elapsedNanos, 60 * 1000ll * 1000ll * 1000ll); // less than a minute
}

TEST_F(GPUDeviceContextTest, MultiContextCopies) {
  // FIXME(GEX-3391): Cross-context copies produce wrong results on Metal.
  ASSIGN_NO_ERR(DeviceContextRef, probe, DeviceContext::create(Device::gpuAPI));
  if (probe->device().api() == Device::metalAPI)
    GTEST_SKIP() << "Known Metal issue (GEX-3391)";

  int numDevices = Device::numberOfDevices(Device::gpuAPI);
  std::cerr << "Number of GPU devices: " << numDevices << "\n";

  // Allocate a vector filled with all available devices prefixed with CPU and
  // GPU[0].
  std::vector<DeviceContextRef> contexts;
  for (int i = 0; i < numDevices + 2; ++i) {
    if (i == 0) {
      ASSIGN_NO_ERR(DeviceContextRef, ctx,
                    DeviceContext::create(Device::cpuAPI, 0));
      contexts.emplace_back(std::move(ctx));
    } else if (i == 1) {
      ASSIGN_NO_ERR(DeviceContextRef, ctx,
                    DeviceContext::create(Device::gpuAPI, 0));
      contexts.emplace_back(std::move(ctx));
    } else {
      ASSIGN_NO_ERR(DeviceContextRef, ctx,
                    DeviceContext::create(Device::gpuAPI, i - 2));
      contexts.emplace_back(std::move(ctx));
    }
  }

  for (int srcDeviceId = 0; srcDeviceId < numDevices + 2; ++srcDeviceId) {
    DeviceContext *srcCtx = contexts[srcDeviceId].getPointer();
    for (int dstDeviceId = 0; dstDeviceId < numDevices + 2; ++dstDeviceId) {
      DeviceContext *dstCtx = contexts[dstDeviceId].getPointer();

      std::cerr << "Copying from contexts[" << srcDeviceId << "] to contexts["
                << dstDeviceId << "].\n";

      const int allocLen = 256;
      char *checkData = new char[allocLen];
      // Initialize checkData with a pattern specific for this run.
      for (int i = 0; i < allocLen; ++i) {
        checkData[i] = (srcDeviceId * 8) + dstDeviceId + i;
      }

      ASSIGN_NO_ERR(DeviceBufferRef, srcHostBuf,
                    srcCtx->createHostBuffer(allocLen));
      ASSIGN_NO_ERR(DeviceBufferRef, srcDevBuf,
                    srcCtx->enqueueCreateBuffer(allocLen));
      ASSIGN_NO_ERR(DeviceBufferRef, dstDevBuf,
                    dstCtx->enqueueCreateBuffer(allocLen));
      ASSIGN_NO_ERR(DeviceBufferRef, dstHostBuf,
                    dstCtx->createHostBuffer(allocLen));

      char *dstHostPtr = static_cast<char *>(dstHostBuf->ptr());
      char *dstPtr = static_cast<char *>(malloc(allocLen));
      EXPECT_NE(dstPtr, nullptr);
      // Initialize with known bad values.
      for (int i = 0; i < allocLen; ++i) {
        // Use a different factor for i than the checkData, so that we can't
        // ever overlap.
        dstPtr[i] = -static_cast<char>((7 + 2 * i) & 127);
        dstHostPtr[i] = -static_cast<char>((7 + 2 * i) & 127);
      }

      // Copy between different buffers.
      memcpy(srcHostBuf->ptr(), checkData, allocLen);
      EXPECT_NO_ERR(srcHostBuf->copyTo(*srcDevBuf));
      EXPECT_NO_ERR(srcDevBuf->copyTo(*dstDevBuf));
      EXPECT_NO_ERR(dstDevBuf->copyTo(dstPtr));
      EXPECT_NO_ERR(srcHostBuf->copyTo(*dstHostBuf));

      // Ensure that all copy operations have finished.
      EXPECT_NO_ERR(dstCtx->synchronize());
      EXPECT_NO_ERR(srcCtx->synchronize());

      EXPECT_EQ(memcmp(dstPtr, checkData, allocLen), 0);
      EXPECT_EQ(memcmp(dstHostPtr, checkData, allocLen), 0);

      free(dstPtr);
      delete[] checkData;
    }
  }
}

TEST_F(GPUDeviceContextTest, CrossContextEvent) {
  ASSIGN_NO_ERR(DeviceContextRef, srcCtx,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, dstCtx,
                DeviceContext::create(Device::gpuAPI));

  const int allocLen = 12 * 1024 * 1024;
  char *checkData = new char[allocLen];
  // Initialize checkData with a pattern specific for this run.
  for (int i = 0; i < allocLen; ++i) {
    checkData[i] = i;
  }

  ASSIGN_NO_ERR(DeviceBufferRef, srcHostBuf,
                srcCtx->createHostBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, dstHostBuf,
                dstCtx->createHostBuffer(allocLen));

  memcpy(srcHostBuf->ptr(), checkData, allocLen);

  ASSIGN_NO_ERR(DeviceBufferRef, srcDevBuf,
                srcCtx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, dstDevBuf,
                dstCtx->enqueueCreateBuffer(allocLen));

  EXPECT_NO_ERR(srcHostBuf->copyTo(*srcDevBuf));

  ASSIGN_NO_ERR(DeviceEventRef, copyDoneEvent, srcCtx->enqueueEvent());

  EXPECT_NO_ERR(dstCtx->enqueueWaitFor(*copyDoneEvent));
  EXPECT_NO_ERR(srcDevBuf->copyTo(*dstDevBuf));
  EXPECT_NO_ERR(dstDevBuf->copyTo(*dstHostBuf));

  // An event based sequence equivalent to: err = dstCtx->synchronize();
  ASSIGN_NO_ERR(DeviceEventRef, syncEvent, dstCtx->enqueueEvent());
  EXPECT_NO_ERR(syncEvent->synchronize());

  EXPECT_EQ(memcmp(dstHostBuf->ptr(), checkData, allocLen), 0);
  delete[] checkData;
}

TEST_F(GPUDeviceContextTest, CrossContextStreamEvent) {
  ASSIGN_NO_ERR(DeviceContextRef, srcCtx,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, dstCtx,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceStreamRef, srcStream, srcCtx->stream());
  ASSIGN_NO_ERR(DeviceStreamRef, dstStream, dstCtx->stream());

  EXPECT_TRUE(*srcCtx == *srcCtx);
  EXPECT_FALSE(*srcCtx == *dstCtx);

  const int allocLen = 12 * 1024 * 1024;
  std::vector<char> checkData(allocLen);
  for (int i = 0; i < allocLen; ++i)
    checkData[i] = static_cast<char>(i * 7);

  ASSIGN_NO_ERR(DeviceBufferRef, srcHostBuf,
                srcCtx->createHostBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, dstHostBuf,
                dstCtx->createHostBuffer(allocLen));
  memcpy(srcHostBuf->ptr(), checkData.data(), allocLen);

  ASSIGN_NO_ERR(DeviceBufferRef, srcDevBuf,
                srcCtx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, dstDevBuf,
                dstCtx->enqueueCreateBuffer(allocLen));

  EXPECT_NO_ERR(srcHostBuf->copyTo(*srcDevBuf));
  ASSIGN_NO_ERR(DeviceEventRef, copyDoneEvent, srcCtx->eventCreate(false));
  EXPECT_NO_ERR(srcStream->eventRecord(*copyDoneEvent));

  EXPECT_NO_ERR(dstStream->enqueueWaitForEvent(*copyDoneEvent));
  EXPECT_NO_ERR(srcDevBuf->copyTo(*dstDevBuf));
  EXPECT_NO_ERR(dstDevBuf->copyTo(*dstHostBuf));
  EXPECT_NO_ERR(dstStream->synchronize());

  EXPECT_EQ(memcmp(dstHostBuf->ptr(), checkData.data(), allocLen), 0);
}

TEST_F(GPUDeviceContextTest, CreatedStreamEventOrdering) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  unsigned baseStreams = ctx->numStreams();
  ASSIGN_NO_ERR(DeviceStreamRef, sideStream, ctx->createStream());
  EXPECT_EQ(ctx->numStreams(), baseStreams + 1);
  ASSIGN_NO_ERR(DeviceContextRef, view, ctx->selectStream(baseStreams));
  ASSIGN_NO_ERR(DeviceContextRef, baseView, ctx->selectStream(0));
  EXPECT_EQ(baseView.getPointer(), ctx.getPointer());
  EXPECT_FALSE(*view == *ctx);
  ASSIGN_NO_ERR(DeviceStreamRef, viewStream, view->stream());
  EXPECT_EQ(viewStream->ptr(), sideStream->ptr());
  EXPECT_TRUE(ctx->selectStream(baseStreams + 1).isError());

  const int allocLen = 12 * 1024 * 1024;
  std::vector<char> checkData(allocLen);
  for (int i = 0; i < allocLen; ++i)
    checkData[i] = static_cast<char>(i * 13);

  ASSIGN_NO_ERR(DeviceBufferRef, srcHostBuf, ctx->createHostBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, dstHostBuf, view->createHostBuffer(allocLen));
  memcpy(srcHostBuf->ptr(), checkData.data(), allocLen);
  ASSIGN_NO_ERR(DeviceBufferRef, srcDevBuf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, dstDevBuf,
                view->enqueueCreateBuffer(allocLen));

  EXPECT_NO_ERR(srcHostBuf->copyTo(*srcDevBuf));
  ASSIGN_NO_ERR(DeviceStreamRef, baseStream, ctx->stream());
  ASSIGN_NO_ERR(DeviceEventRef, copyDoneEvent, ctx->eventCreate(false));
  EXPECT_NO_ERR(baseStream->eventRecord(*copyDoneEvent));
  EXPECT_NO_ERR(sideStream->enqueueWaitForEvent(*copyDoneEvent));
  EXPECT_NO_ERR(srcDevBuf->copyTo(*dstDevBuf));
  EXPECT_NO_ERR(dstDevBuf->copyTo(*dstHostBuf));
  EXPECT_NO_ERR(sideStream->synchronize());

  EXPECT_EQ(memcmp(dstHostBuf->ptr(), checkData.data(), allocLen), 0);
}

TEST_F(GPUDeviceContextTest, BasicEvent) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  // Enqueue with specific event not supported for Metal
  if (ctx->device().api() == Device::metalAPI) {
    GTEST_SKIP() << "Metal not supported";
  }

  const int allocLen = 12 * 1024 * 1024;
  char *checkData = new char[allocLen];
  // Initialize checkData with a pattern specific for this run.
  for (int i = 0; i < allocLen; ++i) {
    checkData[i] = i;
  }

  ASSIGN_NO_ERR(DeviceBufferRef, srcHostBuf, ctx->createHostBuffer(allocLen));
  memcpy(srcHostBuf->ptr(), checkData, allocLen);
  ASSIGN_NO_ERR(DeviceBufferRef, aHostBuf, ctx->createHostBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, bHostBuf, ctx->createHostBuffer(allocLen));

  ASSIGN_NO_ERR(DeviceBufferRef, aDevBuf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, bDevBuf, ctx->enqueueCreateBuffer(allocLen));

  EXPECT_NO_ERR(srcHostBuf->copyTo(*aDevBuf));
  EXPECT_NO_ERR(aDevBuf->copyTo(*aHostBuf));

  // Check enqueueEvent()
  ASSIGN_NO_ERR(DeviceEventRef, aEvent, ctx->enqueueEvent());
  EXPECT_NO_ERR(aEvent->synchronize());
  EXPECT_EQ(memcmp(aHostBuf->ptr(), checkData, allocLen), 0);

  // Check specific event enqueueEvent()
  ASSIGN_NO_ERR(DeviceEventRef, bEvent, ctx->eventCreate(0));

  EXPECT_NO_ERR(srcHostBuf->copyTo(*bDevBuf));
  EXPECT_NO_ERR(bDevBuf->copyTo(*bHostBuf));

  EXPECT_NO_ERR(ctx->enqueueEvent(*bEvent));
  EXPECT_NO_ERR(bEvent->synchronize());
  EXPECT_EQ(memcmp(bHostBuf->ptr(), checkData, allocLen), 0);

  delete[] checkData;
}

TEST_F(GPUDeviceContextTest, GetPciBusId) {
  ASSIGN_NO_ERR(DeviceRef, dev, Device::create(Device::gpuAPI));

  // Querying the PCI bus ID is only supported for CUDA and HIP devices.
  if (dev->api() == Device::cudaAPI || dev->api() == Device::hipAPI) {

    // Check that the PCI bus ID is not empty and contains colons and a dot.
    ASSIGN_NO_ERR(std::string, busId, dev->getPciBusId());
    EXPECT_FALSE(busId.empty());
    EXPECT_NE(busId.find(':'), std::string::npos);
    EXPECT_NE(busId.find('.'), std::string::npos);
  } else {
    // Check that an error is returned.
    EXPECT_ERR(dev->getPciBusId());
  }
}

static void bumpAtomicCounter(void *userData) {
  auto *counter = reinterpret_cast<std::atomic<int> *>(userData);
  counter->fetch_add(1, std::memory_order_relaxed);
}

TEST_F(GPUDeviceContextTest, EnqueueHostFuncRunsCallback) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  if (ctx->device().api() == Device::cudaAPI ||
      ctx->device().api() == Device::hipAPI) {
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

    std::atomic<int> counter{0};
    constexpr int kCount = 8;
    for (int i = 0; i < kCount; ++i)
      EXPECT_NO_ERR(stream->enqueueHostFunc(&bumpAtomicCounter, &counter));

    EXPECT_NO_ERR(stream->synchronize());
    EXPECT_EQ(counter.load(std::memory_order_relaxed), kCount);
  }
}

TEST_F(GPUDeviceContextTest, EnqueueHostFuncUnsupportedOnCpu) {
  ASSIGN_NO_ERR(DeviceContextRef, cpuCtx,
                DeviceContext::create(Device::cpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, cpuStream, cpuCtx->stream());

  std::atomic<int> counter{0};
  EXPECT_ERR(cpuStream->enqueueHostFunc(&bumpAtomicCounter, &counter));
  EXPECT_EQ(counter.load(std::memory_order_relaxed), 0);
}

TEST_F(GPUDeviceContextTest, CompletionFlagAllocResetSignalLoad) {
  // Round-trip the host-side API in isolation: allocation initializes
  // to zero, signal() stores the value, reset() goes back to zero.
  // No GPU stream involved -- this just exercises the device-mapped
  // pinned buffer and the atomic accessors on M::Driver::CompletionFlag.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  if (ctx->device().api() == Device::cudaAPI ||
      ctx->device().api() == Device::hipAPI) {
    ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());

    EXPECT_EQ(flag->load(), 0u) << "fresh allocation should be zero";

    flag->signal(42);
    EXPECT_EQ(flag->load(), 42u);

    flag->signal(0xdeadbeef);
    EXPECT_EQ(flag->load(), 0xdeadbeefu);

    flag->reset();
    EXPECT_EQ(flag->load(), 0u) << "reset should clear back to zero";
  } else {
    ErrorOr<CompletionFlagRef> flagOr = ctx->allocateCompletionFlag();
    ASSERT_TRUE(flagOr.isError());
    EXPECT_THAT(std::string(flagOr.getError()),
                testing::HasSubstr("not supported"));
  }
}

TEST_F(GPUDeviceContextTest,
       EnqueueWaitOnHostValuePreSignaledReturnsImmediately) {
  // If the flag already holds the awaited value when the wait is
  // enqueued, the wait should be a no-op and stream sync should complete.
  // Avoids timing-flakiness while still exercising the wait path end-to-end.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  if (ctx->device().api() == Device::cudaAPI ||
      ctx->device().api() == Device::hipAPI) {
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());
    ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());

    flag->signal(1);
    EXPECT_NO_ERR(stream->enqueueWaitOnHostValue(*flag, 1));
    EXPECT_NO_ERR(stream->synchronize());
  }
}

TEST_F(GPUDeviceContextTest, EnqueueWaitOnHostValueGatesUntilHostSignals) {
  // Enqueue the wait first, then signal from a background thread after
  // a short delay. The stream's synchronize() must not return until
  // signal() lands and the GPU front-end observes value == 1. We don't
  // assert on timing (flaky) -- only that the sequence completes without
  // hang and the worker thread ran before sync returned.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  if (ctx->device().api() == Device::cudaAPI ||
      ctx->device().api() == Device::hipAPI) {
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());
    ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());

    EXPECT_NO_ERR(stream->enqueueWaitOnHostValue(*flag, 1));

    std::atomic<bool> workerDidRun{false};
    std::thread signaller([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      workerDidRun.store(true, std::memory_order_release);
      flag->signal(1);
    });

    EXPECT_NO_ERR(stream->synchronize());
    signaller.join();

    EXPECT_TRUE(workerDidRun.load(std::memory_order_acquire))
        << "stream sync should not have returned before the signal thread "
           "stored the workerDidRun flag";
    EXPECT_EQ(flag->load(), 1u);
  }
}

TEST_F(GPUDeviceContextTest, AllocateCompletionFlagUnsupportedOnCpu) {
  // Sanity: the default `DeviceContext::allocateCompletionFlag` returns
  // an error; only CUDA overrides it. Matches the
  // EnqueueHostFuncUnsupportedOnCpu pattern above.
  ASSIGN_NO_ERR(DeviceContextRef, cpuCtx,
                DeviceContext::create(Device::cpuAPI));
  ErrorOr<CompletionFlagRef> flagOr = cpuCtx->allocateCompletionFlag();
  ASSERT_TRUE(flagOr.isError());
  EXPECT_THAT(std::string(flagOr.getError()),
              testing::HasSubstr("not supported"));
}

TEST_F(GPUDeviceContextTest, EnqueueWaitOnHostValueUnsupportedOnCpu) {
  // Allocate the flag against the CUDA context (the only backend that
  // can produce one), then attempt the wait on the CPU stream. The
  // stream itself must reject the call without dereferencing the flag.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  if (ctx->device().api() == Device::cudaAPI ||
      ctx->device().api() == Device::hipAPI) {
    ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());

    ASSIGN_NO_ERR(DeviceContextRef, cpuCtx,
                  DeviceContext::create(Device::cpuAPI));
    ASSIGN_NO_ERR(DeviceStreamRef, cpuStream, cpuCtx->stream());
    EXPECT_ERR_MSG(cpuStream->enqueueWaitOnHostValue(*flag, /*value=*/1),
                   "not supported");
  }
}

} // namespace M::Driver

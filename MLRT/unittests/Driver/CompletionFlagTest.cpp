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
// Covers the legacy driver. The HAL DeviceContext does not support these
// cases yet.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace M::Driver {

TEST(CompletionFlagTest, CompletionFlagAllocResetSignalLoad) {
  // Round-trip the host-side API in isolation: allocation initializes
  // to zero, signal() stores the value, reset() goes back to zero.
  // No GPU stream involved -- this just exercises the device-mapped
  // pinned buffer and the atomic accessors on M::Driver::CompletionFlag.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  if (ctx->device().api() == Device::cudaAPI ||
      ctx->device().api() == Device::hipAPI ||
      ctx->device().api() == Device::metalAPI) {
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

TEST(CompletionFlagTest, EnqueueWaitOnHostValuePreSignaledReturnsImmediately) {
  // Covers the wait path end to end without depending on timing.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  if (ctx->device().api() == Device::cudaAPI ||
      ctx->device().api() == Device::hipAPI ||
      ctx->device().api() == Device::metalAPI) {
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());
    ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());

    flag->signal(2);
    EXPECT_NO_ERR(stream->enqueueWaitOnFlag(*flag, 1));
    EXPECT_NO_ERR(stream->synchronize());
  }
}

TEST(CompletionFlagTest, EnqueueWaitOnHostValueGatesUntilHostSignals) {
  // Asserts ordering only; a timing assertion would flake.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  if (ctx->device().api() == Device::cudaAPI ||
      ctx->device().api() == Device::hipAPI ||
      ctx->device().api() == Device::metalAPI) {
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());
    ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());

    EXPECT_NO_ERR(stream->enqueueWaitOnFlag(*flag, 1));

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

TEST(CompletionFlagTest, AllocateCompletionFlagUnsupportedOnCpu) {
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

TEST(CompletionFlagTest, EnqueueWaitOnHostValueUnsupportedOnCpu) {
  // Allocate the flag against the CUDA context (the only backend that
  // can produce one), then attempt the wait on the CPU stream. The
  // stream itself must reject the call without dereferencing the flag.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  if (ctx->device().api() == Device::cudaAPI ||
      ctx->device().api() == Device::hipAPI ||
      ctx->device().api() == Device::metalAPI) {
    ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());

    ASSIGN_NO_ERR(DeviceContextRef, cpuCtx,
                  DeviceContext::create(Device::cpuAPI));
    ASSIGN_NO_ERR(DeviceStreamRef, cpuStream, cpuCtx->stream());
    EXPECT_ERR_MSG(cpuStream->enqueueWaitOnFlag(*flag, /*value=*/1),
                   "not supported");
  }
}

TEST(CompletionFlagTest, MetalHostCopyDoesNotBlockBehindFlagWait) {
  // A host-to-device copy from pageable memory must not wait for the flag:
  // the thread that would signal it may be the one copying.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  if (ctx->device().api() != Device::metalAPI)
    GTEST_SKIP() << "only Metal stages pageable host copies";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());
  ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());
  constexpr size_t kCount = 1024;
  std::vector<uint32_t> host(kCount);
  for (size_t i = 0; i < kCount; ++i)
    host[i] = static_cast<uint32_t>(i + 1);
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf,
                ctx->enqueueCreateBuffer(kCount * sizeof(uint32_t)));
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(stream->enqueueWaitOnFlag(*flag, 1));
  std::thread watchdog([&] {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    flag->signal(1);
  });
  const auto start = std::chrono::steady_clock::now();
  EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*devBuf, host.data()));
  const auto elapsed = std::chrono::steady_clock::now() - start;
  // The copy snapshots its source, so the caller may reuse it at once.
  std::fill(host.begin(), host.end(), 0u);
  watchdog.join();
  EXPECT_LT(elapsed, std::chrono::milliseconds(500));

  EXPECT_NO_ERR(ctx->synchronize());
  std::vector<uint32_t> back(kCount, 0);
  EXPECT_NO_ERR(ctx->enqueueCopyFromDevice(back.data(), *devBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  for (size_t i = 0; i < kCount; ++i)
    ASSERT_EQ(back[i], static_cast<uint32_t>(i + 1)) << "at index " << i;
}

} // namespace M::Driver

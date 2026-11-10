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
// Uses only the DeviceContext API, so the same cases run against the legacy
// driver and under --config=hal-device-context.
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

static bool hasFlags(const DeviceContext &ctx) {
  return ctx.device().api() == Device::cudaAPI ||
         ctx.device().api() == Device::hipAPI ||
         ctx.device().api() == Device::metalAPI;
}

static void bumpAtomicCounter(void *userData) {
  auto *counter = reinterpret_cast<std::atomic<int> *>(userData);
  counter->fetch_add(1, std::memory_order_relaxed);
}

TEST(CompletionFlagTest, CompletionFlagAllocResetSignalLoad) {
  // No stream: a reset must read back 0 even where the plugin swaps the event.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  if (hasFlags(*ctx)) {
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

  if (hasFlags(*ctx)) {
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

  if (hasFlags(*ctx)) {
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

TEST(CompletionFlagTest, CompletionFlagResetRegatesQueuedWait) {
  // Pins current behavior of a UB pattern Mach depends on. `gate` holds the
  // stream so the wait on `flag` is still pending when the reset lands.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  if (ctx->device().api() != Device::cudaAPI &&
      ctx->device().api() != Device::hipAPI)
    GTEST_SKIP() << "on Metal the pending wait stays on the replaced event";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());
  ASSIGN_NO_ERR(CompletionFlagRef, gate, ctx->allocateCompletionFlag());
  ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());

  std::atomic<int> counter{0};
  EXPECT_NO_ERR(stream->enqueueWaitOnFlag(*gate, 1));
  EXPECT_NO_ERR(stream->enqueueWaitOnFlag(*flag, 9));
  EXPECT_NO_ERR(stream->enqueueHostFunc(&bumpAtomicCounter, &counter));

  flag->signal(9);
  flag->reset();
  gate->signal(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(counter.load(std::memory_order_acquire), 0)
      << "the reset did not re-gate the queued wait";

  flag->signal(9);
  EXPECT_NO_ERR(stream->synchronize());
  EXPECT_EQ(counter.load(std::memory_order_acquire), 1);
}

TEST(CompletionFlagDeathTest, LowerSignalAbortsOnMetal) {
  // The flag API has no error channel, so a rejected lower signal must abort
  // rather than leave a waiter short of its target.
  if (!MODULAR_HAL_DEVICE_CONTEXT)
    GTEST_SKIP() << "the legacy Metal driver ignores a lower signal";
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  if (ctx->device().api() != Device::metalAPI)
    GTEST_SKIP() << "only Metal rejects a lower signal";

  ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());
  flag->signal(5);
  EXPECT_DEATH(flag->signal(3), "CompletionFlag::signal");
  EXPECT_EQ(flag->load(), 5u);
}

TEST(CompletionFlagTest, CompletionFlagResetKeepsHandle) {
  // A graph payload holds the flag's address, so reset must not replace it,
  // including on Metal, where it swaps the backing event.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  if (!hasFlags(*ctx))
    GTEST_SKIP() << "no completion flags on this device";

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());
  ASSIGN_NO_ERR(CompletionFlagRef, flag, ctx->allocateCompletionFlag());
  CompletionFlag *before = flag.getPointer();

  flag->signal(3);
  flag->reset();
  EXPECT_EQ(flag.getPointer(), before);
  EXPECT_EQ(flag->load(), 0u);

  EXPECT_NO_ERR(stream->enqueueWaitOnFlag(*flag, 2));
  flag->signal(2);
  EXPECT_NO_ERR(stream->synchronize());
  EXPECT_EQ(flag->load(), 2u);
}

TEST(CompletionFlagTest, AllocateCompletionFlagUnsupportedOnCpu) {
  ASSIGN_NO_ERR(DeviceContextRef, cpuCtx,
                DeviceContext::create(Device::cpuAPI));
  ErrorOr<CompletionFlagRef> flagOr = cpuCtx->allocateCompletionFlag();
  ASSERT_TRUE(flagOr.isError());
  EXPECT_THAT(std::string(flagOr.getError()),
              testing::HasSubstr("not supported"));
}

TEST(CompletionFlagTest, EnqueueWaitOnHostValueUnsupportedOnCpu) {
  // A CPU stream must reject a GPU flag rather than hand its handle to the
  // wrong backend.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  if (hasFlags(*ctx)) {
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
  if (MODULAR_HAL_DEVICE_CONTEXT)
    GTEST_SKIP() << "the HAL Metal pageable host copy waits behind the queue";

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

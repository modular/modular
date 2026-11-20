//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Init/Init.h"
#include "gtest/gtest.h"

#include <array>
#include <future>
#include <thread>
#include <vector>

using namespace M;

TEST(ContextTest, CreatesUsingDefaults) {
  Init::Options defaults;
  defaults.withForceDisableCrashReporting();
  auto contextOr = Init::getOrCreateContextWithDefaults("test", defaults);
  ASSERT_FALSE(contextOr.isError());
  auto context = contextOr.takeValue();
  EXPECT_EQ(*context->get<Init::Options>(), defaults);
  EXPECT_EQ(Init::getContext().getPointer(), context.getPointer());
}

TEST(ContextTest, ReusesAndRetainsConfiguredContext) {
  AsyncRT::CPUDeviceOptions cpuOptions;
  cpuOptions.withSingleThreaded();
  cpuOptions.mainWillDonate = false;
  cpuOptions.threadBusyWaitTime = 17;
  Init::Options configured;
  configured.withCPUDeviceOptions(cpuOptions).withForceDisableCrashReporting();
  auto firstOr = Init::getOrCreateContext("session", configured);
  ASSERT_FALSE(firstOr.isError());
  auto first = firstOr.takeValue();

  Init::Options defaults;
  defaults.withCPUDeviceOptions().withForceDisableCrashReporting();
  auto reusedOr = Init::getOrCreateContextWithDefaults("graph", defaults);
  ASSERT_FALSE(reusedOr.isError());
  auto reused = reusedOr.takeValue();
  EXPECT_EQ(first.getPointer(), reused.getPointer());
  EXPECT_EQ(*reused->get<Init::Options>(), configured);
  EXPECT_EQ(first->get<AsyncRT::CPUDevice>(),
            reused->get<AsyncRT::CPUDevice>());

  first.reset();
  EXPECT_EQ(Init::getContext().getPointer(), reused.getPointer());
  // Reuse must not replace the options checked by strict session callers.
  auto matchingOr = Init::getOrCreateContext("session", configured);
  ASSERT_FALSE(matchingOr.isError());
  EXPECT_EQ(matchingOr->getPointer(), reused.getPointer());
}

TEST(ContextTest, ConcurrentDefaultCreationReturnsOneContext) {
  constexpr size_t count = 8;
  std::array<ContextRef, count> contexts;
  std::vector<std::thread> threads;
  std::promise<void> start;
  auto ready = start.get_future().share();
  for (size_t i = 0; i < count; ++i) {
    threads.emplace_back([&, i] {
      ready.wait();
      Init::Options defaults;
      defaults.withForceDisableCrashReporting();
      auto contextOr = Init::getOrCreateContextWithDefaults("test", defaults);
      ASSERT_FALSE(contextOr.isError());
      contexts[i] = contextOr.takeValue();
    });
  }
  start.set_value();
  for (auto &thread : threads) {
    thread.join();
  }
  ASSERT_NE(contexts[0].getPointer(), nullptr);
  for (const auto &context : contexts) {
    EXPECT_EQ(context.getPointer(), contexts[0].getPointer());
  }
}

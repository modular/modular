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
//
// Measures the per-operation cost of the AsyncRT primitives that graph
// execution drives: posting a task and awaiting it, firing AsyncValue waiters
// from a worker, async waiters, and a small parallel-for.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "AsyncRT/Support/Chain.h"
#include "benchmark/benchmark.h"

#include <atomic>
#include <vector>

using namespace M;
using namespace M::AsyncRT;

namespace {

CPUDeviceRef makeDevice(bool mainWillDonate) {
  CPUDeviceOptions options;
  options.numThreads = 8;
  options.mainWillDonate = mainWillDonate;
  return getOrCreateCPUDevice(CPUDeviceSource::Test, options);
}

// One task posted from the calling thread, then awaited.
void BM_AddTaskAwait(benchmark::State &state) {
  CPUDeviceRef device = makeDevice(/*mainWillDonate=*/state.range(0));
  for (auto _ : state) {
    AsyncValueRef<int> result = addTask(*device, [] { return 1; });
    await(result);
    benchmark::DoNotOptimize(result.get());
  }
}
BENCHMARK(BM_AddTaskAwait)->Arg(0)->Arg(1)->UseRealTime();

// A chain of AsyncValues where each waiter emplaces the next one. Every
// emplace runs on a worker, so each link goes through addLocalTask.
void BM_WaiterChain(benchmark::State &state) {
  CPUDeviceRef device = makeDevice(/*mainWillDonate=*/false);
  const size_t length = state.range(0);
  for (auto _ : state) {
    std::vector<AsyncValueRef<Chain>> links;
    links.reserve(length + 1);
    for (size_t i = 0; i <= length; ++i)
      links.push_back(AsyncValueRef<Chain>::allocate(*device));
    for (size_t i = 0; i < length; ++i)
      links[i].andThenSync([next = links[i + 1].copy()]() mutable {
        std::move(next).emplace();
      });
    addTask(*device, [first = links[0].copy()]() mutable {
      std::move(first).emplace();
    });
    await(links[length]);
  }
  state.SetItemsProcessed(state.iterations() * length);
}
BENCHMARK(BM_WaiterChain)->Arg(1024)->UseRealTime();

// Many async waiters on one value, emplaced from a worker.
void BM_AndThenAsyncFanOut(benchmark::State &state) {
  CPUDeviceRef device = makeDevice(/*mainWillDonate=*/false);
  const size_t fanOut = state.range(0);
  for (auto _ : state) {
    auto source = AsyncValueRef<Chain>::allocate(*device);
    auto done = AsyncValueRef<Chain>::allocate(*device);
    auto remaining = std::make_shared<std::atomic<size_t>>(fanOut);
    for (size_t i = 0; i < fanOut; ++i)
      source.andThenAsync([remaining, done = done.copy()]() mutable {
        if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1)
          std::move(done).emplace();
      });
    addTask(*device, [source = source.copy()]() mutable {
      std::move(source).emplace();
    });
    await(done);
  }
  state.SetItemsProcessed(state.iterations() * fanOut);
}
BENCHMARK(BM_AndThenAsyncFanOut)->Arg(1024)->UseRealTime();

// A parallel-for with trivial elements, as CPU kernels issue.
void BM_ParallelForEachN(benchmark::State &state) {
  CPUDeviceRef device = makeDevice(/*mainWillDonate=*/false);
  const size_t count = state.range(0);
  std::vector<int> data(count);
  for (auto _ : state) {
    auto chain = parallelForEachNChain(*device, count,
                                       [&data](size_t i) { data[i] += 1; });
    await(chain);
  }
  benchmark::DoNotOptimize(data.data());
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_ParallelForEachN)->Arg(8)->Arg(64)->UseRealTime();

} // namespace

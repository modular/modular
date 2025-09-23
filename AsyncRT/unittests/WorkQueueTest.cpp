//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "gtest/gtest.h"
#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace M;
using namespace M::AsyncRT;

namespace {

// Enqueue a worker-task that records the result of `shouldRunInlineFor` for a
// given `checkHint`.
static AsyncValueRef<bool> enqueueInlineCheck(Runtime &runtime,
                                              WorkQueue &workQueue,
                                              int dispatchHint, int checkHint) {
  AsyncValueRef<bool> result = AsyncValueRef<bool>::allocate(runtime);
  WorkItem probe([&workQueue, checkHint, ready = result.copy()]() mutable {
    ready.copy().emplace(workQueue.shouldRunInlineFor(checkHint));
  });
  probe.deviceHint = dispatchHint;
  workQueue.addTask(std::move(probe));
  return result;
}

/// Test device-aware scheduling with device hints.
TEST(WorkQueueTest, DeviceHintRouting) {
  RuntimeOptions options;
  options.numThreads = 4;
  options.mainWillDonate = false; // Ensure predictable modulo mapping
  std::unique_ptr<Runtime> runtime = createUniqueRuntime(options);
  WorkQueue *workQueue = runtime->getWorkQueue();

  constexpr int numDevices = 8;
  constexpr int numTasksPerDevice = 10;

  // Track which thread executed each task and per-device thread mapping.
  std::unordered_map<std::thread::id, int> threadTaskCounts;
  std::unordered_map<int, std::vector<std::thread::id>> perDeviceThreads;
  std::mutex mapMutex;

  // Create work items with different device hints.
  std::vector<AsyncValueRef<int>> results;
  results.reserve(numDevices * numTasksPerDevice);
  for (int device = 0; device < numDevices; ++device) {
    for (int task = 0; task < numTasksPerDevice; ++task) {
      results.emplace_back(AsyncValueRef<int>::allocate(*runtime));
      AsyncValueRef<int> &result = results.back();

      WorkItem workItem([device, task, result = result.copy(),
                         &threadTaskCounts, &perDeviceThreads, &mapMutex]() {
        // Record which thread is executing this task and per-device mapping.
        std::thread::id threadId = std::this_thread::get_id();
        {
          std::lock_guard<std::mutex> lock(mapMutex);
          threadTaskCounts[threadId] += 1;
          perDeviceThreads[device].push_back(threadId);
        }
        result.copy().emplace(device * 100 + task);
      });

      workItem.deviceHint = device;
      workQueue->addTask(std::move(workItem));
    }
  }

  // Wait for all tasks to complete.
  for (AsyncValueRef<int> &result : results)
    await(result);

  // Verify all tasks completed
  for (size_t i = 0; i < results.size(); ++i) {
    int device = i / numTasksPerDevice;
    int task = i % numTasksPerDevice;
    EXPECT_EQ(results[i].get(), device * 100 + task);
  }

  // Stronger routing checks:
  // 1) All tasks for a given device run on the same thread.
  for (int d = 0; d < numDevices; ++d) {
    ASSERT_EQ(perDeviceThreads.count(d), 1u)
        << "Missing records for device " << d;
    const std::vector<std::thread::id> &v = perDeviceThreads[d];
    ASSERT_FALSE(v.empty());
    const std::thread::id first = v.front();
    for (const std::thread::id &tid : v)
      EXPECT_EQ(tid, first) << "Device " << d << " ran on multiple threads";
  }

  // 2) Devices d and d+numThreads share the same thread (modulo mapping).
  for (int d = 0; d < 4; ++d) {
    EXPECT_EQ(perDeviceThreads[d].front(), perDeviceThreads[d + 4].front())
        << "Devices " << d << " and " << (d + 4)
        << " should map to same worker";
  }

  // 3) Devices 0..numThreads-1 run on distinct threads.
  std::set<std::thread::id> firstFour;
  for (int d = 0; d < 4; ++d)
    firstFour.insert(perDeviceThreads[d].front());
  EXPECT_EQ(firstFour.size(), 4u)
      << "First four devices should map to four distinct workers";
}

/// Test that negative device hints are handled correctly.
TEST(WorkQueueTest, NegativeDeviceHint) {
  RuntimeOptions options;
  options.numThreads = 4;
  std::unique_ptr<Runtime> runtime = createUniqueRuntime(options);
  WorkQueue *workQueue = runtime->getWorkQueue();

  auto result = AsyncValueRef<int>::allocate(*runtime);

  WorkItem workItem([result = result.copy()]() { result.copy().emplace(42); });

  // Negative device hint should be treated as no preference
  workItem.deviceHint = -5;
  workQueue->addTask(std::move(workItem));

  await(result);
  EXPECT_EQ(result.get(), 42);
}

/// Test device hint with mainWillDonate mode.
TEST(WorkQueueTest, DeviceHintWithMainWillDonate) {
  RuntimeOptions options;
  options.numThreads = 4;
  options.mainWillDonate = true;
  std::unique_ptr<Runtime> runtime = createUniqueRuntime(options);
  WorkQueue *workQueue = runtime->getWorkQueue();

  constexpr int numTasks = 20;
  std::vector<AsyncValueRef<int>> results;
  results.reserve(numTasks);

  // Create tasks that would map to worker 0
  for (int i = 0; i < numTasks; ++i) {
    results.emplace_back(AsyncValueRef<int>::allocate(*runtime));
    AsyncValueRef<int> &result = results.back();

    WorkItem workItem([i, result = result.copy()]() {
      // Small delay to simulate work
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      result.copy().emplace(i);
    });

    // Device hints that would map to worker 0 (0, 4, 8, 12, 16)
    workItem.deviceHint = i * 4;
    workQueue->addTask(std::move(workItem));
  }

  // Tasks should complete without needing await on main thread
  // (they should be redistributed away from worker 0)
  std::promise<void> done;
  std::future<void> fut = done.get_future();
  std::thread checker([&results, d = std::move(done)]() mutable {
    for (AsyncValueRef<int> &result : results)
      await(result);
    d.set_value();
  });

  // Bound the wait to avoid hangs on regression.
  std::future_status status = fut.wait_for(std::chrono::seconds(3));
  EXPECT_EQ(status, std::future_status::ready)
      << "Tasks should complete without main-thread await";

  if (status == std::future_status::ready) {
    checker.join();
  } else {
    // Avoid blocking the test thread; failure already recorded.
    checker.detach();
  }

  // Verify all tasks completed correctly.
  if (status == std::future_status::ready) {
    for (int i = 0; i < numTasks; ++i)
      EXPECT_EQ(results[i].get(), i);
  }
}

/// Test that kNoDevicePreference works correctly.
TEST(WorkQueueTest, NoDevicePreference) {
  RuntimeOptions options;
  options.numThreads = 4;
  std::unique_ptr<Runtime> runtime = createUniqueRuntime(options);
  WorkQueue *workQueue = runtime->getWorkQueue();

  auto result = AsyncValueRef<int>::allocate(*runtime);

  WorkItem workItem([result = result.copy()]() { result.copy().emplace(99); });

  // Should use default scheduling.
  workItem.deviceHint = kNoDevicePreference;
  workQueue->addTask(std::move(workItem));

  await(result);
  EXPECT_EQ(result.get(), 99);
}

TEST(WorkQueueTest, ShouldRunInlineMatchesAssignedWorker) {
  RuntimeOptions options;
  options.numThreads = 4;
  options.mainWillDonate = false;
  std::unique_ptr<Runtime> runtime = createUniqueRuntime(options);
  WorkQueue &workQueue = *runtime->getWorkQueue();

  // Each worker executes a task pinned to its own device hint and reports
  // whether the inline heuristic agrees; the hint-aligned tasks should all come
  // back `true`.
  std::vector<AsyncValueRef<bool>> inlineResults;
  inlineResults.reserve(options.numThreads);
  for (size_t idx = 0; idx < options.numThreads; ++idx) {
    int hint = static_cast<int>(idx);
    inlineResults.emplace_back(
        enqueueInlineCheck(*runtime, workQueue, hint, hint));
  }

  for (size_t idx = 0; idx < inlineResults.size(); ++idx) {
    await(inlineResults[idx]);
    EXPECT_TRUE(inlineResults[idx].get());
  }

  // When a worker is pinned to device 0 but queries device 1, it should decline
  // to inline because it is running on the wrong worker thread.
  AsyncValueRef<bool> mismatch =
      enqueueInlineCheck(*runtime, workQueue, /*dispatchHint=*/0,
                         /*checkHint=*/1);
  await(mismatch);
  EXPECT_FALSE(mismatch.get());
}

TEST(WorkQueueTest, ShouldRunInlineHonorsMainDonation) {
  RuntimeOptions options;
  options.numThreads = 3;
  options.mainWillDonate = true;
  std::unique_ptr<Runtime> runtime = createUniqueRuntime(options);
  WorkQueue &workQueue = *runtime->getWorkQueue();

  // Even though device 0 is rerouted away from the donating worker, the worker
  // that actually executes the task should still report inline = true.
  std::vector<AsyncValueRef<bool>> results;
  for (size_t idx = 0; idx < options.numThreads; ++idx) {
    int hint = static_cast<int>(idx);
    results.emplace_back(enqueueInlineCheck(*runtime, workQueue, hint, hint));
  }

  for (AsyncValueRef<bool> &ready : results) {
    await(ready);
    EXPECT_TRUE(ready.get());
  }
}

TEST(WorkQueueTest, ShouldRunInlineFromForeignThread) {
  RuntimeOptions options;
  options.numThreads = 2;
  std::unique_ptr<Runtime> runtime = createUniqueRuntime(options);
  WorkQueue &workQueue = *runtime->getWorkQueue();

  // Main thread should inline only for "no preference"; other hints belong to
  // workers and should return false here.
  EXPECT_TRUE(workQueue.shouldRunInlineFor(kNoDevicePreference));
  EXPECT_FALSE(workQueue.shouldRunInlineFor(0));
  EXPECT_FALSE(workQueue.shouldRunInlineFor(-5));
}

TEST(WorkQueueTest, ShouldRunInlineSingleThreadedAlwaysTrue) {
  RuntimeOptions options;
  options.singleThreaded = true;
  std::unique_ptr<Runtime> runtime = createUniqueRuntime(options);
  WorkQueue &workQueue = *runtime->getWorkQueue();

  // The single worker should inline regardless of the hint value.
  AsyncValueRef<bool> inlineCheck =
      enqueueInlineCheck(*runtime, workQueue, /*dispatchHint=*/0,
                         /*checkHint=*/3);
  await(inlineCheck);
  EXPECT_TRUE(inlineCheck.get());

  EXPECT_TRUE(workQueue.shouldRunInlineFor(kNoDevicePreference));
}

} // namespace

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/WorkQueue.h"
#include "MLRT/AsyncRT/Runtime/Algorithms.h"
#include "MLRT/AsyncRT/Runtime/AsyncValueRef.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "MLRT/AsyncRT/Runtime/RuntimeManager.h"
#include "Support/Threading/HWInfo.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <atomic>
#include <climits>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

using namespace M;
using namespace M::MLRT;

namespace {

// Enqueue a worker-task that records the result of `shouldRunInlineForTask` for
// a given `checkTaskId`.
static AsyncValueRef<bool> enqueueInlineCheck(Runtime &runtime,
                                              WorkQueue &workQueue,
                                              int dispatchTaskId,
                                              int checkTaskId) {
  AsyncValueRef<bool> result = AsyncValueRef<bool>::allocate(runtime);
  WorkItem probe([&workQueue, checkTaskId, ready = result.copy()]() mutable {
    ready.copy().emplace(workQueue.shouldRunInlineForTask(checkTaskId));
  });
  workQueue.addTask(std::move(probe), dispatchTaskId);
  return result;
}

/// Test task-based scheduling with taskId affinity.
/// With conservative worker 0 avoidance:
/// - 4 workers (0, 1, 2, 3), but we use workers 1, 2, 3 for affinity tasks
/// - taskId = 1 + (hint % 3) for non-negative hints
TEST(WorkQueueTest, TaskIdRouting) {
  RuntimeOptions options;
  options.numThreads = 4;
  options.mainWillDonate = false;
  RuntimeRef runtime = getOrCreateRuntime(RuntimeSource::Test, options);
  WorkQueue *workQueue = runtime->getWorkQueue();

  // Use 3 different taskIds (1, 2, 3) since worker 0 is avoided
  constexpr int numTaskIds = 3;
  constexpr int numTasksPerTaskId = 10;

  // Track which thread executed each task and per-taskId thread mapping.
  std::unordered_map<std::thread::id, int> threadTaskCounts;
  std::unordered_map<int, std::vector<std::thread::id>> perTaskIdThreads;
  std::mutex mapMutex;

  // Create work items with different task IDs (1, 2, 3).
  std::vector<AsyncValueRef<int>> results;
  results.reserve(numTaskIds * numTasksPerTaskId);
  for (int taskId = 1; taskId <= numTaskIds; ++taskId) {
    for (int task = 0; task < numTasksPerTaskId; ++task) {
      results.emplace_back(AsyncValueRef<int>::allocate(*runtime));
      AsyncValueRef<int> &result = results.back();

      WorkItem workItem([taskId, task, result = result.copy(),
                         &threadTaskCounts, &perTaskIdThreads, &mapMutex]() {
        // Record which thread is executing this task and per-taskId mapping.
        std::thread::id threadId = std::this_thread::get_id();
        {
          std::lock_guard<std::mutex> lock(mapMutex);
          threadTaskCounts[threadId] += 1;
          perTaskIdThreads[taskId].push_back(threadId);
        }
        result.copy().emplace(taskId * 100 + task);
      });

      workQueue->addTask(std::move(workItem), taskId);
    }
  }

  // Wait for all tasks to complete.
  for (AsyncValueRef<int> &result : results)
    await(result);

  // Verify all tasks completed
  for (size_t i = 0; i < results.size(); ++i) {
    int taskId = 1 + static_cast<int>(i / numTasksPerTaskId);
    int task = i % numTasksPerTaskId;
    EXPECT_EQ(results[i].get(), taskId * 100 + task);
  }

  // Stronger routing checks:
  // 1) All tasks for a given taskId run on the same thread.
  for (int tid = 1; tid <= numTaskIds; ++tid) {
    ASSERT_EQ(perTaskIdThreads.count(tid), 1u)
        << "Missing records for taskId " << tid;
    const std::vector<std::thread::id> &v = perTaskIdThreads[tid];
    ASSERT_FALSE(v.empty());
    const std::thread::id first = v.front();
    for (const std::thread::id &threadId : v)
      EXPECT_EQ(threadId, first)
          << "TaskId " << tid << " ran on multiple threads";
  }

  // 2) TaskIds 1, 2, 3 run on distinct threads.
  std::set<std::thread::id> workerThreads;
  for (int tid = 1; tid <= numTaskIds; ++tid)
    workerThreads.insert(perTaskIdThreads[tid].front());
  EXPECT_EQ(workerThreads.size(), static_cast<size_t>(numTaskIds))
      << "TaskIds 1-3 should map to three distinct workers";
}

/// Test that negative taskIds are handled correctly (global queue).
TEST(WorkQueueTest, NegativeTaskId) {
  RuntimeOptions options;
  options.numThreads = 4;
  RuntimeRef runtime = getOrCreateRuntime(RuntimeSource::Test, options);
  WorkQueue *workQueue = runtime->getWorkQueue();

  auto result = AsyncValueRef<int>::allocate(*runtime);

  WorkItem workItem([result = result.copy()]() { result.copy().emplace(42); });

  // Negative taskId should use global queue
  workQueue->addTask(std::move(workItem), -5);

  await(result);
  EXPECT_EQ(result.get(), 42);
}

/// Test taskId with mainWillDonate mode.
/// Since we conservatively skip worker 0, all tasks should complete
/// without needing await on main thread.
TEST(WorkQueueTest, TaskIdWithMainWillDonate) {
  RuntimeOptions options;
  options.numThreads = 4;
  options.mainWillDonate = true;
  RuntimeRef runtime = getOrCreateRuntime(RuntimeSource::Test, options);
  WorkQueue *workQueue = runtime->getWorkQueue();

  constexpr int numTasks = 20;
  std::vector<AsyncValueRef<int>> results;
  results.reserve(numTasks);

  // Create tasks with various taskIds (all avoid worker 0)
  for (int i = 0; i < numTasks; ++i) {
    results.emplace_back(AsyncValueRef<int>::allocate(*runtime));
    AsyncValueRef<int> &result = results.back();

    WorkItem workItem([i, result = result.copy()]() {
      // Small delay to simulate work
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      result.copy().emplace(i);
    });

    // Use taskIds 1, 2, 3 (avoiding worker 0)
    int taskId = 1 + (i % 3);
    workQueue->addTask(std::move(workItem), taskId);
  }

  // Tasks should complete without needing await on main thread
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

/// Test that kDefaultTaskId uses global queue scheduling.
TEST(WorkQueueTest, DefaultTaskId) {
  RuntimeOptions options;
  options.numThreads = 4;
  RuntimeRef runtime = getOrCreateRuntime(RuntimeSource::Test, options);
  WorkQueue *workQueue = runtime->getWorkQueue();

  auto result = AsyncValueRef<int>::allocate(*runtime);

  WorkItem workItem([result = result.copy()]() { result.copy().emplace(99); });

  // Should use default (global queue) scheduling.
  workQueue->addTask(std::move(workItem), kDefaultTaskId);

  await(result);
  EXPECT_EQ(result.get(), 99);
}

TEST(WorkQueueTest, ShouldRunInlineMatchesAssignedWorker) {
  RuntimeOptions options;
  options.numThreads = 4;
  options.mainWillDonate = false;
  RuntimeRef runtime = getOrCreateRuntime(RuntimeSource::Test, options);
  WorkQueue &workQueue = *runtime->getWorkQueue();

  // Each worker executes a task pinned to its taskId and reports whether the
  // inline heuristic agrees; tasks dispatched and checked with the same taskId
  // should all come back `true`.
  std::vector<AsyncValueRef<bool>> inlineResults;
  inlineResults.reserve(3);
  // Test with taskIds 1, 2, 3.
  for (int taskId = 1; taskId <= 3; ++taskId) {
    inlineResults.emplace_back(
        enqueueInlineCheck(*runtime, workQueue, taskId, taskId));
  }

  for (size_t idx = 0; idx < inlineResults.size(); ++idx) {
    await(inlineResults[idx]);
    EXPECT_TRUE(inlineResults[idx].get());
  }

  // When a worker is pinned to taskId 1 but queries taskId 2, it should decline
  // to inline because it is running on the wrong worker thread.
  AsyncValueRef<bool> mismatch =
      enqueueInlineCheck(*runtime, workQueue, /*dispatchTaskId=*/1,
                         /*checkTaskId=*/2);
  await(mismatch);
  EXPECT_FALSE(mismatch.get());
}

TEST(WorkQueueTest, ShouldRunInlineHonorsWorkerAffinity) {
  RuntimeOptions options;
  options.numThreads = 3;
  options.mainWillDonate = true;
  RuntimeRef runtime = getOrCreateRuntime(RuntimeSource::Test, options);
  WorkQueue &workQueue = *runtime->getWorkQueue();

  // Enqueue tasks to workers 1 and 2.
  // The worker that actually executes the task should still report inline =
  // true.
  std::vector<AsyncValueRef<bool>> results;
  for (int taskId = 1; taskId <= 2; ++taskId) {
    results.emplace_back(
        enqueueInlineCheck(*runtime, workQueue, taskId, taskId));
  }

  for (AsyncValueRef<bool> &ready : results) {
    await(ready);
    EXPECT_TRUE(ready.get());
  }
}

TEST(WorkQueueTest, ShouldRunInlineFromForeignThread) {
  RuntimeOptions options;
  options.numThreads = 2;
  options.mainWillDonate = false; // Ensure main thread is "foreign"
  RuntimeRef runtime = getOrCreateRuntime(RuntimeSource::Test, options);
  WorkQueue &workQueue = *runtime->getWorkQueue();

  // From a foreign thread (main thread not donating), we should inline only
  // for default/negative taskId; positive taskIds belong to workers and should
  // return false here since we're not on any worker.
  EXPECT_TRUE(workQueue.shouldRunInlineForTask(kDefaultTaskId));
  EXPECT_TRUE(workQueue.shouldRunInlineForTask(-5));
  EXPECT_FALSE(workQueue.shouldRunInlineForTask(0));
  EXPECT_FALSE(workQueue.shouldRunInlineForTask(1));
}

TEST(WorkQueueTest, ShouldRunInlineSingleThreadedAlwaysTrue) {
  RuntimeOptions options;
  options.withSingleThreaded();
  RuntimeRef runtime = getOrCreateRuntime(RuntimeSource::Test, options);
  WorkQueue &workQueue = *runtime->getWorkQueue();

  // The single worker should inline regardless of the taskId value.
  AsyncValueRef<bool> inlineCheck =
      enqueueInlineCheck(*runtime, workQueue, /*dispatchTaskId=*/0,
                         /*checkTaskId=*/3);
  await(inlineCheck);
  EXPECT_TRUE(inlineCheck.get());

  EXPECT_TRUE(workQueue.shouldRunInlineForTask(kDefaultTaskId));
}

//===----------------------------------------------------------------------===//
// Partitioned (NUMA-aware) WorkQueue tests
//===----------------------------------------------------------------------===//

namespace {

/// Returns a NUMA node id with a non-empty CPU list, or an empty optional if
/// the host has no usable NUMA topology (e.g. running in a container that
/// doesn't expose /sys/devices/system/node/).
std::optional<int> firstUsableNumaNode() {
  const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
  if (topologyOr.isError())
    return std::nullopt;
  for (int node : topologyOr->getNumaNodes()) {
    if (!topologyOr->getCpuIdsForNumaNode(node).empty())
      return node;
  }
  return std::nullopt;
}

/// Constructs a Runtime and returns its CompactRuntimePtr for use with the
/// partitioned WorkQueue factory. Holds the RuntimeRef alive for the caller.
struct PartitionedHarness {
  RuntimeRef runtime;
  CompactRuntimePtr runtimePtr;
  std::unique_ptr<WorkQueue> workQueue;

  ~PartitionedHarness() {
    if (workQueue)
      workQueue->shutdown();
  }
};

PartitionedHarness makeHarness(int numaNode) {
  RuntimeOptions options;
  options.withSingleThreaded(); // Minimal "host" runtime just to supply ptr.
  RuntimeRef runtime = getOrCreateRuntime(RuntimeSource::Test, options);
  CompactRuntimePtr ptr = runtime->getCompactPtr();
  std::unique_ptr<WorkQueue> wq = createPartitionedThreadPoolWorkQueue(
      ptr, numaNode, std::chrono::microseconds(200), "🔥 NumaTest");
  return {std::move(runtime), ptr, std::move(wq)};
}

} // namespace

TEST(WorkQueueTest, PartitionedReportsPartition) {
  std::optional<int> nodeOr = firstUsableNumaNode();
  if (!nodeOr)
    GTEST_SKIP() << "host has no NUMA topology available";
  int node = *nodeOr;

  PartitionedHarness h = makeHarness(node);
  EXPECT_EQ(h.workQueue->getNumaNode(), node);

  std::vector<size_t> expected =
      NUMATopology::get()->getCpuIdsForNumaNode(node);
  ArrayRef<size_t> reported = h.workQueue->getCpuIds();
  EXPECT_EQ(reported.size(), expected.size());
  EXPECT_TRUE(std::equal(reported.begin(), reported.end(), expected.begin()));
}

TEST(WorkQueueTest, PartitionedParallelismMatchesCpuIds) {
  std::optional<int> nodeOr = firstUsableNumaNode();
  if (!nodeOr)
    GTEST_SKIP() << "host has no NUMA topology available";

  PartitionedHarness h = makeHarness(*nodeOr);
  EXPECT_EQ(h.workQueue->getParallelismLevel(),
            h.workQueue->getCpuIds().size());
}

#if defined(__linux__)
TEST(WorkQueueTest, PartitionedRunsOnlyOnSelectedCpus) {
  std::optional<int> nodeOr = firstUsableNumaNode();
  if (!nodeOr)
    GTEST_SKIP() << "host has no NUMA topology available";

  PartitionedHarness h = makeHarness(*nodeOr);
  ArrayRef<size_t> cpuIds = h.workQueue->getCpuIds();
  ASSERT_FALSE(cpuIds.empty());
  std::set<size_t> allowedCpus(cpuIds.begin(), cpuIds.end());

  // Dispatch one task per worker with a positive taskId, forcing each task
  // onto its pinned worker. Each task records sched_getcpu() so we can check
  // that every observation lands in the NUMA node's CPU set.
  const size_t numWorkers = cpuIds.size();
  std::vector<AsyncValueRef<int>> observed;
  observed.reserve(numWorkers);
  for (size_t workerID = 0; workerID < numWorkers; ++workerID) {
    observed.emplace_back(AsyncValueRef<int>::allocate(*h.runtime));
    AsyncValueRef<int> &slot = observed.back();
    WorkItem probe([slot = slot.copy()]() mutable {
      // sched_getcpu may return -1 on failure; store as int for inspection.
      slot.copy().emplace(::sched_getcpu());
    });
    h.workQueue->addTask(std::move(probe), static_cast<int>(workerID));
  }

  for (AsyncValueRef<int> &slot : observed)
    await(slot);

  for (size_t i = 0; i < observed.size(); ++i) {
    int cpu = observed[i].get();
    ASSERT_GE(cpu, 0) << "sched_getcpu failed for worker " << i;
    EXPECT_NE(allowedCpus.find(static_cast<size_t>(cpu)), allowedCpus.end())
        << "worker " << i << " ran on CPU " << cpu
        << " which is not in the NUMA partition";
  }
}
#endif // defined(__linux__)

TEST(WorkQueueTest, UnpartitionedReportsNoPartition) {
  RuntimeOptions options;
  options.numThreads = 2;
  options.mainWillDonate = false;
  RuntimeRef runtime = getOrCreateRuntime(RuntimeSource::Test, options);
  WorkQueue *workQueue = runtime->getWorkQueue();

  EXPECT_EQ(workQueue->getCpuIds().size(), 2u);
  EXPECT_EQ(workQueue->getNumaNode(), M::kAnyNumaNode);
}

TEST(WorkQueueTest, PartitionedRejectsUnknownNumaNode) {
  const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
  if (topologyOr.isError())
    GTEST_SKIP() << "host has no NUMA topology available";

  RuntimeOptions options;
  options.withSingleThreaded();
  RuntimeRef runtime = getOrCreateRuntime(RuntimeSource::Test, options);
  CompactRuntimePtr ptr = runtime->getCompactPtr();

  // Pick a NUMA node id that definitely doesn't exist on this host.
  EXPECT_DEATH_IF_SUPPORTED(
      {
        auto wq = createPartitionedThreadPoolWorkQueue(
            ptr, /*numaNode=*/INT_MAX, std::chrono::microseconds(200),
            "NumaTest");
        (void)wq;
      },
      "no CPUs for requested NUMA node");
}

} // namespace

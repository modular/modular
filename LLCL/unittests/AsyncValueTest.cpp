//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/Semaphore.h"

#include "gtest/gtest.h"

using namespace M::LLCL;

std::unique_ptr<Runtime>
TestRuntime(std::function<std::unique_ptr<WorkQueue>()> createWorkQueue) {
  std::unique_ptr<Allocator> allocator =
      createProfilingAllocator(createMallocAllocator());
  std::unique_ptr<WorkQueue> workQueue = createWorkQueue();
  auto runtime =
      std::make_unique<Runtime>(std::move(allocator), std::move(workQueue));
  AsyncValue::registerType<int>();
  AsyncValue::registerType<char>();
  return std::move(runtime);
}

std::unique_ptr<Runtime> TestSingleThreadedRuntime() {
  return TestRuntime(createSingleThreadWorkQueue);
}

std::unique_ptr<Runtime> TestThreadPoolRuntime() {
  // We'll use an arbitrary but large number of threads to encourage
  // contention to tickle concurrency issues.
  return TestRuntime(
      []() { return createThreadPoolWorkQueue(/*numThreads=*/24); });
  createProfilingAllocator(createMallocAllocator());
}

//===----------------------------------------------------------------------===//
// Idiomatic async producer/consumer
//===----------------------------------------------------------------------===//

AsyncValueRef<int> typedProducer(Runtime &runtime) {
  auto result = AsyncValueRef<int>::allocate(runtime);
  addTask(runtime,
          [result = result.copy()]() mutable { std::move(result).emplace(0); });
  return result;
}

int typedConsumer(AsyncValueRef<int> result) { return *result + 1; }

TEST(AsyncValue, TypedProducerConsumer) {
  auto runtime = TestThreadPoolRuntime();
  int i = -1;
  Semaphore consumed;
  AsyncValueRef<int> result = typedProducer(*runtime);
  std::move(result).andThenSync([&i, &consumed](AsyncValueRef<int> &&result) {
    i = typedConsumer(std::move(result));
    consumed.post();
  });
  consumed.wait();
  ASSERT_EQ(i, 1);
}

AnyAsyncValueRef anyProducer(Runtime &runtime) {
  auto result = AnyAsyncValueRef::allocate<int>(runtime);
  addTask(runtime, [result = result.copy()]() mutable {
    std::move(result).emplace<int>(0);
  });
  return result;
}

int anyConsumer(AnyAsyncValueRef result) { return result.get<int>() + 1; }

TEST(AsyncValue, AnyProducerConsumer) {
  auto runtime = TestThreadPoolRuntime();
  int i = -1;
  Semaphore consumed;
  AnyAsyncValueRef result = anyProducer(*runtime);
  std::move(result).andThenSync([&i, &consumed](AnyAsyncValueRef &&result) {
    i = anyConsumer(std::move(result));
    consumed.post();
  });
  consumed.wait();
  ASSERT_EQ(i, 1);
}

//===----------------------------------------------------------------------===//
// No stray references
//===----------------------------------------------------------------------===//

TEST(AsyncValue, SyncConsuming) {
  auto runtime = TestSingleThreadedRuntime();
  auto ref = AnyAsyncValueRef::allocate<int>(*runtime);
  ref.andThenSync([r = ref.copy()]() {
    // At this point r is the only remaining reference due to the use
    // of AsyncValue::emplace below.
    ASSERT_EQ(r.getPointer()->getRefCount(), 1u);
  });
  std::move(ref).emplace<int>(0);
}

TEST(AsyncValue, AsyncShutdown) {
  auto runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  {
    // With single-thredaed work queue, the closure passed to andThenAsync is
    // invoked during the `shutdown()` execution. The test confirms that the
    // reference captured with copy() is the only available reference to the
    // async value when the closure is executed.
    auto ref = AnyAsyncValueRef::allocate<int>(*runtime);
    ref.andThenAsync([&shuttingDown, r = ref.copy()]() {
      ASSERT_TRUE(shuttingDown);
      ASSERT_EQ(r.getPointer()->getRefCount(), 1u);
    });
    std::move(ref).emplace<int>(0);
  }
  shuttingDown = true;
  // Force continuation to run before runtime destroyed.
  runtime->getWorkQueue()->shutdown();
}

TEST(AsyncValue, AsyncThreaded) {
  auto runtime = TestThreadPoolRuntime();
  Semaphore outerRefDestroyed;
  Semaphore continuationTriggered;
  auto ref = AnyAsyncValueRef::allocate<int>(*runtime);
  ref.andThenAsync(
      [r = ref.copy(), &outerRefDestroyed, &continuationTriggered]() {
        continuationTriggered.post();
        outerRefDestroyed.wait();
        ASSERT_EQ(r.getPointer()->getRefCount(), 1u);
      });
  std::move(ref).emplace<int>(0);
  continuationTriggered.wait();
  outerRefDestroyed.post();
}

TEST(AsyncValue, AsyncNoAdditionalRefs) {
  auto runtime = TestSingleThreadedRuntime();
  auto ref = AnyAsyncValueRef::allocate<int>(*runtime);
  ASSERT_EQ(ref.getPointer()->getRefCount(), 1u);
  auto done = AnyAsyncValueRef::allocate<Chain>(*runtime);
  ref.andThenAsync([&ref, done = done.copy()]() mutable {
    // No additional capture of ref has been taken.
    ASSERT_EQ(ref.getPointer()->getRefCount(), 1u);
    std::move(done).emplace<Chain>();
  });
  ref.copy().emplace<int>(0);
  await(done);
}

//===----------------------------------------------------------------------===//
// andThen{Sync,Async}
//===----------------------------------------------------------------------===//

TEST(AsyncValue, TupleSync) {
  auto runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  // With `andThenSync`, the completion function is executed as soon as the
  // last async value is fulfilled, which is `ref2->emplace` in this test.
  // So the original ref1 and ref2 are still valid, and `shuttingDown` is still
  // `false`.
  auto ref1 = AnyAsyncValueRef::allocate<int>(*runtime);
  auto ref2 = AnyAsyncValueRef::allocate<char>(*runtime);
  andThenSync(std::make_tuple(ref1.copy(), ref2.copy()),
              [&shuttingDown](AnyAsyncValueRef ref1, AnyAsyncValueRef ref2) {
                // Confirm that the closure is running after the original
                // `ref` is destroyed.
                ASSERT_EQ(ref1.getPointer()->getRefCount(), 1u);
                ASSERT_EQ(ref2.getPointer()->getRefCount(), 1u);
                ASSERT_FALSE(shuttingDown);
              });
  std::move(ref1).emplace<int>(0);
  std::move(ref2).emplace<char>('a');
  shuttingDown = true;
  // Force continuation to run before runtime destroyed.
  runtime->getWorkQueue()->shutdown();
}

TEST(AsyncValue, TupleAsync) {
  auto runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  {
    // With single-threaded work queue, the closure passed to andThenAsync is
    // invoked during the `shutdown()` execution. The test confirms that the
    // reference captured with copy() is the only available reference to the
    // async value when the closure is executed.
    auto ref1 = AnyAsyncValueRef::allocate<int>(*runtime);
    auto ref2 = AnyAsyncValueRef::allocate<char>(*runtime);
    andThenAsync(std::make_tuple(ref1.copy(), ref2.copy()),
                 [&shuttingDown](AnyAsyncValueRef ref1, AnyAsyncValueRef ref2) {
                   // Confirm that the closure is running after the original
                   // `ref` is destroyed.
                   ASSERT_EQ(ref1.getPointer()->getRefCount(), 1u);
                   ASSERT_EQ(ref2.getPointer()->getRefCount(), 1u);
                   ASSERT_TRUE(shuttingDown);
                 });
    std::move(ref1).emplace<int>(0);
    std::move(ref2).emplace<char>('a');
  }
  shuttingDown = true;
  // Force continuation to run before runtime destroyed.
  runtime->getWorkQueue()->shutdown();
}

TEST(AsyncValue, ArrayCopyingSync) {
  auto runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs[0].copy().emplace<int>(0);
  refs[1].copy().emplace<int>(0);
  // `refs` is copied, so each element has refcount 2 when the completion
  // function is executed. `shuttingDown` is still false.
  andThenSyncCopying(llvm::ArrayRef(refs),
                     [&shuttingDown](llvm::ArrayRef<AnyAsyncValueRef> elts) {
                       ASSERT_EQ(elts[0].getPointer()->getRefCount(), 2u);
                       ASSERT_EQ(elts[1].getPointer()->getRefCount(), 2u);
                       ASSERT_FALSE(shuttingDown);
                     });
  shuttingDown = true;
  // Force continuation to run before runtime destroyed.
  runtime->getWorkQueue()->shutdown();
}

TEST(AsyncValue, ArrayCopyingAsync) {
  auto runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  {
    llvm::SmallVector<AnyAsyncValueRef> refs;
    refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
    refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
    refs[0].copy().emplace<int>(0);
    refs[1].copy().emplace<int>(0);
    // `refs` is copied, but the completion function is executed when the
    // work queue is shutdown. At that time only the copied references are
    // valid but the original `refs` is expired.
    andThenAsyncCopying(llvm::ArrayRef(refs),
                        [&shuttingDown](llvm::ArrayRef<AnyAsyncValueRef> elts) {
                          ASSERT_EQ(elts[0].getPointer()->getRefCount(), 1u);
                          ASSERT_EQ(elts[1].getPointer()->getRefCount(), 1u);
                          ASSERT_TRUE(shuttingDown);
                        });
  }
  shuttingDown = true;
  // Force continuation to run before runtime destroyed.
  runtime->getWorkQueue()->shutdown();
}

TEST(AsyncValue, ArrayMovingSync) {
  auto runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs[0].copy().emplace<int>(0);
  refs[1].copy().emplace<int>(0);
  // `refs` is moved, so each element has refcount 1 when the completion
  // function is executed. `shuttingDown` is still false.
  andThenSyncMoving(
      llvm::MutableArrayRef(refs),
      [&shuttingDown](llvm::MutableArrayRef<AnyAsyncValueRef> elts) {
        ASSERT_EQ(elts[0].getPointer()->getRefCount(), 1u);
        ASSERT_EQ(elts[1].getPointer()->getRefCount(), 1u);
        ASSERT_FALSE(shuttingDown);
      });
  shuttingDown = true;
  // Force continuation to run before runtime destroyed.
  runtime->getWorkQueue()->shutdown();
}

TEST(AsyncValue, ArrayMovingAsync) {
  auto runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  {
    llvm::SmallVector<AnyAsyncValueRef> refs;
    refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
    refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
    refs[0].copy().emplace<int>(0);
    refs[1].copy().emplace<int>(0);
    // With the async version, the completion function is executed with the
    // work queue shuts down. `refs` is moved, so each element has refcount 1
    // when the completion function is executed.
    andThenAsyncMoving(
        llvm::MutableArrayRef(refs),
        [&shuttingDown](llvm::MutableArrayRef<AnyAsyncValueRef> elts) {
          ASSERT_EQ(elts[0].getPointer()->getRefCount(), 1u);
          ASSERT_EQ(elts[1].getPointer()->getRefCount(), 1u);
          ASSERT_TRUE(shuttingDown);
        });
  }
  shuttingDown = true;
  // Force continuation to run before runtime destroyed.
  runtime->getWorkQueue()->shutdown();
}

TEST(AsyncValue, Stress) {
  auto runtime = TestThreadPoolRuntime();
  AsyncValue::registerType<int>();

  const size_t nRounds = 5;
  const size_t nValues = 500;
  // Root AsyncValue.
  auto start = AsyncValueRef<int>::allocate(*runtime);
  // Intermediate AsyncValues.
  llvm::SmallVector<llvm::SmallVector<AsyncValueRef<int>>> refs;
  for (size_t i = 0; i < nRounds; ++i) {
    refs.emplace_back();
    for (size_t j = 0; j < nValues; ++j)
      refs.back().emplace_back(AsyncValueRef<int>::allocate(*runtime));
  }
  // Final AsyncValue.
  auto finish = AsyncValueRef<Chain>::allocate(*runtime);

  // Intermediate dependencies.
  for (size_t i = 0; i < nRounds; ++i) {
    for (size_t j = 0; j < nValues; ++j) {
      const AsyncValueRef<int> &prev = i == 0 ? start : refs[i - 1][j];
      const AsyncValueRef<int> &next = refs[i][j];
      prev.copy().andThenAsync(
          [next = next.copy()](AsyncValueRef<int> &&prev) mutable {
            std::this_thread::sleep_for(std::chrono::microseconds(rand() % 50));
            std::move(next).emplace(prev.get() + 1);
          });
    }
  }

  // Final values and join condition.
  std::atomic<int> sum = 0;
  std::atomic<size_t> waiting = nValues;
  for (size_t j = 0; j < nValues; ++j) {
    std::move(refs[nRounds - 1][j])
        .andThenAsync([&sum, &waiting, finish = finish.copy()](
                          AsyncValueRef<int> &&prev) mutable {
          sum += prev.get();
          if (--waiting == 0)
            std::move(finish).emplace();
        });
  }

  std::move(start).emplace(1);
  await(finish);

  ASSERT_EQ(sum, (nRounds + 1) * nValues);
}

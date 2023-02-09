//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Some unit tests for AsyncValue and friends.
//
// See also GraphRT/lib/Primitives/TestPrimitives.cpp
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/Semaphore.h"

#include "gtest/gtest.h"

using namespace M::LLCL;

enum WorkQueueType { kSingleThread = 0, kThreadPool = 1 };

class AsyncValueTest : public testing::TestWithParam<WorkQueueType> {
protected:
  std::unique_ptr<Runtime> createRuntime() {
    AsyncValue::registerType<int>();
    AsyncValue::registerType<char>();
    AsyncValue::registerType<size_t>();
    std::unique_ptr<Allocator> allocator =
        createLeakCheckAllocator(createMallocAllocator());
    // We'll deliberately oversubscribe threads to tickle more concurrency
    // issues.
    size_t numThreads = std::thread::hardware_concurrency() * 2;
    std::unique_ptr<WorkQueue> workQueue =
        GetParam() == kThreadPool ? createThreadPoolWorkQueue(numThreads)
                                  : createSingleThreadWorkQueue();
    return std::make_unique<Runtime>(std::move(allocator),
                                     std::move(workQueue));
  }
};

INSTANTIATE_TEST_SUITE_P(ManyRuntimes, AsyncValueTest,
                         testing::Values(kSingleThread, kThreadPool));

//===----------------------------------------------------------------------===//
// Idiomatic async producer/consumer
//===----------------------------------------------------------------------===//

AsyncValueRef<int> typedProducer(Runtime &runtime) {
  auto result = AsyncValueRef<int>::allocate(runtime);
  addTask(runtime,
          [result = result.copy()]() mutable { std::move(result).emplace(1); });
  return result;
}

int typedConsumer(AsyncValueRef<int> result) { return *result + 1; }

TEST_P(AsyncValueTest, TypedProducerConsumer) {
  auto runtime = createRuntime();
  AsyncValueRef<int> finished = AsyncValueRef<int>::allocate(*runtime);
  AsyncValueRef<int> result = typedProducer(*runtime);
  std::move(result).andThenSync(
      [finished = finished.copy()](AsyncValueRef<int> &&result) mutable {
        std::move(finished).emplace(typedConsumer(std::move(result)));
      });
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

AnyAsyncValueRef anyProducer(Runtime &runtime) {
  auto result = AnyAsyncValueRef::allocate<int>(runtime);
  addTask(runtime, [result = result.copy()]() mutable {
    std::move(result).emplace<int>(1);
  });
  return result;
}

int anyConsumer(AnyAsyncValueRef result) { return result.get<int>() + 1; }

TEST_P(AsyncValueTest, AnyProducerConsumer) {
  auto runtime = createRuntime();
  auto finished = AsyncValueRef<int>::allocate(*runtime);
  AnyAsyncValueRef result = anyProducer(*runtime);
  std::move(result).andThenSync(
      [finished = finished.copy()](AnyAsyncValueRef &&result) mutable {
        std::move(finished).emplace(anyConsumer(std::move(result)));
      });
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

//===----------------------------------------------------------------------===//
// No stray references
//===----------------------------------------------------------------------===//

TEST_P(AsyncValueTest, SyncConsuming) {
  auto runtime = createRuntime();
  auto finished = AsyncValueRef<int>::allocate(*runtime);
  auto ref = AnyAsyncValueRef::allocate<int>(*runtime);
  ref.andThenSync([ref = ref.copy(), finished = finished.copy()]() mutable {
    // At this point r is the only remaining reference due to the use
    // of AsyncValue::emplace below.
    EXPECT_EQ(ref.getPointer()->getRefCount(), 1u);
    EXPECT_EQ(ref.get<int>(), 1);
    std::move(finished).emplace(2);
  });
  EXPECT_EQ(ref.getPointer()->getRefCount(), 2u);
  std::move(ref).emplace<int>(1);
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

TEST_P(AsyncValueTest, AsyncConsuming) {
  auto runtime = createRuntime();
  auto finished = AsyncValueRef<int>::allocate(*runtime);
  auto ref = AnyAsyncValueRef::allocate<int>(*runtime);
  ref.andThenAsync([ref = ref.copy(), finished = finished.copy()]() mutable {
    // At this point r is the only remaining reference due to the use
    // of AsyncValue::emplace below.
    EXPECT_EQ(ref.getPointer()->getRefCount(), 1u);
    EXPECT_EQ(ref.get<int>(), 1);
    std::move(finished).emplace(2);
  });
  EXPECT_EQ(ref.getPointer()->getRefCount(), 2u);
  std::move(ref).emplace<int>(1);
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

//===----------------------------------------------------------------------===//
// Waiters run off stack
//===----------------------------------------------------------------------===//

TEST_P(AsyncValueTest, EmplacingFromTask) {
  auto runtime = createRuntime();
  auto finished = AsyncValueRef<int>::allocate(*runtime);
  Semaphore canRun;
  addTask(*runtime, [&canRun, runtime = runtime.get(),
                     finished = finished.copy()]() mutable {
    // Run the test inside an LLCL task. Waiter can be scheduled on the
    // same thread.
    auto ref = AsyncValueRef<Chain>::allocate(*runtime);
    ref.andThenSync([&canRun, finished = finished.copy()]() mutable {
      canRun.wait();
      std::move(finished).emplace(1);
    });
    // We'll deadlock if the continuation is run now.
    std::move(ref).emplace();
  });
  canRun.post();
  await(finished);
  EXPECT_EQ(finished.get(), 1);
}

TEST_P(AsyncValueTest, EmplaceOnForeignThread) {
  if (GetParam() != kThreadPool)
    // Can only observe this behaviour with the thread pool workqueue.
    return;

  // Run the test inside the main (ie 'foreign') thread. Waiter will be
  // scheduled as an LLCL task.
  auto runtime = createRuntime();
  auto finished = AsyncValueRef<int>::allocate(*runtime);
  Semaphore canRun;
  auto ref = AsyncValueRef<Chain>::allocate(*runtime);
  ref.andThenSync([&canRun, finished = finished.copy()]() mutable {
    canRun.wait();
    std::move(finished).emplace(1);
  });
  // We'll deadlock if the continuation is run now.
  std::move(ref).emplace();
  canRun.post();
  await(finished);
  EXPECT_EQ(finished.get(), 1);
}

//===----------------------------------------------------------------------===//
// Special andThen{Sync,Async}s from Algorithms
//===----------------------------------------------------------------------===//

TEST_P(AsyncValueTest, TupleAndThenSync) {
  auto runtime = createRuntime();
  auto finished = AsyncValueRef<int>::allocate(*runtime);
  auto ref1 = AnyAsyncValueRef::allocate<int>(*runtime);
  auto ref2 = AnyAsyncValueRef::allocate<char>(*runtime);
  andThenSync(std::make_tuple(ref1.copy(), ref2.copy()),
              [finished = finished.copy()](AnyAsyncValueRef ref1,
                                           AnyAsyncValueRef ref2) mutable {
                // Confirm that the closure is running after the original
                // `ref` is destroyed.
                EXPECT_EQ(ref1.getPointer()->getRefCount(), 1u);
                EXPECT_EQ(ref2.getPointer()->getRefCount(), 1u);
                EXPECT_EQ(ref1.get<int>(), 1);
                EXPECT_EQ(ref2.get<char>(), 'a');
                std::move(finished).emplace(2);
              });
  std::move(ref1).emplace<int>(1);
  std::move(ref2).emplace<char>('a');
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

TEST_P(AsyncValueTest, TupleAndThenAsync) {
  auto runtime = createRuntime();
  auto finished = AsyncValueRef<int>::allocate(*runtime);
  auto ref1 = AnyAsyncValueRef::allocate<int>(*runtime);
  auto ref2 = AnyAsyncValueRef::allocate<char>(*runtime);
  andThenAsync(std::make_tuple(ref1.copy(), ref2.copy()),
               [finished = finished.copy()](AnyAsyncValueRef ref1,
                                            AnyAsyncValueRef ref2) mutable {
                 // Confirm that the closure is running after the original
                 // `ref` is destroyed.
                 EXPECT_EQ(ref1.getPointer()->getRefCount(), 1u);
                 EXPECT_EQ(ref2.getPointer()->getRefCount(), 1u);
                 EXPECT_EQ(ref1.get<int>(), 1);
                 EXPECT_EQ(ref2.get<char>(), 'a');
                 std::move(finished).emplace(2);
               });
  std::move(ref1).emplace<int>(1);
  std::move(ref2).emplace<char>('a');
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

TEST_P(AsyncValueTest, ArrayCopyingSync) {
  auto runtime = createRuntime();
  auto finished = AsyncValueRef<int>::allocate(*runtime);
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs[0].copy().emplace<int>(1);
  refs[1].copy().emplace<int>(2);
  andThenSyncCopying(llvm::ArrayRef(refs),
                     [finished = finished.copy()](
                         llvm::ArrayRef<AnyAsyncValueRef> elts) mutable {
                       // `refs` is copied, so each element has refcount 2 when
                       // the completion function is executed.
                       EXPECT_EQ(elts[0].getPointer()->getRefCount(), 2u);
                       EXPECT_EQ(elts[1].getPointer()->getRefCount(), 2u);
                       EXPECT_EQ(elts[0].get<int>(), 1);
                       EXPECT_EQ(elts[1].get<int>(), 2);
                       std::move(finished).emplace(3);
                     });
  await(finished);
  EXPECT_EQ(finished.get(), 3);
}

TEST_P(AsyncValueTest, ArrayCopyingAsync) {
  auto runtime = createRuntime();
  auto finished = AsyncValueRef<int>::allocate(*runtime);
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs[0].copy().emplace<int>(1);
  refs[1].copy().emplace<int>(2);
  andThenAsyncCopying(llvm::ArrayRef(refs),
                      [finished = finished.copy()](
                          llvm::ArrayRef<AnyAsyncValueRef> elts) mutable {
                        // `refs` is copied, so each element has refcount 2 when
                        // the completion function is executed.
                        EXPECT_EQ(elts[0].getPointer()->getRefCount(), 2u);
                        EXPECT_EQ(elts[1].getPointer()->getRefCount(), 2u);
                        EXPECT_EQ(elts[0].get<int>(), 1);
                        EXPECT_EQ(elts[1].get<int>(), 2);
                        std::move(finished).emplace(3);
                      });
  await(finished);
  EXPECT_EQ(finished.get(), 3);
}

TEST_P(AsyncValueTest, ArrayMovingSync) {
  auto runtime = createRuntime();
  auto finished = AsyncValueRef<int>::allocate(*runtime);
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs[0].copy().emplace<int>(1);
  refs[1].copy().emplace<int>(2);
  andThenSyncMoving(llvm::MutableArrayRef(refs),
                    [finished = finished.copy()](
                        llvm::MutableArrayRef<AnyAsyncValueRef> elts) mutable {
                      // `refs` is moved, so each element has refcount 1 when
                      // the completion function is executed.
                      EXPECT_EQ(elts[0].getPointer()->getRefCount(), 1u);
                      EXPECT_EQ(elts[1].getPointer()->getRefCount(), 1u);
                      EXPECT_EQ(elts[0].get<int>(), 1);
                      EXPECT_EQ(elts[1].get<int>(), 2);
                      std::move(finished).emplace(3);
                    });
  await(finished);
  EXPECT_EQ(finished.get(), 3);
}

TEST_P(AsyncValueTest, ArrayMovingAsync) {
  auto runtime = createRuntime();
  auto finished = AsyncValueRef<int>::allocate(*runtime);
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs[0].copy().emplace<int>(1);
  refs[1].copy().emplace<int>(2);
  andThenAsyncMoving(llvm::MutableArrayRef(refs),
                     [finished = finished.copy()](
                         llvm::MutableArrayRef<AnyAsyncValueRef> elts) mutable {
                       // `refs` is moved, so each element has refcount 1 when
                       // the completion function is executed.
                       EXPECT_EQ(elts[0].getPointer()->getRefCount(), 1u);
                       EXPECT_EQ(elts[1].getPointer()->getRefCount(), 1u);
                       EXPECT_EQ(elts[0].get<int>(), 1);
                       EXPECT_EQ(elts[1].get<int>(), 2);
                       std::move(finished).emplace(3);
                     });
  await(finished);
  EXPECT_EQ(finished.get(), 3);
}

//===----------------------------------------------------------------------===//
// Stress tests
//===----------------------------------------------------------------------===//

TEST_P(AsyncValueTest, Stress) {
  auto runtime = createRuntime();

  const size_t nRounds = 5;
  const size_t nValues = 500;
  // Root AsyncValue.
  auto start = AsyncValueRef<size_t>::allocate(*runtime);
  // Intermediate AsyncValues.
  llvm::SmallVector<llvm::SmallVector<AsyncValueRef<size_t>>> refs;
  for (size_t i = 0; i < nRounds; ++i) {
    refs.emplace_back();
    for (size_t j = 0; j < nValues; ++j)
      refs.back().emplace_back(AsyncValueRef<size_t>::allocate(*runtime));
  }
  // Final AsyncValue.
  auto finish = AsyncValueRef<Chain>::allocate(*runtime);

  // Intermediate dependencies.
  for (size_t i = 0; i < nRounds; ++i) {
    for (size_t j = 0; j < nValues; ++j) {
      const AsyncValueRef<size_t> &prev = i == 0 ? start : refs[i - 1][j];
      const AsyncValueRef<size_t> &next = refs[i][j];
      prev.copy().andThenAsync(
          [next = next.copy()](AsyncValueRef<size_t> &&prev) mutable {
            std::this_thread::sleep_for(std::chrono::microseconds(rand() % 50));
            std::move(next).emplace(prev.get() + 1);
          });
    }
  }

  // Final values and join condition.
  std::atomic<size_t> sum = 0;
  std::atomic<size_t> waiting = nValues;
  for (size_t j = 0; j < nValues; ++j) {
    std::move(refs[nRounds - 1][j])
        .andThenAsync([&sum, &waiting, finish = finish.copy()](
                          AsyncValueRef<size_t> &&prev) mutable {
          sum += prev.get();
          if (--waiting == 0)
            std::move(finish).emplace();
        });
  }

  std::move(start).emplace(1);
  await(finish);

  EXPECT_EQ(sum, (nRounds + 1) * nValues);
}

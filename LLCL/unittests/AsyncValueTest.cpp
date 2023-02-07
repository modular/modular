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
  AsyncValue::registerType<size_t>();
  return std::move(runtime);
}

std::unique_ptr<Runtime> TestSingleThreadedRuntime() {
  return TestRuntime(createSingleThreadWorkQueue);
}

std::unique_ptr<Runtime> TestThreadPoolRuntime() {
  // We'll deliberately oversubscribe threads to tickle more concurrency
  // issues.
  return TestRuntime([]() {
    return createThreadPoolWorkQueue(
        /*numThreads=*/std::thread::hardware_concurrency() * 2);
  });
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
  AsyncValueRef<Chain> finished = AsyncValueRef<Chain>::allocate(*runtime);
  int i = -1;
  AsyncValueRef<int> result = typedProducer(*runtime);
  std::move(result).andThenSync(
      [&i, finished = finished.copy()](AsyncValueRef<int> &&result) mutable {
        i = typedConsumer(std::move(result));
        std::move(finished).emplace();
      });
  await(finished);
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
  AnyAsyncValueRef finished = AnyAsyncValueRef::allocate<Chain>(*runtime);
  int i = -1;
  AnyAsyncValueRef result = anyProducer(*runtime);
  std::move(result).andThenSync(
      [&i, finished = finished.copy()](AnyAsyncValueRef &&result) mutable {
        i = anyConsumer(std::move(result));
        std::move(finished).emplace<Chain>();
      });
  await(finished);
  ASSERT_EQ(i, 1);
}

//===----------------------------------------------------------------------===//
// No stray references
//===----------------------------------------------------------------------===//

TEST(AsyncValue, SyncConsuming) {
  auto runtime = TestSingleThreadedRuntime();
  AsyncValueRef<Chain> finished = AsyncValueRef<Chain>::allocate(*runtime);
  auto ref = AnyAsyncValueRef::allocate<int>(*runtime);
  ref.andThenSync([r = ref.copy(), finished = finished.copy()]() mutable {
    // At this point r is the only remaining reference due to the use
    // of AsyncValue::emplace below.
    ASSERT_EQ(r.getPointer()->getRefCount(), 1u);
    std::move(finished).emplace();
  });
  ASSERT_EQ(ref.getPointer()->getRefCount(), 2u);
  std::move(ref).emplace<int>(0);
  await(finished);
}

TEST(AsyncValue, AsyncConsuming) {
  auto runtime = TestThreadPoolRuntime();
  AsyncValueRef<Chain> finished = AsyncValueRef<Chain>::allocate(*runtime);
  auto ref = AnyAsyncValueRef::allocate<int>(*runtime);
  ref.andThenAsync([ref = ref.copy(), finished = finished.copy()]() mutable {
    // At this point r is the only remaining reference due to the use
    // of AsyncValue::emplace below.
    ASSERT_EQ(ref.getPointer()->getRefCount(), 1u);
    std::move(finished).emplace();
  });
  ASSERT_EQ(ref.getPointer()->getRefCount(), 2u);
  std::move(ref).emplace<int>(0);
  await(finished);
}

//===----------------------------------------------------------------------===//
// andThen{Sync,Async}
//===----------------------------------------------------------------------===//

TEST(AsyncValue, TupleSync) {
  auto runtime = TestSingleThreadedRuntime();
  AsyncValueRef<Chain> finished = AsyncValueRef<Chain>::allocate(*runtime);
  bool shuttingDown = false;
  // With `andThenSync`, the completion function is executed as soon as the
  // last async value is fulfilled, which is `ref2->emplace` in this test.
  // So the original ref1 and ref2 are still valid, and `shuttingDown` is still
  // `false`.
  auto ref1 = AnyAsyncValueRef::allocate<int>(*runtime);
  auto ref2 = AnyAsyncValueRef::allocate<char>(*runtime);
  andThenSync(std::make_tuple(ref1.copy(), ref2.copy()),
              [&shuttingDown, finished = finished.copy()](
                  AnyAsyncValueRef ref1, AnyAsyncValueRef ref2) mutable {
                // Confirm that the closure is running after the original
                // `ref` is destroyed.
                ASSERT_EQ(ref1.getPointer()->getRefCount(), 1u);
                ASSERT_EQ(ref2.getPointer()->getRefCount(), 1u);
                ASSERT_FALSE(shuttingDown);
                std::move(finished).emplace();
              });
  std::move(ref1).emplace<int>(0);
  std::move(ref2).emplace<char>('a');
  shuttingDown = true;
  await(finished);
}

TEST(AsyncValue, TupleAsync) {
  auto runtime = TestSingleThreadedRuntime();
  AsyncValueRef<Chain> finished = AsyncValueRef<Chain>::allocate(*runtime);
  bool shuttingDown = false;
  {
    // With single-threaded work queue, the closure passed to andThenAsync is
    // invoked during the `shutdown()` execution. The test confirms that the
    // reference captured with copy() is the only available reference to the
    // async value when the closure is executed.
    auto ref1 = AnyAsyncValueRef::allocate<int>(*runtime);
    auto ref2 = AnyAsyncValueRef::allocate<char>(*runtime);
    andThenAsync(std::make_tuple(ref1.copy(), ref2.copy()),
                 [&shuttingDown, finished = finished.copy()](
                     AnyAsyncValueRef ref1, AnyAsyncValueRef ref2) mutable {
                   // Confirm that the closure is running after the original
                   // `ref` is destroyed.
                   ASSERT_EQ(ref1.getPointer()->getRefCount(), 1u);
                   ASSERT_EQ(ref2.getPointer()->getRefCount(), 1u);
                   ASSERT_TRUE(shuttingDown);
                   std::move(finished).emplace();
                 });
    std::move(ref1).emplace<int>(0);
    std::move(ref2).emplace<char>('a');
  }
  shuttingDown = true;
  await(finished);
}

TEST(AsyncValue, ArrayCopyingSync) {
  auto runtime = TestSingleThreadedRuntime();
  AsyncValueRef<Chain> finished = AsyncValueRef<Chain>::allocate(*runtime);
  bool shuttingDown = false;
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs[0].copy().emplace<int>(0);
  refs[1].copy().emplace<int>(0);
  // `refs` is copied, so each element has refcount 2 when the completion
  // function is executed. `shuttingDown` is still false.
  andThenSyncCopying(llvm::ArrayRef(refs),
                     [&shuttingDown, finished = finished.copy()](
                         llvm::ArrayRef<AnyAsyncValueRef> elts) mutable {
                       ASSERT_EQ(elts[0].getPointer()->getRefCount(), 2u);
                       ASSERT_EQ(elts[1].getPointer()->getRefCount(), 2u);
                       ASSERT_FALSE(shuttingDown);
                       std::move(finished).emplace();
                     });
  shuttingDown = true;
  await(finished);
}

TEST(AsyncValue, ArrayCopyingAsync) {
  auto runtime = TestSingleThreadedRuntime();
  AsyncValueRef<Chain> finished = AsyncValueRef<Chain>::allocate(*runtime);
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
                        [&shuttingDown, finished = finished.copy()](
                            llvm::ArrayRef<AnyAsyncValueRef> elts) mutable {
                          ASSERT_EQ(elts[0].getPointer()->getRefCount(), 1u);
                          ASSERT_EQ(elts[1].getPointer()->getRefCount(), 1u);
                          ASSERT_TRUE(shuttingDown);
                          std::move(finished).emplace();
                        });
  }
  shuttingDown = true;
  await(finished);
}

TEST(AsyncValue, ArrayMovingSync) {
  auto runtime = TestSingleThreadedRuntime();
  AsyncValueRef<Chain> finished = AsyncValueRef<Chain>::allocate(*runtime);
  bool shuttingDown = false;
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*runtime));
  refs[0].copy().emplace<int>(0);
  refs[1].copy().emplace<int>(0);
  // `refs` is moved, so each element has refcount 1 when the completion
  // function is executed. `shuttingDown` is still false.
  andThenSyncMoving(llvm::MutableArrayRef(refs),
                    [&shuttingDown, finished = finished.copy()](
                        llvm::MutableArrayRef<AnyAsyncValueRef> elts) mutable {
                      ASSERT_EQ(elts[0].getPointer()->getRefCount(), 1u);
                      ASSERT_EQ(elts[1].getPointer()->getRefCount(), 1u);
                      ASSERT_FALSE(shuttingDown);
                      std::move(finished).emplace();
                    });
  shuttingDown = true;
  await(finished);
}

TEST(AsyncValue, ArrayMovingAsync) {
  auto runtime = TestSingleThreadedRuntime();
  AsyncValueRef<Chain> finished = AsyncValueRef<Chain>::allocate(*runtime);
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
        [&shuttingDown, finished = finished.copy()](
            llvm::MutableArrayRef<AnyAsyncValueRef> elts) mutable {
          ASSERT_EQ(elts[0].getPointer()->getRefCount(), 1u);
          ASSERT_EQ(elts[1].getPointer()->getRefCount(), 1u);
          ASSERT_TRUE(shuttingDown);
          std::move(finished).emplace();
        });
  }
  shuttingDown = true;
  await(finished);
}

//===----------------------------------------------------------------------===//
// Stress tests
//===----------------------------------------------------------------------===//

TEST(AsyncValue, Stress) {
  auto runtime = TestThreadPoolRuntime();

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

  ASSERT_EQ(sum, (nRounds + 1) * nValues);
}

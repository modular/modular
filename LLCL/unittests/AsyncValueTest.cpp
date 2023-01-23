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

Runtime &TestSingleThreadedRuntime() {
  static Runtime *runtime = []() {
    std::unique_ptr<Allocator> allocator =
        createProfilingAllocator(createMallocAllocator());
    std::unique_ptr<WorkQueue> workQueue = createSingleThreadWorkQueue();
    runtime = new Runtime(std::move(allocator), std::move(workQueue));
    AsyncValue::registerType<int>();
    AsyncValue::registerType<char>();
    return runtime;
  }();
  return *runtime;
}

Runtime &TestThreadPoolRuntime() {
  static Runtime *runtime = []() {
    std::unique_ptr<Allocator> allocator =
        createProfilingAllocator(createMallocAllocator());
    std::unique_ptr<WorkQueue> workQueue =
        createThreadPoolWorkQueue(/*numThreads=*/10);
    runtime = new Runtime(std::move(allocator), std::move(workQueue));
    AsyncValue::registerType<int>();
    AsyncValue::registerType<char>();
    return runtime;
  }();
  return *runtime;
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
  Runtime &runtime = TestThreadPoolRuntime();
  int i = -1;
  Semaphore consumed;
  AsyncValueRef<int> result = typedProducer(runtime);
  std::move(result).andThenSync([&i, &consumed](AsyncValueRef<int> &&result) {
    i = typedConsumer(std::move(result));
    consumed.post();
  });
  consumed.wait();
  ASSERT_EQ(i, 1);
}

AnyAsyncValueRef anyProducer(Runtime &runtime) {
  auto result = AsyncValue::allocate<int>(runtime);
  addTask(runtime, [result = result.copy()]() mutable {
    AsyncValue::emplace<int>(std::move(result), 0);
  });
  return result;
}

int anyConsumer(AnyAsyncValueRef result) { return result->get<int>() + 1; }

TEST(AsyncValue, AnyProducerConsumer) {
  Runtime &runtime = TestThreadPoolRuntime();
  int i = -1;
  Semaphore consumed;
  AnyAsyncValueRef result = anyProducer(runtime);
  AsyncValue::andThenSync(std::move(result),
                          [&i, &consumed](AnyAsyncValueRef &&result) {
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
  Runtime &runtime = TestSingleThreadedRuntime();
  AnyAsyncValueRef ref = AsyncValue::allocate<int>(runtime);
  ref->andThenSync([r = ref.copy()]() {
    // At this point r is the only remaining reference due to the use
    // of AsyncValue::emplace below.
    ASSERT_EQ(r->getRefCount(), 1u);
  });
  AsyncValue::emplace<int>(std::move(ref), 0);
  runtime.getWorkQueue()->shutdown();
}

TEST(AsyncValue, AsyncShutdown) {
  Runtime &runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  {
    // With single-thredaed work queue, the closure passed to andThenAsync is
    // invoked during the `shutdown()` execution. The test confirms that the
    // reference captured with copy() is the only available reference to the
    // async value when the closure is executed.
    AnyAsyncValueRef ref = AsyncValue::allocate<int>(runtime);
    ref->andThenAsync([&shuttingDown, r = ref.copy()]() {
      ASSERT_TRUE(shuttingDown);
      ASSERT_EQ(r->getRefCount(), 1u);
    });
    ref->emplace<int>(0);
  }
  shuttingDown = true;
  runtime.getWorkQueue()->shutdown();
}

TEST(AsyncValue, AsyncThreaded) {
  Runtime &runtime = TestThreadPoolRuntime();
  Semaphore outerRefDestroyed;
  Semaphore continuationTriggered;
  AnyAsyncValueRef ref = AsyncValue::allocate<int>(runtime);
  ref->andThenAsync(
      [r = ref.copy(), &outerRefDestroyed, &continuationTriggered]() {
        continuationTriggered.post();
        outerRefDestroyed.wait();
        ASSERT_EQ(r->getRefCount(), 1u);
      });
  AsyncValue::emplace<int>(std::move(ref), 0);
  continuationTriggered.wait();
  outerRefDestroyed.post();
  runtime.getWorkQueue()->shutdown();
}

TEST(AsyncValue, AsyncNoAdditionalRefs) {
  Runtime &runtime = TestSingleThreadedRuntime();
  AnyAsyncValueRef ref = AsyncValue::allocate<int>(runtime);
  ASSERT_EQ(ref->getRefCount(), 1u);
  {
    ref->andThenAsync([&ref]() {
      // No additional capture of ref has been taken.
      ASSERT_EQ(ref->getRefCount(), 1u);
    });
    ref->emplace<int>(0);
  }
  runtime.getWorkQueue()->shutdown();
}

//===----------------------------------------------------------------------===//
// andThen{Sync,Async}
//===----------------------------------------------------------------------===//

TEST(AsyncValue, TupleSync) {
  Runtime &runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  // With `andThenSync`, the completion function is executed as soon as the
  // last async value is fulfilled, which is `ref2->emplace` in this test.
  // So the original ref1 and ref2 are still valid, and `shuttingDown` is still
  // `false`.
  AnyAsyncValueRef ref1 = AsyncValue::allocate<int>(runtime);
  AnyAsyncValueRef ref2 = AsyncValue::allocate<char>(runtime);
  andThenSync(std::make_tuple(ref1.copy(), ref2.copy()),
              [&shuttingDown](AnyAsyncValueRef ref1, AnyAsyncValueRef ref2) {
                // Confirm that the closure is running after the original
                // `ref` is destroyed.
                ASSERT_EQ(ref1->getRefCount(), 2u);
                ASSERT_EQ(ref2->getRefCount(), 2u);
                ASSERT_FALSE(shuttingDown);
              });
  ref1->emplace<int>(0);
  ref2->emplace<char>('a');
  shuttingDown = true;
  runtime.getWorkQueue()->shutdown();
}

TEST(AsyncValue, TupleAsync) {
  Runtime &runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  {
    // With single-threaded work queue, the closure passed to andThenAsync is
    // invoked during the `shutdown()` execution. The test confirms that the
    // reference captured with copy() is the only available reference to the
    // async value when the closure is executed.
    AnyAsyncValueRef ref1 = AsyncValue::allocate<int>(runtime);
    AnyAsyncValueRef ref2 = AsyncValue::allocate<char>(runtime);
    andThenAsync(std::make_tuple(ref1.copy(), ref2.copy()),
                 [&shuttingDown](AnyAsyncValueRef ref1, AnyAsyncValueRef ref2) {
                   // Confirm that the closure is running after the original
                   // `ref` is destroyed.
                   ASSERT_EQ(ref1->getRefCount(), 1u);
                   ASSERT_EQ(ref2->getRefCount(), 1u);
                   ASSERT_TRUE(shuttingDown);
                 });
    ref1->emplace<int>(0);
    ref2->emplace<char>('a');
  }
  shuttingDown = true;
  runtime.getWorkQueue()->shutdown();
}

TEST(AsyncValue, ArrayCopyingSync) {
  Runtime &runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AsyncValue::allocate<int>(runtime));
  refs.emplace_back(AsyncValue::allocate<int>(runtime));
  refs[0]->emplace<int>(0);
  refs[1]->emplace<int>(0);
  // `refs` is copied, so each element has refcount 2 when the completion
  // function is executed. `shuttingDown` is still false.
  andThenSyncCopying(llvm::ArrayRef(refs),
                     [&shuttingDown](llvm::ArrayRef<AnyAsyncValueRef> elts) {
                       ASSERT_EQ(elts[0]->getRefCount(), 2u);
                       ASSERT_EQ(elts[1]->getRefCount(), 2u);
                       ASSERT_FALSE(shuttingDown);
                     });
  shuttingDown = true;
  runtime.getWorkQueue()->shutdown();
}

TEST(AsyncValue, ArrayCopyingAsync) {
  Runtime &runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  {
    llvm::SmallVector<AnyAsyncValueRef> refs;
    refs.emplace_back(AsyncValue::allocate<int>(runtime));
    refs.emplace_back(AsyncValue::allocate<int>(runtime));
    refs[0]->emplace<int>(0);
    refs[1]->emplace<int>(0);
    // `refs` is copied, but the completion function is executed when the
    // work queue is shutdown. At that time only the copied references are
    // valid but the original `refs` is expired.
    andThenAsyncCopying(llvm::ArrayRef(refs),
                        [&shuttingDown](llvm::ArrayRef<AnyAsyncValueRef> elts) {
                          ASSERT_EQ(elts[0]->getRefCount(), 1u);
                          ASSERT_EQ(elts[1]->getRefCount(), 1u);
                          ASSERT_TRUE(shuttingDown);
                        });
  }
  shuttingDown = true;
  runtime.getWorkQueue()->shutdown();
}

TEST(AsyncValue, ArrayMovingSync) {
  Runtime &runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AsyncValue::allocate<int>(runtime));
  refs.emplace_back(AsyncValue::allocate<int>(runtime));
  refs[0]->emplace<int>(0);
  refs[1]->emplace<int>(0);
  // `refs` is moved, so each element has refcount 1 when the completion
  // function is executed. `shuttingDown` is still false.
  andThenSyncMoving(
      llvm::MutableArrayRef(refs),
      [&shuttingDown](llvm::MutableArrayRef<AnyAsyncValueRef> elts) {
        ASSERT_EQ(elts[0]->getRefCount(), 1u);
        ASSERT_EQ(elts[1]->getRefCount(), 1u);
        ASSERT_FALSE(shuttingDown);
      });
  shuttingDown = true;
  runtime.getWorkQueue()->shutdown();
}

TEST(AsyncValue, ArrayMovingAsync) {
  Runtime &runtime = TestSingleThreadedRuntime();
  bool shuttingDown = false;
  {
    llvm::SmallVector<AnyAsyncValueRef> refs;
    refs.emplace_back(AsyncValue::allocate<int>(runtime));
    refs.emplace_back(AsyncValue::allocate<int>(runtime));
    refs[0]->emplace<int>(0);
    refs[1]->emplace<int>(0);
    // With the async version, the completion function is executed with the
    // work queue shuts down. `refs` is moved, so each element has refcount 1
    // when the completion function is executed.
    andThenAsyncMoving(
        llvm::MutableArrayRef(refs),
        [&shuttingDown](llvm::MutableArrayRef<AnyAsyncValueRef> elts) {
          ASSERT_EQ(elts[0]->getRefCount(), 1u);
          ASSERT_EQ(elts[1]->getRefCount(), 1u);
          ASSERT_TRUE(shuttingDown);
        });
  }
  shuttingDown = true;
  runtime.getWorkQueue()->shutdown();
}

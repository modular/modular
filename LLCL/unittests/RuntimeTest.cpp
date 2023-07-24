//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Runtime.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::LLCL;

std::unique_ptr<Runtime> createRuntime() {
  std::unique_ptr<Allocator> allocator =
      createLeakCheckAllocator(createMallocAllocator());
  std::unique_ptr<WorkQueue> workQueue = createSingleThreadWorkQueue();
  return std::make_unique<Runtime>(std::move(allocator), std::move(workQueue));
}

//===----------------------------------------------------------------------===//
// Runtime contexts
//===----------------------------------------------------------------------===//

struct ContextA {
  int i = 42;

  ContextA() = default;
  ContextA(int i) : i(i) {}
};

struct ContextB {
  bool b = true;
  char lots[26]; // Give this struct a large but unaligned size

  ContextB() = default;
};

struct ContextC {
  char c = 'a';
};

TEST(RuntimeTest, Contexts) {
  auto runtime = createRuntime();

  ContextA &ContextARef = runtime->emplaceContext<ContextA>(5);
  runtime->emplaceContext<ContextB>();

  ++ContextARef.i;

  ContextA *ContextAPtr = runtime->getContext<ContextA>();
  ContextB *ContextBPtr = runtime->getContext<ContextB>();
  ContextC *ContextCPtr = runtime->getContext<ContextC>();

  ASSERT_NE(ContextAPtr, nullptr);
  EXPECT_EQ(ContextAPtr->i, 6);
  ASSERT_NE(ContextBPtr, nullptr);
  EXPECT_EQ(ContextBPtr->b, true);
  EXPECT_EQ(ContextCPtr, nullptr);

  bool created = false;
  ErrorOr<ContextC *> contextCOr = runtime->createContextIfMissing<ContextC>(
      [&created]() -> ErrorOr<std::unique_ptr<ContextC>> {
        created = true;
        return std::make_unique<ContextC>();
      });
  ASSERT_TRUE(created);
  ASSERT_FALSE(contextCOr.isError());
  ASSERT_EQ((*contextCOr)->c, 'a');

  created = false;
  ErrorOr<ContextC *> contextCAgainOr =
      runtime->createContextIfMissing<ContextC>(
          [&created]() -> ErrorOr<std::unique_ptr<ContextC>> {
            created = true;
            return std::make_unique<ContextC>();
          });
  ASSERT_FALSE(created);
  ASSERT_FALSE(contextCAgainOr.isError());
  ASSERT_EQ(*contextCAgainOr, *contextCOr);
}

#ifndef NDEBUG
TEST(RuntimeTest, Contexturations_ExpectDeath) {
  auto runtime = createRuntime();

  runtime->emplaceContext<ContextA>();

  ASSERT_DEATH_IF_SUPPORTED(runtime->emplaceContext<ContextA>(),
                            "Runtime already holds context of type");
}
#endif

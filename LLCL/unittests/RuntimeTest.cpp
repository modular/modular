//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Runtime.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::LLCL;

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

struct ContextD {
  int x = 1;
  int y = 2;
};

std::unique_ptr<Runtime> createRuntime() {
  auto globalContextObjects = RCRef<SharedGenericUniquePtrSet>::create();
  auto &dRef = globalContextObjects->emplace<ContextD>();
  dRef.x = 3;

  std::unique_ptr<Allocator> allocator =
      createLeakCheckAllocator(createMallocAllocator());
  std::unique_ptr<WorkQueue> workQueue = createSingleThreadWorkQueue();
  return std::make_unique<Runtime>(std::move(allocator), std::move(workQueue),
                                   /*profileFilename=*/"",
                                   std::move(globalContextObjects));
}

TEST(RuntimeTest, Contexts) {
  auto runtime = createRuntime();

  ContextD *contextDPtr = runtime->getContext<ContextD>();
  ASSERT_NE(contextDPtr, nullptr);
  ASSERT_EQ(contextDPtr->x, 3);

  ContextA &contextARef = runtime->emplaceContext<ContextA>(5);
  runtime->emplaceContext<ContextB>();

  ++contextARef.i;

  ContextA *contextAPtr = runtime->getContext<ContextA>();
  ContextB *contextBPtr = runtime->getContext<ContextB>();
  ContextC *contextCPtr = runtime->getContext<ContextC>();

  ASSERT_NE(contextAPtr, nullptr);
  EXPECT_EQ(contextAPtr->i, 6);
  ASSERT_NE(contextBPtr, nullptr);
  EXPECT_EQ(contextBPtr->b, true);
  EXPECT_EQ(contextCPtr, nullptr);

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
                            "set already holds object of type");

  ASSERT_DEATH_IF_SUPPORTED(runtime->emplaceContext<ContextD>(),
                            "set already holds object of type");
}
#endif

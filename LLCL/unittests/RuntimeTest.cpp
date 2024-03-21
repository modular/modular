//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Runtime.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::LLCL;

namespace {

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

std::unique_ptr<Runtime> createRuntime() {
  return LLCL::createUniqueRuntime(LLCL::RuntimeOptions()
                                       .withLeakCheckedAllocator()
                                       .withMainWillNotDonate());
}

TEST(RuntimeTest, Contexts) {
  auto runtime = createRuntime();

  ContextA &contextARef = runtime->context->emplace<ContextA>(5);
  runtime->context->emplace<ContextB>();

  ++contextARef.i;

  ContextA *contextAPtr = runtime->context->get<ContextA>();
  ContextB *contextBPtr = runtime->context->get<ContextB>();
  ContextC *contextCPtr = runtime->context->get<ContextC>();

  ASSERT_NE(contextAPtr, nullptr);
  EXPECT_EQ(contextAPtr->i, 6);
  ASSERT_NE(contextBPtr, nullptr);
  EXPECT_EQ(contextBPtr->b, true);
  EXPECT_EQ(contextCPtr, nullptr);

  bool created = false;
  ErrorOr<ContextC *> contextCOr = runtime->context->createIfMissing<ContextC>(
      [&created]() -> ErrorOr<std::unique_ptr<ContextC>> {
        created = true;
        return std::make_unique<ContextC>();
      });
  ASSERT_TRUE(created);
  ASSERT_FALSE(contextCOr.isError());
  ASSERT_EQ((*contextCOr)->c, 'a');

  created = false;
  ErrorOr<ContextC *> contextCAgainOr =
      runtime->context->createIfMissing<ContextC>(
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

  runtime->context->emplace<ContextA>();

  ASSERT_DEATH_IF_SUPPORTED(runtime->context->emplace<ContextA>(),
                            "set already holds object of type");
}
#endif

/// Test to ensure that we can utilize the full range of indices for runtime.
/// This is mostly meant to be a precursor to check that the full range of
/// runtime indices is available
TEST(RuntimeTest, MaxRuntime) {
  std::vector<std::unique_ptr<Runtime>> allRuntimes;
  for (int i = 0; i < 255; ++i) {
    allRuntimes.emplace(allRuntimes.end(), createRuntime());
  }
  for (int i = 0; i < 255; ++i) {
    allRuntimes[i].reset();
  }
  allRuntimes.clear();
}

/// Test to ensure that we can utilize the full range of indices for runtime. It
/// checks the free indices first, and then fills up the index space. Next it
/// then removes 10 instances from the middle of the range and then attempts to
/// add 10 instances again which should succeed.
TEST(RuntimeTest, MaxRuntimeUtilize) {
  std::vector<std::unique_ptr<Runtime>> allRuntimes;
  uint8_t numRuntimes =
      M::LLCL::Detail::RuntimeTable::getSingleton().numActiveRuntimes();
  for (uint8_t i = 0; i < (255 - numRuntimes); ++i) {
    allRuntimes.emplace(allRuntimes.end(), createRuntime());
  }
  // now remove 10 indices from the middle of the indices range
  for (uint8_t i = 0; i < 10; ++i) {
    allRuntimes[i * 10].reset();
  }
  // now add back 10 runtime instances
  std::vector<std::unique_ptr<Runtime>> newRuntimes;
  for (uint8_t i = 0; i < 10; ++i) {
    newRuntimes.emplace(newRuntimes.end(), createRuntime());
  }
}
} // namespace

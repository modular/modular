//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Runtime.h"

#include "gtest/gtest.h"

using namespace M::LLCL;

std::unique_ptr<Runtime> createRuntime() {
  std::unique_ptr<Allocator> allocator =
      createLeakCheckAllocator(createMallocAllocator());
  std::unique_ptr<WorkQueue> workQueue = createSingleThreadWorkQueue();
  return std::make_unique<Runtime>(std::move(allocator), std::move(workQueue));
}

//===----------------------------------------------------------------------===//
// Runtime configuration
//===----------------------------------------------------------------------===//

struct ConfigA {
  int i = 42;

  ConfigA() = default;
  ConfigA(int i) : i(i) {}
};

struct ConfigB {
  bool b = true;

  ConfigB() = default;
};

struct ConfigC {
  char c = 'a';
};

TEST(RuntimeTest, Configurations) {
  auto runtime = createRuntime();

  ConfigA &configARef = runtime->emplaceConfig<ConfigA>(5);
  runtime->emplaceConfig<ConfigB>();

  ++configARef.i;

  const ConfigA *configAPtr = runtime->getConfig<ConfigA>();
  const ConfigB *configBPtr = runtime->getConfig<ConfigB>();
  const ConfigC *configCPtr = runtime->getConfig<ConfigC>();

  ASSERT_NE(configAPtr, nullptr);
  EXPECT_EQ(configAPtr->i, 6);
  ASSERT_NE(configBPtr, nullptr);
  EXPECT_EQ(configBPtr->b, true);
  EXPECT_EQ(configCPtr, nullptr);
}

TEST(RuntimeTest, Configurations_ExpectDeath) {
  auto runtime = createRuntime();

  runtime->emplaceConfig<ConfigA>();

  EXPECT_DEATH(runtime->emplaceConfig<ConfigA>(),
               "Runtime already holds configuration of type");
}

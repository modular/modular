//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/TypeID.h"

#include "LLCL/Support/Semaphore.h"

#include "gtest/gtest.h"

#include <thread>
#include <vector>

using namespace M;
using namespace LLCL;

template <typename T>
struct Foo {};

template <typename T, typename U>
struct Bar {
  T foo1;
  U foo2;
};

using FooBar = Bar<Foo<int>, bool>;

struct Baz {};

template <>
struct Detail::TypeNameResolver<Baz> {
  static StringRef getTypeName() { return "my_name"; }
};

TEST(TypeID, Smoke) {
  constexpr size_t numThreads = 10;

  {
    // Concurrently register the same types.
    Semaphore registerReady;
    auto registerThreadWorkFn = [&registerReady]() {
      registerReady.wait();
      TypeID::registerType<FooBar>();
      TypeID::registerType<Baz>();
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < numThreads; ++i)
      threads.emplace_back(registerThreadWorkFn);
    for (size_t i = 0; i < numThreads; ++i)
      // Try to trigger a thundering hurd.
      registerReady.post();
    for (auto &thread : threads)
      thread.join();
  }

  std::vector<TypeID> typeIDsA, typeIDsB;
  typeIDsA.resize(numThreads, TypeID());
  typeIDsB.resize(numThreads, TypeID());

  {
    // Concurrently get the same types.
    Semaphore getReady;
    auto getThreadWorkFn = [&getReady, &typeIDsA, &typeIDsB](size_t i) {
      getReady.wait();
      typeIDsA[i] = TypeID::get<FooBar>();
      typeIDsB[i] = TypeID::get<Baz>();
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < numThreads; ++i)
      threads.emplace_back(getThreadWorkFn, i);
    for (size_t i = 0; i < numThreads; ++i)
      // Try to trigger a thundering hurd.
      getReady.post();
    for (auto &thread : threads)
      thread.join();
  }

  EXPECT_EQ(typeIDsA.front().getTypeName(), "Bar<Foo<int>, bool>");
  EXPECT_EQ(typeIDsB.front().getTypeName(), "my_name");

  for (size_t i = 1; i < numThreads; ++i) {
    EXPECT_EQ(typeIDsA[i], typeIDsA.front());
    EXPECT_EQ(typeIDsB[i], typeIDsB.front());
  }
}

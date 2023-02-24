//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/TypeID.h"

#include "LLCL/Support/Semaphore.h"

#include "gtest/gtest.h"

#include <string_view>
#include <thread>
#include <vector>

using namespace M;
using namespace LLCL;

template <typename T>
struct SingleClassTemplate {};

template <typename T, typename U>
struct DoubleClassTemplate {
  T foo1;
  U foo2;
};

using FooBar = DoubleClassTemplate<SingleClassTemplate<int>, bool>;

struct Baz {};

template <>
struct Detail::TypeNameResolver<Baz> {
  // This is how you override the preferred name with a custom type.
  static StringRef getTypeName() { return "my_name"; }
};

struct Foo {};

namespace ns1::ns2 {
struct bar;
}

TEST(TypeID, typeName) {
  using namespace M::LLCL::Detail;
  using namespace std::string_view_literals;
  static_assert("void"sv == typeNameFor<void>());
  static_assert("int"sv == typeNameFor<int>());
  static_assert("fwd"sv == typeNameFor<class fwd>());
  static_assert("Foo"sv == typeNameFor<Foo>());

  static_assert("const int *" == typeNameFor<const int *>());
  static_assert("const int &" == typeNameFor<const int &>());
  static_assert("int **" == typeNameFor<int **>());
  static_assert("int &&" == typeNameFor<int &&>());

  static_assert("ns1::ns2::bar" == typeNameFor<ns1::ns2::bar>());
  static_assert("ns1::ns2::bar[]" == typeNameFor<ns1::ns2::bar[]>());

  static_assert("SingleClassTemplate<void>" ==
                typeNameFor<SingleClassTemplate<void>>());
  static_assert("SingleClassTemplate<int>" ==
                typeNameFor<SingleClassTemplate<int>>());

  // Show how `preferred_name` attribute can come into play
#if defined(_LIBCPP_VERSION)
  static_assert("std::string" == typeNameFor<std::string>());
#else
  static_assert("std::basic_string<char>" == typeNameFor<std::string>());
#endif
}

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

  EXPECT_EQ(typeIDsA.front().getTypeName(),
            "DoubleClassTemplate<SingleClassTemplate<int>, bool>");
  EXPECT_EQ(typeIDsB.front().getTypeName(), "my_name");

  for (size_t i = 1; i < numThreads; ++i) {
    EXPECT_EQ(typeIDsA[i], typeIDsA.front());
    EXPECT_EQ(typeIDsB[i], typeIDsB.front());
  }
}

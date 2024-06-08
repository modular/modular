//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/TimerHeap.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"

#include "gtest/gtest.h"

using namespace M::LLCL;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;
using std::chrono::time_point;

namespace {

struct TimerHeapTest : public testing::Test {
  std::unique_ptr<Runtime> runtime;
  time_point<steady_clock> start;
  TimerHeap heap;

  TimerHeapTest()
      : runtime(createUniqueRuntime()), start(steady_clock::now()) {}

  AsyncValueRef<Chain> in(int64_t ns) {
    AsyncValueRef<Chain> out = AsyncValueRef<Chain>::allocate(*runtime);
    TimerHeap::deadline expiration = steady_clock::now() + nanoseconds(ns);
    heap.push(expiration, out);
    return out;
  }

  void passed(int64_t ns) {
    EXPECT_GE(steady_clock::now(), start + nanoseconds(ns));
  }
};

TEST_F(TimerHeapTest, Serial) {
  await(in(0));
  await(in(10));
  await(in(100));
  await(in(1'000));
  passed(1'000);
}

TEST_F(TimerHeapTest, OutOfOrder) {
  auto a = in(1'000);
  auto b = in(100'000);    // 100us.
  auto c = in(10'000'000); // 10ms.
  await(c);
  passed(10'000'000);
  await(b);
  await(a);
}

TEST_F(TimerHeapTest, Cancelation) {
  auto a = in(1'000);
  auto b = in(100'000);    // 100us.
  auto c = in(10'000'000); // 10ms.
  heap.cancel(a);
  heap.cancel(b);
  await(c);
  passed(10'000'000);
}

} // namespace

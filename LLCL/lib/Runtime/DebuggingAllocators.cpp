//===- DebuggingAllocators.cpp --------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines wrapper allocators that keep track of extra metadata.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Support/Atomics.h"
#include "Support/LLVM.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include <atomic>

using namespace LLCL;

//===----------------------------------------------------------------------===//
// Leak Checking Allocator
//===----------------------------------------------------------------------===//

namespace {
class LeakCheckAllocator : public Allocator {
public:
  explicit LeakCheckAllocator(std::unique_ptr<Allocator> baseAllocator)
      : baseAllocator(std::move(baseAllocator)) {}
  ~LeakCheckAllocator() override { checkLeak(); }

  void *allocateBytes(size_t size, size_t alignment) override {
    ++numAllocations;
    numBytesAllocated.fetch_add(size);
    return baseAllocator->allocateBytes(size, alignment);
  }

  void deallocateBytes(void *ptr, size_t size) override {
    auto curNumAllocations = --numAllocations;
    assert(curNumAllocations >= 0 && "deallocation imbalance");
    numBytesAllocated.fetch_sub(size);
    baseAllocator->deallocateBytes(ptr, size);
  }

  /// Print a message and exit(1) when memory leak is detected.
  void checkLeak() {
    if (numBytesAllocated.load() != 0) {

      llvm::report_fatal_error(
          "Memory leak detected: " + llvm::Twine(numAllocations.load()) +
          " alive allocations, " + llvm::Twine(numBytesAllocated.load()) +
          " alive bytes\n" +
          "Run with other allocators to debug what happened.\n");
    }
  }

  /// This keeps track of how many bytes/allocations are currently alive.
  std::atomic<ssize_t> numBytesAllocated{0}, numAllocations{0};

private:
  std::unique_ptr<Allocator> baseAllocator;
};
} // end anonymous namespace.

/// Create a wrapper allocator that checks to make sure all memory is
/// deallocated when the allocator itself is destroyed.
std::unique_ptr<Allocator>
LLCL::createLeakCheckAllocator(std::unique_ptr<Allocator> baseAllocator) {
  return std::make_unique<LeakCheckAllocator>(std::move(baseAllocator));
}

//===----------------------------------------------------------------------===//
// Profiling Allocator
//===----------------------------------------------------------------------===//

namespace {
class ProfilingAllocator : public LeakCheckAllocator {
public:
  explicit ProfilingAllocator(std::unique_ptr<Allocator> baseAllocator)
      : LeakCheckAllocator(std::move(baseAllocator)) {}

  void *allocateBytes(size_t size, size_t alignment) override {
    void *result = LeakCheckAllocator::allocateBytes(size, alignment);
    ++totalAllocations;

    atomicMax(maxAllocations, numAllocations.load());
    atomicMax(maxBytesAllocated, numBytesAllocated.load());
    return result;
  }

  ~ProfilingAllocator() override {
    printf("LLCL::Allocator profile:\n");
    printf("Max number of allocations = %lld\n",
           (long long)maxAllocations.load());
    printf("Total number of allocations = %lld\n",
           (long long)totalAllocations.load());
    printf("Max number of bytes ever allocated = %lld\n",
           (long long)maxBytesAllocated.load());
    fflush(stdout);

    // If we still have active memory alive, print an error.
    checkLeak();
  }

  std::atomic<ssize_t> maxAllocations{0}, maxBytesAllocated{0};
  std::atomic<int64_t> totalAllocations{0};
};
} // end anonymous namespace.

/// Create a wrapper allocator that prints memory profiling information when it
/// is destroyed.  This also performs leak checks.
std::unique_ptr<Allocator>
LLCL::createProfilingAllocator(std::unique_ptr<Allocator> baseAllocator) {
  return std::make_unique<ProfilingAllocator>(std::move(baseAllocator));
}

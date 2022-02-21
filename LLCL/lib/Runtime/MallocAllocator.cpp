//===- MallocAllocator.cpp - Allocator using malloc/free ------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Allocator.h"
#include "Support/AlignedAlloc.h"

using namespace LLCL;

namespace {
class MallocAllocator : public Allocator {
  // Allocate the specified number of bytes with the specified alignment.
  void *allocateBytes(size_t size, size_t alignment) override {
    return M::alignedAlloc(size, alignment);
  }

  // Deallocate the specified pointer that has the specified size.
  void deallocateBytes(void *ptr, size_t size) override { M::alignedFree(ptr); }
};
} // end anonymous namespace.

std::unique_ptr<Allocator> LLCL::createMallocAllocator() {
  return std::make_unique<MallocAllocator>();
}

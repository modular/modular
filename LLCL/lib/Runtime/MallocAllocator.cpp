//===- MallocAllocator.cpp ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Allocator.h"
#include "Support/AlignedAlloc.h"

using namespace M::LLCL;

namespace {
/// This is an implementation of the Allocator interface that just calls to
/// alignedAlloc/alignedFree, the system allocator implementations.
class MallocAllocator : public Allocator {
  // Allocate the specified number of bytes with the specified alignment.
  void *allocateBytes(size_t size, size_t alignment) override {
    return M::alignedAlloc(alignment, size);
  }

  // Deallocate the specified pointer that has the specified size.
  void deallocateBytes(void *ptr, size_t size) override { M::alignedFree(ptr); }
};
} // namespace

std::unique_ptr<Allocator> M::LLCL::createMallocAllocator() {
  return std::make_unique<MallocAllocator>();
}

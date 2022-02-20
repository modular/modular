//===- MallocAllocator.cpp - Allocator using malloc/free ------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Allocator.h"
#include "llvm/Support/MathExtras.h"

using namespace LLCL;

/// This is a helper to handle host-specific system alignment functions.
static void *alignedAlloc(size_t alignment, size_t size) {
#if defined(__ANDROID__) || defined(OS_ANDROID)
  return memalign(alignment, size);
#else // !__ANDROID__ && !OS_ANDROID
  void *ptr = nullptr;
  assert(alignment >= sizeof(void *) && "caller already checked");
  if (posix_memalign(&ptr, alignment, size) != 0)
    return nullptr;
  return ptr;
#endif
}

namespace {
class MallocAllocator : public Allocator {
  // Allocate the specified number of bytes with the specified alignment.
  void *allocateBytes(size_t size, size_t alignment) override {
    assert(llvm::isPowerOf2_64(alignment) && "non-power-of-2 alignment!");
    if (alignment <= 8)
      return malloc(size);
    return alignedAlloc(alignment, size);
  }

  // Deallocate the specified pointer that has the specified size.
  void deallocateBytes(void *ptr, size_t size) override { free(ptr); }
};
} // end anonymous namespace.

std::unique_ptr<Allocator> LLCL::createMallocAllocator() {
  return std::make_unique<MallocAllocator>();
}

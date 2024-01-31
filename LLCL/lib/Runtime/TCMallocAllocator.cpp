//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Allocator.h"
#include "Support/AlignedAlloc.h"
#ifdef USE_TCMALLOC
#include <gperftools/tcmalloc.h>
#endif
using namespace M;
using namespace M::LLCL;

namespace {
/// This is an implementation of the Allocator interface that just calls to
/// tc_new/tc_delete when TCMalloc is enabled and built or fallback to system
/// allocator otherwise.
class TCMallocAllocator : public Allocator {
  // Allocate the specified number of bytes with the specified alignment.
  void *allocateBytes(size_t size, size_t alignment) override {

#ifdef USE_TCMALLOC
    TimeTraceScope scope(MemAllocFreeProfilerEntry::create("mem.alloc.tcmalloc",
                                                           (uint64_t)size));
    return tc_new_aligned(size, std::align_val_t(alignment));
#else

    TimeTraceScope scope(
        MemAllocFreeProfilerEntry::create("mem.alloc", (uint64_t)size));
    return alignedAlloc(alignment, size);
#endif
  }

  // Deallocate the specified pointer that has the specified size.
  void deallocateBytes(void *ptr, size_t size) override {

#ifdef USE_TCMALLOC
    TimeTraceScope scope(
        MemAllocFreeProfilerEntry::create("mem.free.tcmalloc", (uint64_t)size));
    return tc_delete(ptr);
#else
    TimeTraceScope scope(
        MemAllocFreeProfilerEntry::create("mem.free", (uint64_t)size));
    alignedFree(ptr);
#endif
  }
};
} // namespace

std::unique_ptr<Allocator> M::LLCL::createTCMallocAllocator() {
#ifndef USE_TCMALLOC
  llvm::report_fatal_error("LLCL not built with tcmalloc");
#endif
  return std::make_unique<TCMallocAllocator>();
}

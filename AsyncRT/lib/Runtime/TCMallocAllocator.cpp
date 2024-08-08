//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/Globals/Globals.h"

namespace M::AsyncRT {
#ifdef USE_TCMALLOC

/// This is an implementation of the Allocator interface that just calls to
/// tc_new/tc_delete.
/// NOTE: TCMalloc uses static global variables that do not live within an
/// instance of this class. These methods make a call into RuntimeGlobals.
class TCMallocAllocator : public Allocator {
  /// Allocate the specified number of bytes with the specified alignment.
  void *allocateBytes(size_t size, size_t alignment) override {
    TimeTraceScope scope(MemAllocFreeProfilerEntry::create("mem.alloc.tcmalloc",
                                                           (uint64_t)size));
    return TCMallocGlobals::tc_new(size, alignment);
  }

  /// Deallocate the specified pointer that has the specified size.
  void deallocateBytes(void *ptr, size_t size) override {
    TimeTraceScope scope(
        MemAllocFreeProfilerEntry::create("mem.free.tcmalloc", (uint64_t)size));
    return TCMallocGlobals::tc_delete(ptr);
  }
};
#endif // USE_TCMALLOC

std::unique_ptr<Allocator> createTCMallocAllocator() {
#ifdef USE_TCMALLOC
  return std::make_unique<TCMallocAllocator>();
#else  // USE_TCMALLOC
  llvm::report_fatal_error("--allocator=tcmalloc not built for target. Please "
                           "use a different allocator");
#endif // USE_TCMALLOC
}

} // end namespace M::AsyncRT

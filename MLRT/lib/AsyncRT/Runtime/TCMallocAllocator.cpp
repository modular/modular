//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/Allocator.h"
#include "MLRT/AsyncRT/Runtime/Globals/Globals.h"

namespace M::MLRT {

/// This is an implementation of the Allocator interface that just calls to
/// tc_new/tc_delete.
/// NOTE: TCMalloc uses static global variables that do not live within an
/// instance of this class. These methods make a call into RuntimeGlobals.
class TCMallocAllocator : public Allocator {
  /// Allocate the specified number of bytes with the specified alignment.
  void *allocateBytes(size_t size, size_t alignment) override {
    TimeTraceScope scope(MemAllocFreeProfilerEntry::create("mem.alloc.tcmalloc",
                                                           (uint64_t)size));
    return TCMallocGlobals::tc_new(alignment, size);
  }

  /// Deallocate the specified pointer that has the specified size.
  void deallocateBytes(void *ptr, size_t size) override {
    TimeTraceScope scope(
        MemAllocFreeProfilerEntry::create("mem.free.tcmalloc", (uint64_t)size));
    return TCMallocGlobals::tc_delete(ptr);
  }
};

std::unique_ptr<Allocator> createTCMallocAllocator() {
  return std::make_unique<TCMallocAllocator>();
}

} // end namespace M::MLRT

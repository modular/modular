//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/Allocator.h"
#include "MLRT/AsyncRT/Runtime/Globals/Globals.h"
#include "Support/Log.h"

namespace M::MLRT {

/// This is an implementation of the Allocator interface that just calls to
/// tc_new/tc_delete.
/// NOTE: TCMalloc uses static global variables that do not live within an
/// instance of this class. These methods make a call into RuntimeGlobals.
class TCMallocAllocator : public Allocator {
public:
  TCMallocAllocator() = default;
  explicit TCMallocAllocator(int numaPlacement) : Allocator(numaPlacement) {}

private:
  void *allocateBytes(size_t size, size_t alignment) override {
    TimeTraceScope scope(MemAllocFreeProfilerEntry::create("mem.alloc.tcmalloc",
                                                           (uint64_t)size));
    if (getNumaPlacement() != kAnyNumaNode) {
      // TCMalloc uses a maximum of two partitions, even if there are more than
      // two NUMA nodes, NUMA nodes are mapped to alternating partitions, so
      // here we match this.
      // TODO: This is a know limitation of TCMalloc that we don't intend to
      // fix, instead this will be resolved when we introduce our own allocation
      // library to replace TCMalloc.
      const size_t partition = static_cast<size_t>(getNumaPlacement()) % 2;
      void *ptr = TCMallocGlobals::tc_new(alignment, size, partition);
#if MODULAR_ALLOC_LOGGING
      MLOG_DEBUG("tcmalloc alloc (numa partition {}): ptr={} size={} "
                 "alignment={}",
                 partition, ptr, size, alignment);
#endif
      return ptr;
    }

    void *ptr = TCMallocGlobals::tc_new(alignment, size);
#if MODULAR_ALLOC_LOGGING
    MLOG_DEBUG("tcmalloc alloc: ptr={} size={} alignment={}", ptr, size,
               alignment);
#endif
    return ptr;
  }

  void deallocateBytes(void *ptr, size_t size) override {
    TimeTraceScope scope(
        MemAllocFreeProfilerEntry::create("mem.free.tcmalloc", (uint64_t)size));
#if MODULAR_ALLOC_LOGGING
    MLOG_DEBUG("tcmalloc free: ptr={} size={}", ptr, size);
#endif
    return TCMallocGlobals::tc_delete(ptr);
  }
};

std::unique_ptr<Allocator> createTCMallocAllocator() {
  return std::make_unique<TCMallocAllocator>();
}

std::unique_ptr<Allocator> createTCMallocAllocator(int numaPlacement) {
  return std::make_unique<TCMallocAllocator>(numaPlacement);
}

} // end namespace M::MLRT

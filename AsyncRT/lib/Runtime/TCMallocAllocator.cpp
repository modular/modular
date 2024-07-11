//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Allocator.h"
#include "Support/AlignedAlloc.h"

using namespace M;
using namespace M::AsyncRT;

#ifdef USE_TCMALLOC
#include <gperftools/tcmalloc.h>
namespace {
/// This is an implementation of the Allocator interface that just calls to
/// tc_new/tc_delete.
class TCMallocAllocator : public Allocator {
  /// Allocate the specified number of bytes with the specified alignment.
  void *allocateBytes(size_t size, size_t alignment) override {
    TimeTraceScope scope(MemAllocFreeProfilerEntry::create("mem.alloc.tcmalloc",
                                                           (uint64_t)size));
    return tc_new_aligned(size, std::align_val_t(alignment));
  }

  /// Deallocate the specified pointer that has the specified size.
  void deallocateBytes(void *ptr, size_t size) override {
    TimeTraceScope scope(
        MemAllocFreeProfilerEntry::create("mem.free.tcmalloc", (uint64_t)size));
    return tc_delete(ptr);
  }
};
} // namespace
#endif // USE_TCMALLOC

std::unique_ptr<Allocator> M::AsyncRT::createTCMallocAllocator() {
#ifdef USE_TCMALLOC
  return std::make_unique<TCMallocAllocator>();
#else  // USE_TCMALLOC
  llvm::report_fatal_error("--allocator=tcmalloc not built for target. Please "
                           "use a different allocator");
#endif // USE_TCMALLOC
}

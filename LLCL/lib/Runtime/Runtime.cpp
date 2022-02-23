//===- Runtime.cpp - LLCL Runtime implementation --------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the core LLCL Runtime.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/WorkQueue.h"
using namespace LLCL;

void WorkQueue::vtableAnchor() {}
void Allocator::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// CompactRuntimePtr
//===----------------------------------------------------------------------===//

/// The `CompactRuntimePtr` type provides a pointer compressed version of
/// `Runtime*` that fits in 8 bits.  This allows every AsyncValue to carry a
/// backpointer to the Runtime that allocated them, and allows deallocating the
/// memory for the AsyncValue through the Runtime's allocator.
///
/// This is implemented with a static array of Runtime pointers that are given
/// unique IDs.
static std::atomic<uint8_t> nextRuntimeIndex{0};
static Runtime *allRuntimes[CompactRuntimePtr::kInvalidIndex];

CompactRuntimePtr::CompactRuntimePtr(Runtime *runtime)
    : CompactRuntimePtr(runtime->getCompactPtr()) {}

Runtime *CompactRuntimePtr::get() const {
  assert(index != kInvalidIndex);
  return allRuntimes[index];
}

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

Runtime::Runtime(std::unique_ptr<Allocator> allocator,
                 std::unique_ptr<WorkQueue> workQueue)
    : allocator(std::move(allocator)), workQueue(std::move(workQueue)),
      runtimeIndex(nextRuntimeIndex.fetch_add(1)) {
  // We provide a dense numbering of runtime instances right now, but we could
  // make this fancier to allow deallocating and reusing indexes if needbe.
  assert(runtimeIndex < CompactRuntimePtr::kInvalidIndex &&
         "Created too many Runtimes");
  allRuntimes[runtimeIndex] = this;
}

Runtime::~Runtime() { allRuntimes[runtimeIndex] = nullptr; }

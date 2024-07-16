//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements ReferenceCounted class.
//
//===----------------------------------------------------------------------===//

#include "Support/ReferenceCounted.h"
#include <cstdio>
#include <cstdlib>
using namespace M;

#ifdef MODULAR_DEBUG
/// In debug builds we keep track of the number of reference counted objects,
/// which enables clients to check that none are alive at key moments.  This is
/// a low-tech way to find certain classes of memory leaks.
std::atomic<size_t> M::currentReferenceCountedObjects{0};
#endif // MODULAR_DEBUG

/// Verify that there are no live ReferenceCounted objects that are currently
/// alive and print the specified message and abort if there are.
void M::verifyNoLiveReferenceCountedObjects(const char *errorMessage) {
#ifdef MODULAR_DEBUG
  if (!currentReferenceCountedObjects.load(std::memory_order_relaxed))
    return;

  // Otherwise print an error and crash.
  fprintf(stderr, "AsyncRT internal error %s\n", errorMessage);
  fflush(stderr);
  abort();
#endif // MODULAR_DEBUG
}

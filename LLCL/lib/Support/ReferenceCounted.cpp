//===- ReferenceCounted.cpp -----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements ReferenceCounted class.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/ReferenceCounted.h"
#include <cstdio>
#include <cstdlib>
using namespace LLCL;

#ifndef NDEBUG
/// In debug builds we keep track of the number of reference counted objects,
/// which enables clients to check that none are alive at key moments.  This is
/// a low-tech way to find certain classes of memory leaks.
std::atomic<size_t> LLCL::currentReferenceCountedObjects{0};
#endif

/// Verify that there are no live ReferenceCounted objects that are currently
/// alive and print the specified message and abort if there are.
void LLCL::verifyNoLiveReferenceCountedObjects(const char *errorMessage) {
  if (!currentReferenceCountedObjects.load(std::memory_order_relaxed))
    return;

  // Otherwise print an error and crash.
  fprintf(stderr, "LLCL internal error %s\n", errorMessage);
  fflush(stderr);
  abort();
}
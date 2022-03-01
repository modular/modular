//===- Support/AlignedAlloc.h -----------------------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares alignedAlloc() and alignedFree() for allocating dynamic
// buffers with explicit alignment.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ALIGNED_ALLOC_H
#define SUPPORT_ALIGNED_ALLOC_H

#include <cstddef>
#include <cstdlib>

namespace M {

/// Allocate the a block of memory with the specified size and alignment.
///  NOTE: The returned pointer *must* be deallocated with alignedFree().
/// Deallocating with e.g. free() instead causes runtime issues on Windows that
/// are hard to debug.
void *alignedAlloc(size_t size, size_t alignment);

#ifndef _WIN32
/// alignedFree deallocates a pointer allocated with alignedAlloc.
inline void alignedFree(void *ptr) { free(ptr); }
#else
/// alignedFree deallocates a pointer allocated with alignedAlloc.
void alignedFree(void *ptr);
#endif

} // namespace M

#endif // SUPPORT_ALIGNED_ALLOC_H

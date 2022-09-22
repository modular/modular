//===- Support/AlignedAlloc.h ---------------------------------------------===//
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

#if defined(__AVX512F__)
static constexpr size_t kPreferredMemoryAlignment = 64;
#elif defined(__AVX2__) || defined(__AVX__)
static constexpr size_t kPreferredMemoryAlignment = 32;
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
static constexpr size_t kPreferredMemoryAlignment = 16;
#else
static constexpr size_t kPreferredMemoryAlignment = 16;
#endif

/// Allocate the a block of memory with the specified size and alignment.
///  NOTE: The returned pointer *must* be deallocated with alignedFree().
/// Deallocating with e.g. free() instead causes runtime issues on Windows that
/// are hard to debug.
void *alignedAlloc(size_t alignment, size_t size);

#ifndef _WIN32
/// alignedFree deallocates a pointer allocated with alignedAlloc.
inline void alignedFree(void *ptr) { std::free(ptr); }
#else
/// alignedFree deallocates a pointer allocated with alignedAlloc.
void alignedFree(void *ptr);
#endif

} // namespace M

#endif // SUPPORT_ALIGNED_ALLOC_H

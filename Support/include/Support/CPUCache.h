//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CPUCACHE_H
#define SUPPORT_CPUCACHE_H

#include "Support/ForwardDecls.h"
#include <cstddef>

namespace M {
//===----------------------------------------------------------------------===//
// Cache sizes
//===----------------------------------------------------------------------===//

/// Get the D$ or unified cache size in bytes at a 1-based cache level index.
/// An error is returned if there is an OS error in finding the cache level.  If
/// the cache level does not exist, 0 is returned.
ErrorOr<size_t> getHostCPUCacheSize(size_t cacheLevel);
} // namespace M

#endif // SUPPORT_CPUCACHE_H

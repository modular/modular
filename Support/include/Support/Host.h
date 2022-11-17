//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HOST_H
#define SUPPORT_HOST_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
#define HOST_IS_APPLE_SILICON_PROCESSOR
#endif

namespace M {
ErrorOr<size_t> getHostCPUCacheSize(size_t cacheLevel);
}

#endif // SUPPORT_HOST_H

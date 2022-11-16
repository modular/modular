//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HOST_H
#define SUPPORT_HOST_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

namespace M {
ErrorOr<size_t> getHostCPUCacheSize(size_t cacheLevel);
}

#endif // SUPPORT_HOST_H

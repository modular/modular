//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/AsyncValue.h"
#include "Support/SymbolExport.h"

#include <atomic>

using namespace M::LLCL;

#ifndef _WIN32
[[maybe_unused]] MODULAR_CXX_EXPORT std::atomic<ssize_t>
    M::LLCL::AsyncValue::totalAllocatedAsyncValues{0};
#endif


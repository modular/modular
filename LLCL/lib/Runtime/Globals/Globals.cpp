//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/AsyncValue.h"
#include "Support/SymbolExport.h"

#include <atomic>

using namespace M::LLCL;

[[maybe_unused]] MODULAR_CXX_EXPORT std::atomic<ssize_t>
    M::LLCL::AsyncValue::totalAllocatedAsyncValues{0};

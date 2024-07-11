//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./Memory.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "KGEN/CompilerRT/Registration.h"
#include "Support/SymbolExport.h"

using namespace M;

/// Returns an alignment allocated memory. If the alignment value is not
/// positive, then the default alignment is used.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_AlignedAlloc(ssize_t alignment, ssize_t size) {
  if (alignment <= 0)
    alignment = kPreferredMemoryAlignment;
  return Runtime::getCurrentAllocator()->allocateBytes(size, alignment);
}

/// Frees memory allocated via KGEN_CompilerRT_AlignedAlloc.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AlignedFree(void *ptr) {
  Runtime::getCurrentAllocator()->deallocateBytes(ptr);
}

void M::KGEN::registerMemory(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back(
      {"KGEN_CompilerRT_AlignedAlloc", (void *)&KGEN_CompilerRT_AlignedAlloc});
  funcs.push_back(
      {"KGEN_CompilerRT_AlignedFree", (void *)&KGEN_CompilerRT_AlignedFree});
}

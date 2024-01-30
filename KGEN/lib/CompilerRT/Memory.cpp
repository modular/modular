//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/Registration.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/AlignedAlloc.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"

using namespace M;

/// Returns an alignment allocated memory. If the alignment value is not
/// positive, then the default alignment is used.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_AlignedAlloc(ssize_t alignment, ssize_t size) {
  if (alignment <= 0)
    alignment = kPreferredMemoryAlignment;
  auto rt = Runtime::getCurrentRuntimeOrNull();
  if (rt)
    return rt->getAllocator()->allocateBytes(size, alignment);
  return alignedAlloc(alignment, size);
}

/// Frees memory allocated via KGEN_CompilerRT_AlignedAlloc.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AlignedFree(void *ptr) {
  auto rt = Runtime::getCurrentRuntimeOrNull();
  if (rt) {
    // We do not use size in dealloc for now. May need to
    // change if sized delete is more performant.
    return rt->getAllocator()->deallocateBytes(ptr);
  }
  return alignedFree(ptr);
}

void M::KGEN::registerMemory(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back(
      {"KGEN_CompilerRT_AlignedAlloc", (void *)&KGEN_CompilerRT_AlignedAlloc});
  funcs.push_back(
      {"KGEN_CompilerRT_AlignedFree", (void *)&KGEN_CompilerRT_AlignedFree});
}

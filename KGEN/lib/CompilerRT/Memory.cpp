//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./Memory.h"
#include "MLRT/AsyncRT/Runtime/Globals/Globals.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/AlignedAlloc.h"
#include "Support/SymbolExport.h"
using namespace M;

namespace {
struct {
  void *(*alloc)(size_t alignment,
                 size_t size) = AsyncRT::TCMallocGlobals::tc_new;
  void (*free)(void *ptr) = AsyncRT::TCMallocGlobals::tc_delete;
} constinit static KGEN_Allocators{};
} // namespace

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_SetAsanAllocators() {
  KGEN_Allocators = {.alloc = M::alignedAlloc, .free = M::alignedFree};
}

/// Returns an alignment allocated memory. If the alignment value is not
/// positive, then the default alignment is used.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_AlignedAlloc(ssize_t alignment, ssize_t size) {
  if (alignment <= 0)
    alignment = kPreferredMemoryAlignment;
  return KGEN_Allocators.alloc(alignment, size);
}

/// Frees memory allocated via KGEN_CompilerRT_AlignedAlloc.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AlignedFree(void *ptr) {
  return KGEN_Allocators.free(ptr);
}

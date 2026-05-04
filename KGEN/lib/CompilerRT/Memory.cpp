//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./Memory.h"
#include "MLRT/AsyncRT/Runtime/Globals/Globals.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/AlignedAlloc.h"
#include "Support/Log.h"
#include "Support/SymbolExport.h"
using namespace M;

namespace {
struct {
  void *(*alloc)(size_t alignment, size_t size) = MLRT::TCMallocGlobals::tc_new;
  void (*free)(void *ptr) = MLRT::TCMallocGlobals::tc_delete;
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
  void *ptr = KGEN_Allocators.alloc(alignment, size);
#if MODULAR_ALLOC_LOGGING
  MLOG_DEBUG("mojo alloc: ptr={} size={} alignment={}", ptr, size, alignment);
#endif
  return ptr;
}

/// Frees memory allocated via KGEN_CompilerRT_AlignedAlloc.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AlignedFree(void *ptr) {
#if MODULAR_ALLOC_LOGGING
  MLOG_DEBUG("mojo free: ptr={}", ptr);
#endif
  return KGEN_Allocators.free(ptr);
}

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILERRT_MEMORY_H
#define KGEN_COMPILERRT_MEMORY_H

#include "Support/SymbolExport.h"

#ifndef _MSC_VER
#include <unistd.h>
#endif // _MSC_VER

// Set allocators to system memalign/free to support asan
// this function is NOT thread safe and needs to be called
// before any allocations
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_SetAsanAllocators();

/// Returns an alignment allocated memory. If the alignment value is not
/// positive, then the default alignment is used.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_AlignedAlloc(ssize_t alignment, ssize_t size);

/// Frees memory allocated via KGEN_CompilerRT_AlignedAlloc.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AlignedFree(void *ptr);

#endif // KGEN_COMPILERRT_MEMORY_H

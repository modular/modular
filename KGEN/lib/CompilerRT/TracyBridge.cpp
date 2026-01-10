//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Tracy bridge for Mojo, providing FFI-callable
// functions for creating and ending Tracy profiling zones.
//
//===----------------------------------------------------------------------===//

#include "Support/Profiling/TracyZone.h"
#include "Support/SymbolExport.h"
#include <cstddef>
#include <cstdint>

extern "C" {

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_TracyIsEnabled(void) {
#ifdef TRACY_ENABLE
  return 1;
#else
  return 0;
#endif
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint64_t
KGEN_CompilerRT_TracyZoneBegin(const char *name, size_t nameLen,
                               uint32_t color) {
  return M::tracyZoneBegin("MojoTrace", sizeof("MojoTrace") - 1,
                           "Trace.__enter__", sizeof("Trace.__enter__") - 1,
                           name, nameLen, color);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_TracyZoneEnd(uint64_t packedCtx) {
  M::tracyZoneEnd(packedCtx);
}

} // extern "C"

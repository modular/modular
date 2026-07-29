//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// FFI bridge between Mojo and the M::Profiling Range API. Mojo's planned
// mo.profile range op will lower to KGEN_CompilerRT_Range{Begin,End} via this
// file (not yet implemented). The enable/disable control surface is driven by
// InferenceSession construction auto-start (max-debug.profiling-enabled).
//
//===----------------------------------------------------------------------===//

#include "Support/SymbolExport.h"

#if MODULAR_KGEN_PROFILING_ENABLED
#include "Support/Profiling/Ranges.h"
#include <string_view>
#endif

#include <cstddef>
#include <cstdint>

extern "C" {

// Mojo-callable range begin / end. Both are cheap when no trace is live —
// RangeBegin branches on M::Profiling::isRangeRecordingActive() and RangeEnd
// on this thread's pairing state — so they are safe to call from hot
// kernel-launch paths.
//
// Precondition: `namePtr` must be non-null even when `nameLen == 0`.
// Constructing a `std::string_view` from a null pointer is undefined behavior
// under C++17 and only well-defined under C++20 when the length is zero.
// The planned `mo.profile` lowering (not yet implemented) will pass
// `String.unsafe_ptr()`, which is always non-null, and so will satisfy this
// contract; until then, every C-ABI caller must uphold it themselves.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeBegin(const char *namePtr, size_t nameLen,
                           uint32_t color) {
#if MODULAR_KGEN_PROFILING_ENABLED
  M::Profiling::rangeBegin(std::string_view(namePtr, nameLen), color);
#endif
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeEnd(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  M::Profiling::rangeEnd();
#endif
}

// Step counter advance. MAX's Model::execute() calls this once per
// invocation to drive the warmup/active state machine. Exposed via FFI so
// Mojo-side runtime entry points can also drive it when appropriate.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeStep(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  M::Profiling::step();
#endif
}

// Enable / disable control surface. Driven by InferenceSession's
// construction-time auto-start (max-debug.profiling-enabled).
// TODO(MXTOOLS-190): Dynolog's IPC listener will drive enable/disable
// through these same entry points.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeEnable(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  M::Profiling::enable();
#endif
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeDisable(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  M::Profiling::disable();
#endif
}

// Returns 1 if the profiler is currently enabled, 0 otherwise. Useful from
// Mojo to elide expensive name materialization on the disabled fast path —
// but note this reflects only the session API's enable intent: it stays 0
// during Dynolog daemon-driven on-demand traces, when ranges DO record, so
// eliding on it opts the caller out of daemon-trace annotation (RangeBegin
// itself is one predicted branch when idle, so unconditional calls are fine).
// Returns `size_t` to match the existing `KGEN_CompilerRT_TracyIsEnabled`
// shape so Mojo callers can treat both predicates uniformly.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_RangeIsEnabled(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  return M::Profiling::isEnabled() ? 1 : 0;
#else
  return 0;
#endif
}

} // extern "C"

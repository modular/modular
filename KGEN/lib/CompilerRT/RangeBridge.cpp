//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// FFI bridge between Mojo and the M::Profiling Range API. Mojo's planned
// mo.profile range op will lower to KGEN_CompilerRT_Range{Begin,End} via this
// file (not yet implemented), and the Python kineto_enable/disable bindings
// call the matching control surface.
//
//===----------------------------------------------------------------------===//

#include "Support/Profiling/Range.h"
#include "Support/SymbolExport.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

extern "C" {

// Mojo-callable range begin / end. Both branch on M::Profiling::gKinetoEnabled
// before doing any work, so they are safe to call from hot kernel-launch
// paths even when profiling is disabled.
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
  M::Profiling::rangeBegin(std::string_view(namePtr, nameLen), color);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeEnd(void) {
  M::Profiling::rangeEnd();
}

// Step counter advance. MAX's Model::execute() calls this once per
// invocation to drive the warmup/active state machine. Exposed via FFI so
// Mojo-side runtime entry points can also drive it when appropriate.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeStep(void) {
  M::Profiling::step();
}

// Enable / disable control surface. The Python session.profiling.start() /
// .stop() bindings call into these via nanobind.
// TODO(MXTOOLS-190): Dynolog's IPC listener will drive enable/disable
// through these same entry points.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeEnable(void) {
  M::Profiling::enable();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeDisable(void) {
  M::Profiling::disable();
}

// Returns 1 if the profiler is currently enabled, 0 otherwise. Useful from
// Mojo to elide expensive name materialization on the disabled fast path.
// Returns `size_t` to match the existing `KGEN_CompilerRT_TracyIsEnabled`
// shape so Mojo callers can treat both predicates uniformly.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_RangeIsEnabled(void) {
  return M::Profiling::isEnabled() ? 1 : 0;
}

} // extern "C"

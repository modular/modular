//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file provides helper functions for creating Tracy profiling zones with
// dynamic names. These utilities are shared between the CompilerRT Tracy bridge
// (for Mojo FFI) and C++ tracing code.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_TRACY_ZONE_H
#define SUPPORT_PROFILING_TRACY_ZONE_H

#include "Support/max_license/Tracy/Tracy.h"
#include "llvm/ADT/StringRef.h"
#include <cstddef>
#include <cstdint>

#ifdef TRACY_ENABLE
#include "tracy/TracyC.h"
#endif // TRACY_ENABLE

namespace M {

#ifdef TRACY_ENABLE

/// Begins a Tracy zone with dynamic name allocation.
///
/// This function allocates a source location with the given identifiers and
/// begins a new Tracy zone. The returned packed context encodes the zone
/// context into a single uint64_t, suitable for passing across FFI
/// boundaries.
///
/// source: The source identifier (e.g., "MojoTrace", "ScopedMojoTrace").
/// sourceLen: Length of the source string.
/// func: The function name to display in the profiler.
/// funcLen: Length of the function name string.
/// name: The zone name (typically the trace message).
/// nameLen: Length of the zone name string.
/// color: Optional color value (0 uses Tracy's default).
/// Returns packed context: id in bits 0-31, active in bit 32, and (under
/// TRACY_ON_DEMAND) the low 31 bits of connectionId in bits 33-63.
/// Inactive zones pack to 0.
inline uint64_t tracyZoneBegin(const char *source, size_t sourceLen,
                               const char *func, size_t funcLen,
                               const char *name, size_t nameLen,
                               uint32_t color = 0) {
  uint64_t srcloc = ___tracy_alloc_srcloc_name(
      /*line=*/0, source, sourceLen, func, funcLen, name, nameLen, /*color=*/0);
  TracyCZoneCtx ctx = ___tracy_emit_zone_begin_alloc(srcloc, 1);
  // Inactive zones (e.g. no client connected under TRACY_ON_DEMAND) leave
  // ctx.id and ctx.connectionId uninitialized; don't read them.
  if (!ctx.active)
    return 0;
  if (color != 0)
    TracyCZoneColor(ctx, color);
  uint64_t packed = static_cast<uint64_t>(ctx.id) | (uint64_t{1} << 32);
#ifdef TRACY_ON_DEMAND
  packed |= (static_cast<uint64_t>(ctx.connectionId) & 0x7FFFFFFFu) << 33;
#endif
  return packed;
}

/// Ends a Tracy zone from a packed context.
///
/// packedCtx: The packed context returned by tracyZoneBegin().
///            If 0, this function does nothing.
inline void tracyZoneEnd(uint64_t packedCtx) {
  if (packedCtx == 0)
    return;
  TracyCZoneCtx ctx;
  ctx.id = static_cast<uint32_t>(packedCtx & 0xFFFFFFFFu);
  ctx.active = static_cast<int32_t>((packedCtx >> 32) & 1);
#ifdef TRACY_ON_DEMAND
  ctx.connectionId = packedCtx >> 33;
#endif
  TracyCZoneEnd(ctx);
}

/// Parses a color string to a Tracy color value.
///
/// Uses TRACY_COLOR_* constants from Tracy.h for consistency across the
/// codebase. Supported colors: "red", "blue", "yellow", "green", "orange",
/// "purple", "gray", "teal".
///
/// color: The color name string.
/// Returns the corresponding TRACY_COLOR_* value, or 0 if empty,
/// or TRACY_COLOR_GREEN as default for unrecognized colors.
inline uint32_t parseTracyColor(llvm::StringRef color) {
  if (color.empty())
    return 0;
  if (color == "red")
    return TRACY_COLOR_RED;
  if (color == "blue")
    return TRACY_COLOR_BLUE;
  if (color == "yellow")
    return TRACY_COLOR_YELLOW;
  if (color == "green")
    return TRACY_COLOR_GREEN;
  if (color == "orange")
    return TRACY_COLOR_ORANGE;
  if (color == "purple")
    return TRACY_COLOR_PURPLE;
  if (color == "gray")
    return TRACY_COLOR_GRAY;
  if (color == "teal")
    return TRACY_COLOR_TEAL;
  return TRACY_COLOR_GREEN; // default
}

#else // !TRACY_ENABLE

// No-op implementations when Tracy is disabled.
inline uint64_t tracyZoneBegin(const char *, size_t, const char *, size_t,
                               const char *, size_t, uint32_t = 0) {
  return 0;
}
inline void tracyZoneEnd(uint64_t) {}
inline uint32_t parseTracyColor(llvm::StringRef) { return 0; }

#endif // TRACY_ENABLE

} // namespace M

#endif // SUPPORT_PROFILING_TRACY_ZONE_H

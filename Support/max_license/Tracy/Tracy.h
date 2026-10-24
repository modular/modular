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
// This file provides convenience macros for using Tracy profiler zones.
// Tracy is a real-time, nanosecond resolution frame profiler that can be used
// for performance analysis.
//
// Tracy profiling is controlled by two compile-time defines:
//   - TRACY_ENABLE: Controls whether Tracy is enabled
//   - TRACY_VERBOSITY: Controls verbosity when Tracy is enabled
//       - 0: Disabled (no zones are created)
//       - 1: Minimal (basic zones only, TRACY_ZONE_SCOPED_NCT falls back to NC)
//       - 2: Detailed (full zones including text via TRACY_ZONE_SCOPED_NCT)
//
// The NCT (Name, Color, Text) variant can add overhead due to string creation,
// so level 1 (minimal) disables it while keeping basic zone tracking.
//
// See Support/docs/Tracy.md for more information on using Tracy profiling.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_TRACY_H
#define SUPPORT_PROFILING_TRACY_H

// Tracy profiling color constants.
#define TRACY_COLOR_TEAL 0x54A1A1
#define TRACY_COLOR_RED 0xC46666
#define TRACY_COLOR_PURPLE 0xA559AA
#define TRACY_COLOR_ORANGE 0xF1A226
#define TRACY_COLOR_GRAY 0xA09C96
#define TRACY_COLOR_BLUE 0x3594CC
#define TRACY_COLOR_YELLOW 0xF2C45F
#define TRACY_COLOR_GREEN 0x00FF00

#ifdef TRACY_ENABLE

#include "tracy/Tracy.hpp"

/// Creates a scoped Tracy profiling zone with a name and color.
/// The zone automatically ends when the scope exits.
#define TRACY_ZONE_SCOPED_NC(name, color) ZoneScopedNC(name, color);

/// Creates a scoped Tracy profiling zone with a name, color, and text.
/// The text provides additional information for the profiling zone.
/// The zone automatically ends when the scope exits.
///
/// Note: This macro is only active at TRACY_VERBOSITY >= 2 (detailed).
/// At level 1 (minimal), it falls back to TRACY_ZONE_SCOPED_NC (without text)
/// to avoid string creation overhead.
#if TRACY_VERBOSITY >= 2
#define TRACY_ZONE_SCOPED_NCT(name, color, text)                               \
  std::string _text = text;                                                    \
  ZoneScopedNC(name, color);                                                   \
  ZoneText(_text.c_str(), _text.size());
#else
// At level 1 (minimal), fall back to basic zone without text to avoid overhead.
#define TRACY_ZONE_SCOPED_NCT(name, color, text) ZoneScopedNC(name, color);
#endif

#else

// No-op macros when Tracy profiling is disabled.
#define TRACY_ZONE_SCOPED_NC(name, color)
#define TRACY_ZONE_SCOPED_NCT(name, color, text)

#endif

#endif // SUPPORT_PROFILING_TRACY_H

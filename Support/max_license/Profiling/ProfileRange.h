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
// Scoped host-range macros for the C++ runtime. Each macro marks the rest of
// the enclosing scope. M_PROFILE_RANGE expands by build:
//
//   - Default: an M::Profiling::RangeScope (Support/Profiling/Ranges.h), so
//     the range shows up in MAX profiler traces. With no live trace it costs
//     one relaxed atomic load.
//   - --//:profile_ranges=false (MODULAR_PROFILE_RANGES=0): nothing. The
//     arguments are not evaluated.
//   - --config=tracy (TRACY_ENABLE): a Tracy zone, because Tracy and the MAX
//     profiler plugin are never linked into the same binary.
//
// M_PROFILE_RANGE_TEXT and M_PROFILE_SPAN_TEXT attach `text` only to Tracy
// zones at TRACY_VERBOSITY >= 2. Ranges have no text field, so the other
// builds do not evaluate `text`, and hot paths never build the string.
//
// The MAX profiler attributes each GPU kernel and copy to the innermost range
// open when it was submitted, so the runtime code around and below the op
// (executor, primitives, memory manager, driver bindings, device contexts) uses
// M_PROFILE_SPAN instead, which records an M::Profiling::SpanScope: the span
// shows in the trace, but the work stays attributed to the op. Recording a
// range or a span costs tens of nanoseconds while a capture is live, and only
// a range pushes a correlation id into the vendor tracer.
//
// M_PROFILE_TRACY_ZONE is a Tracy zone and nothing in other builds. Wrappers
// around single vendor API calls use it, because the vendor tracer already
// records each of those calls, and so do wrappers whose work is shorter than
// recording a span (refcounting, accessors).
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_PROFILERANGE_H
#define SUPPORT_PROFILING_PROFILERANGE_H

#define M_PROFILE_COLOR_TEAL 0x54A1A1
#define M_PROFILE_COLOR_RED 0xC46666
#define M_PROFILE_COLOR_PURPLE 0xA559AA
#define M_PROFILE_COLOR_ORANGE 0xF1A226
#define M_PROFILE_COLOR_GRAY 0xA09C96
#define M_PROFILE_COLOR_BLUE 0x3594CC
#define M_PROFILE_COLOR_YELLOW 0xF2C45F
#define M_PROFILE_COLOR_GREEN 0x00FF00

#ifdef TRACY_ENABLE

#include "tracy/Tracy.hpp"
#include <string>

#define M_PROFILE_RANGE(name, color) ZoneScopedNC(name, color)
#define M_PROFILE_SPAN(name, color) ZoneScopedNC(name, color)
#define M_PROFILE_TRACY_ZONE(name, color) ZoneScopedNC(name, color)
#define M_PROFILE_SPAN_TEXT(name, color, text)                                 \
  M_PROFILE_RANGE_TEXT(name, color, text)

#if TRACY_VERBOSITY >= 2
#define M_PROFILE_RANGE_TEXT(name, color, text)                                \
  std::string _mProfileRangeText = text;                                       \
  ZoneScopedNC(name, color);                                                   \
  ZoneText(_mProfileRangeText.c_str(), _mProfileRangeText.size())
#else
#define M_PROFILE_RANGE_TEXT(name, color, text) ZoneScopedNC(name, color)
#endif

#elif MODULAR_PROFILE_RANGES

#include "Support/Profiling/Ranges.h"

#define M_PROFILE_RANGE_CONCAT_IMPL(a, b) a##b
#define M_PROFILE_RANGE_CONCAT(a, b) M_PROFILE_RANGE_CONCAT_IMPL(a, b)

#define M_PROFILE_RANGE(name, color)                                           \
  ::M::Profiling::RangeScope M_PROFILE_RANGE_CONCAT(_mProfileRange,            \
                                                    __LINE__)(name, color)
#define M_PROFILE_RANGE_TEXT(name, color, text) M_PROFILE_RANGE(name, color)
#define M_PROFILE_SPAN(name, color)                                            \
  ::M::Profiling::SpanScope M_PROFILE_RANGE_CONCAT(_mProfileSpan,              \
                                                   __LINE__)(name, color)
#define M_PROFILE_SPAN_TEXT(name, color, text) M_PROFILE_SPAN(name, color)
#define M_PROFILE_TRACY_ZONE(name, color)

#else

#define M_PROFILE_RANGE(name, color)
#define M_PROFILE_RANGE_TEXT(name, color, text)
#define M_PROFILE_SPAN(name, color)
#define M_PROFILE_SPAN_TEXT(name, color, text)
#define M_PROFILE_TRACY_ZONE(name, color)

#endif

#endif // SUPPORT_PROFILING_PROFILERANGE_H

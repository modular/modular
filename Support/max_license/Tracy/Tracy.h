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
// This file maps the Tracy color names that TracyZone.h uses onto the profiling
// range colors. Scoped zones in C++ use the M_PROFILE_RANGE macros in
// Support/max_license/Profiling/ProfileRange.h instead, which expand to Tracy
// zones in --config=tracy builds.
//
// See docs/internal/Tracy.md for more information on using Tracy profiling.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_TRACY_H
#define SUPPORT_PROFILING_TRACY_H

#include "Support/max_license/Profiling/ProfileRange.h"

#define TRACY_COLOR_TEAL M_PROFILE_COLOR_TEAL
#define TRACY_COLOR_RED M_PROFILE_COLOR_RED
#define TRACY_COLOR_PURPLE M_PROFILE_COLOR_PURPLE
#define TRACY_COLOR_ORANGE M_PROFILE_COLOR_ORANGE
#define TRACY_COLOR_GRAY M_PROFILE_COLOR_GRAY
#define TRACY_COLOR_BLUE M_PROFILE_COLOR_BLUE
#define TRACY_COLOR_YELLOW M_PROFILE_COLOR_YELLOW
#define TRACY_COLOR_GREEN M_PROFILE_COLOR_GREEN

#endif // SUPPORT_PROFILING_TRACY_H

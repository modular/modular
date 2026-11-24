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

#ifndef GLIDIR_VIEWS_COMMON_H
#define GLIDIR_VIEWS_COMMON_H

#include "imgui.h"

namespace M {
namespace GlidIR {

// Bookmark color groups shared by the text and graph views. Names are the
// matching colored-circle emoji in UTF-8.
static constexpr int kNumBookmarkColors = 4;
static constexpr ImU32 kBookmarkColors[kNumBookmarkColors] = {
    IM_COL32(220, 70, 70, 255),  // red
    IM_COL32(230, 130, 50, 255), // orange
    IM_COL32(210, 190, 50, 255), // yellow
    IM_COL32(70, 190, 70, 255),  // green
};
static constexpr const char *kBookmarkColorNames[kNumBookmarkColors] = {
    "\xf0\x9f\x94\xb4", // U+1F534
    "\xf0\x9f\x9f\xa0", // U+1F7E0
    "\xf0\x9f\x9f\xa1", // U+1F7E1
    "\xf0\x9f\x9f\xa2", // U+1F7E2
};

static inline void HelpMarker(const char *desc) {
  ImGui::TextDisabled("(?)");
  if (ImGui::BeginItemTooltip()) {
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

} // namespace GlidIR
} // namespace M

#endif // GLIDIR_VIEWS_COMMON_H

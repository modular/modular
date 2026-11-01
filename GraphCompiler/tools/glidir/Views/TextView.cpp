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

#include "TextView.h"

#include "Common.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"

using namespace mlir;

namespace M {
namespace GlidIR {

static constexpr auto kIconTrash = "\xf0\x9f\x97\x91"; // U+1F5D1

// Longer lines are truncated, preserving the line count for the op mapping.
static constexpr size_t kMaxLineWidth = 1024;
static constexpr llvm::StringLiteral kTruncationSuffix = " ... [truncated]";

static std::string truncateLongLines(const std::string &text) {
  std::string out;
  out.reserve(text.size());
  size_t lineStart = 0;
  for (size_t i = 0; i <= text.size(); ++i) {
    if (i == text.size() || text[i] == '\n') {
      size_t lineLen = i - lineStart;
      if (lineLen > kMaxLineWidth) {
        size_t keep = kMaxLineWidth - kTruncationSuffix.size();
        out.append(text, lineStart, keep);
        out.append(kTruncationSuffix.data(), kTruncationSuffix.size());
      } else {
        out.append(text, lineStart, lineLen);
      }
      if (i < text.size())
        out.push_back('\n');
      lineStart = i + 1;
    }
  }
  return out;
}

void TextView::init(ModuleOp root, ModuleOp dumpRoot,
                    const std::string &dumpText,
                    const TextViewer::Palette &palette) {
  textViewer.SetPalette(palette);
  textViewer.SetLanguageDefinition(TextViewer::LanguageDefinition::MLIR());
  textViewer.SetReadOnly(true);
  textViewer.SetText(truncateLongLines(dumpText));
  textAsLines = textViewer.GetTextLines();

  // Reparsed ops' locations point into the dump text; walk both modules in
  // parallel to map original ops to dump lines.
  SmallVector<Operation *> origOps, dumpOps;
  root->walk([&](Operation *op) { origOps.push_back(op); });
  dumpRoot->walk([&](Operation *op) { dumpOps.push_back(op); });
  lineNumberToOp.resize(textViewer.GetTotalLines(), nullptr);
  for (auto &&[origOp, dumpOp] : llvm::zip(origOps, dumpOps)) {
    if (auto loc = dyn_cast<FileLineColLoc>(dumpOp->getLoc())) {
      int line = (int)loc.getLine() - 1; // convert to 0-indexed
      if (line >= 0 && line < (int)lineNumberToOp.size()) {
        lineNumberToOp[line] = origOp;
        opLineNumber[origOp] = line;
      }
    }
  }
}

void TextView::syncBreakpoints() {
  auto &bp = textViewer.GetBreakpoints();
  bp.clear();
  auto &markers = textViewer.GetLineMarkers();
  markers.clear();
  for (int gi = 0; gi < kNumBookmarkColors; ++gi) {
    for (int line : coloredBookmarks[gi]) {
      bp.insert(line);
      markers[line].push_back(kBookmarkColors[gi]);
    }
  }
}

bool TextView::hasAnyBookmarks() const {
  for (auto &group : coloredBookmarks)
    if (!group.empty())
      return true;
  return false;
}

llvm::SmallVector<int> TextView::sortedBookmarksForGroup(int idx) const {
  return SmallVector<int>(coloredBookmarks[idx].begin(),
                          coloredBookmarks[idx].end());
}

llvm::SmallVector<mlir::Operation *> TextView::opsForGroup(int idx) const {
  SmallVector<Operation *> ops;
  for (int lineNo : coloredBookmarks[idx]) {
    int actualLineNo = lineNo - 1;
    if (actualLineNo >= 0 && actualLineNo < (int)lineNumberToOp.size())
      if (Operation *op = lineNumberToOp[actualLineNo])
        ops.push_back(op);
  }
  return ops;
}

void TextView::bookmarkSelection(int minLine, int maxLine, int colorIdx) {
  for (int line = minLine; line <= maxLine; ++line)
    if (line >= 0 && line < (int)lineNumberToOp.size() && lineNumberToOp[line])
      coloredBookmarks[colorIdx].insert(line + 1);
  syncBreakpoints();
}

void TextView::highlightLine(int lineNo) {
  TextViewer::Coordinates coords;
  coords.mLine = lineNo;
  coords.mColumn = 0;
  textViewer.SetCursorPosition(coords);
  textViewer.SetSelection(coords, coords);
  if (treeCallbacks.clearHighlight)
    treeCallbacks.clearHighlight();
}

void TextView::scrollToOp(Operation *op) {
  auto it = opLineNumber.find(op);
  if (it != opLineNumber.end())
    highlightLine(it->second);
}

void TextView::renderTextFilter() {
  // Re-focus after Enter so repeated presses keep working.
  static char buffer[512] = {0};
  static char prevBuffer[512] = {0};
  static bool refocus = false;
  if (refocus) {
    ImGui::SetKeyboardFocusHere(0);
    refocus = false;
  }
  ImGui::SetNextItemWidth(256);
  bool enterPressed = ImGui::InputText("Search IR", buffer, sizeof(buffer),
                                       ImGuiInputTextFlags_EnterReturnsTrue);
  if (enterPressed)
    refocus = true;
  bool filterChanged = strcmp(buffer, prevBuffer) != 0;
  if (filterChanged)
    memcpy(prevBuffer, buffer, sizeof(prevBuffer));
  ImGui::SameLine();
  HelpMarker("Search using a regex");

  for (int i = 0; i < kNumBookmarkColors; ++i) {
    ImGui::SameLine(0, i == 0 ? 8 : 2);
    ImGui::PushID(2000 + i);
    bool isActive = (i == activeBookmarkColor);
    if (isActive) {
      ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(70, 70, 90, 220));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(90, 90, 110, 240));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(50, 50, 70, 255));
    } else {
      ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(0, 0, 0, 0));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(50, 50, 70, 150));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(70, 70, 90, 200));
    }
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.0f, 2.0f));
    if (ImGui::Button(kBookmarkColorNames[i]))
      activeBookmarkColor = i;
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);
    ImGui::PopID();
  }

  if (filterChanged) {
    currentMatchedIndex.reset();
    matchedLines.clear();
    llvm::Regex pattern(buffer);
    for (auto &&[idx, line] : llvm::enumerate(textAsLines)) {
      if (pattern.match(line.c_str()))
        matchedLines.push_back(idx);
    }
  }

  if (matchedLines.empty())
    return;

  ImGui::SameLine();
  bool prev = ImGui::SmallButton("\xe2\xac\x85##ir"); // U+2B05
  ImGui::SameLine();
  bool next = ImGui::SmallButton("\xe2\x9e\xa1##ir") || enterPressed; // U+27A1
  bool somethingWasClicked = prev || next;

  ImGui::SameLine();
  if (!somethingWasClicked && !currentMatchedIndex.has_value()) {
    ImGui::Text("-/%lu", matchedLines.size());
    return;
  }

  // Step from the current match, not the cursor (clicking can move it); on
  // first use, start from the match nearest the cursor.
  if (somethingWasClicked) {
    if (currentMatchedIndex.has_value()) {
      int idx = *currentMatchedIndex;
      if (next)
        idx = (idx + 1) % (int)matchedLines.size();
      else
        idx = (idx == 0) ? (int)matchedLines.size() - 1 : idx - 1;
      currentMatchedIndex = idx;
    } else {
      int cursorLine = textViewer.GetCursorPosition().mLine;
      if (next) {
        currentMatchedIndex = 0;
        for (auto &&[idx, match] : llvm::enumerate(matchedLines))
          if (match >= cursorLine) {
            currentMatchedIndex = idx;
            break;
          }
      } else {
        currentMatchedIndex = (int)matchedLines.size() - 1;
        for (int idx = (int)matchedLines.size() - 1; idx >= 0; --idx)
          if (matchedLines[idx] <= cursorLine) {
            currentMatchedIndex = idx;
            break;
          }
      }
    }
  }

  ImGui::Text("%d/%lu", *currentMatchedIndex + 1, matchedLines.size());

  if (somethingWasClicked) {
    highlightLine(matchedLines[*currentMatchedIndex]);
    if (treeCallbacks.highlightOp)
      treeCallbacks.highlightOp(
          lineNumberToOp[matchedLines[*currentMatchedIndex]]);
  }
}

void TextView::processTextViewerCursorEvents() {
  if (textViewer.IsCursorPositionChanged()) {
    if (treeCallbacks.highlightOp)
      treeCallbacks.highlightOp(
          lineNumberToOp[textViewer.GetCursorPosition().mLine]);
  }

  if (textViewer.DoubleClick()) {
    int cursorLine = textViewer.GetCursorPosition().mLine;
    if (lineNumberToOp[cursorLine]) {
      int line1indexed = cursorLine + 1;
      auto &activeGroup = coloredBookmarks[activeBookmarkColor];
      if (activeGroup.count(line1indexed))
        activeGroup.erase(line1indexed);
      else
        activeGroup.insert(line1indexed);
      syncBreakpoints();
    }
  }
}

void TextView::renderBookmarks() {
  if (!hasAnyBookmarks()) {
    HelpMarker("Double click operations in the IR to bookmark them here");
    return;
  }

  ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(192, 0, 0, 255));
  if (ImGui::SmallButton("Clear all")) {
    for (auto &group : coloredBookmarks)
      group.clear();
    syncBreakpoints();
  }
  ImGui::PopStyleColor();

  const int lineNoWidth =
      (int)std::ceil(std::log10(std::max<size_t>(1, textAsLines.size())));
  const std::string spaces(lineNoWidth, ' ');

  for (int gi = 0; gi < kNumBookmarkColors; ++gi) {
    auto &group = coloredBookmarks[gi];
    if (group.empty())
      continue;

    ImGui::Separator();
    ImGui::Text("%s", kBookmarkColorNames[gi]);

    SmallVector<int> toErase;
    for (int lineNo : group) {
      int actualLineNo = lineNo - 1;
      if (!lineNumberToOp[actualLineNo])
        continue;
      auto &line = textAsLines[actualLineNo];
      std::string lineNoAsStr = std::to_string(lineNo);
      std::string lineNoPreceded =
          llvm::formatv("{}{}", &spaces.c_str()[lineNoAsStr.size()],
                        lineNoAsStr)
              .str();

      ImGui::PushID(lineNo * kNumBookmarkColors + gi);
      ImGui::PushStyleColor(ImGuiCol_Button, kBookmarkColors[gi]);
      if (ImGui::SmallButton(lineNoPreceded.c_str())) {
        highlightLine(actualLineNo);
        if (treeCallbacks.highlightOp)
          treeCallbacks.highlightOp(lineNumberToOp[actualLineNo]);
      }
      ImGui::PopStyleColor();

      ImGui::SameLine(0, 2);
      ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(192, 0, 0, 255));
      bool deleted = ImGui::SmallButton(kIconTrash);
      ImGui::PopStyleColor();
      ImGui::PopID();

      if (deleted) {
        toErase.push_back(lineNo);
      } else {
        ImGui::SameLine();
        const char *trimmed = line.c_str();
        while (*trimmed == ' ' || *trimmed == '\t')
          ++trimmed;
        ImGui::Text("%s", trimmed);
      }
    }
    for (int lineNo : toErase)
      group.erase(lineNo);
    if (!toErase.empty())
      syncBreakpoints();
  }
}

} // namespace GlidIR
} // namespace M

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

#include "SourceView.h"

#include "Common.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Regex.h"

#include <fstream>
#include <limits>
#include <sstream>

using namespace mlir;

namespace M {
namespace GlidIR {

void collectFileLineColLocs(Location loc,
                            SmallVectorImpl<FileLineColLoc> &result) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    result.push_back(fileLoc);
  else if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
    for (Location inner : fusedLoc.getLocations())
      collectFileLineColLocs(inner, result);
}

SourceView::SourceView(ModuleOp root) {
  root.walk([&](Operation *op) {
    SmallVector<FileLineColLoc> locs;
    collectFileLineColLocs(op->getLoc(), locs);
    for (FileLineColLoc loc : locs) {
      std::string path = loc.getFilename().str();

      int fileIdx = -1;
      for (int i = 0; i < (int)files.size(); i++)
        if (files[i].path == path) {
          fileIdx = i;
          break;
        }
      if (fileIdx < 0) {
        fileIdx = (int)files.size();
        SourceFile sf;
        sf.path = path;
        std::ifstream f(path);
        if (f.good()) {
          std::ostringstream ss;
          ss << f.rdbuf();
          sf.content = ss.str();
        }
        files.push_back(std::move(sf));
      }
      int line = std::max(0, (int)loc.getLine() - 1);
      auto &ops = files[fileIdx].lineToOp[line];
      if (!llvm::is_contained(ops, op))
        ops.push_back(op);
    }
  });
}

void SourceView::initViewer(const TextViewer::Palette &palette) {
  TextViewer::Palette p = palette;
  // Flatten token colors so C/C++ highlighting doesn't bleed onto Mojo source.
  ImU32 fg = p[(int)TextViewer::PaletteIndex::Default];
  for (int i = (int)TextViewer::PaletteIndex::Keyword;
       i < (int)TextViewer::PaletteIndex::Background; ++i)
    p[i] = fg;
  textViewer.SetPalette(p);
  textViewer.SetReadOnly(true);
}

void SourceView::renderSourceFilter() {
  if (searchRefocus) {
    ImGui::SetKeyboardFocusHere(0);
    searchRefocus = false;
  }
  ImGui::SetNextItemWidth(256);
  bool enterPressed =
      ImGui::InputText("Search Source", searchBuffer, sizeof(searchBuffer),
                       ImGuiInputTextFlags_EnterReturnsTrue);
  if (enterPressed)
    searchRefocus = true;

  bool filterChanged = strcmp(searchBuffer, prevSearchBuffer) != 0;
  if (filterChanged) {
    memcpy(prevSearchBuffer, searchBuffer, sizeof(prevSearchBuffer));
    searchCurrentIndex.reset();
    searchMatchedLines.clear();
    if (searchBuffer[0] != '\0') {
      llvm::Regex pattern(searchBuffer);
      for (auto &&[idx, line] : llvm::enumerate(cachedTextLines))
        if (pattern.match(line))
          searchMatchedLines.push_back(idx);
    }
  }

  ImGui::SameLine();
  HelpMarker("Search using a regex");

  if (searchMatchedLines.empty())
    return;

  ImGui::SameLine();
  bool prev = ImGui::SmallButton("\xe2\xac\x85##src"); // U+2B05
  ImGui::SameLine();
  bool next = ImGui::SmallButton("\xe2\x9e\xa1##src") || enterPressed; // U+27A1
  bool somethingWasClicked = prev || next;

  ImGui::SameLine();
  if (!somethingWasClicked && !searchCurrentIndex.has_value()) {
    ImGui::Text("-/%zu", searchMatchedLines.size());
    return;
  }

  if (somethingWasClicked) {
    if (searchCurrentIndex.has_value()) {
      int idx = *searchCurrentIndex;
      if (next)
        idx = (idx + 1) % (int)searchMatchedLines.size();
      else
        idx = (idx == 0) ? (int)searchMatchedLines.size() - 1 : idx - 1;
      searchCurrentIndex = idx;
    } else {
      int cursorLine = textViewer.GetCursorPosition().mLine;
      if (next) {
        searchCurrentIndex = 0;
        for (auto &&[idx, match] : llvm::enumerate(searchMatchedLines))
          if (match >= cursorLine) {
            searchCurrentIndex = idx;
            break;
          }
      } else {
        searchCurrentIndex = (int)searchMatchedLines.size() - 1;
        for (int idx = (int)searchMatchedLines.size() - 1; idx >= 0; --idx)
          if (searchMatchedLines[idx] <= cursorLine) {
            searchCurrentIndex = idx;
            break;
          }
      }
    }
    pendingSearchLine = searchMatchedLines[*searchCurrentIndex];
  }

  ImGui::Text("%d/%zu", *searchCurrentIndex + 1, searchMatchedLines.size());
}

void SourceView::render(
    Operation *highlightedOp,
    std::function<void(ArrayRef<Operation *>)> onSourceLineClicked) {
  if (files.empty()) {
    ImGui::TextDisabled("No source locations found in the IR.");
    return;
  }

  bool opChanged = highlightedOp != prevHighlightedOp;
  prevHighlightedOp = highlightedOp;

  // All of a fused loc's leaves in this file contribute, mirroring the
  // source -> IR mapping.
  int highlightedMinLine = -1, highlightedMaxLine = -1;
  if (highlightedOp) {
    SmallVector<FileLineColLoc> locs;
    collectFileLineColLocs(highlightedOp->getLoc(), locs);
    // When the op changes, adopt the file of the first matching loc.
    bool fileSelected = !opChanged;
    for (FileLineColLoc loc : locs) {
      std::string path = loc.getFilename().str();
      for (int i = 0; i < (int)files.size(); i++) {
        if (files[i].path != path)
          continue;
        if (!fileSelected) {
          currentFileIndex = i;
          fileSelected = true;
        }
        if (i == currentFileIndex) {
          int line = std::max(0, (int)loc.getLine() - 1);
          if (highlightedMinLine < 0)
            highlightedMinLine = highlightedMaxLine = line;
          else {
            highlightedMinLine = std::min(highlightedMinLine, line);
            highlightedMaxLine = std::max(highlightedMaxLine, line);
          }
        }
        break;
      }
    }
  }

  // Basename in the button, full path in the tooltip.
  const std::string &fullPath = files[currentFileIndex].path;
  std::string basename = fullPath;
  if (auto pos = fullPath.rfind('/'); pos != std::string::npos)
    basename = fullPath.substr(pos + 1);

  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80.0f);
  if (ImGui::BeginCombo("##srcfile", basename.c_str())) {
    for (int i = 0; i < (int)files.size(); i++) {
      bool sel = (i == currentFileIndex);
      if (ImGui::Selectable(files[i].path.c_str(), sel))
        currentFileIndex = i;
      if (sel)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  if (ImGui::BeginItemTooltip()) {
    ImGui::TextUnformatted(fullPath.c_str());
    ImGui::EndTooltip();
  }

  const auto &file = files[currentFileIndex];

  if (!loadedFileIndex.has_value() || currentFileIndex != *loadedFileIndex) {
    textViewer.SetText(file.content.empty() ? std::string("(file not found)")
                                            : file.content);
    loadedFileIndex = currentFileIndex;
    cachedTextLines = textViewer.GetTextLines();
    searchMatchedLines.clear();
    searchCurrentIndex.reset();
    memset(searchBuffer, 0, sizeof(searchBuffer));
    memset(prevSearchBuffer, 0, sizeof(prevSearchBuffer));
  }

  renderSourceFilter();

  // IR -> Source: before Render(), whose flag reset then suppresses a
  // spurious Source -> IR callback this frame.
  if (opChanged && highlightedMinLine >= 0) {
    textViewer.SetCursorPosition({highlightedMinLine, 0});
    textViewer.SetSelection({highlightedMinLine, 0},
                            {highlightedMaxLine, INT_MAX});
  } else if (pendingSearchLine >= 0) {
    textViewer.SetCursorPosition({pendingSearchLine, 0});
    textViewer.SetSelection({pendingSearchLine, 0},
                            {pendingSearchLine, INT_MAX});
    pendingSearchLine = -1;
  }

  textViewer.Render("SourceText");

  // Source -> IR.
  if (textViewer.IsCursorPositionChanged()) {
    int line = textViewer.GetCursorPosition().mLine;
    auto opIt = file.lineToOp.find(line);
    if (opIt != file.lineToOp.end())
      onSourceLineClicked(opIt->second);
  }
}

} // namespace GlidIR
} // namespace M

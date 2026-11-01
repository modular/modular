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
// Taken from ImGuiColorTextEdit
// (https://github.com/BalazsJako/ImGuiColorTextEdit) and modified for our
// needs: stripped to read-only, added drag selection, right-click detection.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <regex>
#include <string>

#include "TextViewer.h"

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h" // for imGui::GetCurrentWindow()

namespace ImGui {
static inline IMGUI_API ImGuiKey GetKeyIndex(ImGuiKey key) { return key; }
} // namespace ImGui

template <class InputIt1, class InputIt2, class BinaryPredicate>
bool equals(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2,
            BinaryPredicate p) {
  for (; first1 != last1 && first2 != last2; ++first1, ++first2) {
    if (!p(*first1, *first2))
      return false;
  }
  return first1 == last1 && first2 == last2;
}

TextViewer::TextViewer()
    : mLineSpacing(1.0f), mTabSize(4), mWithinRender(false),
      mScrollToCursor(false), mScrollToTop(false), mColorizerEnabled(true),
      mTextStart(20.0f), mLeftMargin(10), mCursorPositionChanged(false),
      mColorRangeMin(0), mColorRangeMax(0),
      mSelectionMode(SelectionMode::Normal), mHandleKeyboardInputs(true),
      mHandleMouseInputs(true), mIgnoreImGuiChild(false),
      mShowWhitespaces(true), mCheckComments(true),
      mStartTime(std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count()),
      mLastClick(-1.0f), mDoubleClick(false), mRightClick(false) {
  SetPalette(GetDarkPalette());
  mLines.push_back(Line());
}

TextViewer::~TextViewer() {}

void TextViewer::SetLanguageDefinition(const LanguageDefinition &aLanguageDef) {
  mLanguageDefinition = aLanguageDef;
  mRegexList.clear();

  for (auto &r : mLanguageDefinition.mTokenRegexStrings)
    mRegexList.push_back(std::make_pair(
        std::regex(r.first, std::regex_constants::optimize), r.second));

  Colorize();
}

void TextViewer::SetPalette(const Palette &aValue) { mPaletteBase = aValue; }

std::string TextViewer::GetText(const Coordinates &aStart,
                                const Coordinates &aEnd) const {
  std::string result;

  auto lstart = aStart.mLine;
  auto lend = aEnd.mLine;
  auto istart = GetCharacterIndex(aStart);
  auto iend = GetCharacterIndex(aEnd);
  size_t s = 0;

  for (int i = lstart; i < lend; i++)
    s += mLines[i].size();

  result.reserve(s + s / 8);

  while (istart < iend || lstart < lend) {
    if (lstart >= (int)mLines.size())
      break;

    auto &line = mLines[lstart];
    if (istart < (int)line.size()) {
      result += line[istart].mChar;
      istart++;
    } else {
      istart = 0;
      ++lstart;
      result += '\n';
    }
  }

  return result;
}

TextViewer::Coordinates TextViewer::GetActualCursorCoordinates() const {
  return SanitizeCoordinates(mState.mCursorPosition);
}

TextViewer::Coordinates
TextViewer::SanitizeCoordinates(const Coordinates &aValue) const {
  auto line = aValue.mLine;
  auto column = aValue.mColumn;
  if (line >= (int)mLines.size()) {
    if (mLines.empty()) {
      line = 0;
      column = 0;
    } else {
      line = (int)mLines.size() - 1;
      column = GetLineMaxColumn(line);
    }
    return Coordinates(line, column);
  } else {
    column = mLines.empty() ? 0 : std::min(column, GetLineMaxColumn(line));
    return Coordinates(line, column);
  }
}

// https://en.wikipedia.org/wiki/UTF-8
// We assume that the char is a standalone character (<128) or a leading byte of
// an UTF-8 code sequence (non-10xxxxxx code)
static int UTF8CharLength(TextViewer::Char c) {
  if ((c & 0xFE) == 0xFC)
    return 6;
  if ((c & 0xFC) == 0xF8)
    return 5;
  if ((c & 0xF8) == 0xF0)
    return 4;
  else if ((c & 0xF0) == 0xE0)
    return 3;
  else if ((c & 0xE0) == 0xC0)
    return 2;
  return 1;
}

TextViewer::Coordinates
TextViewer::ScreenPosToCoordinates(const ImVec2 &aPosition) const {
  ImVec2 origin = ImGui::GetCursorScreenPos();
  ImVec2 local(aPosition.x - origin.x, aPosition.y - origin.y);

  int lineNo = std::max(0, (int)floor(local.y / mCharAdvance.y));

  int columnCoord = 0;

  if (lineNo >= 0 && lineNo < (int)mLines.size()) {
    auto &line = mLines.at(lineNo);

    int columnIndex = 0;
    float columnX = 0.0f;

    while ((size_t)columnIndex < line.size()) {
      float columnWidth = 0.0f;

      if (line[columnIndex].mChar == '\t') {
        float spaceSize =
            ImGui::GetFont()
                ->CalcTextSizeA(ImGui::GetFontSize(), FLT_MAX, -1.0f, " ")
                .x;
        float oldX = columnX;
        float newColumnX = (1.0f + std::floor((1.0f + columnX) /
                                              (float(mTabSize) * spaceSize))) *
                           (float(mTabSize) * spaceSize);
        columnWidth = newColumnX - oldX;
        if (mTextStart + columnX + columnWidth * 0.5f > local.x)
          break;
        columnX = newColumnX;
        columnCoord = (columnCoord / mTabSize) * mTabSize + mTabSize;
        columnIndex++;
      } else {
        char buf[7];
        auto d = UTF8CharLength(line[columnIndex].mChar);
        int i = 0;
        while (i < 6 && d-- > 0)
          buf[i++] = line[columnIndex++].mChar;
        buf[i] = '\0';
        columnWidth =
            ImGui::GetFont()
                ->CalcTextSizeA(ImGui::GetFontSize(), FLT_MAX, -1.0f, buf)
                .x;
        if (mTextStart + columnX + columnWidth * 0.5f > local.x)
          break;
        columnX += columnWidth;
        columnCoord++;
      }
    }
  }

  return SanitizeCoordinates(Coordinates(lineNo, columnCoord));
}

TextViewer::Coordinates
TextViewer::FindWordStart(const Coordinates &aFrom) const {
  Coordinates at = aFrom;
  if (at.mLine >= (int)mLines.size())
    return at;

  auto &line = mLines[at.mLine];
  auto cindex = GetCharacterIndex(at);

  if (cindex >= (int)line.size())
    return at;

  while (cindex > 0 && isspace(line[cindex].mChar))
    --cindex;

  auto cstart = (PaletteIndex)line[cindex].mColorIndex;
  while (cindex > 0) {
    auto c = line[cindex].mChar;
    if ((c & 0xC0) != 0x80) // not UTF code sequence 10xxxxxx
    {
      if (c <= 32 && isspace(c)) {
        cindex++;
        break;
      }
      if (cstart != (PaletteIndex)line[size_t(cindex - 1)].mColorIndex)
        break;
    }
    --cindex;
  }
  return Coordinates(at.mLine, GetCharacterColumn(at.mLine, cindex));
}

TextViewer::Coordinates
TextViewer::FindWordEnd(const Coordinates &aFrom) const {
  Coordinates at = aFrom;
  if (at.mLine >= (int)mLines.size())
    return at;

  auto &line = mLines[at.mLine];
  auto cindex = GetCharacterIndex(at);

  if (cindex >= (int)line.size())
    return at;

  bool prevspace = (bool)isspace(line[cindex].mChar);
  auto cstart = (PaletteIndex)line[cindex].mColorIndex;
  while (cindex < (int)line.size()) {
    auto c = line[cindex].mChar;
    auto d = UTF8CharLength(c);
    if (cstart != (PaletteIndex)line[cindex].mColorIndex)
      break;

    if (prevspace != !!isspace(c)) {
      if (isspace(c))
        while (cindex < (int)line.size() && isspace(line[cindex].mChar))
          ++cindex;
      break;
    }
    cindex += d;
  }
  return Coordinates(aFrom.mLine, GetCharacterColumn(aFrom.mLine, cindex));
}

TextViewer::Coordinates
TextViewer::FindNextWord(const Coordinates &aFrom) const {
  Coordinates at = aFrom;
  if (at.mLine >= (int)mLines.size())
    return at;

  // skip to the next non-word character
  auto cindex = GetCharacterIndex(aFrom);
  bool isword = false;
  bool skip = false;
  if (cindex < (int)mLines[at.mLine].size()) {
    auto &line = mLines[at.mLine];
    isword = isalnum(line[cindex].mChar);
    skip = isword;
  }

  while (!isword || skip) {
    if (at.mLine >= (int)mLines.size()) {
      auto l = std::max(0, (int)mLines.size() - 1);
      return Coordinates(l, GetLineMaxColumn(l));
    }

    auto &line = mLines[at.mLine];
    if (cindex < (int)line.size()) {
      isword = isalnum(line[cindex].mChar);

      if (isword && !skip)
        return Coordinates(at.mLine, GetCharacterColumn(at.mLine, cindex));

      if (!isword)
        skip = false;

      cindex++;
    } else {
      cindex = 0;
      ++at.mLine;
      skip = false;
      isword = false;
    }
  }

  return at;
}

int TextViewer::GetCharacterIndex(const Coordinates &aCoordinates) const {
  if (aCoordinates.mLine >= (int)mLines.size())
    return -1;
  auto &line = mLines[aCoordinates.mLine];
  int c = 0;
  int i = 0;
  for (; i < (int)line.size() && c < aCoordinates.mColumn;) {
    if (line[i].mChar == '\t')
      c = (c / mTabSize) * mTabSize + mTabSize;
    else
      ++c;
    i += UTF8CharLength(line[i].mChar);
  }
  return i;
}

int TextViewer::GetCharacterColumn(int aLine, int aIndex) const {
  if (aLine >= (int)mLines.size())
    return 0;
  auto &line = mLines[aLine];
  int col = 0;
  int i = 0;
  while (i < aIndex && i < (int)line.size()) {
    auto c = line[i].mChar;
    i += UTF8CharLength(c);
    if (c == '\t')
      col = (col / mTabSize) * mTabSize + mTabSize;
    else
      col++;
  }
  return col;
}

int TextViewer::GetLineMaxColumn(int aLine) const {
  if (aLine >= (int)mLines.size())
    return 0;
  auto &line = mLines[aLine];
  int col = 0;
  for (unsigned i = 0; i < line.size();) {
    auto c = line[i].mChar;
    if (c == '\t')
      col = (col / mTabSize) * mTabSize + mTabSize;
    else
      col++;
    i += UTF8CharLength(c);
  }
  return col;
}

bool TextViewer::IsOnWordBoundary(const Coordinates &aAt) const {
  if (aAt.mLine >= (int)mLines.size() || aAt.mColumn == 0)
    return true;

  auto &line = mLines[aAt.mLine];
  auto cindex = GetCharacterIndex(aAt);
  if (cindex >= (int)line.size())
    return true;

  if (mColorizerEnabled)
    return line[cindex].mColorIndex != line[size_t(cindex - 1)].mColorIndex;

  return isspace(line[cindex].mChar) != isspace(line[cindex - 1].mChar);
}

std::string TextViewer::GetWordAt(const Coordinates &aCoords) const {
  auto start = FindWordStart(aCoords);
  auto end = FindWordEnd(aCoords);

  std::string r;

  auto istart = GetCharacterIndex(start);
  auto iend = GetCharacterIndex(end);

  for (auto it = istart; it < iend; ++it)
    r.push_back(mLines[aCoords.mLine][it].mChar);

  return r;
}

ImU32 TextViewer::GetGlyphColor(const Glyph &aGlyph) const {
  if (!mColorizerEnabled)
    return mPalette[(int)PaletteIndex::Default];
  if (aGlyph.mComment)
    return mPalette[(int)PaletteIndex::Comment];
  if (aGlyph.mMultiLineComment)
    return mPalette[(int)PaletteIndex::MultiLineComment];
  auto const color = mPalette[(int)aGlyph.mColorIndex];
  if (aGlyph.mPreprocessor) {
    const auto ppcolor = mPalette[(int)PaletteIndex::Preprocessor];
    const int c0 = ((ppcolor & 0xff) + (color & 0xff)) / 2;
    const int c1 = (((ppcolor >> 8) & 0xff) + ((color >> 8) & 0xff)) / 2;
    const int c2 = (((ppcolor >> 16) & 0xff) + ((color >> 16) & 0xff)) / 2;
    const int c3 = (((ppcolor >> 24) & 0xff) + ((color >> 24) & 0xff)) / 2;
    return ImU32(c0 | (c1 << 8) | (c2 << 16) | (c3 << 24));
  }
  return color;
}

void TextViewer::HandleKeyboardInputs() {
  ImGuiIO &io = ImGui::GetIO();
  auto shift = io.KeyShift;
  auto ctrl = io.ConfigMacOSXBehaviors ? io.KeySuper : io.KeyCtrl;
  auto alt = io.ConfigMacOSXBehaviors ? io.KeyCtrl : io.KeyAlt;

  if (ImGui::IsWindowFocused()) {
    if (ImGui::IsWindowHovered())
      ImGui::SetMouseCursor(ImGuiMouseCursor_TextInput);

    io.WantCaptureKeyboard = true;
    io.WantTextInput = true;

    if (ctrl && !shift && !alt &&
        ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Insert)))
      Copy();
    else if (ctrl && !shift && !alt &&
             ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_C)))
      Copy();
    else if (ctrl && !shift && !alt &&
             ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_X)))
      Copy();
    else if (!ctrl && shift && !alt &&
             ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Delete)))
      Copy();
  }
}

void TextViewer::HandleMouseInputs() {
  ImGuiIO &io = ImGui::GetIO();
  auto shift = io.KeyShift;
  auto ctrl = io.ConfigMacOSXBehaviors ? io.KeySuper : io.KeyCtrl;
  auto alt = io.ConfigMacOSXBehaviors ? io.KeyCtrl : io.KeyAlt;

  mDoubleClick = false;
  if (ImGui::IsWindowHovered()) {
    if (!shift && !alt) {
      auto click = ImGui::IsMouseClicked(0);
      auto doubleClick = ImGui::IsMouseDoubleClicked(0);

      if (click || doubleClick) {
        mState.mCursorPosition = mInteractiveStart = mInteractiveEnd =
            ScreenPosToCoordinates(ImGui::GetMousePos());
        if (ctrl)
          mSelectionMode = SelectionMode::Word;
        else
          mSelectionMode = SelectionMode::Normal;
        SetSelection(mInteractiveStart, mInteractiveEnd, mSelectionMode);

        if (doubleClick) {
          mDoubleClick = true;
        }

        mLastClick = (float)ImGui::GetTime();
      } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        mInteractiveEnd = ScreenPosToCoordinates(ImGui::GetMousePos());
        mState.mCursorPosition = mInteractiveEnd;
        SetSelection(mInteractiveStart, mInteractiveEnd, mSelectionMode);
      }

      if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
        mRightClick = true;
    }
  }
}

void TextViewer::Render() {
  /* Compute mCharAdvance regarding to scaled font size (Ctrl + mouse wheel)*/
  const float fontSize = ImGui::GetFont()
                             ->CalcTextSizeA(ImGui::GetFontSize(), FLT_MAX,
                                             -1.0f, "#", nullptr, nullptr)
                             .x;
  mCharAdvance =
      ImVec2(fontSize, ImGui::GetTextLineHeightWithSpacing() * mLineSpacing);

  /* Update palette with the current alpha from style */
  for (int i = 0; i < (int)PaletteIndex::Max; ++i) {
    auto color = ImGui::ColorConvertU32ToFloat4(mPaletteBase[i]);
    color.w *= ImGui::GetStyle().Alpha;
    mPalette[i] = ImGui::ColorConvertFloat4ToU32(color);
  }

  assert(mLineBuffer.empty());

  auto contentSize = ImGui::GetWindowContentRegionMax();
  auto drawList = ImGui::GetWindowDrawList();
  float longest(mTextStart);

  if (mScrollToTop) {
    mScrollToTop = false;
    ImGui::SetScrollY(0.f);
  }

  ImVec2 cursorScreenPos = ImGui::GetCursorScreenPos();
  auto scrollX = ImGui::GetScrollX();
  auto scrollY = ImGui::GetScrollY();

  auto lineNo = (int)floor(scrollY / mCharAdvance.y);
  auto globalLineMax = (int)mLines.size();
  auto lineMax =
      std::max(0, std::min((int)mLines.size() - 1,
                           lineNo + (int)floor((scrollY + contentSize.y) /
                                               mCharAdvance.y)));

  // mTextStart: width of the largest line number plus two spaces.
  char buf[16];
  snprintf(buf, 16, " %d ", globalLineMax);
  mTextStart = ImGui::GetFont()
                   ->CalcTextSizeA(ImGui::GetFontSize(), FLT_MAX, -1.0f, buf,
                                   nullptr, nullptr)
                   .x +
               mLeftMargin;

  if (!mLines.empty()) {
    float spaceSize = ImGui::GetFont()
                          ->CalcTextSizeA(ImGui::GetFontSize(), FLT_MAX, -1.0f,
                                          " ", nullptr, nullptr)
                          .x;

    while (lineNo <= lineMax) {
      ImVec2 lineStartScreenPos = ImVec2(
          cursorScreenPos.x, cursorScreenPos.y + lineNo * mCharAdvance.y);
      ImVec2 textScreenPos =
          ImVec2(lineStartScreenPos.x + mTextStart, lineStartScreenPos.y);

      auto &line = mLines[lineNo];
      longest = std::max(mTextStart + TextDistanceToLineStart(Coordinates(
                                          lineNo, GetLineMaxColumn(lineNo))),
                         longest);
      Coordinates lineStartCoord(lineNo, 0);
      Coordinates lineEndCoord(lineNo, GetLineMaxColumn(lineNo));

      // Draw selection for the current line
      float sstart = -1.0f;
      float ssend = -1.0f;

      assert(mState.mSelectionStart <= mState.mSelectionEnd);
      if (mState.mSelectionStart <= lineEndCoord)
        sstart = mState.mSelectionStart > lineStartCoord
                     ? TextDistanceToLineStart(mState.mSelectionStart)
                     : 0.0f;
      if (mState.mSelectionEnd > lineStartCoord)
        ssend = TextDistanceToLineStart(mState.mSelectionEnd < lineEndCoord
                                            ? mState.mSelectionEnd
                                            : lineEndCoord);

      if (mState.mSelectionEnd.mLine > lineNo)
        ssend += mCharAdvance.x;

      if (sstart != -1 && ssend != -1 && sstart < ssend) {
        ImVec2 vstart(lineStartScreenPos.x + mTextStart + sstart,
                      lineStartScreenPos.y);
        ImVec2 vend(lineStartScreenPos.x + mTextStart + ssend,
                    lineStartScreenPos.y + mCharAdvance.y);
        drawList->AddRectFilled(vstart, vend,
                                mPalette[(int)PaletteIndex::Selection]);
      }

      // Also anchors the bookmark circles.
      snprintf(buf, 16, "%d  ", lineNo + 1);
      auto lineNoWidth = ImGui::GetFont()
                             ->CalcTextSizeA(ImGui::GetFontSize(), FLT_MAX,
                                             -1.0f, buf, nullptr, nullptr)
                             .x;

      // Bookmark circles, right-anchored before the line number so more
      // circles grow leftward.
      auto markerIt = mLineMarkers.find(lineNo + 1);
      if (markerIt != mLineMarkers.end() && !markerIt->second.empty()) {
        const auto &colors = markerIt->second;
        int numCircles = (int)colors.size();
        float r = mCharAdvance.y * 0.28f;
        float step = r * 0.7f;
        float cy = lineStartScreenPos.y + mCharAdvance.y * 0.5f;
        // Right edge of the last circle sits 2px before the line number text.
        float rightAnchor =
            lineStartScreenPos.x + mTextStart - lineNoWidth - 2.0f;
        float cx = rightAnchor - r - (numCircles - 1) * step;
        for (ImU32 color : colors) {
          drawList->AddCircleFilled({cx, cy}, r, color, 16);
          cx += step;
        }
      }

      auto start = ImVec2(lineStartScreenPos.x + scrollX, lineStartScreenPos.y);

      // Draw error markers
      auto errorIt = mErrorMarkers.find(lineNo + 1);
      if (errorIt != mErrorMarkers.end()) {
        auto end = ImVec2(lineStartScreenPos.x + contentSize.x + 2.0f * scrollX,
                          lineStartScreenPos.y + mCharAdvance.y);
        drawList->AddRectFilled(start, end,
                                mPalette[(int)PaletteIndex::ErrorMarker]);

        if (ImGui::IsMouseHoveringRect(lineStartScreenPos, end)) {
          ImGui::BeginTooltip();
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
          ImGui::Text("Error at line %d:", errorIt->first);
          ImGui::PopStyleColor();
          ImGui::Separator();
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.2f, 1.0f));
          ImGui::Text("%s", errorIt->second.c_str());
          ImGui::PopStyleColor();
          ImGui::EndTooltip();
        }
      }

      // Draw line number (right aligned)
      drawList->AddText(ImVec2(lineStartScreenPos.x + mTextStart - lineNoWidth,
                               lineStartScreenPos.y),
                        mPalette[(int)PaletteIndex::LineNumber], buf);

      if (mState.mCursorPosition.mLine == lineNo) {
        auto focused = ImGui::IsWindowFocused();

        // Highlight the current line (where the cursor is)
        if (!HasSelection()) {
          auto end = ImVec2(start.x + contentSize.x + scrollX,
                            start.y + mCharAdvance.y);
          drawList->AddRectFilled(
              start, end,
              mPalette[(int)(focused ? PaletteIndex::CurrentLineFill
                                     : PaletteIndex::CurrentLineFillInactive)]);
          drawList->AddRect(start, end,
                            mPalette[(int)PaletteIndex::CurrentLineEdge], 1.0f);
        }

        // Render the cursor
        if (focused) {
          auto timeEnd =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
          auto elapsed = timeEnd - mStartTime;
          if (elapsed > 400) {
            float cx = TextDistanceToLineStart(mState.mCursorPosition);
            ImVec2 cstart(textScreenPos.x + cx, lineStartScreenPos.y);
            ImVec2 cend(textScreenPos.x + cx + 1.0f,
                        lineStartScreenPos.y + mCharAdvance.y);
            drawList->AddRectFilled(cstart, cend,
                                    mPalette[(int)PaletteIndex::Cursor]);
            if (elapsed > 800)
              mStartTime = timeEnd;
          }
        }
      }

      // Render colorized text
      auto prevColor = line.empty() ? mPalette[(int)PaletteIndex::Default]
                                    : GetGlyphColor(line[0]);
      ImVec2 bufferOffset;

      for (int i = 0; i < (int)line.size();) {
        auto &glyph = line[i];
        auto color = GetGlyphColor(glyph);

        if ((color != prevColor || glyph.mChar == '\t' || glyph.mChar == ' ') &&
            !mLineBuffer.empty()) {
          const ImVec2 newOffset(textScreenPos.x + bufferOffset.x,
                                 textScreenPos.y + bufferOffset.y);
          drawList->AddText(newOffset, prevColor, mLineBuffer.c_str());
          auto textSize = ImGui::GetFont()->CalcTextSizeA(
              ImGui::GetFontSize(), FLT_MAX, -1.0f, mLineBuffer.c_str(),
              nullptr, nullptr);
          bufferOffset.x += textSize.x;
          mLineBuffer.clear();
        }
        prevColor = color;

        if (glyph.mChar == '\t') {
          auto oldX = bufferOffset.x;
          bufferOffset.x = (1.0f + std::floor((1.0f + bufferOffset.x) /
                                              (float(mTabSize) * spaceSize))) *
                           (float(mTabSize) * spaceSize);
          ++i;

          if (mShowWhitespaces) {
            const auto s = ImGui::GetFontSize();
            const auto x1 = textScreenPos.x + oldX + 1.0f;
            const auto x2 = textScreenPos.x + bufferOffset.x - 1.0f;
            const auto y = textScreenPos.y + bufferOffset.y + s * 0.5f;
            const ImVec2 p1(x1, y);
            const ImVec2 p2(x2, y);
            const ImVec2 p3(x2 - s * 0.2f, y - s * 0.2f);
            const ImVec2 p4(x2 - s * 0.2f, y + s * 0.2f);
            drawList->AddLine(p1, p2, 0x90909090);
            drawList->AddLine(p2, p3, 0x90909090);
            drawList->AddLine(p2, p4, 0x90909090);
          }
        } else if (glyph.mChar == ' ') {
          if (mShowWhitespaces) {
            const auto s = ImGui::GetFontSize();
            const auto x = textScreenPos.x + bufferOffset.x + spaceSize * 0.5f;
            const auto y = textScreenPos.y + bufferOffset.y + s * 0.5f;
            drawList->AddCircleFilled(ImVec2(x, y), 1.5f, 0x80808080, 4);
          }
          bufferOffset.x += spaceSize;
          i++;
        } else {
          auto l = UTF8CharLength(glyph.mChar);
          while (l-- > 0)
            mLineBuffer.push_back(line[i++].mChar);
        }
      }

      if (!mLineBuffer.empty()) {
        const ImVec2 newOffset(textScreenPos.x + bufferOffset.x,
                               textScreenPos.y + bufferOffset.y);
        drawList->AddText(newOffset, prevColor, mLineBuffer.c_str());
        mLineBuffer.clear();
      }

      ++lineNo;
    }

    // Draw a tooltip on known identifiers/preprocessor symbols
    if (ImGui::IsMousePosValid()) {
      auto id = GetWordAt(ScreenPosToCoordinates(ImGui::GetMousePos()));
      if (!id.empty()) {
        auto it = mLanguageDefinition.mIdentifiers.find(id);
        if (it != mLanguageDefinition.mIdentifiers.end()) {
          ImGui::BeginTooltip();
          ImGui::TextUnformatted(it->second.mDeclaration.c_str());
          ImGui::EndTooltip();
        } else {
          auto pi = mLanguageDefinition.mPreprocIdentifiers.find(id);
          if (pi != mLanguageDefinition.mPreprocIdentifiers.end()) {
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(pi->second.mDeclaration.c_str());
            ImGui::EndTooltip();
          }
        }
      }
    }
  }

  ImGui::Dummy(ImVec2((longest + 2), mLines.size() * mCharAdvance.y));

  if (mScrollToCursor) {
    EnsureCursorVisible();
    mScrollToCursor = false;
  }
}

void TextViewer::Render(const char *aTitle, const ImVec2 &aSize, bool aBorder) {
  mWithinRender = true;
  mCursorPositionChanged = false;
  mRightClick = false;

  ImGui::PushStyleColor(
      ImGuiCol_ChildBg,
      ImGui::ColorConvertU32ToFloat4(mPalette[(int)PaletteIndex::Background]));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
  if (!mIgnoreImGuiChild)
    ImGui::BeginChild(aTitle, aSize, aBorder,
                      ImGuiWindowFlags_HorizontalScrollbar |
                          ImGuiWindowFlags_AlwaysHorizontalScrollbar |
                          ImGuiWindowFlags_NoMove);

  if (mHandleKeyboardInputs) {
    HandleKeyboardInputs();
    ImGui::PushTabStop(true);
  }

  if (mHandleMouseInputs)
    HandleMouseInputs();

  ColorizeInternal();
  Render();

  if (mHandleKeyboardInputs)
    ImGui::PopTabStop();

  if (!mIgnoreImGuiChild)
    ImGui::EndChild();

  ImGui::PopStyleVar();
  ImGui::PopStyleColor();

  mWithinRender = false;
}

void TextViewer::SetText(const std::string &aText) {
  mLines.clear();
  mLines.emplace_back(Line());
  for (auto chr : aText) {
    if (chr == '\r') {
      // ignore the carriage return character
    } else if (chr == '\n')
      mLines.emplace_back(Line());
    else {
      mLines.back().emplace_back(Glyph(chr, PaletteIndex::Default));
    }
  }

  mScrollToTop = true;
  Colorize();
}

void TextViewer::SetReadOnly(bool aValue) {
  // The viewer is always read-only; this is a no-op kept for API compatibility.
  (void)aValue;
}

void TextViewer::SetCursorPosition(const Coordinates &aPosition) {
  if (mState.mCursorPosition != aPosition) {
    mState.mCursorPosition = aPosition;
    mCursorPositionChanged = true;
    EnsureCursorVisible();
  }
}

void TextViewer::SetSelection(const Coordinates &aStart,
                              const Coordinates &aEnd, SelectionMode aMode) {
  auto oldSelStart = mState.mSelectionStart;
  auto oldSelEnd = mState.mSelectionEnd;

  mState.mSelectionStart = SanitizeCoordinates(aStart);
  mState.mSelectionEnd = SanitizeCoordinates(aEnd);
  if (mState.mSelectionStart > mState.mSelectionEnd)
    std::swap(mState.mSelectionStart, mState.mSelectionEnd);

  switch (aMode) {
  case TextViewer::SelectionMode::Normal:
    break;
  case TextViewer::SelectionMode::Word: {
    mState.mSelectionStart = FindWordStart(mState.mSelectionStart);
    if (!IsOnWordBoundary(mState.mSelectionEnd))
      mState.mSelectionEnd = FindWordEnd(FindWordStart(mState.mSelectionEnd));
    break;
  }
  case TextViewer::SelectionMode::Line: {
    const auto lineNo = mState.mSelectionEnd.mLine;
    mState.mSelectionStart = Coordinates(mState.mSelectionStart.mLine, 0);
    mState.mSelectionEnd = Coordinates(lineNo, GetLineMaxColumn(lineNo));
    break;
  }
  }

  if (mState.mSelectionStart != oldSelStart ||
      mState.mSelectionEnd != oldSelEnd)
    mCursorPositionChanged = true;
}

bool TextViewer::HasSelection() const {
  return mState.mSelectionEnd > mState.mSelectionStart;
}

void TextViewer::Copy() {
  if (HasSelection()) {
    ImGui::SetClipboardText(GetSelectedText().c_str());
  } else {
    if (!mLines.empty()) {
      std::string str;
      auto &line = mLines[GetActualCursorCoordinates().mLine];
      for (auto &g : line)
        str.push_back(g.mChar);
      ImGui::SetClipboardText(str.c_str());
    }
  }
}

const TextViewer::Palette &TextViewer::GetDarkPalette() {
  const static Palette p = {{
      0xff7f7f7f, // Default
      0xffd69c56, // Keyword
      0xff00ff00, // Number
      0xff7070e0, // String
      0xff70a0e0, // Char literal
      0xffffffff, // Punctuation
      0xff408080, // Preprocessor
      0xffaaaaaa, // Identifier
      0xff9bc64d, // Known identifier
      0xffc040a0, // Preproc identifier
      0xff206020, // Comment (single line)
      0xff406020, // Comment (multi line)
      0xff101010, // Background
      0xffe0e0e0, // Cursor
      0x80a06020, // Selection
      0x800020ff, // ErrorMarker
      0x40f08000, // Breakpoint
      0xff707000, // Line number
      0x40000000, // Current line fill
      0x40808080, // Current line fill (inactive)
      0x40a0a0a0, // Current line edge
  }};
  return p;
}

std::vector<std::string> TextViewer::GetTextLines() const {
  std::vector<std::string> result;

  result.reserve(mLines.size());

  for (auto &line : mLines) {
    std::string text;

    text.resize(line.size());

    for (size_t i = 0; i < line.size(); ++i)
      text[i] = line[i].mChar;

    result.emplace_back(std::move(text));
  }

  return result;
}

std::string TextViewer::GetSelectedText() const {
  return GetText(mState.mSelectionStart, mState.mSelectionEnd);
}

void TextViewer::Colorize(int aFromLine, int aLines) {
  int toLine = aLines == -1 ? (int)mLines.size()
                            : std::min((int)mLines.size(), aFromLine + aLines);
  mColorRangeMin = std::min(mColorRangeMin, aFromLine);
  mColorRangeMax = std::max(mColorRangeMax, toLine);
  mColorRangeMin = std::max(0, mColorRangeMin);
  mColorRangeMax = std::max(mColorRangeMin, mColorRangeMax);
  mCheckComments = true;
}

void TextViewer::ColorizeRange(int aFromLine, int aToLine) {
  if (mLines.empty() || aFromLine >= aToLine)
    return;

  std::string buffer;
  std::cmatch results;
  std::string id;

  int endLine = std::max(0, std::min((int)mLines.size(), aToLine));
  for (int i = aFromLine; i < endLine; ++i) {
    auto &line = mLines[i];

    if (line.empty())
      continue;

    buffer.resize(line.size());
    for (size_t j = 0; j < line.size(); ++j) {
      auto &col = line[j];
      buffer[j] = col.mChar;
      col.mColorIndex = PaletteIndex::Default;
    }

    const char *bufferBegin = &buffer.front();
    const char *bufferEnd = bufferBegin + buffer.size();

    auto last = bufferEnd;

    for (auto first = bufferBegin; first != last;) {
      const char *tokenBegin = nullptr;
      const char *tokenEnd = nullptr;
      PaletteIndex tokenColor = PaletteIndex::Default;

      bool hasTokenizeResult = false;

      if (mLanguageDefinition.mTokenize != nullptr) {
        if (mLanguageDefinition.mTokenize(first, last, tokenBegin, tokenEnd,
                                          tokenColor))
          hasTokenizeResult = true;
      }

      if (hasTokenizeResult == false) {
        for (auto &p : mRegexList) {
          if (std::regex_search(first, last, results, p.first,
                                std::regex_constants::match_continuous)) {
            hasTokenizeResult = true;

            auto &v = *results.begin();
            tokenBegin = v.first;
            tokenEnd = v.second;
            tokenColor = p.second;
            break;
          }
        }
      }

      if (hasTokenizeResult == false) {
        first++;
      } else {
        const size_t tokenLength = tokenEnd - tokenBegin;

        if (tokenColor == PaletteIndex::Identifier) {
          id.assign(tokenBegin, tokenEnd);

          if (!mLanguageDefinition.mCaseSensitive)
            std::transform(id.begin(), id.end(), id.begin(), ::toupper);

          if (!line[first - bufferBegin].mPreprocessor) {
            if (mLanguageDefinition.mKeywords.count(id) != 0)
              tokenColor = PaletteIndex::Keyword;
            else if (mLanguageDefinition.mIdentifiers.count(id) != 0)
              tokenColor = PaletteIndex::KnownIdentifier;
            else if (mLanguageDefinition.mPreprocIdentifiers.count(id) != 0)
              tokenColor = PaletteIndex::PreprocIdentifier;
          } else {
            if (mLanguageDefinition.mPreprocIdentifiers.count(id) != 0)
              tokenColor = PaletteIndex::PreprocIdentifier;
          }
        }

        for (size_t j = 0; j < tokenLength; ++j)
          line[(tokenBegin - bufferBegin) + j].mColorIndex = tokenColor;

        first = tokenEnd;
      }
    }
  }
}

void TextViewer::ColorizeInternal() {
  if (mLines.empty() || !mColorizerEnabled)
    return;

  if (mCheckComments) {
    auto endLine = (int)mLines.size();
    auto endIndex = 0;
    auto commentStartLine = endLine;
    auto commentStartIndex = endIndex;
    auto withinString = false;
    auto withinSingleLineComment = false;
    auto withinPreproc = false;
    auto firstChar =
        true; // there is no other non-whitespace characters in the line before
    auto concatenate = false; // '\' on the very end of the line
    auto currentLine = 0;
    auto currentIndex = 0;
    while (currentLine < endLine || currentIndex < endIndex) {
      auto &line = mLines[currentLine];

      if (currentIndex == 0 && !concatenate) {
        withinSingleLineComment = false;
        withinPreproc = false;
        firstChar = true;
      }

      concatenate = false;

      if (!line.empty()) {
        auto &g = line[currentIndex];
        auto c = g.mChar;

        if (c != mLanguageDefinition.mPreprocChar && !isspace(c))
          firstChar = false;

        if (currentIndex == (int)line.size() - 1 &&
            line[line.size() - 1].mChar == '\\')
          concatenate = true;

        bool inComment = (commentStartLine < currentLine ||
                          (commentStartLine == currentLine &&
                           commentStartIndex <= currentIndex));

        if (withinString) {
          line[currentIndex].mMultiLineComment = inComment;

          if (c == '\"') {
            if (currentIndex + 1 < (int)line.size() &&
                line[currentIndex + 1].mChar == '\"') {
              currentIndex += 1;
              if (currentIndex < (int)line.size())
                line[currentIndex].mMultiLineComment = inComment;
            } else
              withinString = false;
          } else if (c == '\\') {
            currentIndex += 1;
            if (currentIndex < (int)line.size())
              line[currentIndex].mMultiLineComment = inComment;
          }
        } else {
          if (firstChar && c == mLanguageDefinition.mPreprocChar)
            withinPreproc = true;

          if (c == '\"') {
            withinString = true;
            line[currentIndex].mMultiLineComment = inComment;
          } else {
            auto pred = [](const char &a, const Glyph &b) {
              return a == b.mChar;
            };
            auto from = line.begin() + currentIndex;
            auto &startStr = mLanguageDefinition.mCommentStart;
            auto &singleStartStr = mLanguageDefinition.mSingleLineComment;

            if (singleStartStr.size() > 0 &&
                currentIndex + singleStartStr.size() <= line.size() &&
                equals(singleStartStr.begin(), singleStartStr.end(), from,
                       from + singleStartStr.size(), pred)) {
              withinSingleLineComment = true;
            } else if (!withinSingleLineComment &&
                       currentIndex + startStr.size() <= line.size() &&
                       equals(startStr.begin(), startStr.end(), from,
                              from + startStr.size(), pred)) {
              commentStartLine = currentLine;
              commentStartIndex = currentIndex;
            }

            inComment = inComment = (commentStartLine < currentLine ||
                                     (commentStartLine == currentLine &&
                                      commentStartIndex <= currentIndex));

            line[currentIndex].mMultiLineComment = inComment;
            line[currentIndex].mComment = withinSingleLineComment;

            auto &endStr = mLanguageDefinition.mCommentEnd;
            if (currentIndex + 1 >= (int)endStr.size() &&
                equals(endStr.begin(), endStr.end(), from + 1 - endStr.size(),
                       from + 1, pred)) {
              commentStartIndex = endIndex;
              commentStartLine = endLine;
            }
          }
        }
        line[currentIndex].mPreprocessor = withinPreproc;
        currentIndex += UTF8CharLength(c);
        if (currentIndex >= (int)line.size()) {
          currentIndex = 0;
          ++currentLine;
        }
      } else {
        currentIndex = 0;
        ++currentLine;
      }
    }
    mCheckComments = false;
  }

  if (mColorRangeMin < mColorRangeMax) {
    const int increment =
        (mLanguageDefinition.mTokenize == nullptr) ? 10 : 10000;
    const int to = std::min(mColorRangeMin + increment, mColorRangeMax);
    ColorizeRange(mColorRangeMin, to);
    mColorRangeMin = to;

    if (mColorRangeMax == mColorRangeMin) {
      mColorRangeMin = std::numeric_limits<int>::max();
      mColorRangeMax = 0;
    }
    return;
  }
}

float TextViewer::TextDistanceToLineStart(const Coordinates &aFrom) const {
  auto &line = mLines[aFrom.mLine];
  float distance = 0.0f;
  float spaceSize = ImGui::GetFont()
                        ->CalcTextSizeA(ImGui::GetFontSize(), FLT_MAX, -1.0f,
                                        " ", nullptr, nullptr)
                        .x;
  int colIndex = GetCharacterIndex(aFrom);
  for (int it = 0; it < (int)line.size() && it < colIndex;) {
    if (line[it].mChar == '\t') {
      distance = (1.0f + std::floor((1.0f + distance) /
                                    (float(mTabSize) * spaceSize))) *
                 (float(mTabSize) * spaceSize);
      ++it;
    } else {
      auto d = UTF8CharLength(line[it].mChar);
      char tempCString[7];
      int i = 0;
      for (; i < 6 && d-- > 0 && it < (int)line.size(); i++, it++)
        tempCString[i] = line[it].mChar;

      tempCString[i] = '\0';
      distance += ImGui::GetFont()
                      ->CalcTextSizeA(ImGui::GetFontSize(), FLT_MAX, -1.0f,
                                      tempCString, nullptr, nullptr)
                      .x;
    }
  }

  return distance;
}

void TextViewer::EnsureCursorVisible() {
  if (!mWithinRender) {
    mScrollToCursor = true;
    return;
  }

  float scrollX = ImGui::GetScrollX();
  float scrollY = ImGui::GetScrollY();

  auto height = ImGui::GetWindowHeight();
  auto width = ImGui::GetWindowWidth();

  auto top = 1 + (int)ceil(scrollY / mCharAdvance.y);
  auto bottom = (int)ceil((scrollY + height) / mCharAdvance.y);

  auto left = (int)ceil(scrollX / mCharAdvance.x);
  auto right = (int)ceil((scrollX + width) / mCharAdvance.x);

  auto pos = GetActualCursorCoordinates();
  auto len = TextDistanceToLineStart(pos);

  if (pos.mLine < top)
    ImGui::SetScrollY(std::max(0.0f, (pos.mLine - 1) * mCharAdvance.y));
  if (pos.mLine > bottom - 4)
    ImGui::SetScrollY(
        std::max(0.0f, (pos.mLine + 4) * mCharAdvance.y - height));
  if (len + mTextStart < left + 4)
    ImGui::SetScrollX(std::max(0.0f, len + mTextStart - 4));
  if (len + mTextStart > right - 4)
    ImGui::SetScrollX(std::max(0.0f, len + mTextStart + 4 - width));
}

int TextViewer::GetPageSize() const {
  auto height = ImGui::GetWindowHeight() - 20.0f;
  return (int)floor(height / mCharAdvance.y);
}

static bool TokenizeCStyleString(const char *in_begin, const char *in_end,
                                 const char *&out_begin, const char *&out_end) {
  const char *p = in_begin;

  if (*p == '"') {
    p++;

    while (p < in_end) {
      // handle end of string
      if (*p == '"') {
        out_begin = in_begin;
        out_end = p + 1;
        return true;
      }

      // handle escape character for "
      if (*p == '\\' && p + 1 < in_end && p[1] == '"')
        p++;

      p++;
    }
  }

  return false;
}

static bool TokenizeCStyleCharacterLiteral(const char *in_begin,
                                           const char *in_end,
                                           const char *&out_begin,
                                           const char *&out_end) {
  const char *p = in_begin;

  if (*p == '\'') {
    p++;

    // handle escape characters
    if (p < in_end && *p == '\\')
      p++;

    if (p < in_end)
      p++;

    // handle end of character literal
    if (p < in_end && *p == '\'') {
      out_begin = in_begin;
      out_end = p + 1;
      return true;
    }
  }

  return false;
}

static bool TokenizeCStyleIdentifier(const char *in_begin, const char *in_end,
                                     const char *&out_begin,
                                     const char *&out_end) {
  const char *p = in_begin;

  if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || *p == '_') {
    p++;

    while ((p < in_end) &&
           ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
            (*p >= '0' && *p <= '9') || *p == '_'))
      p++;

    out_begin = in_begin;
    out_end = p;
    return true;
  }

  return false;
}

static bool TokenizeCStyleNumber(const char *in_begin, const char *in_end,
                                 const char *&out_begin, const char *&out_end) {
  const char *p = in_begin;

  const bool startsWithNumber = *p >= '0' && *p <= '9';

  if (*p != '+' && *p != '-' && !startsWithNumber)
    return false;

  p++;

  bool hasNumber = startsWithNumber;

  while (p < in_end && (*p >= '0' && *p <= '9')) {
    hasNumber = true;

    p++;
  }

  if (hasNumber == false)
    return false;

  bool isFloat = false;
  bool isHex = false;
  bool isBinary = false;

  if (p < in_end) {
    if (*p == '.') {
      isFloat = true;

      p++;

      while (p < in_end && (*p >= '0' && *p <= '9'))
        p++;
    } else if (*p == 'x' || *p == 'X') {
      // hex formatted integer of the type 0xef80

      isHex = true;

      p++;

      while (p < in_end &&
             ((*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'f') ||
              (*p >= 'A' && *p <= 'F')))
        p++;
    } else if (*p == 'b' || *p == 'B') {
      // binary formatted integer of the type 0b01011101

      isBinary = true;

      p++;

      while (p < in_end && (*p >= '0' && *p <= '1'))
        p++;
    }
  }

  if (isHex == false && isBinary == false) {
    // floating point exponent
    if (p < in_end && (*p == 'e' || *p == 'E')) {
      isFloat = true;

      p++;

      if (p < in_end && (*p == '+' || *p == '-'))
        p++;

      bool hasDigits = false;

      while (p < in_end && (*p >= '0' && *p <= '9')) {
        hasDigits = true;

        p++;
      }

      if (hasDigits == false)
        return false;
    }

    // single precision floating point type
    if (p < in_end && *p == 'f')
      p++;
  }

  if (isFloat == false) {
    // integer size type
    while (p < in_end && (*p == 'u' || *p == 'U' || *p == 'l' || *p == 'L'))
      p++;
  }

  out_begin = in_begin;
  out_end = p;
  return true;
}

static bool TokenizeCStylePunctuation(const char *in_begin, const char *in_end,
                                      const char *&out_begin,
                                      const char *&out_end) {
  (void)in_end;

  switch (*in_begin) {
  case '[':
  case ']':
  case '{':
  case '}':
  case '!':
  case '%':
  case '^':
  case '&':
  case '*':
  case '(':
  case ')':
  case '-':
  case '+':
  case '=':
  case '~':
  case '|':
  case '<':
  case '>':
  case '?':
  case ':':
  case '/':
  case ';':
  case ',':
  case '.':
    out_begin = in_begin;
    out_end = in_begin + 1;
    return true;
  }

  return false;
}

const TextViewer::LanguageDefinition &
TextViewer::LanguageDefinition::CPlusPlus() {
  static bool inited = false;
  static LanguageDefinition langDef;
  if (!inited) {
    static const char *const cppKeywords[] = {"alignas",
                                              "alignof",
                                              "and",
                                              "and_eq",
                                              "asm",
                                              "atomic_cancel",
                                              "atomic_commit",
                                              "atomic_noexcept",
                                              "auto",
                                              "bitand",
                                              "bitor",
                                              "bool",
                                              "break",
                                              "case",
                                              "catch",
                                              "char",
                                              "char16_t",
                                              "char32_t",
                                              "class",
                                              "compl",
                                              "concept",
                                              "const",
                                              "constexpr",
                                              "const_cast",
                                              "continue",
                                              "decltype",
                                              "default",
                                              "delete",
                                              "do",
                                              "double",
                                              "dynamic_cast",
                                              "else",
                                              "enum",
                                              "explicit",
                                              "export",
                                              "extern",
                                              "false",
                                              "float",
                                              "for",
                                              "friend",
                                              "goto",
                                              "if",
                                              "import",
                                              "inline",
                                              "int",
                                              "long",
                                              "module",
                                              "mutable",
                                              "namespace",
                                              "new",
                                              "noexcept",
                                              "not",
                                              "not_eq",
                                              "nullptr",
                                              "operator",
                                              "or",
                                              "or_eq",
                                              "private",
                                              "protected",
                                              "public",
                                              "register",
                                              "reinterpret_cast",
                                              "requires",
                                              "return",
                                              "short",
                                              "signed",
                                              "sizeof",
                                              "static",
                                              "static_assert",
                                              "static_cast",
                                              "struct",
                                              "switch",
                                              "synchronized",
                                              "template",
                                              "this",
                                              "thread_local",
                                              "throw",
                                              "true",
                                              "try",
                                              "typedef",
                                              "typeid",
                                              "typename",
                                              "union",
                                              "unsigned",
                                              "using",
                                              "virtual",
                                              "void",
                                              "volatile",
                                              "wchar_t",
                                              "while",
                                              "xor",
                                              "xor_eq"};
    for (auto &k : cppKeywords)
      langDef.mKeywords.insert(k);

    static const char *const identifiers[] = {
        "abort",   "abs",           "acos",    "asin",          "atan",
        "atexit",  "atof",          "atoi",    "atol",          "ceil",
        "clock",   "cosh",          "ctime",   "div",           "exit",
        "fabs",    "floor",         "fmod",    "getchar",       "getenv",
        "isalnum", "isalpha",       "isdigit", "isgraph",       "ispunct",
        "isspace", "isupper",       "kbhit",   "log10",         "log2",
        "log",     "memcmp",        "modf",    "pow",           "printf",
        "sprintf", "snprintf",      "putchar", "putenv",        "puts",
        "rand",    "remove",        "rename",  "sinh",          "sqrt",
        "srand",   "strcat",        "strcmp",  "strerror",      "time",
        "tolower", "toupper",       "std",     "string",        "vector",
        "map",     "unordered_map", "set",     "unordered_set", "min",
        "max"};
    for (auto &k : identifiers) {
      Identifier id;
      id.mDeclaration = "Built-in function";
      langDef.mIdentifiers.insert(std::make_pair(std::string(k), id));
    }

    langDef.mTokenize = [](const char *in_begin, const char *in_end,
                           const char *&out_begin, const char *&out_end,
                           PaletteIndex &paletteIndex) -> bool {
      paletteIndex = PaletteIndex::Max;

      while (in_begin < in_end && isascii(*in_begin) && isblank(*in_begin))
        in_begin++;

      if (in_begin == in_end) {
        out_begin = in_end;
        out_end = in_end;
        paletteIndex = PaletteIndex::Default;
      } else if (TokenizeCStyleString(in_begin, in_end, out_begin, out_end))
        paletteIndex = PaletteIndex::String;
      else if (TokenizeCStyleCharacterLiteral(in_begin, in_end, out_begin,
                                              out_end))
        paletteIndex = PaletteIndex::CharLiteral;
      else if (TokenizeCStyleIdentifier(in_begin, in_end, out_begin, out_end))
        paletteIndex = PaletteIndex::Identifier;
      else if (TokenizeCStyleNumber(in_begin, in_end, out_begin, out_end))
        paletteIndex = PaletteIndex::Number;
      else if (TokenizeCStylePunctuation(in_begin, in_end, out_begin, out_end))
        paletteIndex = PaletteIndex::Punctuation;

      return paletteIndex != PaletteIndex::Max;
    };

    langDef.mCommentStart = "/*";
    langDef.mCommentEnd = "*/";
    langDef.mSingleLineComment = "//";

    langDef.mCaseSensitive = true;
    langDef.mAutoIndentation = true;

    langDef.mName = "C++";

    inited = true;
  }
  return langDef;
}

const TextViewer::LanguageDefinition &TextViewer::LanguageDefinition::MLIR() {
  static bool inited = false;
  static LanguageDefinition langDef;
  if (!inited) {
    // Control-flow and module-level keywords.
    static const char *const keywords[] = {
        "module", "func", "return", "private",    "public",
        "nested", "loc",  "true",   "false",      "dense",
        "sparse", "unit", "opaque", "unknown",    "callsite",
        "fused",  "inf",  "nan",    "affine_map", "affine_set",
    };
    for (auto &k : keywords)
      langDef.mKeywords.insert(k);

    // Builtin type names shown as known-identifier color.
    static const char *const types[] = {
        "i1",         "i2",         "i4",      "i8",   "i16",      "i32",
        "i64",        "i128",       "si8",     "si16", "si32",     "si64",
        "ui8",        "ui16",       "ui32",    "ui64", "f8E4M3FN", "f8E5M2",
        "f8E4M3FNUZ", "f8E5M2FNUZ", "f16",     "f32",  "f64",      "bf16",
        "f80",        "f128",       "index",   "none", "memref",   "tensor",
        "vector",     "tuple",      "complex",
    };
    for (auto &t : types) {
      Identifier id;
      id.mDeclaration = "MLIR type";
      langDef.mIdentifiers.insert({t, id});
    }

    langDef.mTokenize = [](const char *in_begin, const char *in_end,
                           const char *&out_begin, const char *&out_end,
                           PaletteIndex &paletteIndex) -> bool {
      paletteIndex = PaletteIndex::Max;

      // Skip leading whitespace.
      while (in_begin < in_end && isascii(*in_begin) && isblank(*in_begin))
        in_begin++;

      if (in_begin == in_end) {
        out_begin = out_end = in_end;
        paletteIndex = PaletteIndex::Default;
        return true;
      }

      const char c = *in_begin;

      // SSA values: %name or %0 (also %arg0, %result, etc.)
      if (c == '%') {
        const char *p = in_begin + 1;
        if (p < in_end && *p == '"') {
          // %"..." form (quoted SSA name)
          ++p;
          while (p < in_end && *p != '"') {
            if (*p == '\\' && p + 1 < in_end)
              ++p;
            ++p;
          }
          if (p < in_end)
            ++p; // closing "
        } else {
          while (p < in_end && (isalnum(*p) || *p == '_' || *p == '.'))
            ++p;
        }
        if (p > in_begin + 1) {
          out_begin = in_begin;
          out_end = p;
          paletteIndex = PaletteIndex::Preprocessor;
          return true;
        }
        return false;
      }

      // Block labels: ^bb0, ^entry, etc.
      if (c == '^') {
        const char *p = in_begin + 1;
        while (p < in_end && (isalnum(*p) || *p == '_'))
          ++p;
        if (p > in_begin + 1) {
          out_begin = in_begin;
          out_end = p;
          paletteIndex = PaletteIndex::PreprocIdentifier;
          return true;
        }
        return false;
      }

      // Symbol references: @name or @"name"
      if (c == '@') {
        const char *p = in_begin + 1;
        if (p < in_end && *p == '"') {
          ++p;
          while (p < in_end && *p != '"') {
            if (*p == '\\' && p + 1 < in_end)
              ++p;
            ++p;
          }
          if (p < in_end)
            ++p;
        } else {
          while (p < in_end && (isalnum(*p) || *p == '_'))
            ++p;
        }
        if (p > in_begin + 1) {
          out_begin = in_begin;
          out_end = p;
          paletteIndex = PaletteIndex::KnownIdentifier;
          return true;
        }
        return false;
      }

      // Strings, numbers, identifiers, and punctuation reuse the C helpers.
      if (TokenizeCStyleString(in_begin, in_end, out_begin, out_end))
        paletteIndex = PaletteIndex::String;
      else if (TokenizeCStyleNumber(in_begin, in_end, out_begin, out_end))
        paletteIndex = PaletteIndex::Number;
      else if (TokenizeCStyleIdentifier(in_begin, in_end, out_begin, out_end))
        paletteIndex = PaletteIndex::Identifier; // keyword/type lookup follows
      else if (TokenizeCStylePunctuation(in_begin, in_end, out_begin, out_end))
        paletteIndex = PaletteIndex::Punctuation;

      return paletteIndex != PaletteIndex::Max;
    };

    // MLIR only uses single-line // comments; use /* */ as a sentinel that
    // never appears so the multi-line comment scanner stays inactive.
    langDef.mCommentStart = "/*";
    langDef.mCommentEnd = "*/";
    langDef.mSingleLineComment = "//";

    langDef.mPreprocChar = '\0'; // no preprocessor directives in MLIR
    langDef.mCaseSensitive = true;
    langDef.mAutoIndentation = false;
    langDef.mName = "MLIR";

    inited = true;
  }
  return langDef;
}

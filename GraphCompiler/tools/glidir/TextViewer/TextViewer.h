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
#pragma once

#include "imgui.h"
#include <array>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class TextViewer {
public:
  enum class PaletteIndex {
    Default,
    Keyword,
    Number,
    String,
    CharLiteral,
    Punctuation,
    Preprocessor,
    Identifier,
    KnownIdentifier,
    PreprocIdentifier,
    Comment,
    MultiLineComment,
    Background,
    Cursor,
    Selection,
    ErrorMarker,
    Breakpoint,
    LineNumber,
    CurrentLineFill,
    CurrentLineFillInactive,
    CurrentLineEdge,
    Max
  };

  enum class SelectionMode { Normal, Word, Line };

  // A 0-based cell in the rendered fixed-width grid; a tab spans up to the next
  // tab stop. E.g. with mTabSize = 4, (1, 5) is 'B' in "\tABC".
  struct Coordinates {
    int mLine, mColumn;
    Coordinates() : mLine(0), mColumn(0) {}
    Coordinates(int aLine, int aColumn) : mLine(aLine), mColumn(aColumn) {
      assert(aLine >= 0);
      assert(aColumn >= 0);
    }
    static Coordinates Invalid() {
      static Coordinates invalid(-1, -1);
      return invalid;
    }

    bool operator==(const Coordinates &o) const {
      return mLine == o.mLine && mColumn == o.mColumn;
    }

    bool operator!=(const Coordinates &o) const {
      return mLine != o.mLine || mColumn != o.mColumn;
    }

    bool operator<(const Coordinates &o) const {
      if (mLine != o.mLine)
        return mLine < o.mLine;
      return mColumn < o.mColumn;
    }

    bool operator>(const Coordinates &o) const {
      if (mLine != o.mLine)
        return mLine > o.mLine;
      return mColumn > o.mColumn;
    }

    bool operator<=(const Coordinates &o) const {
      if (mLine != o.mLine)
        return mLine < o.mLine;
      return mColumn <= o.mColumn;
    }

    bool operator>=(const Coordinates &o) const {
      if (mLine != o.mLine)
        return mLine > o.mLine;
      return mColumn >= o.mColumn;
    }
  };

  struct Identifier {
    Coordinates mLocation;
    std::string mDeclaration;
  };

  using Identifiers = std::unordered_map<std::string, Identifier>;
  using Keywords = std::unordered_set<std::string>;
  using ErrorMarkers = std::map<int, std::string>;
  using Breakpoints = std::unordered_set<int>;
  // Maps 1-indexed line numbers to a list of colors; drawn as small stacked
  // circles in the gutter margin, one per bookmark color group on that line.
  using LineMarkers = std::unordered_map<int, std::vector<ImU32>>;
  using Palette = std::array<ImU32, (unsigned)PaletteIndex::Max>;
  using Char = uint8_t;

  struct Glyph {
    Char mChar;
    PaletteIndex mColorIndex = PaletteIndex::Default;
    bool mComment : 1;
    bool mMultiLineComment : 1;
    bool mPreprocessor : 1;

    Glyph(Char aChar, PaletteIndex aColorIndex)
        : mChar(aChar), mColorIndex(aColorIndex), mComment(false),
          mMultiLineComment(false), mPreprocessor(false) {}
  };

  using Line = std::vector<Glyph>;
  using Lines = std::vector<Line>;

  struct LanguageDefinition {
    using TokenRegexString = std::pair<std::string, PaletteIndex>;
    using TokenRegexStrings = std::vector<TokenRegexString>;
    using TokenizeCallback = bool (*)(const char *inBegin, const char *inEnd,
                                      const char *&outBegin,
                                      const char *&outEnd,
                                      PaletteIndex &paletteIndex);

    std::string mName;
    Keywords mKeywords;
    Identifiers mIdentifiers;
    Identifiers mPreprocIdentifiers;
    std::string mCommentStart, mCommentEnd, mSingleLineComment;
    char mPreprocChar;
    bool mAutoIndentation;

    TokenizeCallback mTokenize;

    TokenRegexStrings mTokenRegexStrings;

    bool mCaseSensitive;

    LanguageDefinition()
        : mPreprocChar('#'), mAutoIndentation(true), mTokenize(nullptr),
          mCaseSensitive(true) {}

    static const LanguageDefinition &CPlusPlus();
    static const LanguageDefinition &MLIR();
  };

  TextViewer();
  ~TextViewer();

  bool DoubleClick() const { return mDoubleClick; }
  bool RightClick() const { return mRightClick; }

  void SetLanguageDefinition(const LanguageDefinition &aLanguageDef);
  void SetPalette(const Palette &aValue);

  Breakpoints &GetBreakpoints() { return mBreakpoints; }
  LineMarkers &GetLineMarkers() { return mLineMarkers; }

  void Render(const char *aTitle, const ImVec2 &aSize = ImVec2(),
              bool aBorder = false);
  void SetText(const std::string &aText);
  std::vector<std::string> GetTextLines() const;

  int GetTotalLines() const { return (int)mLines.size(); }

  void SetReadOnly(bool aValue);
  bool IsCursorPositionChanged() const { return mCursorPositionChanged; }

  Coordinates GetCursorPosition() const { return GetActualCursorCoordinates(); }
  void SetCursorPosition(const Coordinates &aPosition);

  void SetSelection(const Coordinates &aStart, const Coordinates &aEnd,
                    SelectionMode aMode = SelectionMode::Normal);
  bool HasSelection() const;

  Coordinates GetSelectionStart() const { return mState.mSelectionStart; }
  Coordinates GetSelectionEnd() const { return mState.mSelectionEnd; }

  void Copy();

  static const Palette &GetDarkPalette();

private:
  using RegexList = std::vector<std::pair<std::regex, PaletteIndex>>;

  struct EditorState {
    Coordinates mSelectionStart;
    Coordinates mSelectionEnd;
    Coordinates mCursorPosition;
  };

  void Colorize(int aFromLine = 0, int aCount = -1);
  void ColorizeRange(int aFromLine = 0, int aToLine = 0);
  void ColorizeInternal();
  float TextDistanceToLineStart(const Coordinates &aFrom) const;
  void EnsureCursorVisible();
  int GetPageSize() const;
  std::string GetText(const Coordinates &aStart, const Coordinates &aEnd) const;
  std::string GetSelectedText() const;
  Coordinates GetActualCursorCoordinates() const;
  Coordinates SanitizeCoordinates(const Coordinates &aValue) const;
  Coordinates ScreenPosToCoordinates(const ImVec2 &aPosition) const;
  Coordinates FindWordStart(const Coordinates &aFrom) const;
  Coordinates FindWordEnd(const Coordinates &aFrom) const;
  Coordinates FindNextWord(const Coordinates &aFrom) const;
  int GetCharacterIndex(const Coordinates &aCoordinates) const;
  int GetCharacterColumn(int aLine, int aIndex) const;
  int GetLineMaxColumn(int aLine) const;
  bool IsOnWordBoundary(const Coordinates &aAt) const;
  std::string GetWordAt(const Coordinates &aCoords) const;
  ImU32 GetGlyphColor(const Glyph &aGlyph) const;

  void HandleKeyboardInputs();
  void HandleMouseInputs();
  void Render();

  float mLineSpacing;
  Lines mLines;
  EditorState mState;

  int mTabSize;
  bool mWithinRender;
  bool mScrollToCursor;
  bool mScrollToTop;
  bool mColorizerEnabled;
  float mTextStart;
  int mLeftMargin;
  bool mCursorPositionChanged;
  int mColorRangeMin, mColorRangeMax;
  SelectionMode mSelectionMode;
  bool mHandleKeyboardInputs;
  bool mHandleMouseInputs;
  bool mIgnoreImGuiChild;
  bool mShowWhitespaces;

  Palette mPaletteBase;
  Palette mPalette;
  LanguageDefinition mLanguageDefinition;
  RegexList mRegexList;

  bool mCheckComments;
  Breakpoints mBreakpoints;
  LineMarkers mLineMarkers;
  ErrorMarkers mErrorMarkers;
  ImVec2 mCharAdvance;
  Coordinates mInteractiveStart, mInteractiveEnd;
  std::string mLineBuffer;
  uint64_t mStartTime;

  float mLastClick;
  bool mDoubleClick;
  bool mRightClick;
};

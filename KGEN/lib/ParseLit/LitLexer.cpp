//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Defines the a Lexer and Token interface for .lit files.
//
//===----------------------------------------------------------------------===//

#include "LitLexer.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

using namespace M;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SMRange;
using llvm::SourceMgr;

// These C macros are often inefficient due to attempt to support unicode, use
// the llvm::isAlpha methods instead.
#define isalpha(x) DO_NOT_USE_SLOW_CTYPE_FUNCTIONS
#define isdigit(x) DO_NOT_USE_SLOW_CTYPE_FUNCTIONS

//===----------------------------------------------------------------------===//
// LitToken
//===----------------------------------------------------------------------===//

SMLoc LitToken::getLoc() const {
  return SMLoc::getFromPointer(spelling.data());
}

SMLoc LitToken::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange LitToken::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

/// Return true if this is one of the keyword token kinds (e.g. kw_pass).
bool LitToken::isKeyword() const {
  switch (kind) {
  default:
    return false;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return true;
#include "LitTokenKinds.def"
  }
}

//===----------------------------------------------------------------------===//
// LitLexer
//===----------------------------------------------------------------------===//

LitLexer::LitLexer(LitSharedState &sharedState,
                   const llvm::MemoryBuffer *buffer)
    : sharedState(sharedState), curBuffer(buffer->getBuffer()),
      curPtr(curBuffer.begin()),
      // Prime the first token.
      curToken(lexTokenImpl()) {}

static StringRef getBuffer(LitSharedState &sharedState,
                           const LitLexerCursor &cursor) {
  unsigned cursorBufferId =
      sharedState.sourceMgr.FindBufferContainingLoc(cursor.getToken().getLoc());
  assert(cursorBufferId && "invalid cursor!");
  const auto *buffer = sharedState.sourceMgr.getMemoryBuffer(cursorBufferId);
  return buffer->getBuffer();
}

LitLexer::LitLexer(LitSharedState &sharedState, const LitLexerCursor &cursor)
    : sharedState(sharedState), curBuffer(getBuffer(sharedState, cursor)),
      curToken(LitToken::eof, {}, 0) {
  cursor.restore(*this);
}

/// Inflate a lightweight SMLoc into an MLIR Location object for addition
/// into the IR.
Location LitLexer::translateLocation(llvm::SMLoc loc) {
  return sharedState.translateLocation(loc);
}

/// Emit an error message and return a LitToken::error token.
LitToken LitLexer::emitError(const char *loc, const Twine &message) {
  mlir::emitError(translateLocation(SMLoc::getFromPointer(loc)), message);
  return formToken(LitToken::error, loc, -1);
}

//===----------------------------------------------------------------------===//
// Lexer Implementation Methods
//===----------------------------------------------------------------------===//

LitToken LitLexer::lexTokenImpl() {
  // This keeps track of the indentation of the current token from the start of
  // the line.  The first byte of the file starts with an indentation of zero,
  // but subsequent tokens always start out by following an existing token, so
  // they aren't at the start of line.
  ssize_t indentation = curPtr == curBuffer.begin() ? 0 : -1;
  const char *tokStart;
  // This is a helper lambda for forming tokens with tokStart and indentation,
  // and optionally incrementing `curPtr` to make some of the conditionals below
  // ergonomic.
  auto formToken = [&](LitToken::Kind kind, size_t incr = 0) -> LitToken {
    curPtr += incr;
    return this->formToken(kind, tokStart, indentation);
  };

  while (true) {
    // This loop is set up so a "continue" can be used to ignore a whitespace
    // character.  Always reset 'tokStart'.
    tokStart = curPtr;
    switch (*curPtr++) {
    case 0:
      // This may either be a nul character in the source file or may be the EOF
      // marker that MemoryBuffer guarantees will be there.
      if (curPtr - 1 == curBuffer.end())
        return formToken(LitToken::eof);

      [[fallthrough]]; // Treat as whitespace.

      // Horizontal whitespace increases the indentation if current token is at
      // start of line.
    case ' ':
    case '\t':
      if (indentation != -1)
        ++indentation;
      continue;

      // Vertical whitespace resets the indentation to zero since anything that
      // comes after it is at the start of the line.
    case '\n':
    case '\r':
    case '\f':
    case '\v':
      indentation = 0;
      continue;

      // Handle \ at end of line by treating it as whitespace instead of
      // tracking the next token as start of line.
    case '\\': {
      // Check that there is only horizontal whitespace before the \n.
      while (*curPtr == ' ' || *curPtr == '\t')
        ++curPtr;
      if (*curPtr == '\n' || *curPtr == '\r') {
        if (*curPtr == '\r' && curPtr[1] == '\n') // Windows new line
          ++curPtr;
        ++curPtr;
        indentation = -1;
        continue;
      }
      return emitError(tokStart,
                       "unexpected '\\' character, isn't at end of line");
    }

    default:
      // Handle identifiers.
      if (llvm::isAlpha(curPtr[-1])) {
        // Raw string literal
        if ((curPtr[-1] == 'r' || curPtr[-1] == 'R') &&
            (*curPtr == '\'' || *curPtr == '"'))
          return lexString(tokStart, indentation);
        return lexIdentifierOrKeyword(tokStart, indentation);
      }

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case '_':
      // Handle identifiers.
      return lexIdentifierOrKeyword(tokStart, indentation);
    case '`':
      return lexBacktickIdentifier(tokStart, indentation);
    case '"':
    case '\'':
      return lexString(tokStart, indentation);
    case '%':
      if (*curPtr == '=')
        return formToken(LitToken::percent_equal, 1);
      return formToken(LitToken::percent);
    case '&':
      if (*curPtr == '=')
        return formToken(LitToken::amp_equal, 1);
      return formToken(LitToken::amp);
    case '(':
      return formToken(LitToken::l_paren);
    case ')':
      return formToken(LitToken::r_paren);
    case '*':
      if (*curPtr == '=')
        return formToken(LitToken::star_equal, 1);
      if (*curPtr == '*') {
        if (curPtr[1] == '=')
          return formToken(LitToken::star_star_equal, 2);
        return formToken(LitToken::star_star, 1);
      }
      return formToken(LitToken::star);
    case '+':
      if (*curPtr == '=')
        return formToken(LitToken::plus_equal, 1);
      return formToken(LitToken::plus);
    case ',':
      return formToken(LitToken::comma);
    case '-':
      if (*curPtr == '=')
        return formToken(LitToken::minus_equal, 1);
      if (*curPtr == '>')
        return formToken(LitToken::minus_greater, 1);
      return formToken(LitToken::minus);
    case '.':
      if (llvm::isDigit(*curPtr))
        return lexFloat(tokStart, indentation);
      if (*curPtr == '.' && curPtr[1] == '.')
        return formToken(LitToken::dot_dot_dot, 2);
      return formToken(LitToken::dot);
    case '/':
      if (*curPtr == '=')
        return formToken(LitToken::slash_equal, 1);
      if (*curPtr == '/') {
        if (curPtr[1] == '=')
          return formToken(LitToken::slash_slash_equal, 2);
        return formToken(LitToken::slash_slash, 1);
      }
      return formToken(LitToken::slash);
    case ':':
      // TODO: Python keeps track of nesting level in the lexer to report
      // mismatched tokens here.  How does that affect error recovery?
      if (*curPtr == '=')
        return formToken(LitToken::colon_equal, 1);
      return formToken(LitToken::colon);
    case ';':
      return formToken(LitToken::semi);
    case '<':
      switch (*curPtr) {
      case '<':
        if (curPtr[1] == '=')
          return formToken(LitToken::less_less_equal, 2);
        return formToken(LitToken::less_less, 1);
      case '=':
        return formToken(LitToken::less_equal, 1);
      case '>':
        return formToken(LitToken::less_greater, 1);
      }
      return formToken(LitToken::less);
    case '=':
      if (*curPtr == '=')
        return formToken(LitToken::equal_equal, 1);
      return formToken(LitToken::equal);
    case '>':
      switch (*curPtr) {
      case '=':
        return formToken(LitToken::greater_equal, 1);
      case '>':
        if (curPtr[1] == '=')
          return formToken(LitToken::right_right_equal, 2);
        return formToken(LitToken::right_right, 1);
      }
      return formToken(LitToken::greater);
    case '@':
      if (*curPtr == '=')
        return formToken(LitToken::at_equal, 1);
      return formToken(LitToken::at);
    case '[':
      return formToken(LitToken::l_square);
    case ']':
      return formToken(LitToken::r_square);
    case '^':
      if (*curPtr == '=')
        return formToken(LitToken::circumflex_equal, 1);
      return formToken(LitToken::circumflex);
    case '{':
      return formToken(LitToken::l_brace);
    case '|':
      if (*curPtr == '=')
        return formToken(LitToken::pipe_equal, 1);
      return formToken(LitToken::pipe);
    case '}':
      return formToken(LitToken::r_brace);
    case '~':
      return formToken(LitToken::tilde);
    case '!':
      if (*curPtr == '=')
        return formToken(LitToken::exclaim_equal, 1);
      return emitError(tokStart, "unexpected character");

    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return lexInteger(tokStart, indentation);

    case '#':
      skipComment();
      indentation = 0; // skipComment eats the \n.
      continue;
    }
  }
}

/// Lex an identifier or keyword that starts with a letter.
///
/// TODO: Python supports unicode in is_potential_identifier_start etc.
///
LitToken LitLexer::lexIdentifierOrKeyword(const char *tokStart,
                                          ssize_t indentation) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_$-]*
  while (llvm::isAlpha(*curPtr) || llvm::isDigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$' || *curPtr == '-')
    ++curPtr;

  StringRef spelling(tokStart, curPtr - tokStart);

  // Check to see if this identifier is a keyword.
  LitToken::Kind kind = llvm::StringSwitch<LitToken::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, LitToken::kw_##SPELLING)
#include "LitTokenKinds.def"
                            .Default(LitToken::identifier);

  return LitToken(kind, spelling, indentation);
}

/// Lex an identifier with backtick syntax, e.g. `ide nt if ier` or `fn`.  These
/// may contain any character other than vertical whitespace and `'s in them and
/// are otherwise interpreted verbatim as an identifier.
LitToken LitLexer::lexBacktickIdentifier(const char *tokStart,
                                         ssize_t indentation) {
  assert(curPtr[-1] == '`');
  while (true) {
    switch (*curPtr++) {
    case '`':
      // Found the end character.
      return LitToken(LitToken::identifier,
                      StringRef(tokStart + 1, curPtr - tokStart - 2),
                      indentation);
    case '\n':
    case '\r':
    case '\v':
    case '\f':
      // Vertical whitespace within a ` is invalid is the end of the comment.
      return emitError(tokStart, "unterminated backtick identifier");
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        return emitError(tokStart, "unterminated backtick identifier");
      }
      [[fallthrough]];
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// Skip a comment line, starting with a '#' and going to end of line.
void LitLexer::skipComment() {
  while (true) {
    switch (*curPtr++) {
    case '\n':
    case '\r':
    case '\v':
    case '\f':
      // Vertical whitespaces is the end of the comment.
      return;
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        return;
      }
      [[fallthrough]];
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// Checks if character \p C is one of the 8 octal digits.
static bool isOctalDigit(char C) { return C >= '0' && C <= '7'; }

/// Lex a string literal.
///
/// stringliteral   ::=  [stringprefix] shortstring
/// stringprefix    ::=  "r" | "R"
/// shortstring     ::=  "'" shortstringitem* "'" | '"' shortstringitem* '"'
/// shortstringitem ::=  shortstringchar | stringescapeseq
/// shortstringchar ::=  <any source character except "\" or newline or the
/// quote> stringescapeseq ::=  "\" <any source character>
///
///
// TODO: support full Python grammar below:
// stringliteral   ::=  [stringprefix](shortstring | longstring)
// stringprefix    ::=  "r" | "u" | "R" | "U" | "f" | "F"
//                      | "fr" | "Fr" | "fR" | "FR" | "rf" | "rF" | "Rf" | "RF"
// shortstring     ::=  "'" shortstringitem* "'" | '"' shortstringitem* '"'
// longstring      ::=  "'''" longstringitem* "'''" | '"""' longstringitem*
// '"""' shortstringitem ::=  shortstringchar | stringescapeseq longstringitem
// ::=  longstringchar | stringescapeseq shortstringchar ::=  <any source
// character except "\" or newline or the quote> longstringchar  ::=  <any
// source character except "\"> stringescapeseq ::=  "\" <any source character>
LitToken LitLexer::lexString(const char *tokStart, ssize_t indentation) {
  curPtr = tokStart;
  bool isRaw = false;
  if (*curPtr == 'r' || *curPtr == 'R') {
    isRaw = true;
    ++curPtr;
  }

  if (*curPtr != '\'' && *curPtr != '"')
    return emitError(tokStart,
                     "expecting a string quoting character: `'` or `\"`");
  char quoteChar = *curPtr;
  ++curPtr;

  while (*curPtr != quoteChar && curPtr != curBuffer.end()) {
    switch (*curPtr++) {
    case '\\':
      if (isRaw) {
        if (curPtr == curBuffer.end())
          return emitError(tokStart, "unterminated string");
        ++curPtr;
        break;
      }

      // Skip escaped characters
      if (isOctalDigit(*curPtr)) {
        // at most 3 octal digits.
        size_t i = 0;
        while (isOctalDigit(*curPtr) && i < 3) {
          ++curPtr;
          i++;
        }
      } else if (*curPtr == 'x') {
        ++curPtr;
        // exactly 2 hex digits.
        size_t i = 0;
        while (llvm::isHexDigit(*curPtr) && i < 2) {
          ++curPtr;
          i++;
        }
        if (i != 2)
          return emitError(
              tokStart,
              "invalid hex escape sequence: exactly two hex digits needed");
      } else {
        if (!llvm::is_contained({'\\', '"', '\'', '\n', '\r', 'a', 'b', 'f',
                                 'n', 'r', 't', 'v'},
                                *curPtr))
          return emitError(tokStart, "invalid escape sequence");
        if (*curPtr == '\r' && curPtr[1] == '\n') // Windows new line
          ++curPtr;
        ++curPtr;
      }
      break;
    case '\n': // newline isn't allowed in a string.
    case '\r':
      return emitError(tokStart, "unterminated string");
    default:
      // Skip over other characters.
      break;
    }
  }
  if (curPtr == curBuffer.end())
    return emitError(tokStart, "unterminated string");
  ++curPtr;
  return formToken(LitToken::string, tokStart, indentation);
}

/// Lex a integer number literal.
///
/// integer      ::=  decinteger | bininteger | octinteger | hexinteger
/// decinteger   ::=  nonzerodigit ("_" | digit)* | "0"+ ("_" | "0")*
/// bininteger   ::=  "0" ("b" | "B") (["_"] bindigit)+
/// octinteger   ::=  "0" ("o" | "O") (["_"] octdigit)+
/// hexinteger   ::=  "0" ("x" | "X") (["_"] hexdigit)+
/// nonzerodigit ::=  "1"..."9"
/// digit        ::=  "0"..."9"
/// bindigit     ::=  "0" | "1"
/// octdigit     ::=  "0"..."7"
/// hexdigit     ::=  digit | "a"..."f" | "A"..."F"
///
/// DIFFERENCES with Python:
/// - Python uses the following more restrictive productions, which
///   disallows `1__9_` for example:
///   decinteger   ::=  nonzerodigit (["_"] digit)* | "0"+ (["_"] "0")*
//    same thing for  bininteger, octinteger and hexinteger
/// - Python warns if the numeric literal is immediately followed by
//    other keyword or identifier.

LitToken LitLexer::lexInteger(const char *tokStart, ssize_t indentation) {
  assert(llvm::isDigit(curPtr[-1]));

  if (curPtr[-1] == '0') {
    if (*curPtr == 'b' || *curPtr == 'B') {
      ++curPtr;
      bool hasDigits = false;
      while (*curPtr == '0' || *curPtr == '1' || *curPtr == '_') {
        hasDigits |= *curPtr != '_';
        ++curPtr;
      }
      if (!hasDigits)
        return emitError(curPtr, "no digits specified for binary literal");
    } else if (*curPtr == 'o' || *curPtr == 'O') {
      ++curPtr;
      bool hasDigits = false;
      while (isOctalDigit(*curPtr) || *curPtr == '_') {
        hasDigits |= *curPtr != '_';
        ++curPtr;
      }
      if (!hasDigits)
        return emitError(curPtr, "no digits specified for octal literal");
    } else if (*curPtr == 'x' || *curPtr == 'X') {
      ++curPtr;
      bool hasDigits = false;
      while (llvm::isHexDigit(*curPtr) || *curPtr == '_') {
        hasDigits |= *curPtr != '_';
        ++curPtr;
      }
      if (!hasDigits)
        return emitError(curPtr, "no digits specified for hex literal");
    } else if (*curPtr == '.' || *curPtr == 'e' || *curPtr == 'E' ||
               *curPtr == 'j' || *curPtr == 'J') {
      return lexFloat(tokStart, indentation);
    } else if (*curPtr == '0' || *curPtr == '_') {
      // Literal zero, ex. 00, 00_0, 0_0_0__0
      // Superset of Python's grammar, we allow consecutive and trailing `_`
      // ex. 0__0_
      do
        ++curPtr;
      while (*curPtr == '0' || *curPtr == '_');
    } else if (llvm::isDigit(*curPtr))
      // ex. 0123
      return emitError(curPtr,
                       "leading zeros in decimal integer literals are not "
                       "permitted; use an 0o prefix for octal integers");
  } else {
    // nonzerodigit
    // Superset of Python's grammar, we allow consecutive and trailing `_`
    // ex. 1__9_
    while (llvm::isDigit(*curPtr) || *curPtr == '_')
      ++curPtr;
  }
  if (*curPtr == '.' || *curPtr == 'e' || *curPtr == 'E' || *curPtr == 'j' ||
      *curPtr == 'J')
    return lexFloat(tokStart, indentation);
  return formToken(LitToken::integer, tokStart, indentation);
}

/// Lex a float number literal.
/// When the function is called tokStart points to "." or a digit.
/// floatnumber   ::=  pointfloat | exponentfloat
/// pointfloat    ::=  [digitpart] fraction | digitpart "."
/// exponentfloat ::=  (digitpart | pointfloat) exponent
/// digitpart     ::=  digit ("_" | digit)*
/// fraction      ::=  "." digitpart
/// exponent      ::=  ("e" | "E") ["+" | "-"] digitpart
///
/// DIFFERENCES with Python:
/// - Python uses the following more restrictive productions, which
///   disallows `1__9_` for example:
///   digitpart     ::=  digit (["_"] digit)*
LitToken LitLexer::lexFloat(const char *tokStart, ssize_t indentation) {
  assert(*tokStart == '.' || llvm::isDigit(*tokStart));
  // lexFloat could have been called from lexInteger so reset curPtr to undo
  // previous increments done by lexInteger
  curPtr = tokStart;
  if (llvm::isDigit(*curPtr)) {
    do
      ++curPtr;
    while (llvm::isDigit(*curPtr) || *curPtr == '_');
  }
  if (*curPtr == '.')
    ++curPtr;
  if (llvm::isDigit(*curPtr)) {
    do
      ++curPtr;
    while (llvm::isDigit(*curPtr) || *curPtr == '_');
  }
  if (*curPtr == 'e' || *curPtr == 'E') {
    ++curPtr;
    if (*curPtr == '+' || *curPtr == '-')
      ++curPtr;
    if (!llvm::isDigit(*curPtr))
      return emitError(curPtr, "expecting a digit after the exponent");
    while (llvm::isDigit(*curPtr) || *curPtr == '_')
      ++curPtr;
  }
  return formToken(LitToken::float_num, tokStart, indentation);
}

static std::string filterUnderscores(StringRef spelling) {
  std::string digits;
  digits.reserve(spelling.size());
  for (auto c : spelling) {
    if (c != '_')
      digits.push_back(c);
  }
  return digits;
}

/// Return the a value for the specified string, which is known to have been
/// lexed as a float literal token.
APFloat LitLexer::getFloatLiteralValue(StringRef spelling) {
  std::string digits = filterUnderscores(spelling);
  spelling = StringRef(digits);
  APFloat num(0.0);
  auto StatusOrErr =
      num.convertFromString(spelling, APFloat::rmNearestTiesToEven);
  assert(!errorToBool(StatusOrErr.takeError()) &&
         "Invalid floating point literal");
  APFloat::opStatus Status = *StatusOrErr;
  assert((Status == APFloat::opOK || Status & APFloat::opInexact) &&
         "Invalid floating point literal");
  return num;
}

/// Return the a string value of `spelling` after the escape sequences are
/// handled. `spelling` is known to have been lexed as a string literal token.
std::string LitLexer::getStringLiteralValue(StringRef spelling) {
  bool isRaw = false;
  if (spelling[0] == 'r' || spelling[0] == 'R') {
    isRaw = true;
    spelling = spelling.drop_front();
  }
  // Drop quotes.
  StringRef bytes = spelling.drop_front().drop_back();

  std::string result;
  result.reserve(bytes.size());
  for (size_t i = 0, end = bytes.size(); i != end;) {
    auto c = bytes[i++];
    if (c != '\\' || isRaw) {
      result.push_back(c);
      continue;
    }

    assert(i + 1 <= end && "invalid string should be caught by lexer");
    auto c1 = bytes[i++];
    switch (c1) {
    case '\\':
    case '"':
    case '\'':
      result.push_back(c1);
      continue;
    case '\n':
      continue;
    case '\r':
      if (bytes[i] == '\n')
        i++;
      continue;
    case 'a':
      result.push_back('\a');
      continue;
    case 'b':
      result.push_back('\b');
      continue;
    case 'f':
      result.push_back('\f');
      continue;
    case 'n':
      result.push_back('\n');
      continue;
    case 'r':
      result.push_back('\r');
      continue;
    case 't':
      result.push_back('\t');
      continue;
    case 'v':
      result.push_back('\v');
      continue;
    case 'x': {
      char hex0 = bytes[i++];
      char hex1 = bytes[i++];
      assert(llvm::isHexDigit(hex0) && llvm::isHexDigit(hex1) &&
             "invalid escape");
      result.push_back((llvm::hexDigitValue(hex0) << 4) |
                       llvm::hexDigitValue(hex1));
      continue;
    }
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7': {
      size_t startDigit = i - 1;
      // At most 3 digits
      while (i < (startDigit + 3) && isOctalDigit(bytes[i]))
        i++;
      unsigned int num;
      bool failed = bytes.slice(startDigit, i).getAsInteger(8, num);
      assert(!failed && "we know this should always work because we lexed it");
      result.push_back(static_cast<char>(num));
      continue;
    }
    default:
      llvm_unreachable(
          "invalid escape sequence: this should have been caught by lexString");
    }
  }
  return result;
}

/// Return the a value for the specified string, which is known to have been
/// lexed as an integer literal token.
APInt LitLexer::getIntegerLiteralValue(StringRef spelling) {
  APInt result;
  unsigned base = 10;
  if (spelling[0] == '0' && spelling.size() > 2) {
    switch (spelling[1]) {
    case 'b':
    case 'B':
      base = 2;
      break;
    case 'o':
    case 'O':
      base = 8;
      break;
    case 'x':
    case 'X':
      base = 16;
      break;
    }
    spelling = spelling.drop_front(2);
  }
  std::string digits = filterUnderscores(spelling);
  spelling = StringRef(digits);
  bool failed = spelling.getAsInteger(base, result);
  assert(!failed && "we know this should always work because we lexed it");
  (void)failed;
  return result;
}

//===----------------------------------------------------------------------===//
// Support methods
//===----------------------------------------------------------------------===//

/// Given a location that is at the start of a line, scan backwards to find
/// the end of the last line that contains a token, or start of the source
/// buffer if there is none.
SMLoc LitLexer::findEndOfPreviousLine(SMLoc loc) const {
  // To find the end of the previous line, we repeatedly segment the buffer into
  // chunks from the current position to the start of the current line and scan
  // it to see if it contains any tokens.  If not, we keep going, if so we use
  // the end of the last token.
  auto locOffset = size_t(loc.getPointer() - curBuffer.data());
  assert(locOffset <= curBuffer.size() && "loc not in current buffer!");
  // Truncate whole buffer to this segment.
  StringRef buffer(curBuffer.data(), locOffset);

  while (1) {
    auto nextNewLine = buffer.find_last_of("\n\r");
    // If we ran out of lines to check, we must be at the start of the buffer.
    // Give up.
    if (nextNewLine == StringRef::npos)
      return loc;

    // Scan from the start of the line to the current position.
    auto *lineStart = curBuffer.data() + nextNewLine;
    LitLexerCursor cursor(lineStart,
                          {LitToken::plus, StringRef(lineStart, 0), 0});
    LitLexer tmpLexer(sharedState, cursor);
    tmpLexer.lexToken();

    // If the token is on this line, then there was at least one token on this
    // line.  Report the error at the end of the line.
    if (tmpLexer.getToken().getLoc().getPointer() < buffer.end())
      return SMLoc::getFromPointer(buffer.end());

    // Otherwise, drop the newline and anything after it and try again.
    buffer = buffer.take_front(nextNewLine);
  }
}

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Defines the a Lexer and Token interface for .mojo files.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/Lexer.h"
#include "Support/IPRational.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"

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
// Token
//===----------------------------------------------------------------------===//

SMLoc Token::getLoc() const { return SMLoc::getFromPointer(spelling.data()); }

SMLoc Token::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange Token::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

/// Return true if this is one of the keyword token kinds (e.g. kw_pass).
bool Token::isKeyword() const {
  switch (kind) {
  default:
    return false;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return true;
#include "KGEN/MojoParser/TokenKinds.def"
  }
}

bool Token::isIdentifier() const {
  return isAny(identifier, escaped_identifier);
}

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

Lexer::Lexer(Diags &diags, StringRef curBuffer, const char *curPtr)
    : diags(diags), curBuffer(curBuffer), curPtr(curPtr),
      curToken(Token::eof, StringRef(), 0), lastLineStart(nullptr),
      lastLineIndent(0) {
  lexToken();
}

Lexer::Lexer(Diags &diags, const llvm::MemoryBuffer *buffer)
    : diags(diags), curBuffer(buffer->getBuffer()), curPtr(curBuffer.begin()),
      curToken(Token::eof, StringRef(), 0), lastLineStart(nullptr),
      lastLineIndent(0) {

  // Prime the first token.
  lexToken();
}

static StringRef findBuffer(llvm::SourceMgr &sourceMgr,
                            const LexerCursor &cursor) {
  unsigned cursorBufferId =
      sourceMgr.FindBufferContainingLoc(cursor.getToken().getLoc());
  assert(cursorBufferId && "invalid cursor!");
  const auto *buffer = sourceMgr.getMemoryBuffer(cursorBufferId);
  return buffer->getBuffer();
}

Lexer::Lexer(Diags &diags, const LexerCursor &cursor)
    : diags(diags), curBuffer(findBuffer(diags.sourceMgr, cursor)),
      curToken(Token::eof, {}, 0) {
  cursor.restore(*this);
}

/// Emit an error message and return a Token::error token.
InflightDiag Lexer::emitErrorAt(const char *loc, const Twine &message) {
  auto diag = diags.emitError(SMLoc::getFromPointer(loc), message);
  formToken(Token::error, loc, -1);
  return diag;
}

/// This function point is the funnel point for all tokens that are lexed.  This
/// updates curToken and does other final checking.
///
/// The tokenStartOffset field is used to indicate tokens whose spelling is
/// artificially shifted from the start of the token, notably things like
/// `x y` are given a spelling of "x y" and don't include the `.
void Lexer::formToken(Token::Kind kind, StringRef spelling, ssize_t indentation,
                      size_t tokenStartOffset) {
  // We're about to form a token.  If the token is at the start of line, make
  // sure the leading indentation of this token and the previous start of line
  // match in spelling, then update our current start-of-line marker.
  if (indentation != -1) {
    // Check that the leading indentation of these two tokens match.
    const char *thisLineStart =
        spelling.data() - indentation - tokenStartOffset;
    if (memcmp(lastLineStart, thisLineStart,
               std::min(indentation, lastLineIndent))) {
      diags.emitError(SMLoc::getFromPointer(spelling.data()),
                      "leading indentation uses inconsistent whitespace (tabs "
                      "and spaces) than previous line");
    }

    lastLineStart = thisLineStart;
    lastLineIndent = indentation;
  }

  curToken = Token(kind, spelling, indentation);
}

//===----------------------------------------------------------------------===//
// Lexer Implementation Methods
//===----------------------------------------------------------------------===//

void Lexer::lexToken() {
  // This keeps track of the indentation of the current token from the start of
  // the line.  The first byte of the file starts with an indentation of zero,
  // but subsequent tokens always start out by following an existing token, so
  // they aren't at the start of line.
  ssize_t indentation = curPtr == curBuffer.begin() ? 0 : -1;
  const char *tokStart;
  // This is a helper lambda for forming tokens with tokStart and indentation,
  // and optionally incrementing `curPtr` to make some of the conditionals below
  // ergonomic.
  auto formToken = [&](Token::Kind kind, size_t incr = 0) {
    curPtr += incr;
    this->formToken(kind, tokStart, indentation);
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
        return this->formToken(Token::eof, tokStart, 0);

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
      emitErrorAt(tokStart, "unexpected '\\' character, isn't at end of line");
      return;
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
      emitErrorAt(tokStart, "unexpected character");
      return;

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
        return formToken(Token::percent_equal, 1);
      return formToken(Token::percent);
    case '&':
      if (*curPtr == '=')
        return formToken(Token::amp_equal, 1);
      return formToken(Token::amp);
    case '(':
      return formToken(Token::l_paren);
    case ')':
      return formToken(Token::r_paren);
    case '*':
      if (*curPtr == '=')
        return formToken(Token::star_equal, 1);
      if (*curPtr == '*') {
        if (curPtr[1] == '=')
          return formToken(Token::star_star_equal, 2);
        return formToken(Token::star_star, 1);
      }
      return formToken(Token::star);
    case '+':
      if (*curPtr == '=')
        return formToken(Token::plus_equal, 1);
      return formToken(Token::plus);
    case ',':
      return formToken(Token::comma);
    case '-':
      if (*curPtr == '=')
        return formToken(Token::minus_equal, 1);
      if (*curPtr == '>')
        return formToken(Token::minus_greater, 1);
      return formToken(Token::minus);
    case '.':
      if (llvm::isDigit(*curPtr))
        return lexFloat(tokStart, indentation);
      if (*curPtr == '.' && curPtr[1] == '.')
        return formToken(Token::dot_dot_dot, 2);
      return formToken(Token::dot);
    case '/':
      if (*curPtr == '=')
        return formToken(Token::slash_equal, 1);
      if (*curPtr == '/') {
        if (curPtr[1] == '=')
          return formToken(Token::slash_slash_equal, 2);
        return formToken(Token::slash_slash, 1);
      }
      return formToken(Token::slash);
    case ':':
      // TODO: Python keeps track of nesting level in the lexer to report
      // mismatched tokens here.  How does that affect error recovery?
      if (*curPtr == '=')
        return formToken(Token::colon_equal, 1);
      return formToken(Token::colon);
    case ';':
      return formToken(Token::semi);
    case '<':
      switch (*curPtr) {
      case '<':
        if (curPtr[1] == '=')
          return formToken(Token::less_less_equal, 2);
        return formToken(Token::less_less, 1);
      case '=':
        return formToken(Token::less_equal, 1);
      case '>':
        return formToken(Token::less_greater, 1);
      }
      return formToken(Token::less);
    case '=':
      if (*curPtr == '=')
        return formToken(Token::equal_equal, 1);
      return formToken(Token::equal);
    case '>':
      switch (*curPtr) {
      case '=':
        return formToken(Token::greater_equal, 1);
      case '>':
        if (curPtr[1] == '=')
          return formToken(Token::right_right_equal, 2);
        return formToken(Token::right_right, 1);
      }
      return formToken(Token::greater);
    case '@':
      if (*curPtr == '=')
        return formToken(Token::at_equal, 1);
      return formToken(Token::at);
    case '[':
      return formToken(Token::l_square);
    case ']':
      return formToken(Token::r_square);
    case '^':
      if (*curPtr == '=')
        return formToken(Token::caret_equal, 1);
      return formToken(Token::caret);
    case '{':
      return formToken(Token::l_brace);
    case '|':
      if (*curPtr == '=')
        return formToken(Token::pipe_equal, 1);
      return formToken(Token::pipe);
    case '}':
      return formToken(Token::r_brace);
    case '~':
      return formToken(Token::tilde);
    case '!':
      if (*curPtr == '=')
        return formToken(Token::exclaim_equal, 1);
      emitErrorAt(tokStart, "unexpected character");
      return;

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
void Lexer::lexIdentifierOrKeyword(const char *tokStart, ssize_t indentation) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_$]*
  while (llvm::isAlpha(*curPtr) || llvm::isDigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$')
    ++curPtr;

  StringRef spelling(tokStart, curPtr - tokStart);

  // Check to see if this identifier is a keyword.
  Token::Kind kind = llvm::StringSwitch<Token::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, Token::kw_##SPELLING)
#include "KGEN/MojoParser/TokenKinds.def"
                         .Default(Token::identifier);

  formToken(kind, tokStart, indentation);
}

/// Lex an identifier with backtick syntax, e.g. `ide nt if ier` or `fn`.  These
/// may contain any character other than vertical whitespace and `'s in them and
/// are otherwise interpreted verbatim as an identifier.
void Lexer::lexBacktickIdentifier(const char *tokStart, ssize_t indentation) {
  assert(curPtr[-1] == '`');
  while (true) {
    switch (*curPtr++) {
    case '`':
      // Found the end character.
      if (curPtr - tokStart - 2 == 0)
        emitErrorAt(tokStart, "empty backtick identifier isn't allowed");

      formToken(Token::escaped_identifier,
                StringRef(tokStart + 1, curPtr - tokStart - 2), indentation,
                /*tokenOffset*/ 1);
      return;
    case '\n':
    case '\r':
    case '\v':
    case '\f':
      // Vertical whitespace within a ` is invalid is the end of the comment.
      emitErrorAt(tokStart, "unterminated backtick identifier");
      return;
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        emitErrorAt(tokStart, "unterminated backtick identifier");
        return;
      }
      [[fallthrough]];
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// Skip a comment line, starting with a '#' and going to end of line.
void Lexer::skipComment() {
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
/// stringliteral   ::=  [stringprefix](shortstring | longstring)
/// stringprefix    ::=  "r" | "u" | "R" | "U" | "f" | "F"
///                      | "fr" | "Fr" | "fR" | "FR" | "rf" | "rF" | "Rf" | "RF"
/// shortstring     ::=  "'" shortstringitem* "'" | '"' shortstringitem* '"'
/// longstring      ::=  "'''" longstringitem* "'''" |
///                      '"""' longstringitem* '"""'
/// shortstringitem ::=  shortstringchar | stringescapeseq
/// longstringitem  ::=  longstringchar | stringescapeseq
/// shortstringchar ::=  <any source character except "\" or newline or the
///                      quote>
/// longstringchar  ::=  <any source character except "\">
/// stringescapeseq ::=  "\" <any source character>
void Lexer::lexString(const char *tokStart, ssize_t indentation) {
  curPtr = tokStart;
  bool isRaw = false;
  bool isTripleQuote = false;
  if (*curPtr == 'r' || *curPtr == 'R') {
    isRaw = true;
    ++curPtr;
  }

  if (*curPtr != '\'' && *curPtr != '"') {
    emitErrorAt(tokStart, "expecting a string quoting character: `'` or `\"`");
    return;
  }
  char quoteChar = *curPtr;
  if ((curPtr[1] == quoteChar && curPtr[2] == quoteChar)) {
    isTripleQuote = true;
    curPtr += 2;
  }
  ++curPtr;

  while (curPtr != curBuffer.end()) {
    switch (*curPtr++) {
    case '\\':
      if (isRaw) {
        if (curPtr == curBuffer.end()) {
          emitErrorAt(tokStart, "unterminated string");
          return;
        }
        // Handle trailing windows style newline.
        if (*curPtr == '\r' && curPtr[1] == '\n')
          ++curPtr;
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
        if (i != 2) {
          emitErrorAt(
              tokStart,
              "invalid hex escape sequence: exactly two hex digits needed");
          return;
        }
      } else if (!llvm::is_contained({'\\', '"', '\'', '\n', '\r', 'a', 'b',
                                      'f', 'n', 'r', 't', 'v'},
                                     *curPtr)) {
        emitErrorAt(curPtr - 1, "invalid escape sequence");
      } else {
        if (*curPtr == '\r' && curPtr[1] == '\n') // Windows newline
          ++curPtr;
        ++curPtr;
      }
      break;
    case '\'':
    case '"':
      // end of short strings.
      if (curPtr[-1] == quoteChar) {
        if (!isTripleQuote)
          return formToken(Token::string, tokStart, indentation);

        // end of long string
        if (curPtr[0] == quoteChar && curPtr[1] == quoteChar) {
          curPtr += 2;
          return formToken(Token::string, tokStart, indentation);
        }
      }
      break;
    case '\n':
    case '\r':
      // newline isn't allowed in a short string.
      if (!isTripleQuote) {
        emitErrorAt(tokStart, "unterminated string");
        return;
      }
      // Skip newline.
      break;
    default:
      // Skip over other characters.
      break;
    }
  }

  emitErrorAt(tokStart, "unterminated string");
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
///   same thing for  bininteger, octinteger and hexinteger
/// - Python warns if the numeric literal is immediately followed by
///   other keyword or identifier.
void Lexer::lexInteger(const char *tokStart, ssize_t indentation) {
  assert(llvm::isDigit(curPtr[-1]));

  if (curPtr[-1] == '0') {
    if (*curPtr == 'b' || *curPtr == 'B') {
      ++curPtr;
      bool hasDigits = false;
      while (*curPtr == '0' || *curPtr == '1' || *curPtr == '_') {
        hasDigits |= *curPtr != '_';
        ++curPtr;
      }
      if (!hasDigits) {
        emitErrorAt(curPtr, "no digits specified for binary literal");
        return;
      }
    } else if (*curPtr == 'o' || *curPtr == 'O') {
      ++curPtr;
      bool hasDigits = false;
      while (isOctalDigit(*curPtr) || *curPtr == '_') {
        hasDigits |= *curPtr != '_';
        ++curPtr;
      }
      if (!hasDigits) {
        emitErrorAt(curPtr, "no digits specified for octal literal");
        return;
      }
    } else if (*curPtr == 'x' || *curPtr == 'X') {
      ++curPtr;
      bool hasDigits = false;
      while (llvm::isHexDigit(*curPtr) || *curPtr == '_') {
        hasDigits |= *curPtr != '_';
        ++curPtr;
      }
      if (!hasDigits) {
        emitErrorAt(curPtr, "no digits specified for hex literal");
        return;
      }
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
    } else if (llvm::isDigit(*curPtr)) {
      // ex. 0123
      emitErrorAt(curPtr, "leading zeros in decimal integer literals are not "
                          "permitted; use an 0o prefix for octal integers");
      return;
    }
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
  formToken(Token::integer, tokStart, indentation);
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
void Lexer::lexFloat(const char *tokStart, ssize_t indentation) {
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
    if (!llvm::isDigit(*curPtr)) {
      emitErrorAt(curPtr, "expecting a digit after the exponent");
      return;
    }
    while (llvm::isDigit(*curPtr) || *curPtr == '_')
      ++curPtr;
  }
  formToken(Token::float_num, tokStart, indentation);
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
IPRational Lexer::getFloatLiteralValue(StringRef spelling) {
  std::string digits = filterUnderscores(spelling);
  spelling = StringRef(digits);
  IPInt numerator(0);

  size_t digitsIndex = 0;
  bool pastDecimal = false;
  bool foundE = false;
  size_t denominatorCounter = 0;
  while (digitsIndex < digits.size()) {
    char digit = digits[digitsIndex];
    if (digit >= '0' && digit <= '9') {
      char decimalValue = digit - '0';
      numerator = numerator * IPInt(10) + IPInt(decimalValue);
      if (pastDecimal)
        ++denominatorCounter;
    } else if (digit == '.' && !pastDecimal) {
      pastDecimal = true;
    } else if (digit == 'e' || digit == 'E') {
      foundE = true;
      ++digitsIndex;
      break;
    } else {
      assert(false && "bad float literal");
    }
    ++digitsIndex;
  }
  IPInt denominator(IPInt(10).exponentiate(denominatorCounter));

  if (foundE) {
    IPInt exponent = 0;
    bool negativeSign = false;
    if (digits[digitsIndex] == '-') {
      negativeSign = true;
      ++digitsIndex;
    } else if (digits[digitsIndex] == '+') {
      ++digitsIndex;
    }
    while (digitsIndex < digits.size()) {
      char digit = digits[digitsIndex];
      if (digit >= '0' && digit <= '9') {
        char decimalValue = digit - '0';
        exponent = exponent * 10;
        exponent = exponent + IPInt(decimalValue);
      }
      ++digitsIndex;
    }
    IPInt exponentMulValue = IPInt(10).exponentiate(exponent);
    if (negativeSign)
      denominator = denominator * exponentMulValue;
    else
      numerator = numerator * exponentMulValue;
  }

  return IPRational(numerator, denominator);
}

/// Return the a string value of `spelling` after the escape sequences are
/// handled. `spelling` is known to have been lexed as a string literal token.
std::string Lexer::getStringLiteralValue(StringRef bytes) {
  bool isRaw = false;
  if (bytes[0] == 'r' || bytes[0] == 'R') {
    isRaw = true;
    bytes = bytes.drop_front();
  }

  // Drop quotes and triple quotes.
  if (bytes.size() >= 6 &&
      (bytes.starts_with("\"\"\"") || bytes.starts_with("'''")))
    bytes = bytes.drop_front(3).drop_back(3);
  else
    bytes = bytes.drop_front().drop_back();

  std::string result;
  result.reserve(bytes.size());
  for (size_t i = 0, end = bytes.size(); i != end;) {
    auto c = bytes[i++];
    if (c != '\\' || isRaw) {
      result.push_back(c);

      // Handle trailing windows style newline.
      if (c == '\r' && i < end && bytes[i] == '\n') {
        result.push_back('\n');
        ++i;
      }
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
      while (i < (startDigit + 3) && i < bytes.size() && isOctalDigit(bytes[i]))
        i++;
      unsigned int num;
      [[maybe_unused]] bool failed =
          bytes.slice(startDigit, i).getAsInteger(8, num);
      assert(!failed && "we know this should always work because we lexed it");
      result.push_back(static_cast<char>(num));
      continue;
    }
    default:
      // Otherwise it is an invalid escape.  It will already have been diagnosed
      // at lexer time.
      result.push_back(c1);
      continue;
    }
  }
  return result;
}

SMLoc Lexer::getStringLiteralStartLoc(StringRef spelling) {
  size_t stringStartOffset = 1;
  if (spelling[0] == 'r' || spelling[0] == 'R')
    ++stringStartOffset;
  // Handle triple quoted strings.
  if (spelling.size() >= 6 &&
      (spelling.starts_with("\"\"\"") || spelling.starts_with("'''")))
    stringStartOffset += 2;
  return SMLoc::getFromPointer(spelling.data() + stringStartOffset);
}

/// Return the a value for the specified string, which is known to have been
/// lexed as an integer literal token.
APInt Lexer::getIntegerLiteralValue(StringRef spelling) {
  APInt result;
  unsigned base = 10;
  if (spelling[0] == '0' && spelling.size() > 2) {
    switch (spelling[1]) {
    case 'b':
    case 'B':
      base = 2;
      spelling = spelling.drop_front(2);
      break;
    case 'o':
    case 'O':
      base = 8;
      spelling = spelling.drop_front(2);
      break;
    case 'x':
    case 'X':
      base = 16;
      spelling = spelling.drop_front(2);
      break;
    }
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
SMLoc Lexer::findEndOfPreviousLine(SMLoc loc) const {
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
    Lexer tmpLexer(diags, curBuffer, lineStart);

    // If the token is on this line, then there was at least one token on this
    // line.  Report the error at the end of the line.
    if (tmpLexer.getToken().getLoc().getPointer() < buffer.end())
      return SMLoc::getFromPointer(buffer.end());

    // Otherwise, drop the newline and anything after it and try again.
    buffer = buffer.take_front(nextNewLine);
  }
}

//===----------------------------------------------------------------------===//
// LexerCrashReporter
//===----------------------------------------------------------------------===//

void LexerCrashReporter::print(raw_ostream &os) const {
  os << "Crash " << message << " at "
     << lexer.diags.translateLocation(SMLoc::getFromPointer(startPtr)) << '\n';

  // We know where the statement started, though the statement may not be the
  // first token on the line.  We know the current lexer position which is the
  // first token we haven't processed (generally the next statement, but might
  // be in the middle of a statement.
  StringRef buffer = lexer.getBuffer();

  // Figure out where the current unconsumed token is: if it is on something
  // that is the start of a line, back it up to the end of the previous line.
  const char *curTokenPtr = lexer.getToken().getSpelling().data();
  auto curLineWithoutWhitespace =
      buffer.drop_back(buffer.end() - curTokenPtr).rtrim(" \t");
  if (curLineWithoutWhitespace.rtrim("\n\r") != curLineWithoutWhitespace)
    curTokenPtr =
        lexer.findEndOfPreviousLine(lexer.getToken().getLoc()).getPointer();

  // This helper prints a line of the source buffer with highlighting to keep
  // track of where things are.
  auto printSourceLine = [&](StringRef sourceLine) {
    os << "    >> " << sourceLine << '\n';
    // Print out ^'s at the start and current token pointer if they exist in the
    // line.
    if (startPtr < sourceLine.begin() && curTokenPtr > sourceLine.end())
      return; // Don't print fully "." lines.

    os << "       ";
    for (const char &c : sourceLine) {
      char charToPrint;
      if (&c == startPtr)
        charToPrint = '^';
      else if (&c == curTokenPtr)
        charToPrint = '<';
      else if (&c > startPtr && &c < curTokenPtr)
        charToPrint = '.';
      else if (c == '\t')
        charToPrint = '\t';
      else
        charToPrint = ' ';
      os << charToPrint;
    }
    // The next token pointer is typically at the \n of the current line.
    if (sourceLine.end() == curTokenPtr)
      os << '<';
    os << '\n';
  };

  // Start by printing the first line of code
  // that we started on, being careful to stay in the source file.
  size_t stmtStartOffset = startPtr - buffer.data();

  size_t prevNewLine = buffer.find_last_of("\n\r", stmtStartOffset);
  prevNewLine = prevNewLine == StringRef::npos ? 0 : prevNewLine + 1;
  size_t nextNewLine = buffer.find_first_of("\n\r", stmtStartOffset);
  nextNewLine = nextNewLine == StringRef::npos ? buffer.size() : nextNewLine;

  StringRef sourceLine =
      StringRef(buffer.data() + prevNewLine, nextNewLine - prevNewLine);
  printSourceLine(sourceLine);

  // If the current token position isn't in the first line, then print a few
  // more lines of context just in case.
  size_t numLinesPrinted = 0;
  while (curTokenPtr > sourceLine.end() && numLinesPrinted++ < 4 &&
         nextNewLine != buffer.size()) {
    size_t nextNextNewLine = buffer.find_first_of("\n\r", nextNewLine + 1);
    nextNextNewLine =
        nextNextNewLine == StringRef::npos ? buffer.size() : nextNextNewLine;
    sourceLine = StringRef(buffer.data() + nextNewLine + 1,
                           nextNextNewLine - (nextNewLine + 1));
    printSourceLine(sourceLine);
    nextNewLine = nextNextNewLine;
  }
}

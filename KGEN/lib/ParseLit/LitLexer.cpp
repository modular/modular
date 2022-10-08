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

/// Get the name of the main buffer so we can rapidly build Location objects
/// on demand.
static StringAttr getMainBufferNameIdentifier(const SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

LitLexer::LitLexer(const SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)),
      curBuffer(
          sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()),
      curPtr(curBuffer.begin()),
      // Prime the first token.
      curToken(lexTokenImpl()) {}

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
Location LitLexer::translateLocation(llvm::SMLoc loc) {
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  return FileLineColLoc::get(bufferNameIdentifier, lineAndColumn.first,
                             lineAndColumn.second);
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
      if (*curPtr == '\n' || *curPtr == '\r' || *curPtr == '\f' ||
          *curPtr == '\v') {
        ++curPtr;
        indentation = -1;
        continue;
      }
      return emitError(tokStart,
                       "unexpected '\\' character, isn't at end of line");
    }

    default:
      // Handle identifiers.
      if (llvm::isAlpha(curPtr[-1]))
        return lexIdentifierOrKeyword(tokStart, indentation);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case '_':
      // Handle identifiers.
      return lexIdentifierOrKeyword(tokStart, indentation);
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
inline bool isOctalDigit(char C) { return C >= '0' && C <= '7'; }

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
  assert(Status == APFloat::opOK ||
         Status & APFloat::opInexact && "Invalid floating point literal");
  return num;
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

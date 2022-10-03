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
  return formToken(LitToken::error, loc);
}

/// Return the indentation level of the specified token.
/// TODO: Evaluate tracking this inline as part of the lexing loop.  We should
/// eval how commonly this is queried to figure out the tradeoff.
Optional<size_t> LitLexer::getIndentation(const LitToken &tok) const {
  // Count the number of horizontal whitespace characters before the token.
  auto *bufStart = curBuffer.begin();

  auto isHorizontalWS = [](char c) -> bool {
    return c == ' ' || c == '\t' || c == ',';
  };
  auto isVerticalWS = [](char c) -> bool {
    return c == '\n' || c == '\r' || c == '\f' || c == '\v';
  };

  size_t indent = 0;
  const auto *ptr = (const char *)tok.getSpelling().data();
  while (ptr != bufStart && isHorizontalWS(ptr[-1]))
    --ptr, ++indent;

  // If the character we stopped at isn't the start of line, then return none.
  if (ptr != bufStart && !isVerticalWS(ptr[-1]))
    return None;

  return indent;
}

//===----------------------------------------------------------------------===//
// Lexer Implementation Methods
//===----------------------------------------------------------------------===//

LitToken LitLexer::lexTokenImpl() {
  while (true) {
    const char *tokStart = curPtr;
    switch (*curPtr++) {
    case 0:
      // This may either be a nul character in the source file or may be the EOF
      // marker that MemoryBuffer guarantees will be there.
      if (curPtr - 1 == curBuffer.end())
        return formToken(LitToken::eof, tokStart);

      [[fallthrough]]; // Treat as whitespace.

    case ' ':
    case '\t':
    case '\n':
    case '\r':
      // Handle whitespace.
      continue;

    default:
      // Handle identifiers.
      if (llvm::isAlpha(curPtr[-1]))
        return lexIdentifierOrKeyword(tokStart);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case '_':
      // Handle identifiers.
      return lexIdentifierOrKeyword(tokStart);
    case '%':
      if (*curPtr == '=')
        return formToken(LitToken::percent_equal, tokStart, 1);
      return formToken(LitToken::percent, tokStart);
    case '&':
      if (*curPtr == '=')
        return formToken(LitToken::amp_equal, tokStart, 1);
      return formToken(LitToken::amp, tokStart);
    case '(':
      return formToken(LitToken::l_paren, tokStart);
    case ')':
      return formToken(LitToken::r_paren, tokStart);
    case '*':
      switch (*curPtr) {
      case '*':
        if (curPtr[1] == '=')
          return formToken(LitToken::star_star_equal, tokStart, 2);
        return formToken(LitToken::star_star, tokStart, 1);
      case '=':
        return formToken(LitToken::star_equal, tokStart, 1);
      }
      return formToken(LitToken::star, tokStart);
    case '+':
      if (*curPtr == '=')
        return formToken(LitToken::plus_equal, tokStart, 1);
      return formToken(LitToken::plus, tokStart);
    case ',':
      return formToken(LitToken::comma, tokStart);
    case '-':
      switch (*curPtr) {
      case '=':
        return formToken(LitToken::minus_equal, tokStart, 1);
      case '>':
        return formToken(LitToken::minus_greater, tokStart, 1);
      }
      return formToken(LitToken::minus, tokStart);
    case '.':
      if (llvm::isDigit(*curPtr))
        return lexFloat(tokStart);
      if (*curPtr == '.' && curPtr[1] == '.')
        return formToken(LitToken::dot_dot_dot, tokStart, 2);
      return formToken(LitToken::dot, tokStart);
    case '/':
      switch (*curPtr) {
      case '/':
        if (curPtr[1] == '=')
          return formToken(LitToken::slash_slash_equal, tokStart, 2);
        return formToken(LitToken::slash_slash, tokStart, 1);
      case '=':
        return formToken(LitToken::slash_equal, tokStart, 1);
      }
      return formToken(LitToken::slash, tokStart);
    case ':':
      // TODO: Python keeps track of nesting level in the lexer to report
      // mismatched tokens here.  How does that affect error recovery?
      if (*curPtr == '=')
        return formToken(LitToken::colon_equal, tokStart, 1);
      return formToken(LitToken::colon, tokStart);
    case ';':
      return formToken(LitToken::semi, tokStart);
    case '<':
      switch (*curPtr) {
      case '<':
        if (curPtr[1] == '=')
          return formToken(LitToken::less_less_equal, tokStart, 2);
        return formToken(LitToken::less_less, tokStart, 1);
      case '=':
        return formToken(LitToken::less_equal, tokStart, 1);
      case '>':
        return formToken(LitToken::less_greater, tokStart, 1);
      }
      return formToken(LitToken::less, tokStart);
    case '=':
      if (*curPtr == '=')
        return formToken(LitToken::equal_equal, tokStart, 1);
      return formToken(LitToken::equal, tokStart);
    case '>':
      switch (*curPtr) {
      case '=':
        return formToken(LitToken::greater_equal, tokStart, 1);
      case '>':
        if (curPtr[1] == '=')
          return formToken(LitToken::right_right_equal, tokStart, 2);
        return formToken(LitToken::right_right, tokStart, 1);
      }
      return formToken(LitToken::greater, tokStart);
    case '@':
      if (*curPtr == '=')
        return formToken(LitToken::at_equal, tokStart, 1);
      return formToken(LitToken::at, tokStart);
    case '[':
      return formToken(LitToken::l_square, tokStart);
    case ']':
      return formToken(LitToken::r_square, tokStart);
    case '^':
      if (*curPtr == '=')
        return formToken(LitToken::circumflex_equal, tokStart, 1);
      return formToken(LitToken::circumflex, tokStart);
    case '{':
      return formToken(LitToken::l_brace, tokStart);
    case '|':
      if (*curPtr == '=')
        return formToken(LitToken::pipe_equal, tokStart, 1);
      return formToken(LitToken::pipe, tokStart);
    case '}':
      return formToken(LitToken::r_brace, tokStart);
    case '~':
      return formToken(LitToken::tilde, tokStart);
    case '!':
      if (*curPtr == '=')
        return formToken(LitToken::exclaim_equal, tokStart, 1);
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
      return lexInteger(tokStart);

    case '#':
      skipComment();
      continue;
    }
  }
}

/// Lex an identifier or keyword that starts with a letter.
///
/// TODO: Python supports unicode in is_potential_identifier_start etc.
///
LitToken LitLexer::lexIdentifierOrKeyword(const char *tokStart) {
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

  return LitToken(kind, spelling);
}

/// Skip a comment line, starting with a '#' and going to end of line.
void LitLexer::skipComment() {
  while (true) {
    switch (*curPtr++) {
    case '\n':
    case '\r':
      // Newline is end of comment.
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

LitToken LitLexer::lexInteger(const char *tokStart) {
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
      return lexFloat(tokStart);
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
    return lexFloat(tokStart);
  return formToken(LitToken::integer, tokStart);
}

LitToken LitLexer::lexFloat(const char *tokStart) {
  return emitError(curPtr, "lexFloat: TODO");
}

/// Return the a value for the specifed string, which is known to have been
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
  std::string digits;
  digits.reserve(spelling.size());
  for (auto c : spelling) {
    if (c != '_')
      digits.push_back(c);
  }
  spelling = StringRef(digits);
  bool failed = spelling.getAsInteger(base, result);
  assert(!failed && "we know this should always work because we lexed it");
  (void)failed;
  return result;
}

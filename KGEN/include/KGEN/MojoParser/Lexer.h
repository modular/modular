//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Defines the a Lexer and Token interface for .mojo files.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_LEXER_H
#define KGEN_MOJOPARSER_LEXER_H

#include "Support/Compiler/Diags.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SMLoc.h"

namespace M {
class Diags;
class IPRational;
} // namespace M

namespace M::KGEN::LIT {
class LexerCursor;
using llvm::SMLoc;

/// This represents a specific token for .mojo files.
class Token {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#include "TokenKinds.def"
  };

  Token(Kind kind, StringRef spelling, ssize_t indentation)
      : kind(kind), spelling(spelling), indentation(indentation) {}

  /// Return the bytes that make up this token in the original source buffer.
  StringRef getSpelling() const { return spelling; }

  /// Return the indentation of this token.
  std::optional<size_t> getIndentation() const {
    if (indentation == -1)
      return std::nullopt;
    return size_t(indentation);
  }

  /// Return true if the token is the first on a line.
  bool isStartOfLine() const { return indentation != -1; }

  // Token classification.
  Kind getKind() const { return kind; }
  bool is(Kind K) const { return kind == K; }

  bool isAny(Kind k1, Kind k2) const { return is(k1) || is(k2); }

  /// Return true if this token is one of the specified kinds.
  template <typename... T>
  bool isAny(Kind k1, Kind k2, Kind k3, T... others) const {
    if (is(k1))
      return true;
    return isAny(k2, k3, others...);
  }

  /// Return true if this token is any one of the specified token kinds.
  bool isAny(ArrayRef<Kind> kinds) const {
    for (auto k : kinds)
      if (kind == k)
        return true;
    return false;
  }

  bool isNot(Kind k) const { return kind != k; }
  bool isNot(ArrayRef<Kind> kinds) const { return !isAny(kinds); }

  /// Return true if this token isn't one of the specified kinds.
  template <typename... T>
  bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  /// Return true if this is one of the keyword token kinds (e.g. kw_pass).
  bool isKeyword() const;

  /// Return true if the kind is either `identifier` or `escaped_identifier`.
  bool isIdentifier() const;

  // Location processing.
  SMLoc getLoc() const;
  SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

private:
  /// Discriminator that indicates the sort of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;

  /// If this token is at the start of a logical source line, then this
  /// specifies the number of bytes the character is indented by.  If the token
  /// is not at the start of line (or follows a \ on the previous line), then
  /// this contains -1.
  ssize_t indentation;
};

/// This implements a lexer for .mojo files.
class Lexer {
public:
  Lexer(Diags &diags, StringRef curBuffer, const char *curPtr);
  Lexer(Diags &diags, const llvm::MemoryBuffer *buffer);
  Lexer(Diags &diags, const LexerCursor &cursor);

  /// Move to the next valid token.
  void lexToken();

  const Token &getToken() const { return curToken; }

  /// Get an opaque pointer into the lexer state that can be restored later.
  LexerCursor getCursor() const;

  /// Return the a value for the specified string, which is known to have been
  /// lexed as an integer literal token.
  static APInt getIntegerLiteralValue(StringRef spelling);
  /// Return the a value for the specified string, which is known to have been
  /// lexed as a float literal token.
  static IPRational getFloatLiteralValue(StringRef spelling);
  /// Return the a string value of `spelling` after the escape sequences are
  /// handled. `spelling` is known to have been lexed as a string literal token.
  static std::string getStringLiteralValue(StringRef spelling);
  /// Return a location to the start of the given string, after stripping the
  /// wrapping quotes.
  static SMLoc getStringLiteralStartLoc(StringRef spelling);

  InflightDiag emitTokenError(const Twine &message) {
    return emitErrorAt(getToken().getSpelling().data(), message);
  }

  /// Given a location that is at the start of a line, scan backwards to find
  /// the end of the last line that contains a token, or start of the source
  /// buffer if there is none.
  SMLoc findEndOfPreviousLine(SMLoc loc) const;

  /// Return the current buffer we are lexing from.
  StringRef getBuffer() const { return curBuffer; }

private:
  void formToken(Token::Kind kind, const char *tokStart, ssize_t indentation,
                 size_t tokenStartOffset = 0) {
    formToken(kind, StringRef(tokStart, curPtr - tokStart), indentation,
              tokenStartOffset);
  }
  void formToken(Token::Kind kind, StringRef spelling, ssize_t indentation,
                 size_t tokenStartOffset = 0);
  InflightDiag emitErrorAt(const char *loc, const Twine &message);

  // Lexer implementation methods.
  void lexIdentifierOrKeyword(const char *tokStart, ssize_t indentation);
  void lexBacktickIdentifier(const char *tokStart, ssize_t indentation);
  void lexInteger(const char *tokStart, ssize_t indentation);
  void lexFloat(const char *tokStart, ssize_t indentation);
  void lexString(const char *tokStart, ssize_t indentation);
  void skipComment();

private:
  /// This the source file diagnostic manager to use.
  Diags &diags;
  /// This is the overall memory buffer that we are lexing from.
  StringRef curBuffer;
  /// This the start of the next byte to lex.
  const char *curPtr;
  /// This is the next token that hasn't been consumed yet.
  Token curToken;

  // This is the start of the last token that was at a beginning of line, and
  // the indentation (in bytes) of that token.
  const char *lastLineStart;
  ssize_t lastLineIndent;

  Lexer(const Lexer &) = delete;
  void operator=(const Lexer &) = delete;
  friend class LexerCursor;
  friend class LexerCrashReporter;
};

/// This is the state captured for a lexer cursor.
class LexerCursor {
public:
  LexerCursor()
      : curPtr(0), curToken(Token(Token::eof, StringRef(), 0)),
        lastLineStart(nullptr), lastLineIndent(0) {}
  LexerCursor(const Lexer &lexer)
      : curPtr(lexer.curPtr), curToken(lexer.getToken()),
        lastLineStart(lexer.lastLineStart),
        lastLineIndent(lexer.lastLineIndent) {}
  LexerCursor(const LexerCursor &cursor) = default;
  LexerCursor &operator=(const LexerCursor &cursor) = default;

  /// Get a cursor that indicates the end of file.  This isn't for continued
  /// lexing, it is for comparisons.
  static LexerCursor getEOF(const llvm::MemoryBuffer *buffer) {
    LexerCursor result;
    result.curPtr = buffer->getBufferEnd() + 1;
    result.curToken = Token(Token::eof, StringRef(result.curPtr, 0), 0);
    return result;
  }

  void restore(Lexer &lexer) const {
    lexer.curPtr = curPtr;
    lexer.curToken = curToken;
    lexer.lastLineStart = lastLineStart;
    lexer.lastLineIndent = lastLineIndent;
  }

  /// Return true if this cursor is default constructed, not valid for lexing.
  bool isInvalid() const { return curPtr == nullptr; }

  /// Return an internal pointer that represents the cursor state without the
  /// current token.
  const char *getState() const { return curPtr; }
  const Token &getToken() const { return curToken; }

  bool operator==(const LexerCursor &rhs) const { return curPtr == rhs.curPtr; }
  bool operator!=(const LexerCursor &rhs) const { return !(*this == rhs); }

private:
  const char *curPtr;
  Token curToken;
  const char *lastLineStart;
  ssize_t lastLineIndent;
};

inline LexerCursor Lexer::getCursor() const { return LexerCursor(*this); }

/// This crash reporter snapshots the current token a lexer is at.  If a crash
/// happens, it then prints out a region of code from that location to the
/// next-unconsumed token to make it easier to debug parser/typechecker/IR
/// emission related bugs.
///
/// The lexer passed in must outlive this crash reporter.
class LexerCrashReporter : public llvm::PrettyStackTraceEntry {
public:
  LexerCrashReporter(Lexer &lexer, const char *message)
      : startPtr(lexer.getToken().getSpelling().data()), lexer(lexer),
        message(message) {}

  LexerCrashReporter(Lexer &lexer, SMLoc loc, const char *message)
      : startPtr(loc.getPointer()), lexer(lexer), message(message) {}

  /// print - Emit information about this stack frame to OS.
  virtual void print(raw_ostream &os) const override;

public:
  const char *startPtr;
  Lexer &lexer;
  const char *message;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_LEXER_H

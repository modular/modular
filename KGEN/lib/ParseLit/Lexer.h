//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Defines the a Lexer and Token interface for .mojo files.
//
//===----------------------------------------------------------------------===//

#ifndef LITLEXER_H
#define LITLEXER_H

#include "LitSharedState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"

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
class Lexer : public LitSharedStateUser {
public:
  Lexer(LitSharedState &sharedState, const llvm::MemoryBuffer *buffer);
  Lexer(LitSharedState &sharedState, const LexerCursor &cursor);

  /// Move to the next valid token.
  void lexToken() { curToken = lexTokenImpl(); }

  const Token &getToken() const { return curToken; }

  /// Get an opaque pointer into the lexer state that can be restored later.
  LexerCursor getCursor() const;

  /// Return the a value for the specified string, which is known to have been
  /// lexed as an integer literal token.
  static APInt getIntegerLiteralValue(StringRef spelling);
  /// Return the a value for the specified string, which is known to have been
  /// lexed as a float literal token.
  static APFloat getFloatLiteralValue(StringRef spelling);
  /// Return the a string value of `spelling` after the escape sequences are
  /// handled. `spelling` is known to have been lexed as a string literal token.
  static std::string getStringLiteralValue(StringRef spelling);

  Token emitTokenError(const Twine &message) {
    return emitErrorAt(getToken().getSpelling().data(), message);
  }

  /// Given a location that is at the start of a line, scan backwards to find
  /// the end of the last line that contains a token, or start of the source
  /// buffer if there is none.
  SMLoc findEndOfPreviousLine(SMLoc loc) const;

  /// Given a valid pointer into a source buffer for some token, return the
  /// length of the token by re-lex'ing it.  This is efficient.
  static size_t getTokenLength(LitSharedState &shared, SMLoc loc);

private:
  Token lexTokenImpl();

  // Helpers.
  Token formToken(Token::Kind kind, const char *tokStart, ssize_t indentation) {
    return Token(kind, StringRef(tokStart, curPtr - tokStart), indentation);
  }

  Token emitErrorAt(const char *loc, const Twine &message);

  // Lexer implementation methods.
  Token lexIdentifierOrKeyword(const char *tokStart, ssize_t indentation);
  Token lexBacktickIdentifier(const char *tokStart, ssize_t indentation);
  Token lexInteger(const char *tokStart, ssize_t indentation);
  Token lexFloat(const char *tokStart, ssize_t indentation);
  Token lexString(const char *tokStart, ssize_t indentation);
  void skipComment();

private:
  Lexer(LitSharedState &shared, StringRef curBuffer, const char *curPtr);

  StringRef curBuffer;
  const char *curPtr;

  /// This is the next token that hasn't been consumed yet.
  Token curToken;

  Lexer(const Lexer &) = delete;
  void operator=(const Lexer &) = delete;
  friend class LexerCursor;
};

/// This is the state captured for a lexer cursor.
class LexerCursor {
public:
  LexerCursor() : LexerCursor(Token(Token::eof, StringRef(), 0)) {}
  LexerCursor(const Lexer &lexer)
      : state(lexer.curPtr), curToken(lexer.getToken()) {}
  LexerCursor(const Token &tok)
      : state(tok.getSpelling().data()), curToken(tok) {}
  LexerCursor(const LexerCursor &cursor) = default;
  LexerCursor &operator=(const LexerCursor &cursor) = default;

  void restore(Lexer &lexer) const {
    lexer.curPtr = state;
    lexer.curToken = curToken;
  }

  /// Return an internal pointer that represents the cursor state without the
  /// current token.
  const char *getState() const { return state; }
  const Token &getToken() const { return curToken; }

  bool operator==(const LexerCursor &rhs) const { return state == rhs.state; }
  bool operator!=(const LexerCursor &rhs) const { return !(*this == rhs); }

private:
  const char *state;
  Token curToken;
};

inline LexerCursor Lexer::getCursor() const { return LexerCursor(*this); }

} // namespace M::KGEN::LIT

#endif // LITLEXER_H

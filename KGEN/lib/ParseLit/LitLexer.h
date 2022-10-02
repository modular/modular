//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Defines the a Lexer and Token interface for .lit files.
//
//===----------------------------------------------------------------------===//

#ifndef LITLEXER_H
#define LITLEXER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"

namespace M::KGEN::LIT {
class LitLexerCursor;

/// This represents a specific token for .lit files.
class LitToken {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#include "LitTokenKinds.def"
  };

  LitToken(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  // Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

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
  bool isAny(ArrayRef<LitToken::Kind> kinds) const {
    for (auto k : kinds)
      if (kind == k)
        return true;
    return false;
  }

  bool isNot(Kind k) const { return kind != k; }

  /// Return true if this token isn't one of the specified kinds.
  template <typename... T>
  bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  /// Return true if this is one of the keyword token kinds (e.g. kw_pass).
  bool isKeyword() const;

  // Location processing.
  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

private:
  /// Discriminator that indicates the sort of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

/// This implements a lexer for .lit files.
class LitLexer {
public:
  LitLexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context);

  const llvm::SourceMgr &getSourceMgr() const { return sourceMgr; }

  /// Move to the next valid token.
  void lexToken() { curToken = lexTokenImpl(); }

  const LitToken &getToken() const { return curToken; }

  mlir::Location translateLocation(llvm::SMLoc loc);

  /// Return the indentation level of the specified token or None if this token
  /// is preceded by another token on the same line.
  Optional<size_t> getIndentation(const LitToken &tok) const;

  /// Get an opaque pointer into the lexer state that can be restored later.
  LitLexerCursor getCursor() const;

  /// Return the a value for the specifed string, which is known to have been
  /// lexed as an integer literal token.
  static APInt getIntegerLiteralValue(StringRef spelling);

private:
  LitToken lexTokenImpl();

  // Helpers.
  LitToken formToken(LitToken::Kind kind, const char *tokStart) {
    return LitToken(kind, StringRef(tokStart, curPtr - tokStart));
  }

  LitToken formToken(LitToken::Kind kind, const char *tokStart, size_t incr) {
    curPtr += incr;
    return LitToken(kind, StringRef(tokStart, curPtr - tokStart));
  }

  LitToken emitError(const char *loc, const Twine &message);

  // Lexer implementation methods.
  LitToken lexIdentifierOrKeyword(const char *tokStart);
  LitToken lexNumber(const char *tokStart);
  void skipComment();

  const llvm::SourceMgr &sourceMgr;
  const mlir::StringAttr bufferNameIdentifier;

  StringRef curBuffer;
  const char *curPtr;

  /// This is the next token that hasn't been consumed yet.
  LitToken curToken;

  LitLexer(const LitLexer &) = delete;
  void operator=(const LitLexer &) = delete;
  friend class LitLexerCursor;
};

/// This is the state captured for a lexer cursor.
class LitLexerCursor {
public:
  LitLexerCursor(const LitLexer &lexer)
      : state(lexer.curPtr), curToken(lexer.getToken()) {}

  void restore(LitLexer &lexer) {
    lexer.curPtr = state;
    lexer.curToken = curToken;
  }

private:
  const char *state;
  LitToken curToken;
};

inline LitLexerCursor LitLexer::getCursor() const {
  return LitLexerCursor(*this);
}

} // namespace M::KGEN::LIT

#endif // LITLEXER_H

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

  LitToken(Kind kind, StringRef spelling, ssize_t indentation)
      : kind(kind), spelling(spelling), indentation(indentation) {}

  /// Return the bytes that make up this token in the original source buffer.
  StringRef getSpelling() const { return spelling; }

  /// Return the indentation of this token.
  Optional<size_t> getIndentation() const {
    if (indentation == -1)
      return None;
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

  /// If this token is at the start of a logical source line, then this
  /// specifies the number of bytes the character is indented by.  If the token
  /// is not at the start of line (or follows a \ on the previous line), then
  /// this contains -1.
  ssize_t indentation;
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

  /// Get an opaque pointer into the lexer state that can be restored later.
  LitLexerCursor getCursor() const;

  /// Return the a value for the specified string, which is known to have been
  /// lexed as an integer literal token.
  static APInt getIntegerLiteralValue(StringRef spelling);
  /// Return the a value for the specified string, which is known to have been
  /// lexed as a float literal token.
  static APFloat getFloatLiteralValue(StringRef spelling);
  /// Return the a string value of `spelling` after the escape sequences are
  /// handled. `spelling` is known to have been lexed as a string literal token.
  static std::string getStringLiteralValue(StringRef spelling);

private:
  LitToken lexTokenImpl();

  // Helpers.
  LitToken formToken(LitToken::Kind kind, const char *tokStart,
                     ssize_t indentation) {
    return LitToken(kind, StringRef(tokStart, curPtr - tokStart), indentation);
  }

  LitToken emitError(const char *loc, const Twine &message);

  // Lexer implementation methods.
  LitToken lexIdentifierOrKeyword(const char *tokStart, ssize_t indentation);
  LitToken lexInteger(const char *tokStart, ssize_t indentation);
  LitToken lexFloat(const char *tokStart, ssize_t indentation);
  LitToken lexString(const char *tokStart, ssize_t indentation);
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
  LitLexerCursor(const LitLexerCursor &cursor) = default;
  LitLexerCursor &operator=(const LitLexerCursor &cursor) = default;

  void restore(LitLexer &lexer) {
    lexer.curPtr = state;
    lexer.curToken = curToken;
  }

  Location getLoc(LitLexer &lexer) const;

private:
  const char *state;
  LitToken curToken;
};

inline LitLexerCursor LitLexer::getCursor() const {
  return LitLexerCursor(*this);
}

} // namespace M::KGEN::LIT

#endif // LITLEXER_H

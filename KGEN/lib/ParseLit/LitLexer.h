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

  bool isNot(Kind k) const { return kind != k; }

  /// Return true if this token isn't one of the specified kinds.
  template <typename... T>
  bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  /// Return true if this is one of the keyword token kinds (e.g. kw_wire).
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
  Optional<unsigned> getIndentation(const LitToken &tok) const;

private:
  LitToken lexTokenImpl();

  // Helpers.
  LitToken formToken(LitToken::Kind kind, const char *tokStart) {
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
};

} // namespace M::KGEN::LIT

#endif // LITLEXER_H

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_COMMON_LSPPROTOCOL_SEMANTICTOKENS_H
#define KGEN_TOOLS_COMMON_LSPPROTOCOL_SEMANTICTOKENS_H

#include "Protocol.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

namespace M::Mojo::LSP {
//===----------------------------------------------------------------------===//
// SemanticToken Kind
//===----------------------------------------------------------------------===//

/// This enum represents all the different kinds of tokens that can be
/// highlighted.
enum class SemanticTokenKind {
  kVariable = 0,
  kSpecialVariable,
  kParameter,
  kFunction,
  kMethod,
  kField,
  kClass,
  kTrait,
  kType,
  kModule,

  kCount
};

/// Convert the given token kind into a string representing the LSP token type.
StringRef toLspSemanticTokenType(SemanticTokenKind kind);

//===----------------------------------------------------------------------===//
// SemanticToken Modifier
//===----------------------------------------------------------------------===//

/// This enum represents all the different modifiers that can be applied to
/// highlighted tokens.
enum class SemanticTokenModifier {
  kCount,
};

/// Convert the given token modifier into a string representing the LSP token
/// modifier.
StringRef toLspSemanticTokenModifier(SemanticTokenModifier modifier);

//===----------------------------------------------------------------------===//
// SemanticToken Token
//===----------------------------------------------------------------------===//

/// This class represents a highlighted token.
struct SemanticToken {
  SemanticToken() : kind(SemanticTokenKind::kCount) {}
  SemanticToken(SemanticTokenKind kind, mlir::lsp::Range range,
                uint32_t modifiers = 0)
      : kind(kind), modifiers(modifiers), range(range) {}

  bool operator==(const SemanticToken &rhs) const;
  bool operator<(const SemanticToken &rhs) const;

  /// Add a modifier to the token.
  SemanticToken &addModifier(SemanticTokenModifier modifier) {
    modifiers |= 1 << static_cast<unsigned>(modifier);
    return *this;
  }

  /// The kind of token this is.
  SemanticTokenKind kind;

  /// Modifiers that affect the token.
  uint32_t modifiers = 0;

  /// The range of the token.
  mlir::lsp::Range range;
};

/// Convert the given tokens into LSP semantic tokens. LSP semantic tokens need
/// to be constructed at the same time, because the position fields of an LSP
/// token are relative to the previous token.
std::vector<mlir::lsp::SemanticToken>
toLspSemanticTokens(ArrayRef<SemanticToken> tokens);

/// Convert the given LSP semantic tokens into the Mojo equivalent. We process
/// all at once because the position fields of an LSP token are relative to the
/// previous token.
std::vector<SemanticToken>
fromLspSemanticTokens(ArrayRef<mlir::lsp::SemanticToken> tokens);

/// Compute the difference between the two sets of tokens.
std::vector<mlir::lsp::SemanticTokensEdit>
diffTokens(ArrayRef<mlir::lsp::SemanticToken> before,
           ArrayRef<mlir::lsp::SemanticToken> after);

} // namespace M::Mojo::LSP

#endif

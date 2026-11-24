//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

/// The only token modifiers are address spaces: address space `n` is modifier
/// bit `n`, named `addressSpace<n>` in the LSP legend. The server passes the
/// integer through unchanged rather than naming address spaces, because their
/// meaning depends on the target; the client decides what to call them. This
/// many address spaces get a modifier, which covers every one the stdlib
/// names; higher, target-specific ones are not reported.
constexpr unsigned kNumSemanticTokenModifiers = 16;

/// Convert the given token modifier bit into a string representing the LSP
/// token modifier.
StringRef toLspSemanticTokenModifier(unsigned modifier);

//===----------------------------------------------------------------------===//
// SemanticToken Token
//===----------------------------------------------------------------------===//

/// This class represents a highlighted token.
struct SemanticToken {
  SemanticToken() : kind(SemanticTokenKind::kCount) {}
  SemanticToken(SemanticTokenKind kind, llvm::lsp::Range range,
                uint32_t modifiers = 0)
      : kind(kind), modifiers(modifiers), range(range) {}

  bool operator==(const SemanticToken &rhs) const;
  bool operator<(const SemanticToken &rhs) const;

  /// Mark the token as living in the given address space, if that address
  /// space has a modifier.
  SemanticToken &setAddressSpace(int64_t addressSpace) {
    if (addressSpace >= 0 && addressSpace < kNumSemanticTokenModifiers)
      modifiers |= 1u << addressSpace;
    return *this;
  }

  /// The kind of token this is.
  SemanticTokenKind kind;

  /// Modifiers that affect the token.
  uint32_t modifiers = 0;

  /// The range of the token.
  llvm::lsp::Range range;
};

/// Convert the given tokens into LSP semantic tokens. LSP semantic tokens need
/// to be constructed at the same time, because the position fields of an LSP
/// token are relative to the previous token.
std::vector<llvm::lsp::SemanticToken>
toLspSemanticTokens(ArrayRef<SemanticToken> tokens);

/// Convert the given LSP semantic tokens into the Mojo equivalent. We process
/// all at once because the position fields of an LSP token are relative to the
/// previous token.
std::vector<SemanticToken>
fromLspSemanticTokens(ArrayRef<llvm::lsp::SemanticToken> tokens);

/// Compute the difference between the two sets of tokens.
std::vector<llvm::lsp::SemanticTokensEdit>
diffTokens(ArrayRef<llvm::lsp::SemanticToken> before,
           ArrayRef<llvm::lsp::SemanticToken> after);

} // namespace M::Mojo::LSP

#endif

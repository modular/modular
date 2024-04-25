//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_DOCUMENT_H
#define KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_DOCUMENT_H

#include "Protocol.h"
#include "Support/LLVMForwardDecls.h"

namespace M {
/// Class representing an in-memory document.
class Document {
public:
  Document(StringRef uri, StringRef text);

  mlir::lsp::URIForFile getURI() const { return uri; }

  StringRef getContents() const { return contents; }

  /// Get the full range of the entire text.
  mlir::lsp::Range getFullRange() const;

  /// Get the position of the first occurrence of the given substring in the
  /// document within a single line.
  std::optional<mlir::lsp::Position> findFirstPos(StringRef substr) const;

  /// Get the position of the last occurrence of the given substring in the
  /// document within a single line.
  std::optional<mlir::lsp::Position> findLastPos(StringRef substr) const;

  /// Get the range of the first occurrence of the given substring in the
  /// document within a single line.
  std::optional<mlir::lsp::Range> findFirstRange(StringRef substr) const;

  /// Get the range of the last occurrence of the given substring in the
  /// document within a single line.
  std::optional<mlir::lsp::Range> findLastRange(StringRef substr) const;

private:
  mlir::lsp::URIForFile uri;
  std::string contents;
  SmallVector<StringRef> lines;
};

} // namespace M

#endif // KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_DOCUMENT_H

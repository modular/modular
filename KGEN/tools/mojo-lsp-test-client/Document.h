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
  Document(StringRef uri, StringRef contents);

  mlir::lsp::URIForFile getURI() const { return uri; }

  StringRef getContents() const { return contents; }

private:
  mlir::lsp::URIForFile uri;
  std::string contents;
};

} // namespace M

#endif // KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_DOCUMENT_H

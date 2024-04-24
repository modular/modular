//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Document.h"

using namespace M;
namespace lsp = mlir::lsp;

Document::Document(StringRef uri, StringRef contents) : contents(contents) {
  if (llvm::Expected<lsp::URIForFile> uriOr = lsp::URIForFile::fromURI(uri))
    this->uri = std::move(*uriOr);
  else
    llvm::report_fatal_error(uriOr.takeError());
}

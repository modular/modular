//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Document.h"

using namespace M;
namespace lsp = mlir::lsp;

Document::Document(StringRef uri, StringRef text) : contents(text) {
  if (llvm::Expected<lsp::URIForFile> uriOr = lsp::URIForFile::fromURI(uri))
    this->uri = std::move(*uriOr);
  else
    llvm::report_fatal_error(uriOr.takeError());

  StringRef(contents).split(lines, '\n');
}

lsp::Range Document::getFullRange() const {
  return {lsp::Position{0, 0}, lsp::Position{(int)lines.size(), 0}};
}

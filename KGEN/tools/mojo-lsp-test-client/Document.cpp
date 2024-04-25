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

std::optional<lsp::Position> Document::findFirstPos(StringRef substr) const {
  if (std::optional<lsp::Range> range = findFirstRange(substr))
    return range->start;

  return {};
}

std::optional<lsp::Position> Document::findLastPos(StringRef substr) const {
  if (std::optional<lsp::Range> range = findLastRange(substr))
    return range->start;

  return {};
}

std::optional<mlir::lsp::Range>
Document::findFirstRange(StringRef substr) const {
  for (size_t line = 0, e = lines.size(); line < e; ++line)
    if (size_t pos = lines[line].find(substr); pos != StringRef::npos)
      return lsp::Range{lsp::Position(line, pos),
                        lsp::Position(line, pos + substr.size())};

  return {};
}

std::optional<mlir::lsp::Range>
Document::findLastRange(StringRef substr) const {
  if (lines.empty())
    return {};
  for (size_t line = lines.size() - 1; line; --line)
    if (size_t pos = lines[line].rfind(substr); pos != StringRef::npos)
      return lsp::Range{lsp::Position(line, pos),
                        lsp::Position(line, pos + substr.size())};

  return {};
}

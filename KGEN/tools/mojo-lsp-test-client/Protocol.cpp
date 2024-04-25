//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"

namespace lsp = mlir::lsp;
using namespace lsp;

llvm::json::Value lsp::toJSON(const TextDocumentItem &params) {
  return llvm::json::Object{{"uri", params.uri},
                            {"languageId", params.languageId},
                            {"text", params.text},
                            {"version", params.version}};
}

llvm::json::Value lsp::toJSON(const DidOpenTextDocumentParams &params) {
  return llvm::json::Object{{"textDocument", params.textDocument}};
}

llvm::json::Value lsp::toJSON(const TextDocumentPositionParams &params) {
  return llvm::json::Object{{"textDocument", params.textDocument},
                            {"position", params.position}};
}

llvm::json::Value lsp::toJSON(const CodeActionContext &context) {
  return llvm::json::Object{{"diagnostics", context.diagnostics},
                            {"only", context.only}};
}

llvm::json::Value lsp::toJSON(const CodeActionParams &params) {
  return llvm::json::Object{{"textDocument", params.textDocument},
                            {"range", params.range},
                            {"context", params.context}};
}

llvm::json::Value lsp::toJSON(const Hover2 &hover) {
  return toJSON(static_cast<Hover>(hover));
}

bool lsp::fromJSON(const llvm::json::Value &value, Hover2 &range,
                   llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("contents", range.contents) &&
         o.mapOptional("range", range.range);
}

bool lsp::fromJSON(const llvm::json::Value &value, MarkupKind &kind,
                   llvm::json::Path path) {
  std::optional<StringRef> str = value.getAsString();
  if (!str)
    return false;
  if (*str == "plaintext") {
    kind = MarkupKind::PlainText;
    return true;
  }
  if (*str == "markdown") {
    kind = MarkupKind::Markdown;
    return true;
  }
  return false;
}

bool lsp::fromJSON(const llvm::json::Value &value, MarkupContent &mc,
                   llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("kind", mc.kind) && o.map("value", mc.value);
}

bool lsp::fromJSON(const llvm::json::Value &value, CodeAction &codeAction,
                   llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("title", codeAction.title) &&
         o.map("kind", codeAction.kind) &&
         o.map("diagnostics", codeAction.diagnostics) &&
         o.map("isPreferred", codeAction.isPreferred) &&
         o.map("edit", codeAction.edit);
}

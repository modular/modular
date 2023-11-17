//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::lsp;

// Helper that doesn't treat `null` and absent fields as failures.
template <typename T>
static bool mapOptOrNull(const llvm::json::Value &params,
                         llvm::StringLiteral prop, T &out,
                         llvm::json::Path path) {
  const llvm::json::Object *o = params.getAsObject();
  assert(o);

  // Field is missing or null.
  auto *v = o->get(prop);
  if (!v || v->getAsNull())
    return true;
  return fromJSON(*v, out, path.field(prop));
}

//===----------------------------------------------------------------------===//
// SignatureInformation
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const SignatureInformation2 &value) {
  assert(!value.label.empty() && "signature information label is required");
  llvm::json::Object result{
      {"label", value.label},
      {"parameters", llvm::json::Array(value.parameters)},
  };
  if (value.documentation)
    result["documentation"] = value.documentation;
  return std::move(result);
}

//===----------------------------------------------------------------------===//
// SignatureHelp
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const SignatureHelp2 &value) {
  assert(value.activeSignature >= 0 &&
         "Unexpected negative value for number of active signatures.");
  assert(value.activeParameter >= 0 &&
         "Unexpected negative value for active parameter index");
  return llvm::json::Object{
      {"activeSignature", value.activeSignature},
      {"activeParameter", value.activeParameter},
      {"signatures", llvm::json::Array(value.signatures)},
  };
}

//===----------------------------------------------------------------------===//
// NotebookCell
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value, NotebookCell &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  if (!o)
    return false;

  int kind = 0;
  if (!mapOptOrNull(value, "kind", kind, path))
    return false;
  result.kind = (NotebookCellKind)kind;

  return o.map("document", result.document);
}

//===----------------------------------------------------------------------===//
// NotebookDocument
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookDocument &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("uri", result.uri) &&
         o.map("notebookType", result.notebookType) &&
         o.map("version", result.version) && o.map("cells", result.cells);
}

//===----------------------------------------------------------------------===//
// DidOpenNotebookDocumentParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         DidOpenNotebookDocumentParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("notebookDocument", result.notebookDocument) &&
         o.map("cellTextDocuments", result.cellTextDocuments);
}

//===----------------------------------------------------------------------===//
// NotebookDocumentChangeEvent
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookCellArrayChange &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  o.map("cells", result.cells);
  return o && o.map("start", result.start) &&
         o.map("deleteCount", result.deleteCount);
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookDocumentChangeEvent::CellsStructure &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);

  o.map("didOpen", result.didOpen);
  o.map("didClose", result.didClose);
  return o && o.map("array", result.array);
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookDocumentChangeEvent::CellsTextContent &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("document", result.document) &&
         o.map("changes", result.changes);
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookDocumentChangeEvent::Cells &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  if (!o)
    return false;

  o.map("structure", result.structure);
  o.map("data", result.data);
  o.map("textContent", result.textContent);
  return true;
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookDocumentChangeEvent &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("cells", result.cells);
}

//===----------------------------------------------------------------------===//
// DidChangeNotebookDocumentParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         DidChangeNotebookDocumentParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("notebookDocument", result.notebookDocument) &&
         o.map("change", result.change);
}

//===----------------------------------------------------------------------===//
// DidCloseNotebookDocumentParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         DidCloseNotebookDocumentParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("notebookDocument", result.notebookDocument) &&
         o.map("cellTextDocuments", result.cellTextDocuments);
}

//===----------------------------------------------------------------------===//
// Semantic Token
//===----------------------------------------------------------------------===//

/// The encoded size of a single semantic token.
static constexpr unsigned kSemanticTokenEncodingSize = 5;

/// Encode the given list of semantic tokens into a JSON array.
static llvm::json::Value encodeTokens(ArrayRef<SemanticToken> tokens) {
  llvm::json::Array result;
  result.reserve(kSemanticTokenEncodingSize * tokens.size());
  for (const SemanticToken &token : tokens) {
    result.push_back(token.deltaLine);
    result.push_back(token.deltaStart);
    result.push_back(token.length);
    result.push_back(token.tokenType);
    result.push_back(token.tokenModifiers);
  }
  assert(result.size() == (kSemanticTokenEncodingSize * tokens.size()));
  return std::move(result);
}

bool SemanticToken::operator==(const SemanticToken &rhs) const {
  return std::tie(deltaLine, deltaStart, length, tokenType, tokenModifiers) ==
         std::tie(rhs.deltaLine, rhs.deltaStart, rhs.length, rhs.tokenType,
                  rhs.tokenModifiers);
}

llvm::json::Value mlir::lsp::toJSON(const SemanticTokens &value) {
  return llvm::json::Object{{"resultId", value.resultId},
                            {"data", encodeTokens(value.tokens)}};
}

//===----------------------------------------------------------------------===//
// SemanticTokensParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &params,
                         SemanticTokensParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(params, path);
  return o && o.map("textDocument", result.textDocument);
}

bool mlir::lsp::fromJSON(const llvm::json::Value &params,
                         SemanticTokensDeltaParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(params, path);
  return o && o.map("textDocument", result.textDocument) &&
         o.map("previousResultId", result.previousResultId);
}

//===----------------------------------------------------------------------===//
// SemanticTokensEdit
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const SemanticTokensEdit &value) {
  return llvm::json::Object{
      {"start", kSemanticTokenEncodingSize * value.startToken},
      {"deleteCount", kSemanticTokenEncodingSize * value.deleteTokens},
      {"data", encodeTokens(value.tokens)}};
}

//===----------------------------------------------------------------------===//
// SemanticTokensOrDelta
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const SemanticTokensOrDelta &value) {
  llvm::json::Object result{{"resultId", value.resultId}};
  if (value.edits)
    result["edits"] = *value.edits;
  if (value.tokens)
    result["data"] = encodeTokens(*value.tokens);
  return std::move(result);
}

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains extensions to the mlir/Tools/lsp-server-support/Protocol.h
// file.
//
// TODO: upstream all the changes in this file, as they are generic.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_PROTOCOL_H
#define KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_PROTOCOL_H

#include "mlir/Tools/lsp-server-support/Protocol.h"

namespace mlir::lsp {
/// Extension of Hover that has a default constructor to make fromJSON happy.
struct Hover2 : Hover {
  using Hover::Hover;
  Hover2() : Hover(Range()){};
};

llvm::json::Value toJSON(const TextDocumentItem &params);

llvm::json::Value toJSON(const DidOpenTextDocumentParams &params);

llvm::json::Value toJSON(const TextDocumentPositionParams &params);

llvm::json::Value toJSON(const CodeActionContext &context);

llvm::json::Value toJSON(const CodeActionParams &params);

llvm::json::Value toJSON(const Hover2 &hover);

bool fromJSON(const llvm::json::Value &value, Hover2 &range,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, MarkupKind &kind,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, MarkupContent &mc,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, CodeAction &codeAction,
              llvm::json::Path path);
} // namespace mlir::lsp

#endif // KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_PROTOCOL_H

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/DiagnosticHandler.h"

using namespace M;

DiagnosticHandler::DiagnosticHandler(MLIRContext *ctx) : ctx(ctx) {
  handlerID = ctx->getDiagEngine().registerHandler([this](Diagnostic &diag) {
    diagnostics.push_back(std::move(diag));
    return mlir::success();
  });
}

DiagnosticHandler::~DiagnosticHandler() { release(); }

void DiagnosticHandler::release() {
  ctx->getDiagEngine().eraseHandler(handlerID);
}

void DiagnosticHandler::emitDiagnostics(
    function_ref<void(Diagnostic &)> emitFn) {
  for (Diagnostic &diag : diagnostics)
    emitFn(diag);
}

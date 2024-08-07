//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/DiagnosticHandler.h"

using namespace M;

DiagnosticHandler::DiagnosticHandler(MLIRContext *ctx) : ctx(ctx) {
  handlerID = ctx->getDiagEngine().registerHandler([this](Diagnostic &diag) {
    // Filter out processing diag that is does not have the same handlerID.
    auto shouldProcess = [handlerID =
                              this->handlerID](mlir::DiagnosticArgument &arg) {
      return (arg.getKind() ==
                  mlir::DiagnosticArgument::DiagnosticArgumentKind::Unsigned &&
              arg.getAsUnsigned() == handlerID);
    };

    SmallVectorImpl<mlir::DiagnosticArgument> &metadata = diag.getMetadata();

    if (llvm::any_of(metadata, shouldProcess)) {
      diagnostics.push_back(std::move(diag));
      return mlir::success();
    }
    return mlir::failure();
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

mlir::DiagnosticEngine::HandlerID DiagnosticHandler::getHandlerID() {
  return handlerID;
}

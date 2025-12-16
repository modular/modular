//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/DiagnosticHandler.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Threading.h"
#include <utility>

using namespace M;

DiagnosticHandler::DiagnosticHandler(MLIRContext *ctx, bool capturePerThread)
    : ctx(ctx), capturePerThread(capturePerThread) {
  threadID = llvm::get_threadid();
  handlerID = ctx->getDiagEngine().registerHandler([this](Diagnostic &diag) {
    if (!this->capturePerThread || this->threadID == llvm::get_threadid()) {
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

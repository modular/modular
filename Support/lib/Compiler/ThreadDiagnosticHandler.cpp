//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/ThreadDiagnosticHandler.h"
#include "llvm/Support/Threading.h"

using namespace M;

ThreadDiagnosticHandler::ThreadDiagnosticHandler(MLIRContext *ctx) : ctx(ctx) {
  handlerID = ctx->getDiagEngine().registerHandler(
      [this, tid = llvm::get_threadid()](Diagnostic &diag) {
        if (llvm::get_threadid() != tid)
          return failure();
        diagnostics.push_back(std::move(diag));
        return mlir::success();
      });
}

ThreadDiagnosticHandler::~ThreadDiagnosticHandler() { release(); }

void ThreadDiagnosticHandler::release() {
  ctx->getDiagEngine().eraseHandler(handlerID);
}

void ThreadDiagnosticHandler::emitDiagnostics(
    function_ref<void(Diagnostic &)> emitFn) {
  for (Diagnostic &diag : diagnostics)
    emitFn(diag);
}

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_THREADDIAGNOSTICHANDLER_H
#define SUPPORT_COMPILER_THREADDIAGNOSTICHANDLER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Diagnostics.h"

namespace M {
/// This diagnostic handler captures MLIR diagnostics emitted only by the thread
/// in which the diagnostic handler was created.
class ThreadDiagnosticHandler {
public:
  ThreadDiagnosticHandler(MLIRContext *ctx);
  ~ThreadDiagnosticHandler();

  /// Emit the diagnostics.
  void emitDiagnostics(function_ref<void(Diagnostic &)> emitFn);

  /// Manually remove the handler from the context.
  void release();

private:
  /// The MLIR context.
  MLIRContext *ctx;
  /// The ID of the registered handler.
  mlir::DiagnosticEngine::HandlerID handlerID = 0;
  /// The captured diagnostics.
  std::vector<Diagnostic> diagnostics;
};
} // namespace M

#endif // SUPPORT_COMPILER_THREADDIAGNOSTICHANDLER_H

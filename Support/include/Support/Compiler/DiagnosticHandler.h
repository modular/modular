//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_DIAGNOSTICHANDLER_H
#define SUPPORT_COMPILER_DIAGNOSTICHANDLER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Diagnostics.h"

namespace M {
/// This diagnostic handler captures MLIR diagnostics emitted into a vector.
class DiagnosticHandler {
public:
  DiagnosticHandler(MLIRContext *ctx, bool capturePerThread = true);
  ~DiagnosticHandler();

  /// Emit the diagnostics.
  void emitDiagnostics(function_ref<void(Diagnostic &)> emitFn);

  /// Manually remove the handler from the context.
  void release();

  /// Get the global HandlerID which is a unique identifier for this Handler.
  mlir::DiagnosticEngine::HandlerID getHandlerID();

  /// Return true if there is any diagnostic to emit
  bool hasDiagnostics() const { return !diagnostics.empty(); }

  /// Return the captured diagnostics
  const std::vector<Diagnostic> &getDiagnostics() const { return diagnostics; }

private:
  /// The MLIR context.
  MLIRContext *ctx;
  /// The ID of the registered handler.
  mlir::DiagnosticEngine::HandlerID handlerID = 0;
  /// The thread ID of the thread that registered the handler.
  uint64_t threadID = 0;
  /// Whether to capture diagnostics from all threads.
  bool capturePerThread = true;
  /// The captured diagnostics.
  std::vector<Diagnostic> diagnostics;
};
} // namespace M

#endif // SUPPORT_COMPILER_DIAGNOSTICHANDLER_H

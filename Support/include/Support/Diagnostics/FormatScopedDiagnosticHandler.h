//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DIAGNOSTICS_FORMATSCOPEDDIAGNOSTICHANDLER_H
#define SUPPORT_DIAGNOSTICS_FORMATSCOPEDDIAGNOSTICHANDLER_H

#include "mlir/IR/Diagnostics.h"

#include <string>
#include <vector>

namespace M {

/// Capture the diagnostics in a given scope. Later, generate a formatted
/// message that can be displayed to a user. No formatting is done in the
/// diagnostics handler in order to save time. The diagnostics are captured in
/// the handler, but only formatted when formatMessage is called.
class FormatScopedDiagnosticHandler : public mlir::ScopedDiagnosticHandler {
public:
  FormatScopedDiagnosticHandler(mlir::MLIRContext *ctx);
  std::string formatMessage() const;

  static void emitDiagLocSeverity(llvm::raw_ostream &os,
                                  const mlir::Diagnostic &diag);
  static void emitDiagnosticToStream(llvm::raw_ostream &os,
                                     const mlir::Diagnostic &diag);

private:
  std::vector<mlir::Diagnostic> diagnostics;
};

} // namespace M

#endif // SUPPORT_DIAGNOSTICS_FORMATSCOPEDDIAGNOSTICHANDLER_H

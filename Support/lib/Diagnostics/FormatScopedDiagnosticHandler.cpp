//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"

using namespace M;
using namespace mlir;

static StringRef severityToString(mlir::DiagnosticSeverity severity) {
  if (severity == mlir::DiagnosticSeverity::Note)
    return "note";
  if (severity == mlir::DiagnosticSeverity::Warning)
    return "warning";
  if (severity == mlir::DiagnosticSeverity::Error)
    return "error";
  if (severity == mlir::DiagnosticSeverity::Remark)
    return "remark";

  llvm_unreachable("Unexpected diagnostic severity enum value");
}

static std::string locationToString(Location location) {
  auto fileLoc = dyn_cast<mlir::FileLineColLoc>(location);
  if (!fileLoc)
    return std::string();

  return Twine(Twine(fileLoc.getFilename()) + ":" + Twine(fileLoc.getLine()) +
               ":" + Twine(fileLoc.getColumn()))
      .str();
}

void FormatScopedDiagnosticHandler::emitDiagLocSeverity(
    raw_ostream &os, const Diagnostic &diag) {
  // Only display the location if it is meaningful.
  std::string location = locationToString(diag.getLocation());
  if (!location.empty())
    os << location << ": ";

  os << severityToString(diag.getSeverity()) << ": ";
  os << diag;
  os << "\n";
}

void FormatScopedDiagnosticHandler::emitDiagnosticToStream(
    raw_ostream &os, const Diagnostic &diag) {
  // First emit the diag itself.
  emitDiagLocSeverity(os, diag);

  // Then display each note, indented two spaces.
  const char *indentation = "  ";
  for (Diagnostic &note : diag.getNotes()) {
    os << indentation;
    emitDiagnosticToStream(os, note);
  }
}

FormatScopedDiagnosticHandler::FormatScopedDiagnosticHandler(MLIRContext *ctx)
    : mlir::ScopedDiagnosticHandler(ctx, [&](Diagnostic &diag) {
        diagnostics.push_back(std::move(diag));
      }) {}

std::string FormatScopedDiagnosticHandler::formatMessage() const {
  std::string message;
  llvm::raw_string_ostream messageStream(message);
  for (auto &diagnostic : diagnostics)
    emitDiagnosticToStream(messageStream, diagnostic);
  return message;
}

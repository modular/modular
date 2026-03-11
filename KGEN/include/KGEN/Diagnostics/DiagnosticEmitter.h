//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Convenience helpers for emitting diagnostics with IDs
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_DIAGNOSTICS_DIAGNOSTICEMITTER_H
#define KGEN_DIAGNOSTICS_DIAGNOSTICEMITTER_H

#include "KGEN/Diagnostics/DiagnosticIDs.h"
#include "KGEN/Diagnostics/DiagnosticRegistry.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"

namespace M::KGEN::Diag {
/// Returns the formatted message for a DiagID with optional runtime arguments.
/// Arguments are formatted via llvm::formatv.
///
/// Examples:
///   p.emitError(loc, diagMsg(DiagID::err_foo))
///   return emitError() << diagMsg(DiagID::err_foo, type, attr)
template <typename... Args>
inline std::string diagMsg(DiagID id, Args &&...args) {
  const DiagnosticInfo *info = DiagnosticRegistry::get().lookup(id);
  if (!info) {
    llvm_unreachable(
        "unknown DiagID; every DiagID must be in DiagnosticIDs.def");
  }
  return llvm::formatv(info->messageTemplate.data(),
                       std::forward<Args>(args)...)
      .str();
}

/// Emit a diagnostic error via a registered DiagID (reference overload).
template <typename OpT, typename... Args>
inline mlir::InFlightDiagnostic emitError(OpT &&op, DiagID id, Args &&...args) {
  return op.emitError(diagMsg(id, std::forward<Args>(args)...));
}

/// Emit a diagnostic error via a registered DiagID (pointer overload).
template <typename OpT, typename... Args>
inline mlir::InFlightDiagnostic emitError(OpT *op, DiagID id, Args &&...args) {
  return op->emitError(diagMsg(id, std::forward<Args>(args)...));
}

/// Emit an error via a registered DiagID at a given location.
template <typename ObjT, typename LocT, typename... Args>
inline auto emitError(ObjT &&obj, LocT loc, DiagID id, Args &&...args) {
  return obj.emitError(loc, diagMsg(id, std::forward<Args>(args)...));
}

/// Emit a warning via a registered DiagID at a given location.
template <typename ObjT, typename LocT, typename... Args>
inline auto emitWarning(ObjT &&obj, LocT loc, DiagID id, Args &&...args) {
  return obj.emitWarning(loc, diagMsg(id, std::forward<Args>(args)...));
}

} // namespace M::KGEN::Diag

#endif // KGEN_DIAGNOSTICS_DIAGNOSTICEMITTER_H

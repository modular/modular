//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// KGEN Diagnostic Registry
//
// This file provides a centralized registry for all KGEN compiler diagnostics.
// Each diagnostic is identified by a DiagID enum value, which maps to a
// category, component, and message template.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_DIAGNOSTICS_DIAGNOSTICREGISTRY_H
#define KGEN_DIAGNOSTICS_DIAGNOSTICREGISTRY_H

#include "KGEN/Diagnostics/DiagnosticIDs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace M::KGEN::Diag {

enum class DiagnosticCategory { Error, Warning, Note };

enum class DiagnosticComponent {
  Parser,
  Type,
  MLIR,
  Trait,
  Lifetime,
  Kernel,
  CodeGen,
  Dialect
};

struct DiagnosticInfo {
  DiagID id;
  llvm::StringRef name; // snake_case, e.g. "err_foo"
  DiagnosticCategory category;
  DiagnosticComponent component;
  llvm::StringRef messageTemplate; // {0},{1},... placeholders
};

/// Singleton registry of all KGEN diagnostics, populated from
/// DiagnosticIDs.def.
class DiagnosticRegistry {
public:
  static const DiagnosticRegistry &get();

  /// Returns nullptr if id >= DiagID::NumDiags.
  const DiagnosticInfo *lookup(DiagID id) const;

  llvm::ArrayRef<DiagnosticInfo> getAllDiagnostics() const;

  llvm::SmallVector<const DiagnosticInfo *>
  getDiagnosticsForComponent(DiagnosticComponent component) const;

private:
  DiagnosticRegistry();
  llvm::SmallVector<DiagnosticInfo> diagnostics; // indexed by DiagID value
};

} // namespace M::KGEN::Diag

#endif // KGEN_DIAGNOSTICS_DIAGNOSTICREGISTRY_H

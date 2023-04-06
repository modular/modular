//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_MOJODIAGNOSTIC_H
#define KGEN_LIB_MOJOLLDB_MOJODIAGNOSTIC_H

#include "lldb/Expression/DiagnosticManager.h"
#include "llvm/Support/SourceMgr.h"
#include <cstdint>

namespace M::KGEN::Mojo {
/// Custom diagnostic type that can contain fix-its.
class MojoDiagnostic : public lldb_private::Diagnostic {
public:
  MojoDiagnostic(StringRef message, llvm::ArrayRef<llvm::SMFixIt> fixits,
                 lldb_private::DiagnosticSeverity severity)
      : Diagnostic(message, severity, lldb_private::eDiagnosticOriginLLVM,
                   kMojoCompilerID),
        fixits(fixits.begin(), fixits.end()) {}

  static bool classof(const Diagnostic *diag) {
    return diag->getKind() == lldb_private::eDiagnosticOriginLLVM &&
           diag->GetCompilerID() == kMojoCompilerID;
  }

  bool HasFixIts() const override { return !fixits.empty(); }

  llvm::ArrayRef<llvm::SMFixIt> getFixIts() const { return fixits; }

private:
  /// This is a random, likely unused number that we can use to identify a Mojo
  /// diagnostic.
  static constexpr uint32_t kMojoCompilerID = UINT32_MAX - 12;
  /// Hold onto a list of the fixits.
  llvm::SmallVector<llvm::SMFixIt> fixits;
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_MOJODIAGNOSTIC_H

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Diagnostics/DiagnosticRegistry.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace M::KGEN::Diag;

// Populate diagnostics in declaration order so DiagID value == vector index.
DiagnosticRegistry::DiagnosticRegistry() {
#define DIAG(ID, Category, Component, Message)                                 \
  diagnostics.push_back({DiagID::ID, #ID, DiagnosticCategory::Category,        \
                         DiagnosticComponent::Component, Message});
#include "KGEN/Diagnostics/DiagnosticIDs.def"
}

const DiagnosticRegistry &DiagnosticRegistry::get() {
  static DiagnosticRegistry instance;
  return instance;
}

const DiagnosticInfo *DiagnosticRegistry::lookup(DiagID id) const {
  auto idx = static_cast<unsigned>(id);
  if (idx >= diagnostics.size())
    return nullptr;
  return &diagnostics[idx];
}

llvm::ArrayRef<DiagnosticInfo> DiagnosticRegistry::getAllDiagnostics() const {
  return diagnostics;
}

llvm::SmallVector<const DiagnosticInfo *>
DiagnosticRegistry::getDiagnosticsForComponent(
    DiagnosticComponent component) const {
  llvm::SmallVector<const DiagnosticInfo *> result;
  for (const auto &diag : diagnostics) {
    if (diag.component == component)
      result.push_back(&diag);
  }
  return result;
}

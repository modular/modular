//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGENToLLVM/Target/TargetLowering.h"

#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

// Default: no target-specific debug type. Defined out-of-line so the header
// only needs a forward declaration of `DebugInfo::DIType`.
DebugInfo::DIType
TargetLowering::buildDebugTypeForDType(mlir::MLIRContext *ctx,
                                       KGENDType dtype) const {
  return {};
}

static llvm::ManagedStatic<TargetLoweringRegistry> theLoweringRegistry;

TargetLoweringRegistry &TargetLoweringRegistry::get() {
  return *theLoweringRegistry;
}

void TargetLoweringRegistry::add(std::unique_ptr<TargetLowering> lowering) {
  Targets.push_back(std::move(lowering));
}

const TargetLowering *
TargetLoweringRegistry::lookup(const llvm::Triple &triple) const {
  // A lowering that resolves `triple` to a different lowering (one it owns,
  // e.g. discovered at runtime) takes precedence over a direct self-match.
  const TargetLowering *directMatch = nullptr;
  for (const std::unique_ptr<TargetLowering> &target : Targets) {
    const TargetLowering *resolved = target->resolve(triple);
    if (!resolved)
      continue;
    if (resolved != target.get())
      return resolved;
    if (!directMatch)
      directMatch = resolved;
  }
  return directMatch;
}

} // namespace M::KGEN

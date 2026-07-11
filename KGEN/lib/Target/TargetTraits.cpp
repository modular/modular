//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Target/TargetTraits.h"

#include "llvm/Support/ManagedStatic.h"
#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

static llvm::ManagedStatic<TargetTraitsRegistry> theTraitsRegistry;

TargetTraitsRegistry &TargetTraitsRegistry::get() { return *theTraitsRegistry; }

void TargetTraitsRegistry::add(std::unique_ptr<TargetTraits> traits) {
  Targets.push_back(std::move(traits));
}

const TargetTraits *
TargetTraitsRegistry::lookup(const llvm::Triple &triple) const {
  // A traits object that resolves `triple` to a different object (one it owns,
  // e.g. per-plugin) takes precedence over a direct self-match.
  const TargetTraits *directMatch = nullptr;
  for (const std::unique_ptr<TargetTraits> &traits : Targets) {
    const TargetTraits *resolved = traits->resolve(triple);
    if (!resolved)
      continue;
    if (resolved != traits.get())
      return resolved;
    if (!directMatch)
      directMatch = resolved;
  }
  return directMatch;
}

} // namespace M::KGEN

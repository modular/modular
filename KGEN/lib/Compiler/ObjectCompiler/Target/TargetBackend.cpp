//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/Target/TargetBackend.h"

#include "llvm/Support/ManagedStatic.h"
#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

static llvm::ManagedStatic<TargetBackendRegistry> TheRegistry;

TargetBackendRegistry &TargetBackendRegistry::get() { return *TheRegistry; }

void TargetBackendRegistry::add(std::unique_ptr<TargetBackend> backend) {
  Backends.push_back(std::move(backend));
}

const TargetBackend *
TargetBackendRegistry::lookup(const llvm::Triple &triple) const {
  for (const std::unique_ptr<TargetBackend> &backend : Backends)
    if (backend->matches(triple))
      return backend.get();
  return nullptr;
}

} // namespace M::KGEN

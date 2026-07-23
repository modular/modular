//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TARGET_HOST_HOSTTRAITS_H
#define KGEN_TARGET_HOST_HOSTTRAITS_H

#include "Target/TargetTraits.h"

#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

struct HostTraits final : TargetTraits {
  llvm::StringRef name() const override { return "host"; }
  bool matches(const llvm::Triple &triple) const override {
    return triple.isX86() || triple.isAArch64() || triple.isARM();
  }
  llvm::StringRef getAsmExtension() const override { return ".s"; }
  llvm::StringRef getLLVMExtension() const override { return ".ll"; }
  llvm::StringRef getObjectExtension() const override { return ".o"; }
  llvm::StringRef defaultCPU(const llvm::Triple &triple) const override;

  /// Shared stateless instance for the backend `traits()`.
  static const HostTraits &get();

protected:
  // bast target that is always registered
  bool isBaseTarget() const override { return true; }
};

} // namespace M::KGEN

#endif // KGEN_TARGET_HOST_HOSTTRAITS_H

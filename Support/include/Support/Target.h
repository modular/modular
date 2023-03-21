//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TARGET_H
#define SUPPORT_TARGET_H

#include "Support/ErrorOr.h"
#include "llvm/Support/CodeGen.h"

#include <memory>

namespace llvm {
class TargetMachine;
}

namespace M {
/// Construct a TargetMachine for the current host. This is by far the most
/// common case in our stack, so this provides a convenient utility for many
/// users.
ErrorOr<std::unique_ptr<llvm::TargetMachine>> getTargetMachineForHost(
    bool isJIT = true,
    llvm::CodeGenOpt::Level optLevel = llvm::CodeGenOpt::Aggressive);

/// Get the host CPU features string with only a list of supported features.
std::string getHostCPUFeatures();
} // namespace M

#endif // SUPPORT_TARGET_H

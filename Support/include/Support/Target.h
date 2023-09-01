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
#include <string>

namespace llvm {
// Forward declare.
class TargetMachine;
} // namespace llvm

namespace M {

/// Returns the LLVM CPU features for the given individual features.
/// The result is of the form "+feature1,+feature2".
std::string encodeFeatures(ArrayRef<std::string> features);

/// Decodes the result of encodeFeatures.
ErrorOr<std::vector<std::string>> decodeFeatures(StringRef encodedFeatures);

/// Construct a TargetMachine for the current host. This is by far the most
/// common case in our stack, so this provides a convenient utility for many
/// users.
ErrorOr<std::unique_ptr<llvm::TargetMachine>> getTargetMachineForHost(
    bool isJIT = true,
    llvm::CodeGenOpt::Level optLevel = llvm::CodeGenOpt::Aggressive);

/// Returns the LLVM CPU features for the host.
std::string getHostCPUFeatures();

} // namespace M

#endif // SUPPORT_TARGET_H

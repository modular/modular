//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains MArchTarget declarations that include modular internal
// dependencies, and will not be included in the static version of
// KGENCompilerRT that links with user mojo object files.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MARCHTARGET_MARCHTARGET_H
#define SUPPORT_MARCHTARGET_MARCHTARGET_H

#include "Support/DeviceSpecs.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "Support/MDialect/MAttrs.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {
// Forward declare.
class TargetMachine;
} // namespace llvm

namespace M {

/// Returns a TargetMachine for the current host.
ErrorOr<std::unique_ptr<llvm::TargetMachine>> getTargetMachineForHost(
    bool isJIT = true,
    llvm::CodeGenOptLevel optLevel = llvm::CodeGenOptLevel::Aggressive);

/// As for `getMArchTargetInfo`, but returned as TargetInfoAttr. The `-mtune`
/// flag is captured in the result, and derived information such as for
/// data layout and SIMD width is filled in.
///
/// TODO: Split into separate MLIR-dependent library. All other functions
/// depend only on LLVMTarget and (unfortunately) clang.
ErrorOr<TargetInfoAttr> getMArchFeatures(MLIRContext *ctx, StringRef march,
                                         StringRef mcpu, StringRef mtune);

} // namespace M

#endif // SUPPORT_MARCHTARGET_MARCHTARGET_H

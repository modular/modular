//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MARCHTARGET_MARCHTARGET_H
#define SUPPORT_MARCHTARGET_MARCHTARGET_H

#include "Support/DeviceSpecs.h"
#include "Support/MDialect/MAttrs.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {
// Forward declare.
class TargetMachine;
} // namespace llvm

namespace M {

/// Returns a TargetInfo describing the host.
ErrorOr<TargetInfo> getHostTargetInfo();

/// Returns the features for the host in "+feature1,+feature2" form.
std::string getHostCPUFeatures();

/// Returns a TargetMachine for the current host.
ErrorOr<std::unique_ptr<llvm::TargetMachine>> getTargetMachineForHost(
    bool isJIT = true,
    llvm::CodeGenOptLevel optLevel = llvm::CodeGenOptLevel::Aggressive);

/// Returns a TargetInfo describing the consequences of the given `-march`,
/// `-mcpu` and `-mtune` settings. These flags have target-dependent behaviour
/// as described in https://gcc.gnu.org/onlinedocs/gcc/. Note that the `-mtune`
/// flag is not captured in the result.
///
/// This method will construct a minimum target triple and feature set using the
/// provided architecture and CPU. Both are optional.
///
/// `-march=native` will use all the features of the host system.
///
/// For X86 architectures, `-march` or `-mcpu` can be used to specify a CPU
/// subtype, like `skylake-avx512`. If `-mcpu=generic`, then `-march` is assumed
/// to be an X86 architecture kind and a generic CPU for that is used.
///
/// For ARM architectures, `-march` specifies the base architecture or `-mcpu`
/// specifies the specific CPU kind. If only an architecture is specified, the
/// default CPU for it is used.
///
/// For AArch64 architectures, `-march` specifies the base architecture or
/// `-mcpu` specifies the specific CPU kind. If only an architecture is
/// specified, `-mcpu=generic` will be used.
///
/// `-mtune` will specify the CPU to specifically tune code for.
ErrorOr<TargetInfo> getMArchTargetInfo(StringRef march, StringRef mcpu,
                                       StringRef mtune);

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

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains MArchTarget declarations that are meant to be accessible
// by the static KGENCompilerRT library that links with user object files.
// Its dependencies are intentionally kept minimal to reduce the size of the
// user binary. Any code that adds additional dependencies should be in
// `MArchTarget.h` instead.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MARCHTARGET_MARCHTARGETMINIMAL_H
#define SUPPORT_MARCHTARGET_MARCHTARGETMINIMAL_H

#include "Support/DeviceSpecs.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {
// Forward declare.
class TargetMachine;
} // namespace llvm

namespace clang {
class TargetOptions;
}

namespace M {

/// Returns the feature set according to clang for the given options, which
/// should include the Triple and CPU.
ErrorOr<std::vector<std::string>>
getFeaturesFromClang(std::shared_ptr<clang::TargetOptions> opts, StringRef cpu);

/// Returns a TargetInfo describing the host.
ErrorOr<TargetInfo> getHostTargetInfo();

/// Returns the features for the host in "+feature1,+feature2" form.
std::string getHostCPUFeatures();

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

} // namespace M

#endif // SUPPORT_MARCHTARGET_MARCHTARGETMINIMAL_H

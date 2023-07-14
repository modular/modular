//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MARCHTARGET_H
#define SUPPORT_MARCHTARGET_H

#include "Support/MDialect/MAttrs.h"

namespace M {
/// Construct a TargetInfoAttr given `-march` and `-mcpu`. These flags have
/// target-dependent behaviour as described in
/// https://gcc.gnu.org/onlinedocs/gcc/.
///
/// This method will construct a minium target triple and feature set using the
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
ErrorOr<TargetInfoAttr> getMArchFeatures(MLIRContext *ctx, StringRef march,
                                         StringRef mcpu);
} // namespace M

#endif // SUPPORT_MARCHTARGET_H

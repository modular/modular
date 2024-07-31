//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_CUSTOMDIALECT_CUSTOMUTILS_H
#define KGEN_CUSTOMDIALECT_CUSTOMUTILS_H

#include "llvm/ADT/StringRef.h"

namespace M::KGEN::Custom {
constexpr llvm::StringLiteral kCustomOpParamsAttrName = "_op_impl_params";
constexpr llvm::StringLiteral kCustomOpImplModuleAttr = "custom.op_impls";
} // namespace M::KGEN::Custom

#endif // KGEN_CUSTOMDIALECT_CUSTOMUTILS_H

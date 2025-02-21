//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOGGPREELAB_HELPERS_H
#define KGEN_LIB_MOGGPREELAB_HELPERS_H

#include "KGEN/LITDialect/LITOps.h"

namespace M::KGEN::MOGGPreElab {

inline bool isXType(LIT::StructType maybeTensor, StringLiteral root,
                    StringLiteral className) {
  if (maybeTensor.getSymbol().getRootReference() != root)
    return false;
  return maybeTensor.getSymbol().getLeafReference() == className;
}

inline bool isExtensibilityTensor(LIT::StructType maybeTensor) {
  return isXType(maybeTensor, "extensibility", "Tensor");
}

inline bool isDPSTensor(LIT::StructType maybeTensor) {
  return maybeTensor.getSymbol().getRootReference().strref().starts_with(
             "tensor") &&
         maybeTensor.getSymbol().getLeafReference() == "ManagedTensorSlice";
}

inline bool isMojoDeviceContextPtr(LIT::StructType maybeCallContextPtr) {
  return isXType(maybeCallContextPtr, "runtime", "DeviceContextPtr");
}

inline bool fnNeedsConformances(LIT::FnOp fnOp) {
  return fnOp.getSourceName() == "execute" ||
         fnOp.getSourceName() == "pytorch_fallback";
}

inline bool isCustomType(LIT::StructType maybeCustom) {
  return !isExtensibilityTensor(maybeCustom);
}

/// Remove the decorators from the function. Return true if any function had the
/// kernel decorators.
bool stripDecorators(LIT::FnOp func);

void stripDecorators(LIT::StructDeclOp structDecl);

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_HELPERS_H

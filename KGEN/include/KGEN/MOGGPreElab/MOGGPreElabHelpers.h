//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOGGPREELAB_HELPERS_H
#define KGEN_LIB_MOGGPREELAB_HELPERS_H

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"

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

/// Helper functions for dealing with the io_spec field of ManagedTensorSlice
///
/// Convert IOSpec enum to and from strings
StringRef toString(IOSpec spec);
std::optional<IOSpec> toIOSpec(StringRef str);
/// Get the param values corresponding to mut/input of IOSpec
std::pair<TypedAttr, TypedAttr> getParams(KGEN::MOGGPreElab::IOSpec ioSpec,
                                          Builder &builder);

bool isOutputIOSpec(IOSpec spec);
bool isFusableIOSpec(IOSpec spec);
bool isMutableIOSpec(IOSpec spec);

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_HELPERS_H

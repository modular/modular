//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ParameterAttr
//===----------------------------------------------------------------------===//

bool ParameterAttr::isSimpleConstant(Attribute attr) {
  // Check for simple builtin-in constants.
  if (attr.isa<FloatAttr, IntegerAttr, StringAttr>())
    return true;

  // Check for an interface.
  if (auto itf = llvm::dyn_cast<ParameterAttr>(attr))
    return itf.isSimpleConstant();

  // If the attribute has sub-elements, walk them and check if each one is a
  // simple constant.
  if (auto itf = llvm::dyn_cast<mlir::SubElementAttrInterface>(attr)) {
    bool allSimple = true;
    auto checkAttr = [&](Attribute attr) {
      if (!allSimple)
        return;
      allSimple &= ParameterAttr::isSimpleConstant(attr);
    };
    itf.walkImmediateSubElements(checkAttr, [](Type) {});
    return allSimple;
  }

  // Otherwise, assume the attribute is not a simple constant.
  return false;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrInterfaces.cpp.inc"

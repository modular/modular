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
    return itf.isConstant();

  // Otherwise, assume the attribute is not a simple constant.
  return false;
}

bool ParameterAttr::compare(Attribute lhs, Attribute rhs) {
  // Simplify the code below - we never have to care about exactly equal values.
  if (lhs == rhs)
    return false;

  // All non-constant expressions are "less than" a constant, since they appear
  // on the right. We handle all simple constants consistently here: they can
  // never occur in the same expression since they have different types.
  if (isSimpleConstant(rhs)) {
    if (!isSimpleConstant(lhs))
      return true;

    // Check built-in attributes.
    if (auto intRhs = llvm::dyn_cast<IntegerAttr>(rhs)) {
      auto intLhs = llvm::dyn_cast<IntegerAttr>(lhs);
      return !intLhs || intLhs.getValue().slt(intRhs.getValue());
    }
    if (auto strRhs = llvm::dyn_cast<StringAttr>(rhs)) {
      auto strLhs = llvm::dyn_cast<StringAttr>(lhs);
      return !strLhs || strLhs.getValue() < strRhs.getValue();
    }
    if (auto fltRhs = llvm::dyn_cast<FloatAttr>(rhs)) {
      auto fltLhs = llvm::dyn_cast<FloatAttr>(lhs);
      return !fltLhs || fltLhs.getValue() < fltRhs.getValue();
    }

    // Otherwise, we must have an interface. Any attribute that doesn't
    // implement one wouldn't be considered a simple constant.
    return !llvm::cast<ParameterAttr>(rhs).isLessThan(lhs);
  }
  if (isSimpleConstant(lhs))
    return false;

  // Check for an interface.
  if (auto itf = llvm::dyn_cast<ParameterAttr>(lhs))
    return itf.isLessThan(rhs);
  if (auto itf = llvm::dyn_cast<ParameterAttr>(rhs))
    return !itf.isLessThan(lhs);

  // Otherwise, we don't know how to compare these attributes. Move them all the
  // way to the left.
  return true;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrInterfaces.cpp.inc"

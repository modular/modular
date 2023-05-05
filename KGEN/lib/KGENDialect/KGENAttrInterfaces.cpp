//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/MDialect/MAttrs.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ParameterAttr
//===----------------------------------------------------------------------===//

namespace {
struct IntegerParameterAttr
    : public ParameterAttr::ExternalModel<IntegerParameterAttr, IntegerAttr> {
  bool isConstant(Attribute attr) const { return true; }
  std::optional<bool> isLessThan(Attribute attr, Attribute rhs) const {
    if (auto intAttr = dyn_cast<IntegerAttr>(rhs))
      return cast<IntegerAttr>(attr).getValue().slt(intAttr.getValue());
    return std::nullopt;
  }
};

struct FloatParameterAttr
    : public ParameterAttr::ExternalModel<FloatParameterAttr, FloatAttr> {
  bool isConstant(Attribute attr) const { return true; }
  std::optional<bool> isLessThan(Attribute attr, Attribute rhs) const {
    if (auto fpAttr = dyn_cast<FloatAttr>(rhs))
      return cast<FloatAttr>(attr).getValue() < fpAttr.getValue();
    return std::nullopt;
  }
};

struct StringParameterAttr
    : public ParameterAttr::ExternalModel<StringParameterAttr, StringAttr> {
  bool isConstant(Attribute attr) const { return true; }
  std::optional<bool> isLessThan(Attribute attr, Attribute rhs) const {
    if (auto strAttr = dyn_cast<StringAttr>(rhs))
      return cast<StringAttr>(attr).getValue() < strAttr.getValue();
    return std::nullopt;
  }
};

struct TypeParameterAttr
    : public ParameterAttr::ExternalModel<TypeParameterAttr, TypeAttr> {
  bool isConstant(Attribute attr) const {
    return !isParameterizedType(cast<TypeAttr>(attr).getValue());
  }
};

struct PointerParameterAttr
    : public ParameterAttr::ExternalModel<PointerParameterAttr, PointerAttr> {
  bool isConstant(Attribute attr) const {
    return !isParameterizedType(cast<PointerAttr>(attr).getType());
  }
};
} // namespace

void KGENDialect::injectAttrInterfaces() {
  IntegerAttr::attachInterface<IntegerParameterAttr>(*getContext());
  FloatAttr::attachInterface<FloatParameterAttr>(*getContext());
  StringAttr::attachInterface<StringParameterAttr>(*getContext());
  TypeAttr::attachInterface<TypeParameterAttr>(*getContext());
  PointerAttr::attachInterface<PointerParameterAttr>(*getContext());
}

bool ParameterAttr::isSimpleConstant(Attribute attr) {
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

    // Otherwise, we must have an interface. Any attribute that doesn't
    // implement one wouldn't be considered a simple constant.
    if (auto result = llvm::cast<ParameterAttr>(lhs).isLessThan(rhs))
      return *result;
    if (auto result = llvm::cast<ParameterAttr>(rhs).isLessThan(lhs))
      return !*result;
    return false;
  }
  if (isSimpleConstant(lhs))
    return false;

  // Check for an interface.
  if (auto itf = llvm::dyn_cast<ParameterAttr>(lhs)) {
    if (auto result = itf.isLessThan(rhs))
      return *result;
  }
  if (auto itf = llvm::dyn_cast<ParameterAttr>(rhs)) {
    if (auto result = itf.isLessThan(lhs))
      return !(*result);
  }

  // Otherwise, we don't know how to compare these attributes.
  return false;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrInterfaces.cpp.inc"

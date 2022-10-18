//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// POPDialect
//===----------------------------------------------------------------------===//

void POPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/POPDialect/POPTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                                TypedAttr size, TypedAttr elementType) {
  if (!size.getType().isa<IndexType>())
    return emitError() << "expected size expression to be index type";
  if (!elementType.getType().isa<MLIRTypeType>())
    return emitError() << "expected size expression to be !kgen.mlirtype";
  return success();
}

void ArrayType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrs,
    function_ref<void(Type)> walkTypes) const {
  walkAttrs(getSize());
  walkAttrs(getElementType());
}

Type ArrayType::replaceImmediateSubElements(ArrayRef<Attribute> attrs,
                                            ArrayRef<Type> types) const {
  assert(types.empty() && attrs.size() == 2 && "expected 2 sub-attributes");
  return get(getContext(), attrs[0].cast<TypedAttr>(),
             attrs[1].cast<TypedAttr>());
}

Optional<int64_t> ArrayType::getResolvedSize() const {
  if (auto intAttr = getSize().dyn_cast<IntegerAttr>())
    return intAttr.getInt();
  return {};
}

Type ArrayType::getResolvedElementType() const {
  if (auto typeCst = dyn_cast_if_present<TypeConstantAttr>(getElementType()))
    return typeCst.getValue();
  return nullptr;
}

ArrayType ArrayType::get(TypedAttr size, TypedAttr elementType) {
  return get(size.getContext(), size, elementType);
}

ArrayType ArrayType::get(TypedAttr size, Type elementType) {
  return get(size.getContext(), size, TypeConstantAttr::get(elementType));
}

ArrayType ArrayType::get(int64_t size, Type elementType) {
  return get(Builder(elementType.getContext()).getIndexAttr(size), elementType);
}

ArrayType ArrayType::get(ValueRange elements) {
  assert(!elements.empty() && "expected non-empty elements");
  auto firstElement = elements.front();
  assert(llvm::all_of(elements,
                      [firstType = firstElement.getType()](Value v) {
                        return v.getType() == firstType;
                      }) &&
         "expected same element types");
  return get(elements.size(), firstElement.getType());
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

LogicalResult PointerType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  TypedAttr type) {
  if (type && !type.getType().isa<MLIRTypeType>())
    return emitError() << "type parameter for pointer must be a !kgen.mlirtype";
  return success();
}

void PointerType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getElementType());
}

Type PointerType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                              ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 1 && replTypes.empty());
  return PointerType::get(replAttrs[0]);
}

Type PointerType::getResolvedElementType() const {
  if (auto typeCst = getElementType().dyn_cast<TypeConstantAttr>())
    return typeCst.getValue();
  return nullptr;
}

PointerType PointerType::get(TypedAttr elementType) {
  return PointerType::get(elementType.getContext(), elementType);
}

PointerType PointerType::get(Type elementType) {
  return PointerType::get(TypeConstantAttr::get(elementType));
}

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

LogicalResult ScalarType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 TypedAttr dtype) {
  if (!dtype.getType().isa<DTypeType>())
    return emitError() << "parameter for scalar type must be a !kgen.dtype";
  return success();
}

void ScalarType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getDType());
}

Type ScalarType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                             ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 1 && replTypes.empty());
  return ScalarType::get(replAttrs[0]);
}

ScalarType ScalarType::get(TypedAttr dtype) {
  return get(dtype.getContext(), dtype);
}

ScalarType ScalarType::get(MLIRContext *ctx, KGENDType dtype) {
  return get(ctx, DTypeConstantAttr::get(ctx, dtype));
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

LogicalResult SIMDType::verify(function_ref<InFlightDiagnostic()> emitError,
                               TypedAttr size, TypedAttr dtype) {
  if (!size || !dtype)
    return emitError() << "simd type requires size and dtype";
  if (!size.getType().isIndex())
    return emitError() << "size parameter for simd must have type `index`";
  if (!dtype.getType().isa<DTypeType>())
    return emitError() << "type parameter for simd must be a !kgen.dtype";
  return success();
}

void SIMDType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getSize());
  walkAttrsFn(getDType());
}

Type SIMDType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                           ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 2 && replTypes.empty());
  return SIMDType::get(replAttrs[0], replAttrs[1]);
}

Optional<int64_t> SIMDType::getResolvedSize() const {
  if (auto intAttr = getSize().dyn_cast<IntegerAttr>())
    return intAttr.getInt();
  return {};
}

SIMDType SIMDType::get(TypedAttr size, TypedAttr dtype) {
  return get(size.getContext(), size, dtype);
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<TypedAttr> elementTypes) {
  for (auto &elementType : llvm::enumerate(elementTypes)) {
    if (!elementType.value().getType().isa<MLIRTypeType>())
      return emitError() << "struct element type at index "
                         << elementType.index() << " is not a !kgen.mlirtype";
  }
  return success();
}

void StructType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrs,
    function_ref<void(Type)> walkTypes) const {
  for (TypedAttr elementType : getElementTypes())
    walkAttrs(elementType);
}

Type StructType::replaceImmediateSubElements(ArrayRef<Attribute> attrs,
                                             ArrayRef<Type> types) const {
  assert(types.empty() && attrs.size() == getNumElements() &&
         "expected same number of sub-attributes as element types");
  SmallVector<TypedAttr> elementTypes;
  elementTypes.reserve(attrs.size());
  for (Attribute attr : attrs)
    elementTypes.push_back(attr.cast<TypedAttr>());
  return get(getContext(), elementTypes);
}

LogicalResult
StructType::resolveElementTypes(SmallVectorImpl<Type> &elementTypes) const {
  for (TypedAttr elementType : getElementTypes()) {
    if (auto type = elementType.dyn_cast<TypeConstantAttr>())
      elementTypes.push_back(type.getValue());
    else
      return failure();
  }
  return success();
}

SmallVector<Type> StructType::getParameterizedElementTypes() const {
  SmallVector<Type> elementTypes;
  elementTypes.reserve(getNumElements());
  for (TypedAttr elementType : getElementTypes())
    elementTypes.push_back(ParamRefType::get(elementType));
  return elementTypes;
}

StructType StructType::get(ArrayRef<Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least one element type");
  SmallVector<TypedAttr> elementTypeExprs;
  elementTypeExprs.reserve(elementTypes.size());
  for (Type elementType : elementTypes)
    elementTypeExprs.push_back(TypeConstantAttr::get(elementType));
  return get(elementTypes.front().getContext(), elementTypeExprs);
}

//===----------------------------------------------------------------------===//
// VariantType
//===----------------------------------------------------------------------===//

/// Canonicalize the possible types of a variant. Deduplicate the types.
static SmallVector<TypedAttr>
canonicalizeVariantTypes(ArrayRef<TypedAttr> types) {
  SmallVector<TypedAttr> deduplicatedTypes;
  SmallPtrSet<Attribute, 4> seenTypes;
  deduplicatedTypes.reserve(types.size());
  for (TypedAttr type : types)
    if (seenTypes.insert(type).second)
      deduplicatedTypes.push_back(type);
  return deduplicatedTypes;
}

VariantType VariantType::get(MLIRContext *ctx, ArrayRef<TypedAttr> types) {
  return Base::get(ctx, canonicalizeVariantTypes(types));
}

void VariantType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrs,
    function_ref<void(Type)> walkTypes) const {
  for (TypedAttr type : getTypes())
    walkAttrs(type);
}

Type VariantType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                              ArrayRef<Type> replTypes) const {
  assert(replTypes.empty() && replAttrs.size() == getTypes().size() &&
         "expected same number of sub-attributes as variant types");
  SmallVector<TypedAttr> variantTypes;
  variantTypes.reserve(replAttrs.size());
  for (Attribute attr : replAttrs)
    variantTypes.push_back(attr.cast<TypedAttr>());
  return get(getContext(), variantTypes);
}

Optional<int64_t> VariantType::getTypeIndex(Type type) const {
  for (auto &variantType : llvm::enumerate(getTypes()))
    if (ParamRefType::get(variantType.value()) == type)
      return variantType.index();
  return {};
}

SmallVector<Type> VariantType::getParameterizedElementTypes() const {
  SmallVector<Type> types;
  types.reserve(getTypes().size());
  for (TypedAttr type : getTypes())
    types.push_back(ParamRefType::get(type));
  return types;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/POPDialect/POPTypes.cpp.inc"

//===- POPTypes.cpp -------------------------------------------------------===//
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

Optional<int64_t> ArrayType::resolveSize() const {
  if (auto intAttr = getSize().dyn_cast<IntegerAttr>())
    return intAttr.getInt();
  return {};
}

Type ArrayType::resolveElementType() const {
  if (auto typeCst = getElementType().dyn_cast_or_null<TypeConstantAttr>())
    return typeCst.getValue();
  return nullptr;
}

ArrayType ArrayType::get(TypedAttr size, TypedAttr elementType) {
  return get(size.getContext(), size, elementType);
}

ArrayType ArrayType::get(TypedAttr size, Type elementType) {
  return get(size.getContext(), size, TypeConstantAttr::get(elementType));
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
  assert(types.empty() && attrs.size() == getElementTypes().size() &&
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

/// Parse a comma-separated list of type parameter values.
static ParseResult
parseArrayOfTypeExprs(AsmParser &p, FailureOr<SmallVector<TypedAttr>> &values) {
  values.emplace();
  return p.parseCommaSeparatedList([&]() -> ParseResult {
    FailureOr<TypedAttr> value;
    if (failed(parseTypeParamValue(p, value)))
      return failure();
    values->push_back(*value);
    return success();
  });
}

/// Print a comma-separated list of type parameter values.
static void printArrayOfTypeExprs(AsmPrinter &p, ArrayRef<TypedAttr> values) {
  llvm::interleaveComma(
      values, p, [&](TypedAttr value) { printTypeParamValue(p, value); });
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/POPDialect/POPTypes.cpp.inc"

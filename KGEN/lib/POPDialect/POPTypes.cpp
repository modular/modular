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

//===----------------------------------------------------------------------===//
// POPDialect
//===----------------------------------------------------------------------===//

void POP::POPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/POPDialect/POPTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

LogicalResult
POP::ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                       TypedAttr size, TypedAttr elementType) {
  if (!size.getType().isa<IndexType>())
    return emitError() << "expected size expression to be index type";
  if (!elementType.getType().isa<MLIRTypeType>())
    return emitError() << "expected size expression to be !kgen.mlirtype";
  return success();
}

void POP::ArrayType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrs,
    function_ref<void(Type)> walkTypes) const {
  walkAttrs(getSize());
  walkAttrs(getElementType());
}

Type POP::ArrayType::replaceImmediateSubElements(ArrayRef<Attribute> attrs,
                                                 ArrayRef<Type> types) const {
  assert(types.empty() && attrs.size() == 2 && "expected 2 sub-attributes");
  return get(getContext(), attrs[0].cast<TypedAttr>(),
             attrs[1].cast<TypedAttr>());
}

Optional<int64_t> POP::ArrayType::resolveSize() const {
  if (auto intAttr = getSize().dyn_cast<IntegerAttr>())
    return intAttr.getInt();
  return {};
}

Type POP::ArrayType::resolveElementType() const {
  if (auto typeCst = getElementType().dyn_cast_or_null<TypeConstantAttr>())
    return typeCst.getValue();
  return nullptr;
}

POP::ArrayType POP::ArrayType::get(TypedAttr size, TypedAttr elementType) {
  return get(size.getContext(), size, elementType);
}

POP::ArrayType POP::ArrayType::get(TypedAttr size, Type elementType) {
  return get(size.getContext(), size, TypeConstantAttr::get(elementType));
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/POPDialect/POPTypes.cpp.inc"

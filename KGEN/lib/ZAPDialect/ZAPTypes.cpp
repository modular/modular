//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ZAPDialect/ZAPTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace ZAP;

//===----------------------------------------------------------------------===//
// ZAPDialect
//===----------------------------------------------------------------------===//

void ZAPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/ZAPDialect/ZAPTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

LogicalResult
BufferType::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                   TypedAttr size, TypedAttr dtype) {
  if (size && !size.getType().isIndex())
    return emitError() << "size parameter for buffer must have type `index`";
  if (dtype && !dtype.getType().isa<DTypeType>())
    return emitError() << "type parameter for buffer must be a !kgen.dtype";
  return success();
}

void BufferType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getSize());
  walkAttrsFn(getDType());
}

Type BufferType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                             ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 2 && replTypes.empty());
  return BufferType::get(getContext(), replAttrs[0], replAttrs[1]);
}

Optional<int64_t> BufferType::getResolvedSize() const {
  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(getSize()))
    return intAttr.getInt();
  return {};
}

Optional<DType> BufferType::getResolvedDType() const {
  if (auto dtype = dyn_cast_if_present<DTypeConstantAttr>(getDType()))
    return dtype.getDType();
  return {};
}

POP::PointerType BufferType::getPointerType() const {
  Type elementType;
  if (TypedAttr dtype = getDType())
    elementType = POP::ScalarType::get(dtype);
  else
    elementType = POP::ScalarType::get(getContext(), DType::invalid);

  return POP::PointerType::get(elementType);
}

POP::ScalarType BufferType::getElementType() const {
  assert(getDType() && "expected buffer with known dtype");
  return POP::ScalarType::get(getDType());
}

BufferType BufferType::get(TypedAttr size, TypedAttr dtype) {
  return get(size.getContext(), size, dtype);
}

BufferType BufferType::get(MLIRContext *ctx, int64_t size, DType dtype) {
  return get(OpBuilder(ctx).getIndexAttr(size),
             DTypeConstantAttr::get(ctx, dtype));
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/ZAPDialect/ZAPTypes.cpp.inc"

//===- MetaTypes.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MetaDialect/MetaTypes.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Casting Between Meta and Builtin
//===----------------------------------------------------------------------===//

LogicalResult M::KGEN::checkMetaCastedTypes(
    function_ref<InFlightDiagnostic(StringRef)> emitError, Type metaTy,
    Type standardTy,
    function_ref<LogicalResult(DTypeConstantAttr)> checkDType) {
  if (auto scalarTy = metaTy.dyn_cast<ScalarType>()) {
    // Check that the data types match.
    if (auto dtype = scalarTy.getDType().dyn_cast<DTypeConstantAttr>();
        dtype && failed(checkDType(dtype)))
      return emitError("incompatible scalar data type");
    return success();
  }

  // Check that the standard type is a rank 1 vector with matching dimensions.
  auto simdTy = metaTy.cast<SIMDType>();
  auto vectorTy = standardTy.dyn_cast<VectorType>();
  if (!vectorTy)
    return emitError("expected a vector type");
  if (vectorTy.getNumScalableDims() != 0)
    return emitError("vector type should not be scalable");
  if (vectorTy.getRank() != 1)
    return emitError("expected a rank 1 vector");
  if (auto size = simdTy.getSize().dyn_cast<IntegerAttr>();
      size.getInt() != vectorTy.getShape().front())
    return emitError("dimensions do not match");
  if (auto dtype = simdTy.getDType().dyn_cast<DTypeConstantAttr>();
      dtype && failed(checkDType(dtype)))
    return emitError("element types do not match");
  return success();
}

LogicalResult M::KGEN::checkMetaCastedTypes(
    function_ref<InFlightDiagnostic(StringRef)> emitError, Type metaTy,
    Type standardTy) {
  return checkMetaCastedTypes(
      emitError, metaTy, standardTy, [standardTy](DTypeConstantAttr dtype) {
        return success(dtype.isCompatibleWith(standardTy));
      });
}

//===----------------------------------------------------------------------===//
// custom<ParamDTypeValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamDTypeValue(AsmParser &p,
                                        FailureOr<TypedAttr> &result) {
  TypedAttr retValue;
  if (failed(parseParamValue(p, retValue, p.getBuilder().getType<DTypeType>())))
    return failure();
  result = retValue;
  return success();
}

static void printParamDTypeValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, value);
}

//===----------------------------------------------------------------------===//
// custom<OptionalParamDTypeValue>
//===----------------------------------------------------------------------===//

static ParseResult parseOptionalParamDTypeValue(AsmParser &p,
                                                FailureOr<TypedAttr> &result) {
  if (succeeded(p.parseOptionalQuestion())) {
    result = TypedAttr();
    return success();
  }
  return parseParamDTypeValue(p, result);
}

static void printOptionalParamDTypeValue(AsmPrinter &p, Attribute value) {
  if (!value) {
    p << '?';
    return;
  }
  printParamDTypeValue(p, value);
}

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

LogicalResult
ScalarType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
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

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

LogicalResult
SIMDType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
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

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

LogicalResult
BufferType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
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

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

LogicalResult
PointerType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    TypedAttr dtype) {
  if (dtype && !dtype.getType().isa<DTypeType>())
    return emitError() << "type parameter for pointer must be a !kgen.dtype";
  return success();
}

void PointerType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getDType());
}

Type PointerType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                              ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 1 && replTypes.empty());
  return PointerType::get(replAttrs[0]);
}

//===----------------------------------------------------------------------===//
// Dialect Type Parsing and Printing
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/MetaDialect/MetaTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// MetaDialect type support
//===----------------------------------------------------------------------===//

void MetaDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/MetaDialect/MetaTypes.cpp.inc"
      >();
}

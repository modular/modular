//===- MetaDialect.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Meta dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MetaDialect/MetaDialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/MetaDialect/MetaOps.h"
#include "KGEN/MetaDialect/MetaTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// custom<ParamDTypeValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamDTypeValue(AsmParser &p,
                                        FailureOr<Attribute> &result) {
  Attribute retValue;
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
                                                FailureOr<Attribute> &result) {
  if (succeeded(p.parseOptionalQuestion())) {
    result = Attribute();
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
                   Attribute dtype) {
  if (!dtype.getType().isa<DTypeType>())
    return emitError() << "parameter for scalar type must be a !kgen.dtype";
  return success();
}

void ScalarType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getDtype());
}

mlir::SubElementTypeInterface ScalarType::replaceImmediateSubAttribute(
    ArrayRef<std::pair<size_t, Attribute>> replacements) const {
  if (replacements.empty())
    return *this;
  assert(replacements.size() == 1 && replacements[0].first == 0 &&
         "only have one sub-attribute");
  return ScalarType::get(replacements[0].second);
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

LogicalResult
SIMDType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                 Attribute size, Attribute dtype) {
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
  walkAttrsFn(getDtype());
}

mlir::SubElementTypeInterface SIMDType::replaceImmediateSubAttribute(
    ArrayRef<std::pair<size_t, Attribute>> replacements) const {
  Attribute attrs[2] = {getSize(), getDtype()};

  for (auto entry : replacements) {
    assert(entry.first < 2);
    attrs[entry.first] = entry.second;
  }
  return SIMDType::get(attrs[0], attrs[1]);
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

LogicalResult
BufferType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   Attribute size, Attribute dtype) {
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
  walkAttrsFn(getDtype());
}

mlir::SubElementTypeInterface BufferType::replaceImmediateSubAttribute(
    ArrayRef<std::pair<size_t, Attribute>> replacements) const {
  Attribute attrs[2] = {getSize(), getDtype()};

  for (auto entry : replacements) {
    assert(entry.first < 2);
    attrs[entry.first] = entry.second;
  }
  return BufferType::get(attrs[0], attrs[1]);
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

LogicalResult
PointerType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    Attribute dtype) {
  if (dtype && !dtype.getType().isa<DTypeType>())
    return emitError() << "type parameter for pointer must be a !kgen.dtype";
  return success();
}

void PointerType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getDtype());
}

mlir::SubElementTypeInterface PointerType::replaceImmediateSubAttribute(
    ArrayRef<std::pair<size_t, Attribute>> replacements) const {
  if (replacements.empty())
    return *this;
  assert(replacements.size() == 1 && replacements[0].first == 0 &&
         "only have one sub-attribute");
  return PointerType::get(replacements[0].second);
}

//===----------------------------------------------------------------------===//
// Dialect Type Parsing and Printing
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/MetaDialect/MetaTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/MetaDialect/MetaDialect.cpp.inc"

void MetaDialect::initialize() {
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/MetaDialect/MetaTypes.cpp.inc"
      >();
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "KGEN/MetaDialect/Meta.cpp.inc"
      >();
}

/// Registered hook to materialize a constant operation from a "meta" dialect
/// op that is folded.
Operation *MetaDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  // Integer constants can materialize into something specific.  We need this
  // for ops that fold in the context of kgen.kernel.
  // TODO: What should our primitive arithmetic ops be, arith?  It doesn't
  // support signful math well.
  // if (auto intType = type.dyn_cast<IntegerType>())
  //   if (auto attrValue = value.dyn_cast<IntegerAttr>())
  //     return builder.create<ConstantOp>(loc, type, attrValue);

  if (isValidParameterExpr(value))
    return builder.create<ParamValueOp>(loc, type, value);
  return nullptr;
}

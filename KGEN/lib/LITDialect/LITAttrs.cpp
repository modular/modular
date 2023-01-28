//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// LITDialect
//===----------------------------------------------------------------------===//

void LITDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "KGEN/LITDialect/LITAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ArgumentDefaultAttr
//===----------------------------------------------------------------------===//

/// Reject default arguments with negative indices.
LogicalResult
DefaultArgumentAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            IntegerAttr index, TypedAttr value) {
  if (index.getValue().isNegative())
    return emitError() << "index cannot be negative";

  return success();
}

/// Reject default argument arrays that include multiple defaults for the same
/// argument index.
LogicalResult
DefaultArgumentArrayAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<DefaultArgumentAttr> attrs) {
  llvm::SmallDenseSet<int64_t> indices;
  for (const DefaultArgumentAttr &attr : attrs) {
    int64_t index = attr.getIndex().getInt();
    if (!indices.insert(index).second)
      return emitError() << "cannot specify more than one default argument for "
                            "the same index "
                         << index;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StructAttr
//===----------------------------------------------------------------------===//

static ParseResult parseStructElements(
    AsmParser &p,
    FailureOr<SmallVector<std::pair<StringAttr, TypedAttr>>> &values) {
  values.emplace();
  StringAttr name;
  Type type;
  TypedAttr value;
  auto parseElt = [&]() -> ParseResult {
    if (parseParamName(p, name) || parseColonTypeOrIndex(p, type) ||
        p.parseEqual() || parseParamValue(p, value, type))
      return failure();
    values->emplace_back(name, value);
    return success();
  };
  return p.parseCommaSeparatedList(AsmParser::Delimiter::Braces, parseElt);
}

static void
printStructElements(AsmPrinter &p,
                    ArrayRef<std::pair<StringAttr, TypedAttr>> values) {
  p << '{';
  llvm::interleaveComma(values, p,
                        [&](const std::pair<StringAttr, TypedAttr> &value) {
                          printParamName(p, value.first);
                          printColonTypeOrIndex(p, value.second.getType());
                          p << " = ";
                          printParamValue(p, value.second);
                        });
  p << '}';
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

TypedAttr StructExtractAttr::get(TypedAttr structValue, StructFieldOp fieldOp) {
  auto structType = ::cast<DeclRefType>(structValue.getType());
  ParameterEvaluator evaluator(structType.getParamValues());
  auto resultType = evaluator.getReboundType(fieldOp.getType());
  return get(structValue, fieldOp.getNameAttr(), resultType);
}

TypedAttr StructExtractAttr::get(TypedAttr structValue, StringAttr field,
                                 Type resultType) {
  return get(structValue.getContext(), structValue, field, resultType);
}

TypedAttr StructExtractAttr::get(MLIRContext *context, TypedAttr structValue,
                                 StringAttr field, Type resultType) {
  if (auto value = dyn_cast_if_present<StructAttr>(structValue)) {
    auto it = llvm::find_if(value.getValues(),
                            [&](const auto &p) { return p.first == field; });
    if (it != value.getValues().end())
      return it->second;
  }

  return Base::get(context, structValue, field, resultType);
}

// FIXME(Issue #7779): this shouldn't be needed.
// https://github.com/modularml/modular/issues/7779
Attribute
StructExtractAttr::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 2 && replTypes.size() == 1);
  auto structAttr = ::dyn_cast<TypedAttr>(replAttrs[0]);
  auto fieldAttr = ::dyn_cast<StringAttr>(replAttrs[1]);
  if (!structAttr || !fieldAttr)
    return {};
  if (structAttr == getStructValue() && fieldAttr == getField())
    return *this;
  assert(::isa<DeclRefType>(structAttr.getType()));
  return StructExtractAttr::get(getContext(), structAttr, fieldAttr,
                                replTypes[0]);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/LITDialect/LITAttrs.cpp.inc"

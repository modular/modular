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
// StructAttr
//===----------------------------------------------------------------------===//

static ParseResult
parseStructElements(AsmParser &p,
                    SmallVector<std::pair<StringAttr, TypedAttr>> &values) {
  StringAttr name;
  Type type;
  TypedAttr value;
  auto parseElt = [&]() -> ParseResult {
    if (parseParamName(p, name) || parseColonTypeOrIndex(p, type) ||
        p.parseEqual() || parseParamValue(p, value, type))
      return failure();
    values.emplace_back(name, value);
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

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/LITDialect/LITAttrs.cpp.inc"

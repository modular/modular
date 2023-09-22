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
// FnMetadataAttr
//===----------------------------------------------------------------------===//

FnMetadataAttr FnMetadataAttr::get(MLIRContext *ctx, unsigned numInputs) {
  auto emptyStr = StringAttr::get(ctx);
  SmallVector<StringAttr> names(numInputs, emptyStr);
  return get(ctx, names, {}, {});
}

LogicalResult
FnMetadataAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<StringAttr> argNames,
                       ArrayRef<TypedAttr> defaultArguments,
                       ArrayRef<TypedAttr> defaultParameters) {
  for (StringAttr name : argNames)
    if (!name)
      return emitError() << "arg name cannot be null";
  return success();
}

FnMetadataAttrInterface
FnMetadataAttr::getWithBoundArgs(size_t numBound) const {
  size_t numArgs = getArgNames().size() - numBound;

  ArrayRef<StringAttr> newArgNames = getArgNames().drop_front(numBound);
  ArrayRef<TypedAttr> newDefaultArgs = getDefaultArguments();
  if (numArgs < newDefaultArgs.size())
    newDefaultArgs = newDefaultArgs.take_back(numArgs);
  return get(getContext(), newArgNames, newDefaultArgs);
}

FnMetadataAttrInterface
FnMetadataAttr::getWithBoundParams(const llvm::BitVector &boundParams) const {
  SmallVector<TypedAttr> newDefaultParams;

  size_t numParams = boundParams.size();
  ssize_t defaultIdx = getDefaultParameters().size() - numParams;
  for (size_t idx = 0; idx < numParams; ++idx, ++defaultIdx) {
    // TODO: handle parameter names when available.
    if (defaultIdx >= 0 && !boundParams[idx])
      newDefaultParams.emplace_back(getDefaultParameters()[defaultIdx]);
  }

  return get(getContext(), getArgNames(), getDefaultArguments(),
             newDefaultParams);
}

LogicalResult FnMetadataAttr::verifySignature(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<Type> inputParamTypes, ArrayRef<Type> resultParamTypes,
    FunctionType values, ArrayRef<ValueInputConvention> inputConventions,
    FnEffects effects) const {
  // Verify default arguments and parameters.
  auto verifyDefaults = [emitError](ArrayRef<TypedAttr> defaults,
                                    ArrayRef<Type> types,
                                    StringRef kind) -> LogicalResult {
    if (defaults.size() > types.size()) {
      return emitError() << "there are more default " << kind
                         << "s than inputs : " << defaults.size() << " > "
                         << types.size();
    }
    for (auto [defaultsIndex, value] : llvm::enumerate(defaults)) {
      size_t index = types.size() - defaults.size() + defaultsIndex;
      Type expected = types[index];
      if (value.getType() != expected) {
        return emitError() << kind << " #" << index << " has type " << expected
                           << " but default " << kind << " has type "
                           << value.getType();
      }
    }
    return success();
  };

  if (failed(verifyDefaults(getDefaultArguments(), values.getInputs(),
                            "argument")) ||
      failed(
          verifyDefaults(getDefaultParameters(), inputParamTypes, "parameter")))
    return failure();

  // Verify input conventions.
  size_t numInputConv = inputConventions.size();
  if (getArgNames().size() != numInputConv) {
    return emitError() << "number of argument names does not match number of "
                          "input conventions: "
                       << getArgNames().size() << " != " << numInputConv;
  }

  for (auto [i, argType, conv] :
       llvm::enumerate(values.getInputs(), inputConventions)) {
    Type type = argType;
    // Verify variadics.
    if (effects.hasVarArgs() && effects.isVarArg(values.getNumInputs(), i)) {
      auto variadic = ::dyn_cast<VariadicType>(type);
      if (!variadic) {
        return emitError() << "argument #" << i
                           << " in signature with varargs should be a "
                              "`!kgen.variadic` but got: "
                           << type;
      }
      type = variadic.getElementAsType();
    }
    // Verify argument conventions.
    if (SignatureType::hasAddress(conv)) {
      if (::isa<PointerType, RefType>(type))
        break;
      return emitError() << "argument #" << i << " with convention '"
                         << stringifyEnum(conv)
                         << "' in signature type should be a `!kgen.pointer` "
                            "or `!lit.ref` but got: "
                         << type;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UnboundMLIROperationAttr
//===----------------------------------------------------------------------===//

Type UnboundMLIROperationAttr::getType() const {
  return mlir::NoneType::get(getContext());
}

//===----------------------------------------------------------------------===//
// NoneAttr
//===----------------------------------------------------------------------===//

Type NoneAttr::getType() const { return LIT::NoneType::get(getContext()); }

//===----------------------------------------------------------------------===//
// LifetimeType
//===----------------------------------------------------------------------===//

Type LifetimeAttr::getType() const { return LifetimeType::get(getContext()); }

//===----------------------------------------------------------------------===//
// StructAttr
//===----------------------------------------------------------------------===//

static ParseResult
parseStructElements(AsmParser &p,
                    SmallVector<std::tuple<StringAttr, TypedAttr>> &values) {
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
                    ArrayRef<std::tuple<StringAttr, TypedAttr>> values) {
  p << '{';
  llvm::interleaveComma(values, p, [&](const auto &value) {
    printParamName(p, std::get<0>(value));
    printColonTypeOrIndex(p, std::get<1>(value).getType());
    p << " = ";
    printParamValue(p, std::get<1>(value));
  });
  p << '}';
}

LogicalResult
StructAttr::verifySymbolUses(Operation *module,
                             mlir::LockedSymbolTableCollection &symtab,
                             Location loc) const {
  auto structDecl =
      symtab.lookupSymbolIn<StructDeclOp>(module, getType().getSymbol());
  if (!structDecl)
    return emitError(loc) << "struct attribute type " << getType().getSymbol()
                          << " does not refer to a struct declaration";

  ParameterEvaluator evaluator(getType().getParamValues());
  auto fields = structDecl.getFieldDecls();
  unsigned numFields = std::distance(fields.begin(), fields.end());
  if (numFields != getValues().size()) {
    return (emitError(loc) << "struct declaration expected " << numFields
                           << " fields but struct attribute has "
                           << getValues().size())
               .attachNote(structDecl.getLoc())
           << "see struct declaration here";
  }

  for (auto [fieldDecl, value, i] :
       llvm::zip(fields, getValues(), llvm::seq<unsigned>(0, numFields))) {
    StringAttr nameInDecl = fieldDecl.getNameAttr();
    if (nameInDecl != std::get<0>(value)) {
      return (emitError(loc)
              << "struct attribute field name " << std::get<0>(value)
              << " at position #" << i << " does not match the name "
              << nameInDecl << " in the struct declaration")
                 .attachNote(structDecl.getLoc())
             << "see struct declaration here";
    }

    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (reboundType != std::get<1>(value).getType()) {
      return (emitError(loc)
              << "struct attribute field #" << i << " has type "
              << std::get<1>(value).getType()
              << " but corresponding struct field " << fieldDecl.getNameAttr()
              << " expected " << reboundType)
                 .attachNote(structDecl.getLoc())
             << "see struct declaration here";
    }
  }

  return success();
}

bool LIT::StructAttr::isConstant() const {
  return llvm::all_of(getValues(), [&](const auto &value) {
    return ParameterAttr::isSimpleConstant(std::get<1>(value));
  });
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
    auto it = llvm::find_if(value.getValues(), [&](const auto &p) {
      return std::get<0>(p) == field;
    });
    if (it != value.getValues().end())
      return std::get<1>(*it);
  }

  return Base::get(context, structValue, field, resultType);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/LITDialect/LITAttrs.cpp.inc"

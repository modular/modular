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
#include "KGEN/LITDialect/LITUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
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
//
//===----------------------------------------------------------------------===//

ArgParamListAttr ArgParamListAttr::get(MLIRContext *context) {
  return ArgParamListAttr::get(context, {}, {}, {}, {}, {}, {});
}

ArgParamListAttr ArgParamListAttr::get(MLIRContext *context,
                                       ArrayRef<StringAttr> names,
                                       ArrayRef<PassingKind> passingKinds) {
  return ArgParamListAttr::get(context, names, passingKinds, {}, {}, {}, {});
}

LogicalResult ArgParamListAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<StringAttr> names,
    ArrayRef<PassingKind> passingKinds, ArrayRef<TypedAttr> defaultPos,
    ArrayRef<TypedAttr> defaultKwOnly, ArrayRef<size_t> variadicIndices,
    ArrayRef<size_t> packIndices) {
  size_t numEl = names.size();
  if (size_t numPassingKinds = passingKinds.size(); numEl != numPassingKinds) {
    return emitError() << "number of argument/parameter names and passing "
                          "kinds does not match: "
                       << numEl << " vs. " << numPassingKinds;
  }
  for (StringAttr name : names)
    if (!name)
      return emitError() << "argument/parameter name cannot be null";

  if (failed(verifyPassingKinds(emitError, passingKinds, defaultPos.size(),
                                defaultKwOnly.size(), "arguments/parameter")))
    return failure();

  // We verified the passing kinds' order and number, so we can use a handler.
  DefaultValueHandler defaultHandler(passingKinds, defaultPos, defaultKwOnly);
  auto verifyVariadicIdx = [&](size_t idx, bool isPack) -> LogicalResult {
    if (idx >= numEl) {
      return emitError() << "variadic " << (isPack ? "pack " : "")
                         << "index must be less than the number of elements: "
                         << idx << " vs. " << numEl;
    }
    if (defaultHandler.getDefault(idx))
      return emitError() << "variadic " << (isPack ? "pack " : "")
                         << "cannot have a default value";
    if (passingKinds[idx] != PassingKind::PosOrKw) {
      if (isPack) {
        return emitError()
               << "passing kind of variadic pack must be pos_or_kw, got "
               << stringifyPassingKind(passingKinds[idx]);
      }
      if (passingKinds[idx] != PassingKind::KwOnly) {
        return emitError()
               << "passing kind of variadic must be pos_or_kw or kw, got "
               << stringifyPassingKind(passingKinds[idx]);
      }
    }
    return success();
  };
  if (size_t numPacks = packIndices.size(); packIndices.size() > 1) {
    return emitError() << "more than 1 variadic pack not allowed in an "
                          "argument/parameter list";
  } else if (numPacks == 1) {
    if (failed(verifyVariadicIdx(packIndices[0], /*isPack=*/true)))
      return failure();
  }

  for (size_t idx : variadicIndices)
    if (failed(verifyVariadicIdx(idx, /*isPack=*/false)))
      return failure();

  return success();
}

ArgParamListAttr
ArgParamListAttr::cloneWith(ArrayRef<StringAttr> names,
                            ArrayRef<PassingKind> passingKinds) const {
  return ArgParamListAttr::get(getContext(), names, passingKinds,
                               getDefaultPos(), getDefaultKwOnly(),
                               /*variadicIndices=*/{}, /*packIndices=*/{});
}

//===----------------------------------------------------------------------===//
// FnMetadataAttr
//===----------------------------------------------------------------------===//

FnMetadataAttr FnMetadataAttr::get(MLIRContext *context) {
  auto list = ArgParamListAttr::get(context);
  return FnMetadataAttr::get(context, list, list, 0, VariadicEffects());
}

FnMetadataAttr FnMetadataAttr::get(ArgParamListAttr argListAttrs,
                                   ArgParamListAttr paramListAttrs,
                                   size_t numImplicitLifetimeDecls,
                                   VariadicEffects variadicEffects) {
  return get(argListAttrs.getContext(), argListAttrs, paramListAttrs,
             numImplicitLifetimeDecls, variadicEffects);
}

FnMetadataAttr FnMetadataAttr::get(ArgParamListAttr argListAttrs,
                                   size_t numImplicitLifetimeDecls,
                                   VariadicEffects variadicEffects) {
  MLIRContext *ctx = argListAttrs.getContext();
  return get(ctx, argListAttrs, ArgParamListAttr::get(ctx),
             numImplicitLifetimeDecls, variadicEffects);
}

FnMetadataAttrInterface
FnMetadataAttr::getWithBoundPosArgs(size_t numBound) const {
  ArrayRef<PassingKind> passingKinds = getArgPassingKinds();
  size_t numPositional = countNumPositional(passingKinds);
  assert(numBound <= numPositional && "only positional arguments can be bound");

  ArrayRef<StringAttr> newArgNames = getArgNames().drop_front(numBound);
  ArrayRef<PassingKind> newArgPassingKind = passingKinds.drop_front(numBound);

  ArrayRef<TypedAttr> newDefaultPosArgs = getDefaultPosArgs();
  size_t numArgs = numPositional - numBound;
  if (numArgs < newDefaultPosArgs.size())
    newDefaultPosArgs = newDefaultPosArgs.take_back(numArgs);

  auto newArgListAttrs = ArgParamListAttr::get(
      getContext(), newArgNames, newArgPassingKind, newDefaultPosArgs,
      getDefaultKwOnlyArgs(), /*variadicIndices=*/{}, /*packIndices=*/{});
  return get(newArgListAttrs, getParamListAttrs(),
             getNumImplicitLifetimeDecls(), getVariadicEffects());
}

FnMetadataAttrInterface
FnMetadataAttr::getWithBoundParams(const llvm::BitVector &boundParams) const {
  SmallVector<TypedAttr> newDefaultPosParams;
  SmallVector<TypedAttr> newDefaultKwOnlyParams;
  SmallVector<StringAttr> newParamNames;
  SmallVector<PassingKind> newParamPassingKinds;

  auto defaultHandler = DefaultValueHandler::getDefaultParamHandler(*this);
  for (size_t idx = 0; idx < boundParams.size(); ++idx) {
    if (!boundParams[idx]) {
      newParamNames.emplace_back(getParamNames()[idx]);
      newParamPassingKinds.emplace_back(getParamPassingKinds()[idx]);
      if (TypedAttr defaultOr = defaultHandler.getPosDefault(idx))
        newDefaultPosParams.emplace_back(defaultOr);
      else if (TypedAttr defaultOr = defaultHandler.getKwOnlyDefault(idx))
        newDefaultKwOnlyParams.emplace_back(defaultOr);
    }
  }

  auto newParamAttrs = ArgParamListAttr::get(
      getContext(), newParamNames, newParamPassingKinds, newDefaultPosParams,
      newDefaultKwOnlyParams, /*variadicIndices=*/{}, /*packIndices=*/{});
  return get(getArgListAttrs(), newParamAttrs, getNumImplicitLifetimeDecls(),
             getVariadicEffects());
}

FnMetadataAttrInterface
FnMetadataAttr::prependPosParams(size_t numNewParams) const {
  auto emptyStr = StringAttr::get(getContext());
  SmallVector<StringAttr> newParamNames(numNewParams, emptyStr);
  llvm::append_range(newParamNames, getParamNames());
  SmallVector<PassingKind> newPassingKinds(numNewParams, PassingKind::PosOnly);
  llvm::append_range(newPassingKinds, getParamPassingKinds());

  return get(getArgListAttrs(),
             getParamListAttrs().cloneWith(newParamNames, newPassingKinds),
             getNumImplicitLifetimeDecls(), getVariadicEffects());
}

LogicalResult FnMetadataAttr::verifySignature(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<Type> inputParamTypes, ArrayRef<Type> resultParamTypes,
    FunctionType values, ArrayRef<ArgConvention> argConventions,
    FnEffects effects) const {
  if (getParamNames().size() != inputParamTypes.size()) {
    return emitError() << "number of parameter names doesn't match number of "
                          "input parameter types";
  }

  // Verify input conventions.
  size_t numInputs = values.getNumInputs();
  if (size_t numArgConv = argConventions.size(); numInputs != numArgConv) {
    return emitError()
           << "number of arguments does not match number of input conventions: "
           << numInputs << " != " << numArgConv;
  }
  if (size_t numArgName = getArgNames().size(); numArgName != numInputs) {
    return emitError()
           << "number of arguments does not match number of argument names: "
           << numInputs << " != " << numArgName;
  }

  for (auto [i, argType, conv] :
       llvm::enumerate(values.getInputs(), argConventions)) {
    Type type = argType;
    // Verify variadics.

    if (isVarArg(i)) {
      auto variadic = ::dyn_cast<VariadicType>(type);
      if (!variadic) {
        return emitError() << "argument #" << i
                           << " in signature with varargs should be a "
                              "`!kgen.variadic` but got: "
                           << type;
      }
      type = variadic.getElementType();
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

  if (failed(verifyDefaultTypes(emitError, getDefaultPosArgs(),
                                getDefaultKwOnlyArgs(), getArgPassingKinds(),
                                values.getInputs(), "argument",
                                argConventions)) ||
      failed(verifyDefaultTypes(
          emitError, getDefaultPosParams(), getDefaultKwOnlyParams(),
          getParamPassingKinds(), inputParamTypes, "parameter")))
    return failure();

  return success();
}

bool FnMetadataAttr::hasVarArgs() const {
  return getVariadicEffects().get(VariadicEffects::Impl::VarArg);
}

bool FnMetadataAttr::hasPackVarArgs() const {
  return getVariadicEffects().get(VariadicEffects::Impl::PackVarArg);
}

bool FnMetadataAttr::hasKwVarArgs() const {
  return getVariadicEffects().get(VariadicEffects::Impl::KWVarArg);
}

bool FnMetadataAttr::hasParamVarArgs() const {
  return getVariadicEffects().get(VariadicEffects::Impl::ParamVarArg);
}

bool FnMetadataAttr::isVarArg(size_t idx) const {
  if (!hasVarArgs())
    return false;

  size_t numArgs = getArgPassingKinds().size();
  // If the function has keyword varargs, the vararg index is the second last.
  // Otherwise, it's the last.
  return (idx + 1 + hasKwVarArgs()) == numArgs;
}

bool FnMetadataAttr::isKwVarArg(size_t idx) const {
  if (!hasKwVarArgs())
    return false;
  return idx + 1 == getArgPassingKinds().size();
}

bool FnMetadataAttr::isPackVarArg(size_t idx) const {
  if (!hasPackVarArgs())
    return false;
  return idx + 1 == getArgPassingKinds().size();
}

static ParseResult parseVariadicEffects(AsmParser &p,
                                        VariadicEffects &variadicEffects) {
  if (succeeded(p.parseOptionalKeyword("none"))) {
    variadicEffects = VariadicEffects();
    return success();
  }

  auto effectsValue = VariadicImpl::VariadicEffects::None;
  StringRef kw;
  while (succeeded(p.parseOptionalKeyword(
      &kw, {"vararg", "packvararg", "kwvararg", "param_vararg"}))) {
    effectsValue |= *VariadicImpl::symbolizeVariadicEffects(kw);

    // No vertical bar? We're done. It's not a parse error, but it does mean we
    // can't specify more effects.
    if (failed(p.parseOptionalVerticalBar())) {
      variadicEffects = VariadicEffects(effectsValue);
      return success();
    }
  }

  return failure();
}

static void printVariadicEffects(AsmPrinter &p,
                                 VariadicEffects variadicEffects) {
  p << VariadicImpl::stringifyVariadicEffects(variadicEffects.getImpl());
}

namespace M::KGEN::LIT {
static llvm::hash_code hash_value(VariadicEffects effects) {
  return llvm::hash_value(static_cast<uint16_t>(effects.getImpl()));
}
} // namespace M::KGEN::LIT

//===----------------------------------------------------------------------===//
// UnboundMLIROperationAttr
//===----------------------------------------------------------------------===//

Type UnboundMLIROperationAttr::getType() const {
  return mlir::NoneType::get(getContext());
}

//===----------------------------------------------------------------------===//
// BindTypeAttr
//===----------------------------------------------------------------------===//

static ParseResult parseBindTypeParams(AsmParser &p,
                                       SmallVectorImpl<TypedAttr> &values,
                                       TypedAttr typeValue) {
  auto metatype = dyn_cast<MetaTypeType>(typeValue.getType());
  if (!metatype) {
    return p.emitError(p.getCurrentLocation(),
                       "'bind_type' expected a metatyped type value");
  }

  ParameterEvaluator evaluator;
  auto eachFn = [&](Type type) {
    if (failed(parseParamValue(p, values.emplace_back(),
                               evaluator.getReboundType(type))))
      return failure();
    evaluator.addInputValue(values.back());
    return mlir::success();
  };
  auto betweenFn = [&] { return p.parseComma(); };
  return failableInterleave(metatype.getSignature().getInputParamTypes(),
                            std::move(eachFn), std::move(betweenFn));
}

static void printBindTypeParams(AsmPrinter &p, ArrayRef<TypedAttr> values,
                                TypedAttr typeValue) {
  llvm::interleaveComma(values, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
}

LogicalResult BindTypeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   TypedAttr typeValue,
                                   ArrayRef<TypedAttr> values,
                                   MetaTypeType type) {
  auto metatype = ::dyn_cast<MetaTypeType>(typeValue.getType());
  if (!metatype)
    return emitError() << "'bind_type' expected a metatyped type value";

  // Check the bound values against the input parameter signature. Allow partial
  // binding.
  ArrayRef<Type> inputTypes = metatype.getSignature().getInputParamTypes();
  if (values.size() != inputTypes.size()) {
    return emitError()
           << "'bind_type' has wrong number of input parameters: have "
           << values.size() << " but expected " << inputTypes.size();
  }
  ParameterEvaluator valueSubst;
  for (auto [i, type, value] : llvm::enumerate(inputTypes, values)) {
    Type expected = valueSubst.getReboundType(type);
    valueSubst.addInputValue(value);
    if (expected == value.getType())
      continue;
    return emitError() << "'bind_type' parameter #" << i << " has type "
                       << value.getType() << " but type expected " << expected;
  }

  if (metatype.getParamValues().size() != type.getParamValues().size()) {
    return emitError() << "'bind_type' result metatype should have "
                       << type.getParamValues().size()
                       << " parameter values, but got "
                       << metatype.getParamValues().size();
  }
  auto it = values.begin();
  for (auto [i, old, next] :
       llvm::enumerate(metatype.getParamValues(), type.getParamValues())) {
    if (::isa<UnboundAttr>(old)) {
      if (*it++ != next) {
        return emitError() << "'bind_type' result metatype parameter #" << i
                           << " does not match corresponding input parameter";
      }
    } else if (old != next) {
      return emitError() << "'bind_type' cannot change the value of parameter #"
                         << i;
    }
  }

  // Ignore unbound values.
  SmallVector<Type> expected;
  ArrayRef<Type> resultTypes = type.getSignature().getInputParamTypes();
  ParameterEvaluator typeSubst;
  for (auto [type, value] : llvm::zip(inputTypes, values)) {
    if (::isa<UnboundAttr>(value)) {
      expected.push_back(typeSubst.getReboundType(type));
      typeSubst.addInputValue(ParamIndexRefAttr::get(
          /*depth=*/0, /*isResult=*/false, expected.size() - 1,
          expected.back()));
    } else {
      typeSubst.addInputValue(value);
    }
  }
  if (resultTypes.size() != expected.size()) {
    return emitError() << "'bind_type' result metatype signature should have "
                       << expected.size() << " input parameters";
  }
  for (auto [i, unbound, type] : llvm::enumerate(expected, resultTypes)) {
    if (unbound != type)
      return emitError() << "result signature parameter #" << i
                         << " expected to be " << unbound << " but got "
                         << type;
  }
  return success();
}

/// Infer the result type for `BindTypeAttr`.
static MetaTypeType getBindTypeResultType(TypedAttr typeValue,
                                          ArrayRef<TypedAttr> values) {
  auto metatype = cast<MetaTypeType>(typeValue.getType());
  SmallVector<TypedAttr> bindings;
  auto it = values.begin();
  for (TypedAttr value : metatype.getParamValues()) {
    if (isa<UnboundAttr>(value))
      bindings.push_back(*it++);
    else
      bindings.push_back(value);
  }
  assert(it == values.end() && "expected all bindings to be consumed");
  return metatype.bind(bindings);
}

/// Entry point for the constructor for `BindTypeAttr`, which folds on
/// construction.
static TypedAttr getOrFoldBindType(TypedAttr typeValue,
                                   ArrayRef<TypedAttr> values,
                                   MetaTypeType type) {
  // Assume the inputs are verified. If the type value is a `DeclRefType` then
  // bind it and return a type constant.
  if (auto typeCst = dyn_cast<TypeConstantAttr>(typeValue)) {
    if (auto decl = dyn_cast<DeclRefType>(typeCst.getValue())) {
      auto bound =
          DeclRefType::get(decl.getSymbol(), type.getParamValues(), type);
      return TypeConstantAttr::get(bound, type);
    }
  }
  return BindTypeAttr::Base::get(type.getContext(), typeValue, values, type);
}

TypedAttr BindTypeAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                   MLIRContext *ctx, TypedAttr typeValue,
                                   ArrayRef<TypedAttr> values,
                                   MetaTypeType type) {
  if (failed(verify(emitError, typeValue, values, type)))
    return {};
  return getOrFoldBindType(typeValue, values, type);
}

TypedAttr BindTypeAttr::get(MLIRContext *ctx, TypedAttr typeValue,
                            ArrayRef<TypedAttr> values, MetaTypeType type) {
  return getOrFoldBindType(typeValue, values, type);
}

//===----------------------------------------------------------------------===//
// LifetimeUnionAttr
//===----------------------------------------------------------------------===//

static bool unionArgCompare(TypedAttr lhs, TypedAttr rhs) {
  // Ignore LifetimeMutCastAttr's for comparison.
  return ParameterAttr::compare(LifetimeMutCastAttr::strip(lhs),
                                LifetimeMutCastAttr::strip(rhs));
}

TypedAttr LifetimeUnionAttr::get(ArrayRef<TypedAttr> operandsIn,
                                 LifetimeType type) {

  // Canonicalize the operands, sorting by name/index and eliminating raw
  // #lit.lifetime members.
  SmallVector<TypedAttr> operands(operandsIn);

  // Preprocess operands.
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    assert(operands[i].getType() == type &&
           "all members of a lifetime union must have matching type");
    // Drop #lit.lifetime, they carry no information.
    if (::isa<LifetimeAttr>(operands[i])) {
      operands[i] = operands.back();
      operands.pop_back();
      --e, --i;
      continue;
    }

    // Flatten any of the same operation into the operand list:
    // `(union x, (union y, z))` => `(union x, y, z)`.
    if (auto subexpr = ::dyn_cast<LifetimeUnionAttr>(operands[i])) {
      operands[i] = operands.back();
      operands.pop_back();
      operands.append(subexpr.getOperands().begin(),
                      subexpr.getOperands().end());
      // No need to check these operands, they've already been checked when
      // the subunion was formed.
      --e, --i;
      continue;
    }
  }

  // Impose an ordering on the operands, sorting by name where possible - but
  // predictably ordered w.r.t. each other.
  llvm::stable_sort(operands, unionArgCompare);

  // Remove duplicates which will now be sorted next to each other.
  if (operands.size() > 1) {
    for (size_t i = 0, e = operands.size() - 1; i != e; ++i) {
      if (operands[i] != operands[i + 1])
        continue;

      operands.erase(operands.begin() + i + 1);
      --e, --i;
    }
  }

  // If no results, return a plain lifetime attr.
  if (operands.empty())
    return LifetimeAttr::get(type);
  if (operands.size() == 1)
    return operands[0];

  auto resultType = ::cast<LifetimeType>(operands[0].getType());
  return LifetimeUnionAttr::Base::get(type.getContext(), operands, resultType);
}

//===----------------------------------------------------------------------===//
// LifetimeMutCastAttr
//===----------------------------------------------------------------------===//

TypedAttr LifetimeMutCastAttr::get(TypedAttr operand, TypedAttr isMutable) {
  auto curTy = ::cast<LifetimeType>(operand.getType());
  if (curTy.isMutable() == isMutable)
    return operand;

  // Fold some common cases to canonicalize.
  // mutcast(mutcast(x)) -> mutcast(x), often canceling out.
  if (auto mutCast = ::dyn_cast<LifetimeMutCastAttr>(operand))
    return get(mutCast.getOperand(), isMutable);

  // Push into union so it cancels out.
  if (auto unionAttr = ::dyn_cast<LifetimeUnionAttr>(operand)) {
    SmallVector<TypedAttr> elts;
    for (auto elt : unionAttr.getOperands())
      elts.push_back(LifetimeMutCastAttr::get(elt, isMutable));
    return LifetimeUnionAttr::get(elts, LifetimeType::get(isMutable));
  }

  auto context = curTy.getContext();
  return LifetimeMutCastAttr::Base::get(context, operand,
                                        LifetimeType::get(isMutable));
}

TypedAttr LifetimeMutCastAttr::get(TypedAttr operand, Type type) {
  assert(::isa<LifetimeType>(type) && ::isa<LifetimeType>(operand.getType()) &&
         "#lit.lifetime.union always has !lit.lifetime type");
  if (operand.getType() == type)
    return operand;
  return get(operand, ::cast<LifetimeType>(type).isMutable());
}

TypedAttr LifetimeMutCastAttr::get(TypedAttr operand, bool isMutable) {
  auto operandType = ::cast<LifetimeType>(operand.getType());
  if (operandType.isMutableKnown(isMutable))
    return operand;
  return get(operand, BoolAttr::get(operand.getContext(), isMutable));
}

//===----------------------------------------------------------------------===//
// LITStructAttr
//===----------------------------------------------------------------------===//

static ParseResult
parseStructElements(AsmParser &p,
                    SmallVector<std::tuple<StringAttr, TypedAttr>> &values) {
  std::string name;
  Type type;
  TypedAttr value;
  auto parseElt = [&]() -> ParseResult {
    if (p.parseKeywordOrString(&name) || parseColonTypeOrIndex(p, type) ||
        p.parseEqual() || parseParamValue(p, value, type))
      return failure();
    values.emplace_back(StringAttr::get(p.getContext(), name), value);
    return success();
  };
  return p.parseCommaSeparatedList(AsmParser::Delimiter::Braces, parseElt);
}

static void
printStructElements(AsmPrinter &p,
                    ArrayRef<std::tuple<StringAttr, TypedAttr>> values) {
  p << '{';
  llvm::interleaveComma(values, p, [&](const auto &value) {
    p.printKeywordOrString(std::get<0>(value));
    printColonTypeOrIndex(p, std::get<1>(value).getType());
    p << " = ";
    printParamValue(p, std::get<1>(value));
  });
  p << '}';
}

LogicalResult
LITStructAttr::verifySymbolUses(Operation *module,
                                mlir::LockedSymbolTableCollection &symtab,
                                Location loc) const {
  SymbolRefAttr symbolRef = getType().getSymbol();
  auto structDecl = symtab.lookupSymbolIn<StructDeclOp>(module, symbolRef);
  if (!structDecl) {
    return emitError(loc) << "struct attribute type " << symbolRef
                          << " does not refer to a struct declaration";
  }

  ParameterEvaluator evaluator(structDecl.getInputParams(),
                               getType().getParamValues());

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

bool LITStructAttr::isConstant() const {
  return llvm::all_of(getValues(), [&](const auto &value) {
    return ParameterAttr::isSimpleConstant(std::get<1>(value));
  });
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

TypedAttr LIT::StructExtractAttr::get(TypedAttr structValue,
                                      StructFieldOp fieldOp) {
  auto structType = ::cast<DeclRefType>(structValue.getType());
  ParameterEvaluator evaluator(fieldOp.getParentOp().getInputParams(),
                               structType.getParamValues());
  auto resultType = evaluator.getReboundType(fieldOp.getType());
  return get(structValue, fieldOp.getNameAttr(), resultType);
}

TypedAttr LIT::StructExtractAttr::get(TypedAttr structValue, StringAttr field,
                                      Type resultType) {
  return get(structValue.getContext(), structValue, field, resultType);
}

TypedAttr LIT::StructExtractAttr::get(MLIRContext *context,
                                      TypedAttr structValue, StringAttr field,
                                      Type resultType) {
  if (auto value = dyn_cast_if_present<LITStructAttr>(structValue)) {
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

#include "KGEN/LITDialect/LITEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/LITDialect/LITAttrs.cpp.inc"

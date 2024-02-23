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
// PogsAttr
//===----------------------------------------------------------------------===//

PogsAttr PogsAttr::get(MLIRContext *context) {
  return PogsAttr::get(context, {}, {}, {}, {}, {}, {});
}

PogsAttr PogsAttr::get(MLIRContext *context,
                                       ArrayRef<StringAttr> names,
                                       ArrayRef<PassingKind> passingKinds) {
  return PogsAttr::get(context, names, passingKinds, {}, {}, {}, {});
}

LogicalResult PogsAttr::verify(
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

PogsAttr
PogsAttr::cloneWith(ArrayRef<StringAttr> names,
                            ArrayRef<PassingKind> passingKinds) const {
  return PogsAttr::get(getContext(), names, passingKinds,
                               getDefaultPos(), getDefaultKwOnly(),
                               getVariadicIndices(), getPackIndices());
}

bool PogsAttr::isVariadic(size_t idx) const {
  return llvm::is_contained(getVariadicIndices(), idx);
}

bool PogsAttr::isPack(size_t idx) const {
  return llvm::is_contained(getPackIndices(), idx);
}

//===----------------------------------------------------------------------===//
// FnMetadataAttr
//===----------------------------------------------------------------------===//

FnMetadataAttr FnMetadataAttr::get(MLIRContext *context) {
  auto list = PogsAttr::get(context);
  return FnMetadataAttr::get(context, list, list, 0);
}

FnMetadataAttr FnMetadataAttr::get(PogsAttr argListAttrs,
                                   PogsAttr paramListAttrs,
                                   size_t numImplicitLifetimeDecls) {
  return get(argListAttrs.getContext(), argListAttrs, paramListAttrs,
             numImplicitLifetimeDecls);
}

FnMetadataAttr FnMetadataAttr::get(PogsAttr argListAttrs,
                                   size_t numImplicitLifetimeDecls) {
  MLIRContext *ctx = argListAttrs.getContext();
  return get(ctx, argListAttrs, PogsAttr::get(ctx),
             numImplicitLifetimeDecls);
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

  /// We drop varidic/pack indices if needed, and adjust the rest.
  SmallVector<size_t> newVariadicIndices;
  for (size_t idx : getArgListAttrs().getVariadicIndices())
    if (idx >= numBound)
      newVariadicIndices.push_back(idx - numBound);
  SmallVector<size_t> newPackIndices;
  for (size_t idx : getArgListAttrs().getPackIndices())
    if (idx >= numBound)
      newPackIndices.push_back(idx - numBound);

  auto newArgListAttrs = PogsAttr::get(
      getContext(), newArgNames, newArgPassingKind, newDefaultPosArgs,
      getDefaultKwOnlyArgs(), newVariadicIndices, newPackIndices);
  return get(newArgListAttrs, getParamListAttrs(),
             getNumImplicitLifetimeDecls());
}

FnMetadataAttrInterface
FnMetadataAttr::getWithBoundParams(const llvm::BitVector &boundParams) const {
  SmallVector<TypedAttr> newDefaultPosParams;
  SmallVector<TypedAttr> newDefaultKwOnlyParams;
  SmallVector<StringAttr> newParamNames;
  SmallVector<PassingKind> newParamPassingKinds;

  DefaultValueHandler defaultHandler(getParamListAttrs());
  size_t numParams = boundParams.size();
  for (size_t idx = 0; idx < numParams; ++idx) {
    if (!boundParams[idx]) {
      newParamNames.emplace_back(getParamNames()[idx]);
      newParamPassingKinds.emplace_back(getParamPassingKinds()[idx]);
      if (TypedAttr defaultOr = defaultHandler.getPosDefault(idx))
        newDefaultPosParams.emplace_back(defaultOr);
      else if (TypedAttr defaultOr = defaultHandler.getKwOnlyDefault(idx))
        newDefaultKwOnlyParams.emplace_back(defaultOr);
    }
  }

  ArrayRef<size_t> variadicIndices = getParamListAttrs().getVariadicIndices();
  ArrayRef<size_t> packIndices = getParamListAttrs().getPackIndices();
  SmallVector<size_t> newVariadicIndices;
  SmallVector<size_t> newPackIndices;
  if (!variadicIndices.empty() || !packIndices.empty()) {
    // We need to calculate the cumulatives number of bound parameters to adjust
    // the variadic and packs indices.
    SmallVector<size_t> cumSum{boundParams[0]};
    cumSum.reserve(numParams);
    for (size_t idx = 1; idx < numParams; ++idx)
      cumSum.push_back(cumSum[idx - 1] + boundParams[idx]);

    // We drop an index that corresponds to a bound parameter, and adjust the
    // rest according to how many preceding parameters are bound.
    for (size_t idx : variadicIndices)
      if (!boundParams[idx])
        newVariadicIndices.push_back(idx - cumSum[idx]);
    for (size_t idx : packIndices)
      if (!boundParams[idx])
        newPackIndices.push_back(idx - cumSum[idx]);
  }

  auto newParamAttrs = PogsAttr::get(
      getContext(), newParamNames, newParamPassingKinds, newDefaultPosParams,
      newDefaultKwOnlyParams, newVariadicIndices, newPackIndices);
  return get(getArgListAttrs(), newParamAttrs, getNumImplicitLifetimeDecls());
}

FnMetadataAttrInterface
FnMetadataAttr::prependPosParams(size_t numNewParams,
                                 ArrayRef<size_t> variadicIndices) const {
  if (numNewParams == 0)
    return *this;

  auto emptyStr = StringAttr::get(getContext());
  SmallVector<StringAttr> newParamNames(numNewParams, emptyStr);
  llvm::append_range(newParamNames, getParamNames());
  SmallVector<PassingKind> newPassingKinds(numNewParams, PassingKind::PosOnly);
  llvm::append_range(newPassingKinds, getParamPassingKinds());

  // We need to prepend the new variadic indices, and offset the existing ones.
  PogsAttr oldParamListAttr = getParamListAttrs();
  SmallVector<size_t> newVariadicIndices(variadicIndices);
  for (size_t idx : oldParamListAttr.getVariadicIndices())
    newVariadicIndices.push_back(idx + numNewParams);

  auto newParamListAttr = PogsAttr::get(
      getContext(), newParamNames, newPassingKinds, getDefaultPosParams(),
      getDefaultKwOnlyParams(), newVariadicIndices,
      oldParamListAttr.getPackIndices());
  return get(getArgListAttrs(), newParamListAttr,
             getNumImplicitLifetimeDecls());
}

std::pair<SmallVector<size_t>, size_t>
LIT::getContextualVariadicIndices(ArrayRef<Operation *> ops) {
  std::pair<SmallVector<size_t>, size_t> res;
  auto &[variadicIndices, numNewParams] = res;
  for (Operation *op : ops) {
    // Count the number of new parameters.
    size_t idxOffset = numNewParams;
    numNewParams += ::cast<DeclInterface>(op).getInputParams().size();

    // If we are dealing with a struct or trait, we collect the variadics.
    PogsAttr paramListAttr;
    if (auto structDecl = ::dyn_cast<StructDeclOp>(op))
      paramListAttr = structDecl.getSignature().getParamListAttrs();
    else if (auto traitDecl = ::dyn_cast<TraitDeclOp>(op))
      paramListAttr = traitDecl.getSignature().getParamListAttrs();
    else
      continue;

    // The variadic indices need to be offset to correspond to the location in
    // the collected array.
    for (size_t idx : paramListAttr.getVariadicIndices())
      variadicIndices.push_back(idxOffset + idx);
  }
  return res;
}

FnMetadataAttrInterface
FnMetadataAttr::prependPosParamsFromOps(ArrayRef<Operation *> ops) const {
  auto [variadicIndices, numNewParams] = getContextualVariadicIndices(ops);
  return prependPosParams(numNewParams, variadicIndices);
}

LogicalResult FnMetadataAttr::verifySignature(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<Type> inputParamTypes, ArrayRef<Type> resultParamTypes,
    FunctionType values, ArrayRef<ArgConvention> argConventions,
    FnEffects effects) const {
  if (!resultParamTypes.empty())
    return emitError() << "expected no result parameters";

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
  return !getArgListAttrs().getVariadicIndices().empty();
}

bool FnMetadataAttr::hasPackVarArgs() const {
  return !getArgListAttrs().getPackIndices().empty();
}

bool FnMetadataAttr::hasParamVarArgs() const {
  return !getParamListAttrs().getVariadicIndices().empty();
}

bool FnMetadataAttr::isVarArg(size_t idx) const {
  return llvm::is_contained(getArgListAttrs().getVariadicIndices(), idx);
}

bool FnMetadataAttr::isPackVarArg(size_t idx) const {
  return llvm::is_contained(getArgListAttrs().getPackIndices(), idx);
}

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

    // It takes just one InvalidRefLifetimeAttr to make the whole list be
    // invalid for references.
    if (::isa<InvalidRefLifetimeAttr>(operands[i]))
      return operands[i];

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

  // Singletons don't need a cast, just form one with the new mutability.
  if (::isa<LifetimeAttr>(operand))
    return LifetimeAttr::get(isMutable);
  if (::isa<InvalidRefLifetimeAttr>(operand))
    return InvalidRefLifetimeAttr::get(isMutable);

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

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
// PogListAttr
//===----------------------------------------------------------------------===//

static ParseResult parsePackInfo(AsmParser &p, ssize_t &idx,
                                 std::optional<ArgConvention> &conv) {
  conv.emplace();
  FailureOr<ssize_t> idxResult = mlir::FieldParser<ssize_t>::parse(p);
  if (failed(idxResult))
    return failure();
  idx = *idxResult;
  if (p.parseComma())
    return failure();
  FailureOr<ArgConvention> convResult =
      mlir::FieldParser<ArgConvention>::parse(p);
  if (failed(convResult))
    return failure();
  conv = *convResult;
  return success();
}

static void printPackInfo(AsmPrinter &p, ssize_t idx,
                          const std::optional<ArgConvention> &conv) {
  p << idx << ", " << *conv;
}

PogListAttr PogListAttr::get(MLIRContext *context) {
  return PogListAttr::get(context, {}, {});
}

PogListAttr PogListAttr::get(MLIRContext *context,
                             ArrayRef<PogMetadataAttr> pogs) {
  return PogListAttr::get(context, pogs, {}, {});
}

PogListAttr PogListAttr::get(MLIRContext *context,
                             ArrayRef<PogMetadataAttr> pogs,
                             ArrayRef<TypedAttr> defaultPos,
                             ArrayRef<TypedAttr> defaultKwOnly) {
  return PogListAttr::get(context, pogs, defaultPos, defaultKwOnly, -1,
                          std::nullopt);
}

PogListAttr PogListAttr::get(MLIRContext *context, ArrayRef<StringAttr> names,
                             ArrayRef<PassingKind> passingKinds) {
  SmallVector<PogMetadataAttr> pogs;
  for (auto [name, passingKind] : llvm::zip(names, passingKinds))
    pogs.emplace_back(PogMetadataAttr::get(name, passingKind));

  return PogListAttr::get(context, pogs);
}

PogListAttr PogListAttr::get(MLIRContext *context, ArrayRef<StringAttr> names,
                             ArrayRef<PassingKind> passingKinds,
                             ArrayRef<TypedAttr> defaultPos,
                             ArrayRef<TypedAttr> defaultKwOnly,
                             ArrayRef<size_t> variadicIndices,
                             ssize_t packIndex,
                             std::optional<ArgConvention> origPackConvention) {
  return PogListAttr::get(context, toPogs(names, passingKinds, variadicIndices),
                          defaultPos, defaultKwOnly, packIndex,
                          std::move(origPackConvention));
}

LogicalResult
PogListAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                    ArrayRef<PogMetadataAttr> pogs,
                    ArrayRef<TypedAttr> defaultPos,
                    ArrayRef<TypedAttr> defaultKwOnly, ssize_t packIndex,
                    std::optional<ArgConvention> origPackConvention) {
  size_t numEl = pogs.size();
  for (PogMetadataAttr pogAttr : pogs)
    if (!pogAttr.getName())
      return emitError() << "argument/parameter name cannot be null";

  SmallVector<PassingKind> passingKinds = llvm::map_to_vector(
      pogs, [](PogMetadataAttr pogAttr) { return pogAttr.getPassingKind(); });
  if (failed(verifyPassingKinds(emitError, pogs, defaultPos.size(),
                                defaultKwOnly.size(), "arguments/parameter")))
    return failure();

  // We verified the passing kinds' order and number, so we can use a handler.
  DefaultValueHandler defaultHandler(pogs, defaultPos, defaultKwOnly);
  auto verifyVariadicIdx = [&](size_t idx, bool isPack) -> LogicalResult {
    if (idx >= numEl) {
      return emitError() << "variadic " << (isPack ? "pack " : "")
                         << "index must be less than the number of elements: "
                         << idx << " vs. " << numEl;
    }
    if (TypedAttr varDefault = defaultHandler.getDefault(idx)) {
      if (::isa<UnknownAttr>(varDefault))
        return success();
      return emitError() << "default value of variadic "
                         << (isPack ? "pack " : "") << "must be UnknownAttr";
    }
    return success();
  };

  if (packIndex != -1) {
    if (failed(verifyVariadicIdx(packIndex, /*isPack=*/true)))
      return failure();
    if (!origPackConvention)
      return emitError() << "pack convention not specified";
  } else {
    if (origPackConvention)
      return emitError() << "pack convention specified without pack";
  }

  bool seenInferred = true;
  for (auto [idx, pogAttr] : llvm::enumerate(pogs)) {
    if (pogAttr.getPassingKind() == PassingKind::Inferred) {
      if (!seenInferred) {
        return emitError()
               << "'inferred' parameter follows non-inferred parameter";
      }
    } else {
      seenInferred = false;
    }
    if (pogAttr.isVariadic() &&
        failed(verifyVariadicIdx(idx, /*isPack=*/false)))
      return failure();
  }

  return success();
}

PogListAttr PogListAttr::cloneWith(ArrayRef<PogMetadataAttr> pogs) const {
  return PogListAttr::get(getContext(), pogs, getDefaultPos(),
                          getDefaultKwOnly(), getPackIndex(),
                          getOrigPackConvention());
}

bool PogListAttr::isVariadic(size_t idx) const {
  return getPogs()[idx].isVariadic();
}

bool PogListAttr::isPack(size_t idx) const {
  return getPackIndex() == ssize_t(idx);
}

bool PogListAttr::isPosVariadic(size_t idx) const {
  return llvm::is_contained({PassingKind::PosOnly, PassingKind::PosOrKw},
                            getPassingKind(idx)) &&
         isVariadic(idx);
}

bool PogListAttr::isKwVariadic(size_t idx) const {
  return isVariadic(idx) && getPassingKind(idx) == PassingKind::KwOnly;
}

bool PogListAttr::hasVariadic() const {
  return llvm::any_of(
      getPogs(), [](PogMetadataAttr pogAttr) { return pogAttr.isVariadic(); });
}

bool PogListAttr::hasPack() const { return getPackIndex() != -1; }

bool PogListAttr::hasKwVariadics() const {
  for (size_t idx = 0, e = getPogs().size(); idx < e; ++idx)
    if (isKwVariadic(idx))
      return true;
  return false;
}

StringAttr PogListAttr::getName(size_t idx) const {
  return getPogs()[idx].getName();
}

PassingKind PogListAttr::getPassingKind(size_t idx) const {
  return getPogs()[idx].getPassingKind();
}

/// Create a variadic mask of given length from a list of variadic indices.
static SmallVector<bool> toMask(ArrayRef<size_t> indices, size_t length) {
  SmallVector<bool> variadicMask(length, false);
  for (size_t idx : indices)
    variadicMask[idx] = true;
  return variadicMask;
}

SmallVector<PogMetadataAttr>
PogListAttr::toPogs(ArrayRef<StringAttr> names,
                    ArrayRef<PassingKind> passingKinds,
                    ArrayRef<size_t> indices) {
  SmallVector<PogMetadataAttr> pogs;
  for (auto [name, passingKind, isVariadic] :
       llvm::zip(names, passingKinds, toMask(indices, names.size())))
    pogs.push_back(PogMetadataAttr::get(name, passingKind, isVariadic));
  return pogs;
}

//===----------------------------------------------------------------------===//
// FnMetadataAttr
//===----------------------------------------------------------------------===//

FnMetadataAttr FnMetadataAttr::get(MLIRContext *context) {
  auto list = PogListAttr::get(context);
  return FnMetadataAttr::get(context, list, list, 0);
}

FnMetadataAttr FnMetadataAttr::get(MLIRContext *ctx, size_t numParams,
                                   size_t numArgs,
                                   size_t numImplicitLifetimeDecls) {
  SmallVector<PogMetadataAttr> params, args;
  auto normal = PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly,
                                     /*isVariadic=*/false);
  params.resize(numParams, normal);
  args.resize(numArgs, normal);
  return FnMetadataAttr::get(PogListAttr::get(ctx, args),
                             PogListAttr::get(ctx, params),
                             numImplicitLifetimeDecls);
}

FnMetadataAttr FnMetadataAttr::get(PogListAttr argListAttrs,
                                   PogListAttr paramListAttrs,
                                   size_t numImplicitLifetimeDecls) {
  return get(argListAttrs.getContext(), argListAttrs, paramListAttrs,
             numImplicitLifetimeDecls);
}

FnMetadataAttr FnMetadataAttr::get(PogListAttr argListAttrs,
                                   size_t numImplicitLifetimeDecls) {
  MLIRContext *ctx = argListAttrs.getContext();
  return get(ctx, argListAttrs, PogListAttr::get(ctx),
             numImplicitLifetimeDecls);
}

FnMetadataAttrInterface
FnMetadataAttr::getWithBoundPosArgs(size_t numBound) const {
  PogListAttr argListAttrs = getArgListAttrs();

  size_t numPositional = countNumPositional(argListAttrs);
  assert(numBound <= numPositional && "only positional arguments can be bound");

  ArrayRef<PogMetadataAttr> newPogs =
      argListAttrs.getPogs().drop_front(numBound);

  ArrayRef<TypedAttr> newDefaultPosArgs = getDefaultPosArgs();
  size_t numArgs = numPositional - numBound;
  if (numArgs < newDefaultPosArgs.size())
    newDefaultPosArgs = newDefaultPosArgs.take_back(numArgs);

  /// If needed, we adjust the pack index.
  ssize_t packIdx = argListAttrs.getPackIndex();
  if (argListAttrs.hasPack() && packIdx >= ssize_t(numBound))
    packIdx -= numBound;

  auto newArgListAttrs = PogListAttr::get(
      getContext(), newPogs, newDefaultPosArgs, getDefaultKwOnlyArgs(), packIdx,
      argListAttrs.getOrigPackConvention());
  return get(newArgListAttrs, getParamListAttrs(),
             getNumImplicitLifetimeDecls());
}

FnMetadataAttrInterface
FnMetadataAttr::getWithBoundParams(const llvm::BitVector &boundParams) const {
  SmallVector<TypedAttr> newDefaultPosParams;
  SmallVector<TypedAttr> newDefaultKwOnlyParams;
  SmallVector<PogMetadataAttr> newPogs;

  PogListAttr paramListAttr = getParamListAttrs();
  DefaultValueHandler defaultHandler(paramListAttr);
  size_t numParams = boundParams.size();
  for (size_t idx = 0; idx < numParams; ++idx) {
    if (!boundParams[idx]) {
      newPogs.emplace_back(paramListAttr.getPogs()[idx]);
      if (TypedAttr defaultOr = defaultHandler.getPosDefault(idx))
        newDefaultPosParams.emplace_back(defaultOr);
      else if (TypedAttr defaultOr = defaultHandler.getKwOnlyDefault(idx))
        newDefaultKwOnlyParams.emplace_back(defaultOr);
    }
  }

  auto newParamAttrs = PogListAttr::get(
      getContext(), newPogs, newDefaultPosParams, newDefaultKwOnlyParams);
  return get(getArgListAttrs(), newParamAttrs, getNumImplicitLifetimeDecls());
}

/// Get a new metadata attribute for a signature with the given number of
/// positional input parameters prepended to the signature. An additional
/// array of bool corresponding to the variadic mask of the prepended
/// parameters is also required.
FnMetadataAttr
FnMetadataAttr::prependPosParams(size_t numNewParams,
                                 ArrayRef<bool> variadicMask) const {
  assert(variadicMask.size() == numNewParams);
  if (numNewParams == 0)
    return *this;

  auto emptyStr = StringAttr::get(getContext());
  SmallVector<PogMetadataAttr> newPogs =
      llvm::map_to_vector(variadicMask, [&](bool isVariadic) {
        return PogMetadataAttr::get(emptyStr, PassingKind::PosOnly, isVariadic);
      });

  PogListAttr oldParamListAttr = getParamListAttrs();
  SmallVector<PogMetadataAttr> mergedPogs;
  for (size_t iNew = 0, iOld = 0, eOld = oldParamListAttr.getPogs().size(),
              eNew = newPogs.size();
       iOld < eOld || iNew < eNew;) {
    // Put inferred parameters first.
    if (iOld < eOld && oldParamListAttr.getPogs()[iOld].getPassingKind() ==
                           PassingKind::Inferred) {
      mergedPogs.push_back(oldParamListAttr.getPogs()[iOld]);
      iOld++;
    } else if (iNew < eNew) {
      mergedPogs.push_back(newPogs[iNew]);
      iNew++;
    } else {
      mergedPogs.push_back(oldParamListAttr.getPogs()[iOld]);
      iOld++;
    }
  }

  assert(oldParamListAttr.getPackIndex() && "no param packs");
  auto newParamListAttr =
      PogListAttr::get(getContext(), mergedPogs, getDefaultPosParams(),
                       getDefaultKwOnlyParams());
  return get(getArgListAttrs(), newParamListAttr,
             getNumImplicitLifetimeDecls());
}

SmallVector<bool> LIT::getContextualVariadicMask(ArrayRef<Operation *> ops) {
  SmallVector<bool> variadicMask;
  for (Operation *op : ops) {
    // If we are dealing with a struct or trait, we concatenate their variadic
    // masks.
    PogListAttr paramListAttr;
    if (auto structDecl = ::dyn_cast<StructDeclOp>(op))
      paramListAttr = structDecl.getSignature().getParamListAttrs();
    else if (auto traitDecl = ::dyn_cast<TraitDeclOp>(op))
      paramListAttr = traitDecl.getSignature().getParamListAttrs();
    else
      continue;

    for (PogMetadataAttr pogAttr : paramListAttr.getPogs())
      variadicMask.emplace_back(pogAttr.isVariadic());
  }
  return variadicMask;
}

FnMetadataAttrInterface
FnMetadataAttr::prependPosParamsFromOps(ArrayRef<Operation *> ops) const {
  SmallVector<bool> variadicMask = getContextualVariadicMask(ops);
  return prependPosParams(variadicMask.size(), variadicMask);
}

LogicalResult FnMetadataAttr::verifySignature(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<Type> inputParamTypes, ArrayRef<Type> resultParamTypes,
    FunctionType values, ArrayRef<ArgConvention> argConventions,
    FnEffects effects) const {
  if (!resultParamTypes.empty())
    return emitError() << "expected no result parameters";

  PogListAttr paramListAttr = getParamListAttrs();
  if (paramListAttr.getPogs().size() != inputParamTypes.size()) {
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
  if (size_t numArgs = getNumArgs(); numArgs != numInputs) {
    return emitError()
           << "number of arguments does not match number of argument names: "
           << numInputs << " != " << numArgs;
  }

  for (auto [i, argType, conv] :
       llvm::enumerate(values.getInputs(), argConventions)) {
    if (conv == ArgConvention::ByRefResult && i != values.getNumInputs() - 1)
      return emitError() << "'byref_result' argument must be the last argument";
    if (conv == ArgConvention::InitSelf && i != 0)
      return emitError() << "'init_self' argument must be the first argument";

    Type type = argType;

    // Verify variadics.
    if (isPosVarArg(i)) {
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
      if (::isa<RefType>(type))
        continue;
      return emitError() << "argument #" << i << " with convention '"
                         << stringifyEnum(conv)
                         << "' in signature type should be a `!kgen.pointer` "
                            "or `!lit.ref` but got: "
                         << type;
    }
  }

  if (failed(verifyDefaultTypes(
          emitError, getDefaultPosArgs(), getDefaultKwOnlyArgs(),
          getArgListAttrs(), values.getInputs(), "argument", argConventions)) ||
      failed(verifyDefaultTypes(emitError, getDefaultPosParams(),
                                getDefaultKwOnlyParams(), paramListAttr,
                                inputParamTypes, "parameter")))
    return failure();

  return success();
}

size_t FnMetadataAttr::getNumArgs() const {
  return getArgListAttrs().getPogs().size();
}

bool FnMetadataAttr::hasVarArgs() const {
  return getArgListAttrs().hasVariadic();
}

bool FnMetadataAttr::hasPackVarArgs() const {
  return getArgListAttrs().getPackIndex() != -1;
}

bool FnMetadataAttr::hasParamVarArgs() const {
  return getParamListAttrs().hasVariadic();
}

bool FnMetadataAttr::hasKwVarArgs() const {
  return getArgListAttrs().hasKwVariadics();
}

bool FnMetadataAttr::isAnyVarArg(size_t idx) const {
  return getArgListAttrs().isVariadic(idx) || isPackVarArg(idx);
}

bool FnMetadataAttr::isPosVarArg(size_t idx) const {
  return getArgListAttrs().isPosVariadic(idx);
}

bool FnMetadataAttr::isKwVarArg(size_t idx) const {
  return getArgListAttrs().isKwVariadic(idx);
}

bool FnMetadataAttr::isPackVarArg(size_t idx) const {
  return getArgListAttrs().isPack(idx);
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

BindTypeAttr BindTypeAttr::getFromBytecode(TypedAttr typeValue,
                                           ArrayRef<TypedAttr> values,
                                           AnyStructType type) {
  return Base::get(type.getContext(), typeValue, values, type);
}

static ParseResult parseBindTypeParams(AsmParser &p,
                                       SmallVectorImpl<TypedAttr> &values,
                                       TypedAttr typeValue) {
  auto metatype = dyn_cast<AnyStructType>(typeValue.getType());
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
                                   AnyStructType type) {
  auto metatype = ::dyn_cast<AnyStructType>(typeValue.getType());
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
      typeSubst.addInputValue(
          ParamIndexRefAttr::get(expected.size() - 1, expected.back()));
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
static AnyStructType getBindTypeResultType(TypedAttr typeValue,
                                           ArrayRef<TypedAttr> values) {
  auto metatype = cast<AnyStructType>(typeValue.getType());
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
                                   AnyStructType type) {
  // Assume the inputs are verified. If the type value is a `StructType` then
  // bind it and return a type constant.
  if (auto typeCst = dyn_cast<TypeConstantAttr>(typeValue)) {
    if (auto decl = dyn_cast<LIT::StructType>(typeCst.getMlirType())) {
      auto bound =
          LIT::StructType::get(decl.getSymbol(), type.getParamValues(), type);
      // StructType has identical type/value representation.
      return TypeConstantAttr::get(bound, bound, type);
    }
  }
  return BindTypeAttr::Base::get(type.getContext(), typeValue, values, type);
}

TypedAttr BindTypeAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                   MLIRContext *context, TypedAttr typeValue,
                                   ArrayRef<TypedAttr> values,
                                   AnyStructType type) {
  if (failed(verify(emitError, typeValue, values, type)))
    return {};
  return getOrFoldBindType(typeValue, values, type);
}

TypedAttr BindTypeAttr::get(MLIRContext *context, TypedAttr typeValue,
                            ArrayRef<TypedAttr> values, AnyStructType type) {
  return getOrFoldBindType(typeValue, values, type);
}

//===----------------------------------------------------------------------===//
// LifetimeUnionAttr
//===----------------------------------------------------------------------===//

LifetimeUnionAttr
LifetimeUnionAttr::getFromBytecode(ArrayRef<TypedAttr> operands,
                                   LifetimeType type) {
  return Base::get(type.getContext(), operands, type);
}

static bool unionArgCompare(TypedAttr lhs, TypedAttr rhs) {
  // Ignore LifetimeMutCastAttr's for comparison.
  return ParameterAttr::compare(LifetimeMutCastAttr::strip(lhs),
                                LifetimeMutCastAttr::strip(rhs));
}

static void removeDuplicates(SmallVectorImpl<TypedAttr> &operands) {
  if (operands.size() > 1) {
    for (size_t i = 0, e = operands.size() - 1; i != e; ++i) {
      if (operands[i] != operands[i + 1])
        continue;

      operands.erase(operands.begin() + i + 1);
      --e, --i;
    }
  }
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
  removeDuplicates(operands);

  // If no results, return a plain lifetime attr.
  if (operands.empty())
    return LifetimeAttr::get(type);
  if (operands.size() == 1)
    return operands[0];

  auto resultType = ::cast<LifetimeType>(operands[0].getType());
  return LifetimeUnionAttr::Base::get(type.getContext(), operands, resultType);
}

TypedAttr LifetimeUnionAttr::get(MLIRContext *ctx,
                                 ArrayRef<TypedAttr> lifetimes) {
  // In the empty case, the lifetime is mutable.
  if (lifetimes.empty())
    return LifetimeUnionAttr::get({}, LifetimeType::get(ctx, /*mutable=*/true));

  auto getMut = [](TypedAttr lifetime) {
    return ::cast<LifetimeType>(lifetime.getType()).getIsMutable();
  };

  // If all the parametric mutabilities of the lifetimes are the same, then use
  // that mutability. Otherwise, the overall lifetime is immutable.
  TypedAttr mutability = getMut(lifetimes.front());
  bool needMutCast = false;
  for (TypedAttr other : lifetimes.drop_front()) {
    TypedAttr otherMut = getMut(other);
    if (otherMut == mutability)
      continue;
    mutability = BoolAttr::get(ctx, false);
    needMutCast = true;
    break;
  }

  SmallVector<TypedAttr> newLifetimes;
  if (needMutCast) {
    for (TypedAttr lifetime : lifetimes)
      newLifetimes.push_back(LifetimeMutCastAttr::get(lifetime, mutability));
    lifetimes = newLifetimes;
  }

  return LifetimeUnionAttr::get(lifetimes, LifetimeType::get(mutability));
}

//===----------------------------------------------------------------------===//
// LifetimeMutCastAttr
//===----------------------------------------------------------------------===//

LifetimeMutCastAttr LifetimeMutCastAttr::getFromBytecode(TypedAttr operand,
                                                         LifetimeType type) {
  return Base::get(type.getContext(), operand, type);
}

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
         "#lit.lifetime.mutcast always has !lit.lifetime type");
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
// ImplicitLifetimeRefAttr
//===----------------------------------------------------------------------===//

bool ImplicitLifetimeRefAttr::isConstant() const { return false; }

std::optional<bool> ImplicitLifetimeRefAttr::isLessThan(Attribute rhs) const {
  auto ref = ::dyn_cast<ImplicitLifetimeRefAttr>(rhs);
  if (!ref)
    return false;
  return std::make_tuple(getDepth(), getIndex()) <
         std::make_tuple(ref.getDepth(), ref.getIndex());
}

IndexRefAttrInterface
ImplicitLifetimeRefAttr::replace(size_t depth, size_t index,
                                 ArrayRef<Attribute> attrs,
                                 ArrayRef<Type> types) const {
  assert(attrs.empty() && types.size() == 1);
  return ImplicitLifetimeRefAttr::get(depth, index, types.front());
}

//===----------------------------------------------------------------------===//
// LifetimeSetAttr
//===----------------------------------------------------------------------===//

LifetimeSetAttr LifetimeSetAttr::getFromBytecode(ArrayRef<TypedAttr> operands,
                                                 LifetimeSetType type) {
  return Base::get(type.getContext(), operands, type);
}

LifetimeSetAttr LifetimeSetAttr::get(MLIRContext *ctx,
                                     ArrayRef<TypedAttr> operands,
                                     LifetimeSetType type) {
  return get(operands, type);
}

LifetimeSetAttr LifetimeSetAttr::get(MLIRContext *ctx,
                                     ArrayRef<TypedAttr> operands) {
  return get(operands, LifetimeSetType::get(ctx));
}

LifetimeSetAttr LifetimeSetAttr::get(ArrayRef<TypedAttr> operands,
                                     LifetimeSetType type) {
  SmallVector<TypedAttr> newOperands;
  for (TypedAttr operand : operands) {
    // Recursively flatten sets into each other. We know this one is already
    // flattened.
    if (auto set = ::dyn_cast<LifetimeSetAttr>(operand)) {
      llvm::append_range(newOperands, set.getOperands());
      continue;
    }
    // This doesn't carry any information. Just drop it.
    if (::isa<LifetimeAttr>(operand))
      continue;
    // Break up unions into their constituents without mutcasts.
    if (auto unionAttr = ::dyn_cast<LifetimeUnionAttr>(operand)) {
      for (TypedAttr lifetime : unionAttr.getOperands())
        newOperands.push_back(LifetimeMutCastAttr::strip(lifetime));
      continue;
    }
    newOperands.push_back(LifetimeMutCastAttr::strip(operand));
  }

  // Now sort the operands by mutability and value.
  llvm::stable_sort(newOperands, [&](TypedAttr lhs, TypedAttr rhs) {
    TypedAttr lhsMut = ::cast<LifetimeType>(lhs.getType()).isMutable();
    TypedAttr rhsMut = ::cast<LifetimeType>(rhs.getType()).isMutable();
    if (ParameterAttr::compare(lhsMut, rhsMut))
      return true;
    if (ParameterAttr::compare(rhsMut, lhsMut))
      return false;
    return ParameterAttr::compare(lhs, rhs);
  });
  removeDuplicates(newOperands);

  return Base::get(type.getContext(), newOperands, type);
}

//===----------------------------------------------------------------------===//
// LifetimeSetUnionAttr
//===----------------------------------------------------------------------===//

LifetimeSetUnionAttr LifetimeSetUnionAttr::getFromBytecode(TypedAttr value,
                                                           LifetimeType type) {
  return Base::get(type.getContext(), value, type);
}

TypedAttr LifetimeSetUnionAttr::get(TypedAttr value, LifetimeType type) {
  // Fold `set.union(set) -> union`.
  if (auto set = ::dyn_cast<LifetimeSetAttr>(value)) {
    return LifetimeMutCastAttr::get(
        LifetimeUnionAttr::get(type.getContext(), set.getOperands()), type);
  }
  return Base::get(type.getContext(), value, type);
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

ErrorOr<TypedAttr> LIT::createUninitializedValueOf(Type type,
                                                   InterpreterState &state) {
  auto declRef = dyn_cast<StructType>(type);
  if (!declRef)
    return {UnknownAttr::get(type)};
  SmallVector<std::tuple<StringAttr, TypedAttr>> values;
  auto decl = cast_or_null<StructDeclOp>(
      state.lookupTypeDefinition(declRef.getSymbol()));
  if (!decl)
    return Error("didn't find struct decl");
  ParameterEvaluator evaluator(decl.getParams(), declRef.getParamValues());
  for (StructFieldOp field : decl.getFieldDecls()) {
    Type type = evaluator.getReboundType(field.getType());
    ErrorOr<TypedAttr> value = createUninitializedValueOf(type, state);
    if (value.isError())
      return value.takeError();
    values.emplace_back(field.getNameAttr(), value.takeValue());
  }
  return LITStructAttr::get(values, declRef);
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

LIT::StructExtractAttr
LIT::StructExtractAttr::getFromBytecode(TypedAttr structValue, StringAttr field,
                                        Type type) {
  return Base::get(type.getContext(), structValue, field, type);
}

TypedAttr LIT::StructExtractAttr::get(TypedAttr structValue,
                                      StructFieldOp fieldOp) {
  auto structType = ::cast<StructType>(structValue.getType());
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
// StructGERAttr
//===----------------------------------------------------------------------===//

LogicalResult
StructGERAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      TypedAttr value, StringAttr field, Type type) {
  if (::isa<SymbolicPointerAttr, StructGERAttr>(value))
    return success();
  return emitError() << "base value must be a SymbolicPointerAttr or "
                        "StructGERAttr, but got "
                     << value;
}

//===----------------------------------------------------------------------===//
// RefPackAttr
//===----------------------------------------------------------------------===//

static ParseResult parsePackElements(AsmParser &p,
                                     SmallVector<TypedAttr> &values,
                                     RefPackType packType) {
  auto variadic = packType.getVariadicIfResolved();
  if (!variadic)
    return p.emitError(p.getCurrentLocation())
           << "lit.ref.pack attribute expected a variadic constant, but got "
           << packType.getVariadic();

  // Parse one element for each type in the list.
  return failableInterleave(
      variadic.getValues(),
      [&](TypedAttr eltType) {
        return parseParamValue(
            p, values.emplace_back(),
            packType.getElementRefTypeFor(ParamRefType::get(eltType)));
      },
      [&] { return p.parseComma(); });
}

static void printPackElements(AsmPrinter &p, ArrayRef<TypedAttr> values,
                              RefPackType type) {
  llvm::interleaveComma(values, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
}

OptionalParseResult RefPackType::parseValue(AsmParser &p,
                                            TypedAttr &value) const {
  if (failed(p.parseOptionalLess()))
    return std::nullopt;
  SmallVector<TypedAttr> values;
  if (failed(parsePackElements(p, values, *this)))
    return failure();

  value = RefPackAttr::get(values, *this);
  return p.parseGreater();
}

LogicalResult RefPackType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto packAttr = ::dyn_cast<RefPackAttr>(value);
  if (!packAttr)
    return failure();

  p << "<";
  printPackElements(p, packAttr.getValues(), *this);
  p << ">";
  return success();
}

LogicalResult RefPackAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<TypedAttr> values,
                                  RefPackType packType) {
  auto variadic = packType.getVariadicIfResolved();
  if (!variadic)
    return emitError()
           << "pack attribute expected a variadic constant, but got "
           << packType.getVariadic();

  ArrayRef<TypedAttr> expected = variadic.getValues();
  if (values.size() != expected.size())
    return emitError() << "pack attribute type requires " << expected.size()
                       << " elements, but got " << values.size();

  // Check that the element constants have the right types.
  for (auto [i, value, type] : llvm::enumerate(values, expected)) {
    auto eltType = packType.getElementRefTypeFor(ParamRefType::get(type));
    if (value.getType() != eltType)
      return emitError() << "pack attribute element #" << i << " has type "
                         << value.getType() << " but expected " << type;
  }
  return success();
}

bool RefPackAttr::isConstant() const {
  return llvm::all_of(getValues(), ParameterAttr::isSimpleConstant);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/LITDialect/LITAttrs.cpp.inc"

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
#include "llvm/ADT/SmallPtrSet.h"
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

PogListAttr PogListAttr::get(MLIRContext *context, size_t numPogs) {
  SmallVector<PogMetadataAttr> pogs;
  for (size_t i = 0; i != numPogs; ++i)
    pogs.push_back(PogMetadataAttr::get(context));
  return PogListAttr::get(context, pogs);
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

  for (auto [idx, pogAttr] : llvm::enumerate(pogs)) {
    if (pogAttr.getPassingKind() == PassingKind::Inferred && idx != 0 &&
        pogs[idx - 1].getPassingKind() != PassingKind::Inferred) {
      return emitError()
             << "'inferred' parameter follows non-inferred parameter";
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
  for (size_t idx = 0, e = size(); idx != e; ++idx)
    if (isKwVariadic(idx))
      return true;
  return false;
}

bool PogListAttr::hasInferredParams() const {
  ArrayRef<PogMetadataAttr> params = getPogs();
  return !params.empty() &&
         params.front().getPassingKind() == PassingKind::Inferred;
}

StringAttr PogListAttr::getName(size_t idx) const {
  return getPogs()[idx].getName();
}

PassingKind PogListAttr::getPassingKind(size_t idx) const {
  return getPogs()[idx].getPassingKind();
}

/// Return the number of leading "implicit" parameters by PassingKinds.
size_t PogListAttr::getNumImplicit() const {
  size_t numImplicit = 0;
  for (auto pog : getPogs()) {
    if (pog.getPassingKind() == PassingKind::Implicit)
      ++numImplicit;
    else
      break;
  }
  return numImplicit;
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

LogicalResult
PogListAttr::verifyGenerator(function_ref<InFlightDiagnostic()> emitError,
                             ArrayRef<Type> inputParamTypes, Type body) const {
  if (size() != inputParamTypes.size()) {
    return emitError()
           << "number of pog names doesn't match number of pog types";
  }

  if (failed(verifyDefaultTypes(emitError, getDefaultPos(), getDefaultKwOnly(),
                                *this, inputParamTypes, "pog")))
    return failure();
  return success();
}

void PogListAttr::printGenerator(AsmPrinter &p, GeneratorType generator) const {
  ArrayRef<Type> paramTypes = generator.getInputParamTypes();

  p << "!lit.generator<";
  if (auto sig = ::dyn_cast<FnType>(generator.getBody())) {
    // Special case for FnType Generators: Skip the empty angle brackets, and
    // print the FnType without any additional wrapper.
    if (!paramTypes.empty())
      LIT::printOptionalParamSignature(p, paramTypes, *this);
    printFnType(p, sig);
  } else {
    if (paramTypes.empty())
      p << "<>";
    else
      LIT::printOptionalParamSignature(p, paramTypes, *this);
    printKGENType(p, generator.getBody());
  }
  p << '>';
}

GeneratorMetadataAttrInterface
PogListAttr::getWithBoundParams(const llvm::BitVector &boundParams) const {
  SmallVector<TypedAttr> newDefaultPosParams;
  SmallVector<TypedAttr> newDefaultKwOnlyParams;
  SmallVector<PogMetadataAttr> newPogs;

  DefaultValueHandler defaultHandler(*this);
  size_t numParams = boundParams.size();
  for (size_t idx = 0; idx < numParams; ++idx) {
    if (!boundParams[idx]) {
      newPogs.emplace_back(getPogs()[idx]);
      if (TypedAttr defaultOr = defaultHandler.getPosDefault(idx))
        newDefaultPosParams.emplace_back(defaultOr);
      else if (TypedAttr defaultOr = defaultHandler.getKwOnlyDefault(idx))
        newDefaultKwOnlyParams.emplace_back(defaultOr);
    }
  }

  return PogListAttr::get(getContext(), newPogs, newDefaultPosParams,
                          newDefaultKwOnlyParams);
}

/// Get a new metadata attribute for a generator with the given number of
/// positional input parameters prepended to the generator. An additional
/// array of bool corresponding to the variadic mask of the prepended
/// parameters is also required.
PogListAttr PogListAttr::prependPosParams(size_t numNewParams,
                                          ArrayRef<bool> variadicMask) const {
  assert(variadicMask.size() == numNewParams);
  if (numNewParams == 0)
    return *this;

  auto emptyStr = StringAttr::get(getContext());
  SmallVector<PogMetadataAttr> newPogs =
      llvm::map_to_vector(variadicMask, [&](bool isVariadic) {
        return PogMetadataAttr::get(emptyStr, PassingKind::PosOnly, isVariadic);
      });

  SmallVector<PogMetadataAttr> mergedPogs;
  for (size_t iNew = 0, iOld = 0, eOld = size(), eNew = newPogs.size();
       iOld < eOld || iNew < eNew;) {
    // Put inferred parameters first.
    if (iOld < eOld && getPassingKind(iOld) == PassingKind::Inferred) {
      mergedPogs.push_back(getPogs()[iOld]);
      iOld++;
    } else if (iNew < eNew) {
      mergedPogs.push_back(newPogs[iNew]);
      iNew++;
    } else {
      mergedPogs.push_back(getPogs()[iOld]);
      iOld++;
    }
  }

  assert(getPackIndex() && "no param packs");
  return PogListAttr::get(getContext(), mergedPogs, getDefaultPos(),
                          getDefaultKwOnly());
}

GeneratorMetadataAttrInterface
PogListAttr::prependPosParamsFromOps(ArrayRef<Operation *> ops) const {
  SmallVector<bool> variadicMask = getContextualVariadicMask(ops);
  return prependPosParams(variadicMask.size(), variadicMask);
}

//===----------------------------------------------------------------------===//
// FnMetadataAttr
//===----------------------------------------------------------------------===//

static ParseResult parseFnMetadataOrigins(AsmParser &p, TypedAttr &origins) {
  auto type = OriginSetType::get(p.getContext());
  if (failed(p.parseOptionalComma())) {
    origins = OriginSetAttr::get({}, type);
    return success();
  }
  return parseParamValue(p, origins, type);
}

static void printFnMetadataOrigins(AsmPrinter &p, TypedAttr origins) {
  if (isEmptyOriginSet(origins))
    return;
  p << ", ";
  printParamValue(p, origins);
}

FnMetadataAttr FnMetadataAttr::get(MLIRContext *context) {
  auto list = PogListAttr::get(context);
  return FnMetadataAttr::get(
      list, /*numImplicitOriginDecls=*/0, /*captureOrigins=*/nullptr,
      /*isNestedOriginExclusivityCheckingDisabled=*/false);
}

FnMetadataAttr FnMetadataAttr::get(MLIRContext *ctx, size_t numArgs,
                                   size_t numImplicitOriginDecls) {
  SmallVector<PogMetadataAttr> args;
  auto normal = PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly,
                                     /*isVariadic=*/false);
  args.resize(numArgs, normal);
  return FnMetadataAttr::get(
      PogListAttr::get(ctx, args), numImplicitOriginDecls,
      /*captureOrigins=*/nullptr,
      /*isNestedOriginExclusivityCheckingDisabled=*/false);
}

FnMetadataAttr FnMetadataAttr::get(PogListAttr argListAttrs,
                                   size_t numImplicitOriginDecls,
                                   TypedAttr captureOrigins,
                                   bool nestedOriginFlag) {
  MLIRContext *ctx = argListAttrs.getContext();
  if (!captureOrigins)
    captureOrigins = OriginSetAttr::get(ctx, {});
  return get(ctx, argListAttrs, numImplicitOriginDecls, captureOrigins,
             nestedOriginFlag);
}

FnMetadataAttr FnMetadataAttr::get(PogListAttr argListAttrs,
                                   size_t numImplicitOriginDecls) {
  return get(argListAttrs, numImplicitOriginDecls, /*captureOrigins=*/nullptr,
             /*isNestedOriginExclusivityCheckingDisabled=*/false);
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
  return get(newArgListAttrs, getNumImplicitOriginDecls(), getCaptureOrigins(),
             getIsNestedOriginExclusivityCheckingDisabled());
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

LogicalResult FnMetadataAttr::verifyFuncType(
    function_ref<InFlightDiagnostic()> emitError, FunctionType values,
    ArrayRef<ArgConvention> argConventions, FnEffects effects) const {
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
    if (hasAddress(conv)) {
      if (::isa<RefType>(type))
        continue;
      return emitError()
             << "argument #" << i << " with convention '" << stringifyEnum(conv)
             << "' in signature type should be a `!lit.ref` but got: " << type;
    }
  }

  if (failed(verifyDefaultTypes(
          emitError, getDefaultPosArgs(), getDefaultKwOnlyArgs(),
          getArgListAttrs(), values.getInputs(), "argument", argConventions)))
    return failure();
  return success();
}

FnMetadataAttr FnMetadataAttr::addCaptureOrigins(TypedAttr origins) {
  auto type = OriginType::get(getContext(), /*isMutable=*/true);
  SmallVector<TypedAttr> originUnion{
      OriginSetUnionAttr::get(getCaptureOrigins(), type),
      OriginSetUnionAttr::get(origins, type)};
  return get(getArgListAttrs(), getNumImplicitOriginDecls(),
             OriginSetAttr::get(getContext(), originUnion),
             getIsNestedOriginExclusivityCheckingDisabled());
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

void FnMetadataAttr::printFuncType(AsmPrinter &p, FuncType sig) const {
  p << "!lit.fn<";
  auto signature = ::cast<FnType>(sig);
  printFnType(p, signature);
  p << '>';
}

//===----------------------------------------------------------------------===//
// UnboundMLIROperationAttr
//===----------------------------------------------------------------------===//

Type UnboundMLIROperationAttr::getType() const {
  return mlir::NoneType::get(getContext());
}

//===----------------------------------------------------------------------===//
// UnpackedAttr
//===----------------------------------------------------------------------===//

static ParseResult parseUnpackKind(AsmParser &p, bool &kwOnly) {
  if (succeeded(p.parseOptionalKeyword("kw"))) {
    kwOnly = true;
    return success();
  }
  kwOnly = false;
  return p.parseKeyword("pos");
}

static void printUnpackKind(AsmPrinter &p, bool kwOnly) {
  p << (kwOnly ? "kw" : "pos");
}

//===----------------------------------------------------------------------===//
// BindTypeAttr
//===----------------------------------------------------------------------===//

BindTypeAttr BindTypeAttr::getFromBytecode(TypedAttr typeValue,
                                           ArrayRef<TypedAttr> values,
                                           StructMetaType type) {
  return Base::get(type.getContext(), typeValue, values, type);
}

static ParseResult parseBindTypeParams(AsmParser &p,
                                       SmallVectorImpl<TypedAttr> &values,
                                       TypedAttr typeValue) {
  auto metatype = dyn_cast<StructMetaType>(typeValue.getType());
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
                                   StructMetaType type) {
  auto metatype = ::dyn_cast<StructMetaType>(typeValue.getType());
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
static StructMetaType getBindTypeResultType(TypedAttr typeValue,
                                            ArrayRef<TypedAttr> values) {
  auto metatype = cast<StructMetaType>(typeValue.getType());
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
                                   StructMetaType type) {
  // Assume the inputs are verified. If the type value is a `StructType` then
  // bind it and return a type constant.
  if (auto typeCst = dyn_cast<TypeParamAttr>(typeValue)) {
    if (auto decl = dyn_cast<LIT::StructType>(typeCst.getMlirType())) {
      auto bound = LIT::StructType::get(decl.getSymbol(), type.getParamValues(),
                                        type.getSignature());
      // StructType has identical type/value representation.
      return TypeParamAttr::get(bound, bound, type);
    }
  }
  return BindTypeAttr::Base::get(type.getContext(), typeValue, values, type);
}

TypedAttr BindTypeAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                   MLIRContext *context, TypedAttr typeValue,
                                   ArrayRef<TypedAttr> values,
                                   StructMetaType type) {
  if (failed(verify(emitError, typeValue, values, type)))
    return {};
  return getOrFoldBindType(typeValue, values, type);
}

TypedAttr BindTypeAttr::get(MLIRContext *context, TypedAttr typeValue,
                            ArrayRef<TypedAttr> values, StructMetaType type) {
  return getOrFoldBindType(typeValue, values, type);
}

//===----------------------------------------------------------------------===//
// StaticOriginAttr
//===----------------------------------------------------------------------===//

// Origins are treated as simple constants, allowing folding of function calls.
bool StaticOriginAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// OriginUnionAttr
//===----------------------------------------------------------------------===//

OriginUnionAttr OriginUnionAttr::getFromBytecode(ArrayRef<TypedAttr> operands,
                                                 OriginType type) {
  return Base::get(type.getContext(), operands, type);
}

static bool unionArgCompare(TypedAttr lhs, TypedAttr rhs) {
  // Ignore OriginMutCastAttr's for comparison.
  return ParameterAttr::compare(OriginMutCastAttr::strip(lhs),
                                OriginMutCastAttr::strip(rhs));
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

TypedAttr OriginUnionAttr::get(ArrayRef<TypedAttr> operandsIn,
                               OriginType type) {

  // Canonicalize the operands, sorting by name/index and eliminating raw
  // #lit.any.origin members.
  llvm::SetVector<TypedAttr> operandSet;

  // Preprocess operands.
  for (TypedAttr operand : operandsIn) {
    assert(operand.getType() == type &&
           "all members of a origin union must have matching type");
    // Union{a, b, #lit.any.origin} => #lit.any.origin since #lit.origin
    // represents "possibly anything".
    if (::isa<AnyOriginAttr>(operand))
      return operand;

    // Flatten any of the same operation into the operand list:
    // `(union x, (union y, z))` => `(union x, y, z)`.
    if (auto subexpr = ::dyn_cast<OriginUnionAttr>(operand)) {
      operandSet.insert(subexpr.getOperands().begin(),
                        subexpr.getOperands().end());
    } else {
      operandSet.insert(operand);
    }
  }

  SmallVector<TypedAttr> operands = operandSet.takeVector();

  // Impose an ordering on the operands, sorting by name where possible - but
  // predictably ordered w.r.t. each other.
  llvm::stable_sort(operands, unionArgCompare);

  // If one result, use it.
  if (operands.size() == 1)
    return operands[0];

  // Otherwise, for multiple elements actually form a union.
  return OriginUnionAttr::Base::get(type.getContext(), operands, type);
}

TypedAttr OriginUnionAttr::get(MLIRContext *ctx, ArrayRef<TypedAttr> origins) {
  // In the empty case, the origin is immutable.
  if (origins.empty())
    return OriginUnionAttr::get(OriginType::get(ctx, /*mutable=*/false));

  auto getMut = [](TypedAttr origin) {
    return ::cast<OriginType>(origin.getType()).getIsMutable();
  };

  // The resultant mutability is the worst case of the input mutabilities.
  TypedAttr mutability = getMut(origins.front());
  bool needMutCast = false;
  for (TypedAttr other : origins.drop_front()) {
    TypedAttr otherMut = getMut(other);
    if (otherMut == mutability)
      continue;
    mutability = ParamOperatorAttr::get(POC::And, mutability, otherMut);
    needMutCast = true;
  }

  SmallVector<TypedAttr> newOrigins;
  if (needMutCast) {
    for (TypedAttr origin : origins)
      newOrigins.push_back(OriginMutCastAttr::get(origin, mutability));
    origins = newOrigins;
  }

  return OriginUnionAttr::get(origins, OriginType::get(mutability));
}

// Origins unions are simple constants if all their elements are.
bool OriginUnionAttr::isConstant() const {
  for (auto op : getOperands())
    if (!ParameterAttr::isSimpleConstant(op))
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// OriginMutCastAttr
//===----------------------------------------------------------------------===//

bool OriginMutCastAttr::isLessThan(Attribute rhs) const {
  // Compare the underlying references.
  auto cast = ::cast<OriginMutCastAttr>(rhs);
  return ParameterAttr::compare(getOperand(), cast.getOperand());
}

OriginMutCastAttr OriginMutCastAttr::getFromBytecode(TypedAttr operand,
                                                     OriginType type) {
  return Base::get(type.getContext(), operand, type);
}

TypedAttr OriginMutCastAttr::get(TypedAttr operand, TypedAttr isMutable) {
  auto curTy = ::cast<OriginType>(operand.getType());
  if (curTy.isMutable() == isMutable)
    return operand;

  // Fold some common cases to canonicalize.
  // mutcast(mutcast(x)) -> mutcast(x), often canceling out.
  if (auto mutCast = ::dyn_cast<OriginMutCastAttr>(operand))
    return get(mutCast.getOperand(), isMutable);

  // Singletons don't need a cast, just form one with the new mutability.
  if (::isa<AnyOriginAttr>(operand))
    return AnyOriginAttr::get(isMutable);

  // Push into union so it cancels out.
  if (auto unionAttr = ::dyn_cast<OriginUnionAttr>(operand)) {
    SmallVector<TypedAttr> elts;
    for (auto elt : unionAttr.getOperands())
      elts.push_back(OriginMutCastAttr::get(elt, isMutable));
    return OriginUnionAttr::get(elts, OriginType::get(isMutable));
  }

  auto context = curTy.getContext();
  return OriginMutCastAttr::Base::get(context, operand,
                                      OriginType::get(isMutable));
}

TypedAttr OriginMutCastAttr::get(TypedAttr operand, Type type) {
  assert(::isa<OriginType>(type) && ::isa<OriginType>(operand.getType()) &&
         "#lit.origin.mutcast always has !lit.origin type");
  if (operand.getType() == type)
    return operand;
  return get(operand, ::cast<OriginType>(type).isMutable());
}

TypedAttr OriginMutCastAttr::get(TypedAttr operand, bool isMutable) {
  auto operandType = ::cast<OriginType>(operand.getType());
  if (operandType.isMutableKnown(isMutable))
    return operand;
  return get(operand, BoolAttr::get(operand.getContext(), isMutable));
}

// Casts are simple constants if their base is.
bool OriginMutCastAttr::isConstant() const {
  return ParameterAttr::isSimpleConstant(getOperand());
}

//===----------------------------------------------------------------------===//
// OriginFieldAttr
//===----------------------------------------------------------------------===//

TypedAttr OriginFieldAttr::get(TypedAttr structOrigin, StringAttr field) {
  // Check to see if there are any permutations we can fold.

  // If we have the global mutable origin, treat it conservatively by
  // returning the global mutable origin.  It isn't wise to try to derive
  // information from something where origins have been casted away.
  if (::isa<AnyOriginAttr>(structOrigin))
    return structOrigin;

  // We push any mutability casts outside of ourselves.
  //     mutcast(x).myfield => mutcast(x.myfield)
  if (auto mutCast = ::dyn_cast<OriginMutCastAttr>(structOrigin)) {
    auto inner = OriginFieldAttr::get(mutCast.getOperand(), field);
    return OriginMutCastAttr::get(inner, mutCast.getType());
  }

  // We push this inside a origin.union as well, so we get the union on the
  // outside.
  if (auto unionAttr = ::dyn_cast<OriginUnionAttr>(structOrigin)) {
    SmallVector<TypedAttr> elts;
    for (auto elt : unionAttr.getOperands())
      elts.push_back(OriginFieldAttr::get(elt, field));
    // Field accesses don't affect mutability, so we use the same type.
    return OriginUnionAttr::get(elts, unionAttr.getType());
  }

  // The structOriginRef must have a OriginType, which we propagate.
  auto structType = ::cast<OriginType>(structOrigin.getType());
  return OriginFieldAttr::Base::get(structOrigin.getContext(), structOrigin,
                                    field, structType);
}

OriginFieldAttr OriginFieldAttr::getFromBytecode(TypedAttr structOrigin,
                                                 StringAttr field,
                                                 OriginType type) {
  return Base::get(type.getContext(), structOrigin, field, type);
}

// Fields are simple constants if their base is.
bool OriginFieldAttr::isConstant() const {
  return ParameterAttr::isSimpleConstant(getBase());
}

bool OriginFieldAttr::isLessThan(Attribute rhs) const {
  auto rhsField = ::cast<OriginFieldAttr>(rhs);
  if (getBase() == rhsField.getBase())
    return getField().getValue() < rhsField.getField().getValue();
  return ParameterAttr::compare(getBase(), rhsField.getBase());
}

//===----------------------------------------------------------------------===//
// IndirectOriginAttr
//===----------------------------------------------------------------------===//

TypedAttr IndirectOriginAttr::get(TypedAttr baseOrigin) {
  // Check to see if there are any permutations we can fold.

  // If we have the global mutable origin, treat it conservatively by
  // returning the global mutable origin.  It isn't wise to try to derive
  // information from something where origins have been casted away.
  if (::isa<AnyOriginAttr>(baseOrigin))
    return baseOrigin;

  // We push any mutability casts outside of ourselves.
  //     mutcast(x)[] => mutcast(x[])
  if (auto mutCast = ::dyn_cast<OriginMutCastAttr>(baseOrigin)) {
    auto inner = IndirectOriginAttr::get(mutCast.getOperand());
    return OriginMutCastAttr::get(inner, mutCast.getType());
  }

  // We push this inside a origin.union as well, so we get the union on the
  // outside.
  if (auto unionAttr = ::dyn_cast<OriginUnionAttr>(baseOrigin)) {
    SmallVector<TypedAttr> elts;
    for (auto elt : unionAttr.getOperands())
      elts.push_back(IndirectOriginAttr::get(elt));
    // Field accesses don't affect mutability, so we use the same type.
    return OriginUnionAttr::get(elts, unionAttr.getType());
  }

  // The result type is the same as baseOrigin's.
  auto baseType = ::cast<OriginType>(baseOrigin.getType());
  return IndirectOriginAttr::Base::get(baseOrigin.getContext(), baseOrigin,
                                       baseType);
}

// Fields are simple constants if their base is.
bool IndirectOriginAttr::isConstant() const {
  return ParameterAttr::isSimpleConstant(getBase());
}

//===----------------------------------------------------------------------===//
// ImplicitOriginRefAttr
//===----------------------------------------------------------------------===//

// Origins are treated as simple constants, allowing folding of function calls.
bool ImplicitOriginRefAttr::isConstant() const { return true; }

// Make sure that these are implicitly sorted w.r.t. each other when KGEN
// canonicalizes them.
bool ImplicitOriginRefAttr::isLessThan(Attribute rhs) const {
  auto ref = ::cast<ImplicitOriginRefAttr>(rhs);
  return std::make_tuple(getDepth(), getIndex()) <
         std::make_tuple(ref.getDepth(), ref.getIndex());
}

IndexRefAttrInterface
ImplicitOriginRefAttr::replace(size_t depth, size_t index,
                               ArrayRef<Attribute> attrs,
                               ArrayRef<Type> types) const {
  assert(attrs.empty() && types.size() == 1);
  return ImplicitOriginRefAttr::get(depth, index, types.front());
}

//===----------------------------------------------------------------------===//
// OriginSetAttr
//===----------------------------------------------------------------------===//

OriginSetAttr OriginSetAttr::getFromBytecode(ArrayRef<TypedAttr> operands,
                                             OriginSetType type) {
  return Base::get(type.getContext(), operands, type);
}

TypedAttr OriginSetAttr::get(MLIRContext *ctx, ArrayRef<TypedAttr> operands,
                             OriginSetType type) {
  return get(operands, type);
}

TypedAttr OriginSetAttr::get(MLIRContext *ctx, ArrayRef<TypedAttr> operands) {
  return get(operands, OriginSetType::get(ctx));
}

TypedAttr OriginSetAttr::get(ArrayRef<TypedAttr> operands, OriginSetType type) {
  SmallVector<TypedAttr> newOperands;
  for (TypedAttr operand : operands) {
    // If we have the global mutable origin, treat it conservatively by
    // returning the global mutable origin.  It isn't wise to try to derive
    // information from something where origins have been casted away.
    if (::isa<AnyOriginAttr>(operand)) {
      newOperands.push_back(operand);
      continue;
    }
    // Break up unions into their constituents without mutcasts.
    if (auto unionAttr = ::dyn_cast<OriginUnionAttr>(operand)) {
      for (TypedAttr origin : unionAttr.getOperands())
        newOperands.push_back(OriginMutCastAttr::strip(origin));
      continue;
    }
    newOperands.push_back(OriginMutCastAttr::strip(operand));
  }

  // Now sort the operands by mutability and value.
  llvm::stable_sort(newOperands, [&](TypedAttr lhs, TypedAttr rhs) {
    TypedAttr lhsMut = ::cast<OriginType>(lhs.getType()).isMutable();
    TypedAttr rhsMut = ::cast<OriginType>(rhs.getType()).isMutable();
    if (ParameterAttr::compare(lhsMut, rhsMut))
      return true;
    if (ParameterAttr::compare(rhsMut, lhsMut))
      return false;
    return ParameterAttr::compare(lhs, rhs);
  });
  removeDuplicates(newOperands);

  // If we find a set union and there is only one operand, collapse the union.
  if (newOperands.size() == 1)
    if (auto setUnion = ::dyn_cast<OriginSetUnionAttr>(newOperands.front()))
      return setUnion.getValue();

  return Base::get(type.getContext(), newOperands, type);
}

//===----------------------------------------------------------------------===//
// OriginSetUnionAttr
//===----------------------------------------------------------------------===//

OriginSetUnionAttr OriginSetUnionAttr::getFromBytecode(TypedAttr value,
                                                       OriginType type) {
  return Base::get(type.getContext(), value, type);
}

TypedAttr OriginSetUnionAttr::get(TypedAttr value, OriginType type) {
  // Fold `set.union(set) -> union`.
  if (auto set = ::dyn_cast<OriginSetAttr>(value)) {
    return OriginMutCastAttr::get(
        OriginUnionAttr::get(type.getContext(), set.getOperands()), type);
  }
  return Base::get(type.getContext(), value, type);
}

//===----------------------------------------------------------------------===//
// LITStructAttr
//===----------------------------------------------------------------------===//

TypedAttr LITStructAttr::get(MLIRContext *ctx,
                             ArrayRef<std::tuple<StringAttr, TypedAttr>> values,
                             StructType type) {
  // If we are forming a struct from an single extract from the same type,
  // canonicalize it away so we get type equality for important types like
  // Origin and AddressSpace etc.
  if (values.size() == 1) {
    auto [fieldName, value] = values[0];
    if (auto extract = ::dyn_cast<LIT::StructExtractAttr>(value))
      if (extract.getField() == fieldName &&
          extract.getStructValue().getType() == type)
        return extract.getStructValue();
  }

  return LITStructAttr::Base::get(ctx, values, type);
}

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

bool LIT::StructExtractAttr::isConstant() const { return false; }

bool LIT::StructExtractAttr::isLessThan(Attribute rhs) const {
  // Compare the underlying references if the fields are the same.
  auto rhsExtract = ::cast<LIT::StructExtractAttr>(rhs);
  if (getField() == rhsExtract.getField())
    return ParameterAttr::compare(getStructValue(),
                                  rhsExtract.getStructValue());
  return getField().getValue() < rhsExtract.getField().getValue();
}

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

  // Fold UnknownAttr for convenience.
  if (::isa<UnknownAttr>(structValue))
    return UnknownAttr::get(resultType);

  return Base::get(context, structValue, field, resultType);
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
            packType.getElementRefTypeFor(ParamType::get(eltType)));
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
    auto eltType = packType.getElementRefTypeFor(ParamType::get(type));
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

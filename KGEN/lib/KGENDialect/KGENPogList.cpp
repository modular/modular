//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENPogUtils.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/STLExtras.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// PogListAttr builders
//===----------------------------------------------------------------------===//

// `getChecked` wrappers matching each user-defined `get` builder below.
// TableGen auto-generates the body of `getChecked` only for the canonical
// (all-fields) builder; the user-defined builders get only a declaration in
// the generated header.  The KGEN dialect's nanobind bindings
// (`KGEN/include/KGEN/KGENDialect/Nanobind/KGENAttrs.cpp.inc`) reference
// `getChecked` for every builder, so each overload must have a definition
// that links into `max._core`.  Each wrapper massages its arguments the same
// way its `get` counterpart does and then forwards to the canonical
// `getChecked`, so MLIR's verifier runs on the constructed attribute.
// TODO(MOCO-4014): Drop these wrappers once the nanobind generator stops
// referencing per-builder `getChecked` (or TableGen synthesizes them).
PogListAttr PogListAttr::get(MLIRContext *context) {
  return PogListAttr::get(context, /*pogs=*/{}, /*bodyConstraints=*/{},
                          ArgConvention::ByRefError);
}

PogListAttr
PogListAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        MLIRContext *context) {
  return PogListAttr::getChecked(emitError, context, /*pogs=*/{},
                                 /*bodyConstraints=*/{},
                                 ArgConvention::ByRefError);
}

PogListAttr PogListAttr::get(MLIRContext *context, size_t numPogs) {
  return PogListAttr::get(context, numPogs, /*bodyConstraints=*/{});
}

PogListAttr
PogListAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        MLIRContext *context, size_t numPogs) {
  return PogListAttr::getChecked(emitError, context, numPogs,
                                 /*bodyConstraints=*/{});
}

PogListAttr PogListAttr::get(MLIRContext *context, size_t numPogs,
                             ArrayRef<ConstraintAttr> bodyConstraints) {
  SmallVector<PogMetadataAttr> pogs;
  for (size_t i = 0; i != numPogs; ++i)
    pogs.push_back(PogMetadataAttr::get(context));
  return PogListAttr::get(context, pogs, bodyConstraints,
                          ArgConvention::ByRefError);
}

PogListAttr
PogListAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        MLIRContext *context, size_t numPogs,
                        ArrayRef<ConstraintAttr> bodyConstraints) {
  SmallVector<PogMetadataAttr> pogs;
  for (size_t i = 0; i != numPogs; ++i)
    pogs.push_back(PogMetadataAttr::get(context));
  // Explicit `ArrayRef` cast picks the canonical 5-arg `getChecked` over the
  // `StorageUserBase::getChecked<Args...>` template, which would otherwise
  // bind `Args = SmallVector<...>` as a perfect match.
  return PogListAttr::getChecked(emitError, context,
                                 ArrayRef<PogMetadataAttr>(pogs),
                                 bodyConstraints, ArgConvention::ByRefError);
}

PogListAttr PogListAttr::get(MLIRContext *context,
                             ArrayRef<PogMetadataAttr> pogs) {
  return PogListAttr::get(context, pogs, /*bodyConstraints=*/{},
                          ArgConvention::ByRefError);
}

PogListAttr
PogListAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        MLIRContext *context, ArrayRef<PogMetadataAttr> pogs) {
  return PogListAttr::getChecked(emitError, context, pogs,
                                 /*bodyConstraints=*/{},
                                 ArgConvention::ByRefError);
}

PogListAttr PogListAttr::get(MLIRContext *context, ArrayRef<StringAttr> names,
                             ArrayRef<PassingKind> passingKinds) {
  SmallVector<PogMetadataAttr> pogs;
  for (auto [name, passingKind] : llvm::zip(names, passingKinds))
    pogs.emplace_back(PogMetadataAttr::get(name, passingKind));

  return PogListAttr::get(context, pogs);
}

PogListAttr
PogListAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        MLIRContext *context, ArrayRef<StringAttr> names,
                        ArrayRef<PassingKind> passingKinds) {
  SmallVector<PogMetadataAttr> pogs;
  for (auto [name, passingKind] : llvm::zip(names, passingKinds))
    pogs.emplace_back(PogMetadataAttr::get(name, passingKind));
  return PogListAttr::getChecked(emitError, context,
                                 ArrayRef<PogMetadataAttr>(pogs));
}

PogListAttr
PogListAttr::get(MLIRContext *context, ArrayRef<StringAttr> names,
                 ArrayRef<PassingKind> passingKinds,
                 ArrayRef<VariadicKind> argVariadics,
                 ArrayRef<TypedAttr> defaults,
                 std::optional<ArgConvention> origVariadicConvention,
                 ArrayRef<SmallVector<ConstraintAttr>> paramConstraints,
                 ArrayRef<ConstraintAttr> bodyConstraints) {
  return PogListAttr::get(
      context,
      toPogs(names, passingKinds, argVariadics, defaults, paramConstraints),
      bodyConstraints,
      origVariadicConvention.value_or(ArgConvention::ByRefError));
}

PogListAttr PogListAttr::getChecked(
    function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<StringAttr> names, ArrayRef<PassingKind> passingKinds,
    ArrayRef<VariadicKind> argVariadics, ArrayRef<TypedAttr> defaults,
    std::optional<ArgConvention> origVariadicConvention,
    ArrayRef<SmallVector<ConstraintAttr>> paramConstraints,
    ArrayRef<ConstraintAttr> bodyConstraints) {
  SmallVector<PogMetadataAttr> pogs =
      toPogs(names, passingKinds, argVariadics, defaults, paramConstraints);
  return PogListAttr::getChecked(
      emitError, context, ArrayRef<PogMetadataAttr>(pogs), bodyConstraints,
      origVariadicConvention.value_or(ArgConvention::ByRefError));
}

PogListAttr PogListAttr::cloneWith(ArrayRef<PogMetadataAttr> pogs) const {
  return PogListAttr::get(getContext(), pogs, getBodyConstraints(),
                          getOrigVariadicConvention());
}

//===----------------------------------------------------------------------===//
// PogListAttr accessors
//===----------------------------------------------------------------------===//

VariadicKind PogListAttr::getVariadicKind(size_t idx) const {
  return getPogs()[idx].getVariadic();
}

bool PogListAttr::isAnyVarArg(size_t idx) const {
  return getPogs()[idx].isAnyVarArg();
}

bool PogListAttr::isPack(size_t idx) const { return getPogs()[idx].isPack(); }

bool PogListAttr::isPosVarArg(size_t idx) const {
  return getPogs()[idx].isPosVarArg();
}

bool PogListAttr::isKwVarArg(size_t idx) const {
  return getPogs()[idx].isKwVarArg();
}

bool PogListAttr::hasAnyVarArg() const {
  return llvm::any_of(
      getPogs(), [](PogMetadataAttr pogAttr) { return pogAttr.isAnyVarArg(); });
}

bool PogListAttr::hasPackVarArg() const {
  return llvm::any_of(getPogs(),
                      [](PogMetadataAttr pogAttr) { return pogAttr.isPack(); });
}

bool PogListAttr::hasKwVarArg() const {
  for (size_t idx = 0, e = size(); idx != e; ++idx)
    if (isKwVarArg(idx))
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

SmallVector<PogMetadataAttr> PogListAttr::toPogs(
    ArrayRef<StringAttr> names, ArrayRef<PassingKind> passingKinds,
    ArrayRef<VariadicKind> variadics, ArrayRef<TypedAttr> defaults,
    ArrayRef<SmallVector<ConstraintAttr>> paramConstraints) {
  SmallVector<PogMetadataAttr> pogs;
  for (auto [idx, name, passingKind] : llvm::enumerate(names, passingKinds)) {
    VariadicKind variadic =
        variadics.empty() ? VariadicKind::None : variadics[idx];
    ArrayRef<ConstraintAttr> constraints = paramConstraints.empty()
                                               ? ArrayRef<ConstraintAttr>()
                                               : paramConstraints[idx];
    TypedAttr defaultVal = defaults.empty() ? TypedAttr() : defaults[idx];
    pogs.push_back(PogMetadataAttr::get(name, passingKind, variadic, defaultVal,
                                        constraints));
  }
  return pogs;
}

//===----------------------------------------------------------------------===//
// PogListAttr generator hooks
//===----------------------------------------------------------------------===//

LogicalResult
PogListAttr::verifyGenerator(function_ref<InFlightDiagnostic()> emitError,
                             ArrayRef<Type> inputParamTypes, Type body) const {
  if (size() != inputParamTypes.size()) {
    return emitError()
           << "number of pog names doesn't match number of pog types";
  }

  return success();
}

PogListAttr PogListAttr::getSpecializedMetadata(
    ParameterEvaluator &evaluator, const llvm::BitVector &boundParams,
    function_ref<InFlightDiagnostic()> emitError) const {

  // Now update POG metadata.
  SmallVector<PogMetadataAttr> newPogs;

  size_t numParams = boundParams.size();
  bool hasAnyVarArg = false;
  for (auto [idx, pog] : llvm::enumerate(getPogs().take_front(numParams))) {
    if (!boundParams[idx]) {
      auto newPog = evaluator.getReboundAttribute(pog);
      newPogs.emplace_back(cast<PogMetadataAttr>(newPog));
      hasAnyVarArg |= pog.isAnyVarArg();
    }
  }
  auto varargsConvention = getOrigVariadicConvention();
  if (!hasAnyVarArg)
    varargsConvention = ArgConvention::ByRefError;

  SmallVector<ConstraintAttr> newBodyConstraints;
  for (ConstraintAttr constraint : getBodyConstraints()) {
    newBodyConstraints.push_back(
        cast<ConstraintAttr>(evaluator.getReboundAttribute(constraint)));
  }

  return PogListAttr::get(getContext(), newPogs, newBodyConstraints,
                          varargsConvention);
}

/// Get a new metadata attribute for a generator with the given number of
/// positional input parameters prepended to the generator.
PogListAttr PogListAttr::prependAsInferredParams(
    ArrayRef<StringAttr> names, ArrayRef<TypedAttr> defaults,
    ArrayRef<SmallVector<ConstraintAttr>> paramConstraints,
    ArrayRef<ConstraintAttr> bodyConstraints) const {
  assert((defaults.empty() || defaults.size() == names.size()) &&
         "defaults is either empty (no default column) or parallel to names");
  assert(
      (paramConstraints.empty() || paramConstraints.size() == names.size()) &&
      "paramConstraints is either empty or parallel to names");

  SmallVector<PogMetadataAttr> newPogs;
  for (auto [i, name] : llvm::enumerate(names)) {
    TypedAttr newDefault{};
    if (!defaults.empty() && defaults[i])
      newDefault = defaults[i];
    SmallVector<ConstraintAttr> newConstraints;
    if (!paramConstraints.empty())
      llvm::append_range(newConstraints, paramConstraints[i]);
    if (i + 1 == names.size())
      llvm::append_range(newConstraints, bodyConstraints);
    // Strip off variadic kinds and turn the parameter into infer-only too.
    newPogs.push_back(PogMetadataAttr::get(name, PassingKind::Inferred,
                                           VariadicKind::None, newDefault,
                                           newConstraints));
  }

  llvm::append_range(newPogs, getPogs());

  // If there are no new parameters to prepend, concat the body constraints onto
  // the existing body constraints.
  if (newPogs.empty()) {
    SmallVector<ConstraintAttr> newBodyConstraints(bodyConstraints);
    llvm::append_range(newBodyConstraints, getBodyConstraints());
    return PogListAttr::get(getContext(), newPogs, newBodyConstraints,
                            getOrigVariadicConvention());
  }

  return PogListAttr::get(getContext(), newPogs, getBodyConstraints(),
                          getOrigVariadicConvention());
}

PogListAttr
PogListAttr::prependContextualParamsFromOps(ArrayRef<StringAttr> newParams,
                                            ArrayRef<Operation *> ops) const {
  return prependAsInferredParams(newParams, /*defaults=*/{});
}

//===----------------------------------------------------------------------===//
// PogListAttr::verify
//===----------------------------------------------------------------------===//

LogicalResult PogListAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<PogMetadataAttr> pogs,
                                  ArrayRef<ConstraintAttr> bodyConstraints,
                                  ArgConvention origVariadicConvention) {
  size_t numEl = pogs.size();
  for (PogMetadataAttr pogAttr : pogs)
    if (!pogAttr.getName())
      return emitError() << "argument/parameter name cannot be null";

  SmallVector<PassingKind> passingKinds = llvm::map_to_vector(
      pogs, [](PogMetadataAttr pogAttr) { return pogAttr.getPassingKind(); });
  if (failed(verifyPassingKinds(emitError, pogs, "arguments/parameter")))
    return failure();

  // We verified the passing kinds' order and number, so we can use a handler.
  bool sawVariadic = false;
  auto verifyVariadicIdx = [&](size_t idx, VariadicKind kind) -> LogicalResult {
    bool isPack = kind == VariadicKind::PackVarArg;
    if (isPack || kind == VariadicKind::PosVarArg) {
      sawVariadic = true;
      if (origVariadicConvention == ArgConvention::ByRefError)
        return emitError() << "variadic convention not specified";
    }

    if (idx >= numEl) {
      return emitError() << "variadic " << (isPack ? "pack " : "")
                         << "index must be less than the number of elements: "
                         << idx << " vs. " << numEl;
    }
    if (TypedAttr varDefault = pogs[idx].getDefaultValue()) {
      if (::isa<UnknownAttr>(varDefault))
        return success();
      return emitError() << "default value of variadic "
                         << (isPack ? "pack " : "") << "must be UnknownAttr";
    }
    return success();
  };

  for (auto [idx, pogAttr] : llvm::enumerate(pogs)) {
    if (pogAttr.getPassingKind() == PassingKind::Inferred && idx != 0 &&
        pogs[idx - 1].getPassingKind() != PassingKind::Inferred) {
      return emitError()
             << "'inferred' parameter follows non-inferred parameter";
    }
    if (pogAttr.isAnyVarArg() &&
        failed(verifyVariadicIdx(idx, pogAttr.getVariadic())))
      return failure();
  }

  if (origVariadicConvention != ArgConvention::ByRefError && !sawVariadic)
    return emitError() << "pack convention specified without pack";

  return success();
}

//===----------------------------------------------------------------------===//
// PogListAttr::printGenerator
//===----------------------------------------------------------------------===//

void PogListAttr::printGenerator(AsmPrinter &p, GeneratorType generator) const {
  ArrayRef<Type> paramTypes = generator.getInputParamTypes();

  // TODO(MOCO-3957): Flip this literal to `!kgen.generator<` once the
  // textual-mnemonic retirement lands. Keeping the LIT-flavored prefix for
  // now preserves byte-identical IR with the pre-move state.
  p << "!lit.generator<";
  // FuncType bodies carrying `FnMetadataAttrInterface` metadata (i.e. LIT
  // `FnType`) get a special-case: skip the empty angle brackets, then ask the
  // metadata to emit the FuncType inline (no `!lit.fn` prefix and no extra
  // wrapping). The interface-dispatched call keeps this file free of a
  // libLITDialect.a link dependency.
  if (auto sig = ::dyn_cast<FuncType>(generator.getBody())) {
    if (auto metadata = sig.getMetadata()) {
      KGEN::printOptionalParamSignature(p, paramTypes, *this,
                                        /*omitEmptyAngleBrackets=*/true);
      metadata.printFuncTypeInline(p, sig);
      p << '>';
      return;
    }
  }
  KGEN::printOptionalParamSignature(p, paramTypes, *this);
  printKGENType(p, generator.getBody());
  p << '>';
}

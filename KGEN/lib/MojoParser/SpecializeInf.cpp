//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Specialization inference helpers for closure conformance.
//
//===----------------------------------------------------------------------===//

#include "SpecializeInf.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamMatcher.h"

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/IRValues.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

extern bool checkConventionsConvertible(ArgConvention expectedConv,
                                        ArgConvention actualConv);

SpecializeInf::SpecializeInf(ASTDecl &declScope, const ExprNode *expr,
                             ArrayRef<Type> declaredParamTypes,
                             PogListAttr declaredParamPogs, SMLoc defaultLoc,
                             bool discardError)
    : InferenceState(declScope, declaredParamTypes, declaredParamPogs,
                     defaultLoc, discardError),
      expr(expr) {}

LogicalResult SpecializeInf::matchValueType(ASTType actualType, size_t argIdx,
                                            ASTType expectedType,
                                            PogListAttr argPogs) {
  expectedType = evaluator.getReboundType(expectedType);
  if (actualType.isEqualCanon(expectedType))
    return success();

  ParamMatcher matcher(expr, *this, /*allowImplicitConversions=*/false);
  ParamMatcher::FailableScope simpleEqualityFailableScope(matcher);
  if (succeeded(matcher.matchTypes(actualType, expectedType)))
    return success();
  simpleEqualityFailableScope.revert();

  if (auto nonmaterializableTarget =
          actualType.getNonmaterializableTarget(getShared())) {
    ParamMatcher::FailableScope failableScope(matcher);
    if (succeeded(matcher.matchTypes(nonmaterializableTarget, expectedType)))
      return success();
    failableScope.revert();
  }

  return failure();
}

LogicalResult SpecializeInf::matchArgument(Type actualType,
                                           ArgConvention actualConvention,
                                           size_t argIdx,
                                           ASTType origExpectedType,
                                           ArgConvention expectedConvention,
                                           PogListAttr argPogs) {
  ASTType expectedType = evaluator.getReboundType(origExpectedType);
  ParamMatcher matcher(expr, *this, /*allowImplicitConversions=*/false);

  switch (expectedConvention) {
  case ArgConvention::OwnedReg:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::Mut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError: {
    if (actualConvention != ArgConvention::Mut &&
        actualConvention != ArgConvention::MutRef &&
        !isResultSlot(actualConvention))
      return failure();
    ASTType actualRVType =
        RefType::stripRefConvention(actualType, actualConvention);
    if (failed(matcher.matchTypes(actualRVType,
                                  expectedType.getReferenceElementType())))
      return failure();
    return success();
  }
  case ArgConvention::Ref:
  case ArgConvention::MutRef: {
    auto expectedRef = sugarCast<RefType>(expectedType);
    if (hasAddress(actualConvention) && !isResultSlot(actualConvention)) {
      auto valueRefType = dyn_cast<RefType>(actualType);
      if (!valueRefType)
        return failure();
      if (actualConvention != ArgConvention::Mut &&
          actualConvention != ArgConvention::MutRef &&
          actualConvention != ArgConvention::Ref &&
          !valueRefType.isMutableKnown(false))
        valueRefType = valueRefType.getWithMutability(false);

      if (failed(matcher.matchTypes(valueRefType.getElementType(),
                                    expectedRef.getElementType())))
        return failure();
      expectedType = evaluator.getReboundType(expectedType);
      if (!paramFinder.hasReferences(expectedType)) {
        if (IREmitter::canZeroCostConvert(valueRefType, expectedType,
                                          getShared()))
          return success();
        return failure();
      }
      if (succeeded(matcher.matchTypes(valueRefType, expectedType)))
        return success();
      return failure();
    }

    if (expectedRef.isMutableKnown(true))
      return failure();

    auto anyOrigin =
        AnyOriginAttr::get(expectedRef.getContext(), /*isMut=*/false);
    ParamMatcher::FailableScope failableScope1(matcher);
    if (failed(
            matcher.matchSingleEltStruct(anyOrigin, expectedRef.getOrigin())))
      failableScope1.revert();

    auto addrSpace =
        IntegerAttr::get(IndexType::get(expectedRef.getContext()), 0);
    ParamMatcher::FailableScope failableScope2(matcher);
    if (failed(matcher.matchSingleEltStruct(addrSpace,
                                            expectedRef.getAddressSpace())))
      failableScope2.revert();
    [[fallthrough]];
  }
  case ArgConvention::OwnedMem:
  case ArgConvention::DeinitMem:
  case ArgConvention::ReadMem:
    expectedType = expectedType.getReferenceElementType();
    break;
  case ArgConvention::ReadReg:
    break;
  }

  ASTType actualRVType =
      hasAddress(actualConvention)
          ? ASTType(RefType::stripRefConvention(actualType, actualConvention))
          : ASTType(actualType);
  return matchValueType(actualRVType, argIdx, expectedType, argPogs);
}

FailureOr<SmallVector<TypedAttr>>
SpecializeInf::inferSpecialization(FnTypeGeneratorType target, FnOp actualFn) {
  FnTypeGeneratorType actualSig = actualFn.getFuncTypeGenerator();

  // The target may have a ByRefResult slot that the actual lacks when the
  // actual returns in-register. Allow that convention size difference.
  bool targetHasExtraResultSlot =
      target.hasMemoryOnlyResult() && !actualSig.hasMemoryOnlyResult();
  size_t expectedConvSize = target.getArgConventions().size();
  size_t actualConvSize = actualSig.getArgConventions().size();
  if (targetHasExtraResultSlot) {
    if (expectedConvSize != actualConvSize + 1)
      return mlir::failure();
  } else if (expectedConvSize != actualConvSize) {
    return mlir::failure();
  }

  for (auto [actualConv, expectedConv] :
       llvm::zip(actualSig.getArgConventions(), target.getArgConventions())) {
    bool actualIsResult = isResultSlot(actualConv);
    bool expectedIsResult = isResultSlot(expectedConv);
    if (actualIsResult || expectedIsResult) {
      if (actualConv != expectedConv)
        return mlir::failure();
      continue;
    }
    if (!checkConventionsConvertible(expectedConv, actualConv))
      return mlir::failure();
  }

  Block &body = actualFn.getBodyRegion().front();
  assert(body.getNumArguments() == actualSig.getNumArguments() &&
         "fn body arguments should align with signature arguments");

  if (!actualSig.hasMemoryOnlyResult()) {
    ParamRefRemapper remapper(actualFn.getInputParams());
    Type actualResultType = remapper.replace(actualSig.getUserResultType());
    Type expectedResultType =
        evaluator.getReboundType(target.getUserResultType());
    ParamMatcher matcher(expr, *this, /*allowImplicitConversions=*/false);
    if (failed(matcher.matchTypes(actualResultType, expectedResultType)))
      return mlir::failure();
  }

  PogListAttr argPogs = target.getArgListAttrs();
  for (auto [expectedArgIdx, expectedConvention] :
       llvm::enumerate(target.getArgConventions())) {
    // Skip the extra ByRefResult slot — the actual doesn't have one.
    // Result type compatibility is verified above.
    if (targetHasExtraResultSlot && isResultSlot(expectedConvention))
      continue;
    ArgConvention actualConvention =
        actualSig.getArgConventions()[expectedArgIdx];
    Type expectedType =
        evaluator.getReboundType(target.getArgument(expectedArgIdx));
    Type actualType = body.getArgument(expectedArgIdx).getType();
    if (failed(matchArgument(actualType, actualConvention, expectedArgIdx,
                             expectedType, expectedConvention, argPogs)))
      return mlir::failure();
  }

  for (TypedAttr binding : evaluator.getIndexBindings()) {
    if (!binding)
      return mlir::failure();
    assert(!sugarIsa<UnboundAttr>(binding));
  }

  SmallVector<TypedAttr> specialization;
  specialization.reserve(evaluator.getIndexBindings().size());
  for (TypedAttr binding : evaluator.getIndexBindings())
    specialization.push_back(binding);
  return specialization;
}

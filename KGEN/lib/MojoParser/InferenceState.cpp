//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "InferenceState.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/lib/MojoParser/ExprNodes.h"
#include "KGEN/lib/MojoParser/IREmitter.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

OptionalDiag::OptionalDiag(SharedState &shared, SMLoc defaultLoc,
                           bool discardError)
    : discardError(discardError), diag(std::nullopt) {
  getDiagClosure = [=, &shared,
                    this](std::optional<SMLoc> loc) -> MojoInflightDiag & {
    this->diag = shared.emitError(loc ? *loc : defaultLoc);
    return *this->diag;
  };
}

llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc>)>
OptionalDiag::getDiag() {
  return getDiagClosure;
}

InferenceState::InferenceState(ASTDecl &declScope,
                               ArrayRef<Type> declaredParamTypes,
                               PogListAttr declaredParamPogs, SMLoc defaultLoc,
                               bool discardError,
                               DeferredTypingContext *deferredTypingContext)
    : deferredTypingContext(deferredTypingContext), declScope(declScope),
      shared(declScope.getShared()), evaluator(shared.getParameterEvaluator()),
      declaredParamTypes(declaredParamTypes),
      declaredParamPogs(declaredParamPogs),
      diag(shared, defaultLoc, discardError) {
  for (size_t i = 0; i != declaredParamTypes.size(); ++i)
    evaluator.appendIndexBinding(TypedAttr());
}

LogicalResult InferenceState::setInferredValue(size_t paramIdx,
                                               TypedAttr paramVal,
                                               bool isDefaulted) {
  paramVal = evaluator.getReboundAttribute(paramVal);
  ASTType targetType = evaluator.getReboundType(declaredParamTypes[paramIdx]);

  // If the parameter being inferred is a type, and if the source value is
  // non-mterializable, infer to the materialized type. This ensures that things
  // like `def foo[T: AnyType](a: T):` infer `foo(1)` to T=Int instead of
  // IntLiteral.
  if (!isDefaulted && LIT::isTypeExpr(paramVal)) {
    IREmitter emitter(declScope, EC_TypeParamValue);
    if (auto nmTarget = ASTType(paramVal).getNonmaterializableTarget(shared)) {
      TypedAttr nmTargetAttr = PValue(nmTarget).get();
      FailureOr<bool> typeUpCastable = IREmitter::canMetaTypeUpCastTo(
          shared, declScope.getLoc(), nmTargetAttr.getType(), targetType,
          &declScope);
      // If the nonmaterializable type can be upcast to the target type, then
      // make sure we infer to the nonmaterializable type:
      //    def example[T: TrivialRegisterPassable](a: T): ...
      //    example(1) # T should be Int, not IntLiteral.
      if (succeeded(typeUpCastable) && typeUpCastable.value()) {
        SyntheticNode expr(declScope.getLoc());
        paramVal = emitter.emitPValue({nmTargetAttr, &expr}, EC_TypeParamValue,
                                      targetType);
        assert(paramVal && "must be convertible");
        ++numImplicitConversions;
      }
    }
  }

  // Type must be equal
  assert(targetType.isEqualCanon(paramVal.getType()));

  // now align sugar
  if (paramVal.getType() != targetType)
    paramVal = ParamOperatorAttr::getRebind(paramVal, targetType);

  evaluator.overwriteIndexBinding(paramIdx, paramVal);

  if (isa<UnboundAttr>(paramVal))
    return success();

  ArrayRef<ConstraintAttr> constraints =
      declaredParamPogs.getPogs()[paramIdx].getConstraints();
  if (constraints.empty())
    return success();

  // Verify all constraints are satisfied, collecting unprovable constraints.
  ConstraintResult result = checkConstraints(
      declScope, declaredParamPogs, constraints, /*origConstraints=*/{},
      diag.getDiag(), &unprovableConstraints, &evaluator);

  // TODO: how about we just emitting unprovable error here right away?
  return success(result == ConstraintResult::Satisfied);
}

namespace {
/// Walks an attribute/type tree and reports whether it still contains values
/// that are not yet bound. A `ParamIndexRefAttr` is "not yet bound" if the
/// evaluator does not have a binding for it, or the binding for it is
/// `UnboundAttr`.
///
/// The walker mirrors the depth bookkeeping in `ParamIndexRefAttrFinder`
/// (incrementing depth on entering nested parameter scopes) so index refs into
/// inner scopes are correctly ignored. Results are memoized on (depth,
/// opaque-pointer) so deeply shared sub-trees in parameter expressions are not
/// re-walked.
class ConcretenessChecker {
public:
  ConcretenessChecker(ArrayRef<TypedAttr> bindings) : bindings(bindings) {}
  bool isConcrete(Attribute attr) { return !hasUnresolvedImpl(attr, 0); }

private:
  ArrayRef<TypedAttr> bindings;
  DenseMap<std::pair<size_t, const void *>, bool> cache;

  template <typename T>
  bool hasUnresolvedImpl(T value, size_t depth) {
    if (!value)
      return false;

    std::pair<size_t, const void *> cacheKey(depth, value.getAsOpaquePointer());
    if (auto it = cache.find(cacheKey); it != cache.end())
      return it->second;

    // Entering a nested parameter scope reframes "our" depth-0 refs as
    // depth-1 from the inner perspective, so bump depth before checking.
    if constexpr (std::is_base_of_v<Type, T>)
      if (isa<ParameterScopeTypeInterface>(value))
        ++depth;
    if constexpr (std::is_base_of_v<Attribute, T>)
      if (isa<ParameterScopeAttrInterface>(value))
        ++depth;

    bool unresolved = false;
    if constexpr (std::is_base_of_v<Attribute, T>) {
      if (auto indexRef = dyn_cast<ParamIndexRefAttr>(value))
        unresolved = indexRef.getDepth() == depth &&
                     (indexRef.getIndex() >= bindings.size() ||
                      isa<UnboundAttr>(bindings[indexRef.getIndex()]));
    }

    if (!unresolved) {
      value.walkImmediateSubElements(
          [&](Attribute subAttr) {
            unresolved = unresolved || hasUnresolvedImpl(subAttr, depth);
          },
          [&](Type subType) {
            unresolved = unresolved || hasUnresolvedImpl(subType, depth);
          });
    }

    cache[cacheKey] = unresolved;
    return unresolved;
  }
};
} // namespace

LogicalResult InferenceState::checkBodyConstraints() {
  ArrayRef<ConstraintAttr> bodyConstraints =
      declaredParamPogs.getBodyConstraints();
  if (bodyConstraints.empty())
    return success();

  // Filter to only the body constraints that are fully concrete under the
  // current bindings.
  ConcretenessChecker concreteness(evaluator.getIndexBindings());
  SmallVector<ConstraintAttr> concreteConstraints;
  concreteConstraints.reserve(bodyConstraints.size());
  for (ConstraintAttr constraint : bodyConstraints)
    if (concreteness.isConcrete(constraint.getProposition()))
      concreteConstraints.push_back(constraint);

  if (concreteConstraints.empty())
    return success();

  // Verify that no concrete constraints are violated. Any unprovable concrete
  // constraints are recorded in `bodyUnprovableConstraints`. The caller is
  // responsible for surfacing any errors if appropriate.
  ConstraintResult result =
      LIT::checkConstraints(declScope, declaredParamPogs, concreteConstraints,
                            /*origConstraints=*/{}, diag.getDiag(),
                            &bodyUnprovableConstraints, &evaluator);
  return success(result != ConstraintResult::Violated);
}

TypedAttr
VerifiedParamBindings::specializeStructType(StructDeclOp structOp) const {
  assert(*this && "specializing with failed inference bindings");
  return PValue(structOp.bindReference(getValues()));
}

TypedAttr
VerifiedParamBindings::specializeGenerator(TypedAttr generator) const {
  assert(*this && "specializing with failed inference bindings");
  // Special case for structs types: If the generator is a struct meta type,
  // bind the corresponding struct type directly.
  if (auto structMetaType = sugarDynCast<StructMetaType>(generator.getType())) {
    LIT::StructType boundType =
        structMetaType.getType().bindUnbound(getValues());
    return TypeParamAttr::get(boundType, StructMetaType::get(boundType));
  }

  // Otherwise, the type of the generator must be a GeneratorType.
  assert(sugarIsa<GeneratorType>(generator.getType()) &&
         "generator type expected");
  return BindParamsAttr::get(generator, getValues(), evaluationContext);
}

GeneratorType
VerifiedParamBindings::specializeGeneratorType(GeneratorType genType) const {
  assert(*this && "specializing with failed inference bindings");
  return genType.getSpecializedGenerator(getValues(), evaluationContext,
                                         /*emitErrorFn=*/{});
}

void InferenceState::dump() const {
  auto &os = llvm::errs() << "ParamInf:\n";
  for (auto [idx, value] : llvm::enumerate(evaluator.getIndexBindings())) {
    os << "  *(0," << idx << ") = ";
    if (value)
      os << value;
    else
      os << "<not yet set> : "
         << const_cast<InferenceState *>(this)->evaluator.getReboundType(
                declaredParamTypes[idx]);
    os << "\n";
  }
}

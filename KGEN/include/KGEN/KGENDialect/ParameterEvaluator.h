//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_PARAMETEREVALUATOR_H
#define KGEN_KGENDIALECT_PARAMETEREVALUATOR_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/ParameterReplacer.h"
#include "Support/ForwardDecls.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
class LockedSymbolTableCollection;
} // namespace mlir

namespace M {
class Error;
} // namespace M

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Helper methods for inspecting possibly-parameterized attributes and types.
//===----------------------------------------------------------------------===//

// NOTE: None of these are particularly efficient, because they walk the whole
// IR tree without caching.

/// Given a parameter expression, walk it and return any references to named
/// parameters.  This fails if an invalid parameter expression exists.
void collectParameterReferences(Attribute attr,
                                SmallVectorImpl<ParamDeclRefAttr> &results,
                                bool &hasConstExpr);

/// Given a potentially-parameterized MLIR type, walk it and return any
/// references to named parameters.  This fails if an invalid parameter
/// expression exists.
void collectParameterReferences(Type type,
                                SmallVectorImpl<ParamDeclRefAttr> &results,
                                bool &hasConstExpr);

/// Return true if the specified type contains parameter references, e.g.
/// `!pop.scalar<dt>` returns true, but `!pop.scalar<f32>` returns false.
bool isParameterizedType(Type type);

//===----------------------------------------------------------------------===//
// ParameterEvaluationContext
//===----------------------------------------------------------------------===//

/// This class is used by ParameterEvaluator to evaluate
/// ContextuallyEvaluatedAttrInterface instances, which are attributes whose
/// evaluation may be context-dependent. Sub-classes can store state to help
/// with evaluation.
///
/// The use of this separate context/policy provider class allows
/// ParameterEvaluators, which are stateful and may need to be instantiated
/// multiple times, to be decoupled from the logic of evaluating attributes.
class ParameterEvaluationContext {
public:
  virtual ~ParameterEvaluationContext() = default;

  /// Evaluate the provided attribute. If the attribute is not evaluatable,
  /// return failure(). This does not indicate an unexpected situation, but
  /// rather no further evaluation was possible.
  virtual FailureOr<TypedAttr>
  evaluateExpression(ContextuallyEvaluatedAttrInterface attr) = 0;
};

/// An evaluation context that exposes a LockedSymbolTableCollection.
class SymTabEvaluationContext : public ParameterEvaluationContext {
public:
  Operation *module;
  mlir::LockedSymbolTableCollection &symtab;

  SymTabEvaluationContext(Operation *module,
                          mlir::LockedSymbolTableCollection &symtab)
      : module(module), symtab(symtab) {}

  FailureOr<TypedAttr>
  evaluateExpression(ContextuallyEvaluatedAttrInterface attr) override;

private:
  FailureOr<TypedAttr> evaluateGetWitness(GetWitnessAttr getWitness);
  FailureOr<TypedAttr> inlineApply(ParamOperatorAttr getWitness);
  Operation *getStructInstIfResolved(TypedAttr typeVal);
};

//===----------------------------------------------------------------------===//
// ParameterEvaluator
//===----------------------------------------------------------------------===//

/// This class keeps a set of defined parameter bindings and is used to evaluate
/// and simplify parameter expressions based on those values.
///
/// This class keeps track of two kinds of parameter bindings at the same time:
///
/// 1. Decl-based (name-based) bindings, which are used to substitute
///    ParamDeclRefAttrs.
/// 2. Index-based bindings, which are used to substitute ParamIndexRefAttrs.
///
/// - Rules for ParamDeclRefAttr:
/// If the name referenced by the ParamDeclRefAttr is not registered with the
/// evaluator, the attribute is left unchanged.
///
/// - Rules for ParamIndexRefAttr:
/// The attr/type initially passed to the evaluator is considered to be at
/// depth `inputDepth` (0 for most cases, but can be overridden by the user).
/// Only index references that point back to index bindings at this depth are
/// candidates for substitution (see IRAIDAI and PSTIAIRAID for details).
///
/// In addition, the evaluator will assert if the index is not less than the
/// size of all the index bindings. If partial substitution is being performed
/// (i.e. not all the index bindings are registered, only a given prefix of the
/// index bindings are registered), the user can manually set the total number
/// of index bindings so that the assertion is only triggered for real errors.
class ParameterEvaluator : public ParameterReplacer<ParameterEvaluator> {
public:
  /// Instantiate a new parameter evaluator with the given parameter values.
  ParameterEvaluator(ArrayRef<ParamDeclAttr> paramDecls,
                     ArrayRef<TypedAttr> declBindings);
  /// Instantiate a new parameter evaluator with the given input parameters.
  ParameterEvaluator(ArrayRef<TypedAttr> declBindings);

  /// Instantiate a new parameter evaluator with the given parameter values.
  ParameterEvaluator(
      DenseMap<StringAttr, TypedAttr> declBindings =
          DenseMap<StringAttr, TypedAttr>(),
      ArrayRef<TypedAttr> indexBindings = SmallVector<TypedAttr>(),
      size_t inputDepth = 0)
      : declBindings(std::move(declBindings)),
        indexBindings(std::move(indexBindings)), inputDepth(inputDepth) {}

  /// Set the evaluation context to use.
  void setEvaluationContext(ParameterEvaluationContext *context) {
    evaluationContext = context;
  }
  ParameterEvaluationContext *getEvaluationContext() const {
    return evaluationContext;
  }

  /// Set a value for the specified parameter declaration to the specified
  /// simplified value.
  void setDeclBinding(StringAttr name, TypedAttr value,
                      bool overwrite = false) {
    assert(overwrite ||
           !declBindings.count(name) && "parameter already declared!");
    declBindings[name] = value;
  }
  void setDeclBinding(ParamDeclAttr decl, TypedAttr value,
                      bool overwrite = false) {
    setDeclBinding(decl.getName(), value, overwrite);
  }

  void setRewritten(
      const DenseMap<std::pair<size_t, const void *>, const void *> &value) {
    rewritten = value;
  }

  const DenseMap<std::pair<size_t, const void *>, const void *> &
  getRewritten() {
    return rewritten;
  }

  bool overwriteDeclBinding(ParamDeclAttr decl, TypedAttr value) {
    auto iter = declBindings.find(decl.getName());
    bool exist = iter != declBindings.end();
    declBindings[decl.getName()] = value;
    return exist;
  }

  /// Iterate over the current parameter values.
  const DenseMap<StringAttr, TypedAttr> &getDeclBindings() const {
    return declBindings;
  }

  /// Overwrite the current set of parameter values.
  void setDeclBindings(const DenseMap<StringAttr, TypedAttr> &values) {
    declBindings = values;
  }

  /// Get the specified type with any nested parameter expressions rewritten.
  Type getReboundType(Type type) { return replace(type); }

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.
  Attribute getReboundAttribute(Attribute attr) { return replace(attr); }

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.
  TypedAttr getReboundAttribute(TypedAttr attr) { return replace(attr); }

  TypedAttr getFailableReboundAttribute(TypedAttr attr) {
    return failableReplace(attr);
  }

  /// Dump the parameter evaluator state.
  void dump() const;

  /// Append an index-based parameter binding.
  void appendIndexBinding(TypedAttr value) { indexBindings.push_back(value); }
  void overwriteIndexBinding(size_t idx, TypedAttr value) {
    assert(idx < indexBindings.size() && "invalid index");
    indexBindings[idx] = value;
  }

  /// Return the number of input parameter values that have been added.
  size_t getNumIndexBindings() const { return indexBindings.size(); }
  /// Get all the input parameters.
  ArrayRef<TypedAttr> getIndexBindings() const { return indexBindings; }

  /// Set the relative input depth.
  void setInputDepth(size_t depth) { inputDepth = depth; }
  size_t getInputDepth() const { return inputDepth; }

  void clearCache() { rewritten.clear(); }

private:
  // CRTP methods.
  Type doReplace(Type type, size_t rootDepth);
  Attribute doReplace(Attribute attr, size_t rootDepth);
  friend class ParameterReplacer<ParameterEvaluator>;

  /// Handle the `cond` operator. This needs to return a tri-state: whether the
  /// condition can be narrowed to an integer constant and whether we need to
  /// suspend, which is that the bool represents.
  std::pair<IntegerAttr, bool> narrowCondOp(Attribute attr, size_t rootDepth);

  /// These are the name-based parameter bindings.
  DenseMap<StringAttr, TypedAttr> declBindings;

  /// These are the top-level index-based parameter bindings. This list is
  /// allowed to contain null entries.  When encountered, the parameter replacer
  /// will leave ParamIndexRefAttr referring to them unchanged (actually will
  /// remap the type if needed).
  SmallVector<TypedAttr> indexBindings;

  /// The optional context to use for evaluating contextually evaluated
  /// attributes.
  ParameterEvaluationContext *evaluationContext = nullptr;

public:
  /// The relative depth from the generator where the index-based parameter
  /// bindings are from. This is zero for most applications, but should be set
  /// accordingly when substituting attributes or types inside a generator, see
  /// PSTIAIRAID.
  size_t inputDepth = 0;
};

//===----------------------------------------------------------------------===//
// Helper methods involving parameter evaluation.
//===----------------------------------------------------------------------===//

/// A partially specialized input parameter specification.
struct PartiallySpecializedInputParams {
  ParameterEvaluator evaluator;
  SmallVector<Type, 16> unboundParamTypes;
  llvm::BitVector boundParams;

  /// Given an input parameter specification `paramTypes` and the full set of
  /// bindings `paramBindings`, create a partially specialized input parameter
  /// specification.
  ///
  /// The bindings may be partially specified, with holes represented by
  /// UnboundAttrs.
  static std::optional<PartiallySpecializedInputParams>
  from(ArrayRef<Type> paramTypes, ArrayRef<TypedAttr> paramBindings,
       ParameterEvaluationContext *evaluationContext,
       function_ref<InFlightDiagnostic()> emitErrorFn);
};

//===----------------------------------------------------------------------===//
// ParameterEvaluator for ParametricElaborator that
// does parametric interpreting
//===----------------------------------------------------------------------===//
class ParametricParameterEvaluator : public ParameterEvaluator {
public:
  /// Instantiate a new parameter evaluator with the given parameter values.
  ParametricParameterEvaluator(ArrayRef<ParamDeclAttr> paramDecls,
                               ArrayRef<TypedAttr> declBindings);
  /// Instantiate a new parameter evaluator with the given input parameters.
  ParametricParameterEvaluator(ArrayRef<TypedAttr> declBindings);

  /// Instantiate a new parameter evaluator with the given parameter values.
  ParametricParameterEvaluator(
      DenseMap<StringAttr, TypedAttr> declBindings =
          DenseMap<StringAttr, TypedAttr>(),
      ArrayRef<TypedAttr> indexBindings = SmallVector<TypedAttr>(),
      size_t inputDepth = 0);

  /// Fields added for interpreting parametric functions
  /// for memorizing ParameterReplacer's rewritten cache
  /// to avoid having to rebuild the cache when re-entering the same
  /// parameter domain with the same parameters.

  /// Operation as part of cache key.
  Operation *cachedOpKey = nullptr;
  /// Region as part of the cache key.
  Region *cachedRegionKey = nullptr;
  /// Parameters as part of the cache key.
  ParameterExprArrayAttr cachedAttrKey;
  /// A flag to note if the cache has been memorized.
  bool foundCached = false;
};

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_PARAMETEREVALUATOR_H

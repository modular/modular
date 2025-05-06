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
  SymTabEvaluationContext(Operation *module,
                          mlir::LockedSymbolTableCollection &symtab)
      : module(module), symtab(symtab) {}

  FailureOr<TypedAttr>
  evaluateExpression(ContextuallyEvaluatedAttrInterface attr) override;

  Operation *module;
  mlir::LockedSymbolTableCollection &symtab;
};

//===----------------------------------------------------------------------===//
// ParameterEvaluator
//===----------------------------------------------------------------------===//

/// This class keeps a set of defined parameter values and is used to evaluate
/// and simplify parameter expressions based on those values.
class ParameterEvaluator : public ParameterReplacer<ParameterEvaluator> {
public:
  /// Instantiate a new parameter evaluator with the given parameter values.
  ParameterEvaluator(ArrayRef<ParamDeclAttr> paramDecls,
                     ArrayRef<TypedAttr> paramValues);
  /// Instantiate a new parameter evaluator with the given input parameters.
  ParameterEvaluator(ArrayRef<TypedAttr> paramValues);

  /// Instantiate a new parameter evaluator with the given parameter values.
  ParameterEvaluator(DenseMap<StringAttr, Attribute> paramValues =
                         DenseMap<StringAttr, Attribute>())
      : paramValues(std::move(paramValues)) {}

  /// Set the evaluation context to use.
  void setEvaluationContext(ParameterEvaluationContext *context) {
    evaluationContext = context;
  }

  /// Set a value for the specified parameter declaration to the specified
  /// simplified value.
  void setParameterValue(StringAttr name, Attribute value) {
    assert(!paramValues.count(name) && "parameter already declared!");
    paramValues[name] = value;
  }
  void setParameterValue(ParamDeclAttr decl, Attribute value) {
    setParameterValue(decl.getName(), value);
  }

  /// Iterate over the current parameter values.
  const DenseMap<StringAttr, Attribute> &getParameterValues() const {
    return paramValues;
  }

  /// Overwrite the current set of parameter values.
  void setParameterValues(const DenseMap<StringAttr, Attribute> &values) {
    paramValues = values;
  }

  /// Get the specified type with any nested parameter expressions rewritten.
  Type getReboundType(Type type) { return replace(type); }

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.
  Attribute getReboundAttribute(Attribute attr) { return replace(attr); }

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.
  TypedAttr getReboundAttribute(TypedAttr attr) { return replace(attr); }

  /// Dump the parameter evaluator state.
  void dump() const;

  /// Add an input parameter binding.
  void addInputValue(TypedAttr value) { inputParamValues.push_back(value); }
  /// Set the relative input depth.
  void setInputDepth(size_t depth) { inputDepth = depth; }

  /// Return the number of input parameter values that have been added.
  size_t getNumInputParams() const { return inputParamValues.size(); }
  /// Add an input parameter.
  void addInputParam(TypedAttr param) { inputParamValues.push_back(param); }
  /// Get all the input parameters.
  ArrayRef<TypedAttr> getInputParams() const { return inputParamValues; }

private:
  // CRTP methods.
  Type doReplace(Type type, size_t rootDepth);
  Attribute doReplace(Attribute attr, size_t rootDepth);
  friend class ParameterReplacer<ParameterEvaluator>;

  /// Handle the `cond` operator. This needs to return a tri-state: whether the
  /// condition can be narrowed to an integer constant and whether we need to
  /// suspend, which is that the bool represents.
  std::pair<IntegerAttr, bool> narrowCondOp(Attribute attr, size_t rootDepth);

  /// These are the bound parameter values, captured in simplified form.
  DenseMap<StringAttr, Attribute> paramValues;

  /// These are the top-level input parameters to use when rebinding a
  /// signature.
  SmallVector<TypedAttr> inputParamValues;

  /// The relative depth from the signature where the input parameters are from.
  /// This is zero for most applications, but should be set accordingly when
  /// substituting attributes or types inside a signature.
  size_t inputDepth = 0;

  /// The optional context to use for evaluating contexually evaluated
  /// attributes.
  ParameterEvaluationContext *evaluationContext = nullptr;
};
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_PARAMETEREVALUATOR_H

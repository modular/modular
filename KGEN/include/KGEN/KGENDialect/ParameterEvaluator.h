//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_PARAMETEREVALUATOR_H
#define KGEN_KGENDIALECT_PARAMETEREVALUATOR_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/ForwardDecls.h"
#include "llvm/ADT/DenseMap.h"

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
// ParameterEvaluator
//===----------------------------------------------------------------------===//

/// This class keeps a set of defined parameter values and is used to evaluate
/// and simplify parameter expressions based on those values.
class ParameterEvaluator {
public:
  virtual ~ParameterEvaluator() = default;

  /// Instantiate a new parameter evaluator with the given parameter values.
  ParameterEvaluator(ArrayRef<ParamDeclAttr> paramDecls,
                     ArrayRef<TypedAttr> paramValues);

  /// Instantiate a new parameter evaluator with the given parameter values.
  ParameterEvaluator(DenseMap<StringAttr, Attribute> paramValues =
                         DenseMap<StringAttr, Attribute>())
      : paramValues(std::move(paramValues)) {}

  /// Return true if there are no remappings installed.
  bool empty() const { return paramValues.empty(); }

  /// Clear out the cache of rewritten attrs and types. This is needed because
  /// parameters can be redefined, and the rewritten attr/type may no longer be
  /// valid.
  void clearCache() { rewritten.clear(); }

  /// Set a value for the specified parameter declaration to the specified
  /// simplified value.
  void setParameterValue(StringAttr name, Attribute value) {
    assert(!paramValues.count(name) && "parameter already declared!");
    paramValues[name] = value;
  }
  void setParameterValue(ParamDeclAttr decl, Attribute value) {
    setParameterValue(decl.getName(), value);
  }

  /// Set or overwrite the value of a parameter.
  void setOrOverwriteParameterValue(StringAttr name, Attribute value) {
    paramValues[name] = value;
  }
  void setOrOverwriteParameterValue(ParamDeclAttr decl, Attribute value) {
    setOrOverwriteParameterValue(decl.getName(), value);
  }

  /// Iterate over the current parameter values.
  const DenseMap<StringAttr, Attribute> &getParameterValues() const {
    return paramValues;
  }

  /// Get the specified type with any nested parameter expressions rewritten.
  Type getReboundType(Type type);

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.  The substituted attributes are not necessarily fully folded:
  /// for that use concretizeParameterExpr.
  Attribute getReboundAttribute(Attribute attr);

  /// Evaluate a potentially symbolic expression. This hook should be overridden
  /// to process symbol expressions using some global state, including a symbol
  /// table.
  virtual FailureOr<TypedAttr> evaluateExpression(ParamOperatorAttr op);

  /// Dump the parameter evaluator state.
  void dump() const;

  /// Add an input parameter binding.
  void addInputValue(Attribute value) { inputParamValues.push_back(value); }
  /// Add a result parameter binding.
  void addResultValue(Attribute value) { resultParamValues.push_back(value); }
  /// Set the relative input depth.
  void setInputDepth(size_t depth) { inputDepth = depth; }

private:
  /// These are the bound parameter values, captured in simplified form.
  DenseMap<StringAttr, Attribute> paramValues;

  /// These are the top-level input parameters to use when rebinding a
  /// signature.
  SmallVector<Attribute> inputParamValues;
  /// These are the top-level result parameters to use when rebinding a
  /// signature.
  SmallVector<Attribute> resultParamValues;
  /// The current depth from the root signature, if there is one.
  size_t rootDepth = 0;
  /// The relative depth from the signature where the input parameters are from.
  /// This is zero for most applications, but should be set accordingly when
  /// substituting attributes or types inside a signature.
  size_t inputDepth = 0;

  /// This caches attributes and Types with parameter references rebound, and
  /// remembers complex attributes that don't have parameter subexprs (noted as
  /// being rebound to themselves).
  DenseMap<std::pair<size_t, const void *>, const void *> rewritten;
};
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_PARAMETEREVALUATOR_H

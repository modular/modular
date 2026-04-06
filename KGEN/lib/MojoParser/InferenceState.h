//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Shared state for parameter inference implementations.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_INFERENCESTATE_H
#define KGEN_MOJOPARSER_INFERENCESTATE_H

#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/MojoParser/Constraints.h"
#include "KGEN/MojoParser/MojoDiags.h"
#include "ParserEvaluationContext.h"

#include <cstddef>

namespace M::KGEN::LIT {

class ASTDecl;
class ParamMatcher;
class SharedState;

/// Holds an optional in-flight diagnostic that can be discarded (e.g. for
/// try-style inference). InferenceState uses this as its diagnostic sink.
class OptionalDiag {
public:
  OptionalDiag(SharedState &shared, SMLoc defaultLoc, bool discardError);

  // Allow moving
  OptionalDiag(OptionalDiag &&other) = default;
  OptionalDiag &operator=(OptionalDiag &&other) = default;

  // Don't allow copying
  OptionalDiag(const OptionalDiag &) = delete;
  OptionalDiag &operator=(const OptionalDiag &) = delete;

  ~OptionalDiag() {
    if (discardError && diag.has_value())
      diag->abandon();
  }

  void release() { discardError = false; }

  /// Callback that returns a MojoInflightDiag for the given location (or
  /// default). Invoking the callback records that an error was emitted.
  llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc>)> getDiag();

  bool hasErrorEmitted() const { return diag.has_value(); }

  MojoInflightDiag &attachNote(SMLoc loc) & {
    diag->attachNote(loc);
    return *diag;
  }

  /// Take the emitted diagnostic if any.
  std::optional<MojoInflightDiag> takeMojoDiag() { return std::move(diag); }

private:
  bool discardError;
  std::optional<MojoInflightDiag> diag;
  std::function<MojoInflightDiag &(std::optional<SMLoc>)> getDiagClosure;
};

class InferenceState {
public:
  InferenceState(ASTDecl &declScope, ArrayRef<Type> declaredParamTypes,
                 PogListAttr declaredParamPogs, SMLoc defaultLoc,
                 bool discardError);
  virtual ~InferenceState() = default;

  ASTDecl &getDeclScope() const { return declScope; }
  SharedState &getShared() const { return shared; }

  /// Return a diagnostic emitter for the given location (or default).
  MojoInflightDiag &getMojoDiag(std::optional<SMLoc> loc) {
    return diag.getDiag()(loc);
  }

  LogicalResult setInferredValue(size_t paramIdx, TypedAttr paramVal,
                                 bool isDefaulted = false);
  virtual bool isExplicitlyUnbound(size_t paramIdx) const = 0;

  SmallVector<ConstraintAttr> unprovableConstraints;

protected:
  friend class ParamMatcher;

  ASTDecl &declScope;
  SharedState &shared;
  ParameterEvaluator evaluator;
  ParamIndexRefAttrFinder paramFinder;
  ArrayRef<Type> declaredParamTypes;
  PogListAttr declaredParamPogs;

public:
  OptionalDiag diag;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_INFERENCESTATE_H

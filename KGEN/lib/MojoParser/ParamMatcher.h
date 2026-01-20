//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the ParamMatcher for Mojo parsers to "co-iterate" two
// types/parameters in order to extract parameter value to be inferred.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PARAMMATCHER_H
#define KGEN_MOJOPARSER_PARAMMATCHER_H

#include "ParamInf.h"

#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/MojoParser/ASTType.h"
#include "Support/ADT/SmartVariant.h"

namespace M::KGEN::LIT {

class ExprNode;
class ParamInf;
class SharedState;

//===----------------------------------------------------------------------===//
// MatchFailure
//===----------------------------------------------------------------------===//

/// These are the different failure modes that we know happen.
struct MatchFailure {
  /// This failure happens when a parameter is found of the wrong type.
  struct TypeConflict {
    size_t paramIdx; // TODO: Render this name.
    ASTType paramType, argParamType;
  };

  /// This failure happens when a parameter is inferred to two different values.
  struct ValueConflict {
    size_t paramIdx;
    TypedAttr v1, v2;
  };

  /// This failure happens when merge* is called, but the expected type/value
  /// still has an unresolved dependent type which can't be inferred.
  struct DependsOnUnresolved {
    size_t paramIdx;
  };

  /// This failure happens when parameter is inferred, yet the constraint
  /// attached on it can not be proved.
  struct UnprovableConstraints {
    size_t paramIdx;
  };

  /// This failure hasn't been categorized yet.
  /// FIXME: Remove this.
  struct Unclassified {};

  template <typename Failure>
  MatchFailure(Failure info) : info(info) {}

  // Describe what went wrong.
  void addExplanation(MojoInflightDiag &diag) const;

  /// If this failure is due to an unresolved parameter, return the index of the
  /// parameter.
  std::optional<size_t> getIfDependentOnUnresolved() const {
    if (isa<DependsOnUnresolved>(info)) {
      return cast<DependsOnUnresolved>(info).paramIdx;
    }
    return std::nullopt;
  }

private:
  SmartVariant<TypeConflict, ValueConflict, DependsOnUnresolved, Unclassified,
               UnprovableConstraints>
      info;
};

/// This class implements logic for match parameters between an actual value
/// present at a call site, and expected value in the callee.  The former value
/// is always concrete, but the later may contain symbolic parameters from the
/// callees signature that we're trying to infer.  It is also possible this
/// candidate is completely invalid!  The result of match is one of several
/// cases:
///
/// - Match: the parameters match.
/// - Error: the parameters do not match.  The error code is set to indicate
///   the first reason.
/// - Retry: the parameters matched and led to a parameter getting inferred! The
///   parameter # is set to indicate which one.
///
/// You might wonder why we need to retry matching from a root when inferring a
/// parameter.  It turns out that some values can only be matched after
/// simplification.  Consider a situation like:
///
///    struct S[a: Int, b: Int]:
///    fn take[v: Int](s: S[v, v+1]):
///
/// In this case, we *must* stop after inferring the value of `v`, backtrack
/// up call call stack, and then substitute the value of `v` into the expected
/// type.  If we don't do this, we won't be able to match calls that pass,
/// A[1, 2] because the "v+1=2" knowledge can only be had by substituting which
/// allows the Int addition to fold.
class ParamMatcher {
public:
  ParamMatcher(const ExprNode *expr, ParamInf &state,
               bool allowImplicitConversion);
  ~ParamMatcher() {}

  /// This is set when an error is encountered.
  std::optional<MatchFailure> failureReason;

  enum ResultCode { Match, Error };
  ResultCode matchTypes(Type actualType, Type expectedType);
  ResultCode matchParams(TypedAttr actualAttr, TypedAttr expectedAttr);
  ResultCode matchFunctionTypes(FnTypeGeneratorType actual,
                                FnTypeGeneratorType expected);
  ResultCode matchSingleEltStruct(TypedAttr actual, TypedAttr expected);

  void resetError() { failureReason.reset(); }

private:
  ResultCode error(MatchFailure &&reason) {
    failureReason = std::move(reason);
    return Error;
  }

private:
  /// This is how many signature types deep inference is inside parameter
  /// expressions and determines which index references we match against.
  ///
  /// As we search for param-refs, recursively, we'll be recursing past
  /// `FnTypeGeneratorType`s (and other `ParameterScopeTimeInterface`s),
  /// which changes what param-ref depths we're watching for; the param-refs'
  /// depths would be greater (have to reach further outward so to speak, past
  /// more generator types) to reference param-decls in the
  /// ParameterInferenceState's original scope. paramIndexRefDepth tracks that
  /// number.
  ///
  /// In other words, these paramIndexRefDepth adjustments are for
  /// depth-aware searching, see PSTIAIRAID.
  size_t paramIndexRefDepth = 0;

  /// This is the expression we're inferring within.
  const ExprNode *const expr;
  ParamInf &state;
  SharedState &shared;

  // Whether we allow implicit conversion when matching, e.g., function type
  // conversion.
  bool allowImplicitConversions;

  /// NOTE: this serves a COMPLETELY different purpose from the evaluator in
  /// ParamInf. It is used to bind the parameter defined in a
  /// `ParameterScopedAttr/TypeInterface`. Considering that we are inferring:
  ///
  /// fn foo[
  ///     t : fn [p: Int](...) -> ParamType[p]
  //  ]():
  //     pass
  ///
  /// and we are matching:
  /// actual:   fn [p: Int] () -> ParamType[*(0, 1)]
  /// expected: fn [x: Int] () -> ParamType[*(0, 1)]
  ///
  /// Before pulling out and matching `ParamType[*(0, 1)]`, we need to bind
  /// `*(0, 1)` to the a concrete dummy value before matching. It is important
  /// since both `*(0, 1)`s are bound in the current scope (FnTypeGenerator),
  /// which is NOT the same scope (foo) as we are inferring parameters!
  ParserParameterEvaluator scopedBinder;
  void appendLocallyDefinedParam(Type paramType);
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARAMMATCHER_H

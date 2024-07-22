//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the components for overload fitness evaluation.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_OVERLOADFITNESS_H
#define KGEN_MOJOPARSER_OVERLOADFITNESS_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/MojoParser/IRValues.h"
#include "Support/Compiler/Diags.h"

namespace M::KGEN::LIT {
class CallOperands;
class LITSignatureType;
class PogListAttr;
struct TypeCheckScopeInfo;

/// This struct indicates whether a signature can be successfully applied to a
/// parameter binding and argument list. If so, it keeps track of several
/// metrics that allow comparing different candidates, and if not, it indicates
/// the reason for the mismatch.
class OverloadFitness {
public:
  OverloadFitness(OverloadFitness &&other)
      : paramBindings(other.paramBindings),
        diag(other.diag ? std::optional<InflightDiag>(other.takeDiag())
                        : std::nullopt),
        payload(other.payload) {}

  ~OverloadFitness() {
    if (diag)
      takeDiag().abandon();
  }

  /// Return the parameter bindings if the candidate is valid.
  ParameterExprArrayAttr getParamBindings() const {
    assert(isValid());
    return paramBindings;
  }

  /// Return the number of implicit conversions if the candidate is valid.
  size_t getNumImplicitConversions() const {
    assert(isValid());
    return payload.numImplicitConversions;
  }

  /// Returns whether this fitness is strictly better than another one.
  bool isBetter(const OverloadFitness &other) const;

  /// Consume the diagnostic if the candidate is not valid.
  InflightDiag takeDiag() {
    assert(!isValid());
    return std::move(*diag);
  }

  /// Return whether the candidate was valid.
  bool isValid() const { return !diag; }

  /// Determine whether the specified signature can be invoked with the
  /// parameter bindings specified in `callable` and the arguments specified in
  /// `callOperands`.
  ///
  /// The 'funcIfDirect' member is set if this is a direct call, or null if
  /// indirect.  It can be used to tune diagnostics.
  static OverloadFitness evaluate(LITSignatureType signature,
                                  ASTDecl *funcIfDirect,
                                  const OverloadSet &callable,
                                  const CallOperands &callOperands,
                                  bool allowImplicitConversions);

  /// Determine whether the specified signature can be invoked with the
  /// parameter bindings specified in `callable`.
  static OverloadFitness evaluate(ArrayRef<Type> paramTypes,
                                  PogListAttr paramListAttr,
                                  const OverloadSet &callable,
                                  bool allowImplicitConversions);

  enum ArgTypeMismatchKind {
    kValidType,   //< No argument type mismatch.
    kNotLValue,   //< By-ref argument requires an lvalue, but got an rvalue.
    kWrongLVType, //< By-ref argument and provided l-value types mismatch.
    kWrongType,   //< An argument value not convertible to the expected type.
  };

private:
  /// For valid candidates, this defines the parameter bindings to use.
  ParameterExprArrayAttr paramBindings;
  /// The diagnostic for invalid candidates, or null for valid ones.
  std::optional<InflightDiag> diag = std::nullopt;

  /// Describes the metrics that can be used to compare candidates.
  struct Payload {
    /// The number of implicit conversions required.  Normal implicit
    /// conversions count as 2 each, non-materializable value conversions count
    /// as 1.
    size_t numImplicitConversions = 0;

    /// For each mismatch in "preferred" argument convention, penalize the
    /// overload. This is to resolve ambiguities that can arise from synthesized
    /// thunks for converting calling conventions.
    size_t numMismatchedConventions = 0;
    /// Whether the candidate has a (non-empty) variadic argument.
    bool passesVarArgArgument = false;
    /// Whether the bindings include variadic parameters.
    bool hasVariadicParams = false;

    /// Return a numeric value that allows easy comparison of boolean metrics.
    int8_t getBoolMask() const;
  } payload;

  OverloadFitness(InflightDiag &&diag) : diag(std::move(diag)) {}
  OverloadFitness(ParameterExprArrayAttr paramBindings)
      : paramBindings(paramBindings) {}

  /// Check the expected type against the provided operand. This identifies any
  /// problems with the operand type and also returns the type to be used for
  /// error propagation.
  std::pair<ArgTypeMismatchKind, ASTType>
  checkOneOperand(ASTExprAnd<AnyValue> operand,
                  ArgConvention expectedConvention, ASTType expectedType,
                  bool allowImplicitConversions, SMLoc loc,
                  const TypeCheckScopeInfo &scopeInfo);
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_OVERLOADFITNESS_H

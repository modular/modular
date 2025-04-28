//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PARAMBINDINGS_H
#define KGEN_MOJOPARSER_PARAMBINDINGS_H

#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/MojoParser/CallOperands.h"

namespace M::KGEN {
class ParameterExprArrayAttr;
class ParameterEvaluator;
} // namespace M::KGEN

namespace M::KGEN::LIT {
using llvm::SMLoc;
class DeclResolver;
class ExprNode;
class FnOp;
class FnTypeGeneratorType;
class PogListAttr;
class PValue;
class StructDeclOp;
class TypeSignatureType;

//===----------------------------------------------------------------------===//
// ParamBindings
//===----------------------------------------------------------------------===//

/// This class holds a work-in-progress set of parameter bindings for a type or
/// function declaration.  Some of the bindings may be pre-checked, others may
/// not be.  They are eventually resolved and diagnosed with the
/// verifyBindings() method.
///
/// Consider something like one of these:
///    SomeType[param1].method[param2](...args...)
///    OuterType[param1].InnerType[param2]
///
/// The type parameters (param1) will be bound typed-checked, and the param2
/// will be bound as the TypedAttr value of param2.  We cannot type check the
/// bindings until overload resolution has resolved which 'method' we are
/// talking about and when inference is complete, so we keep a flag.
class ParamBindings {
public:
  /// This is the declaration that we do name lookup against.
  ASTDecl &declScope;
  SharedState &shared;

  /// Initialize ParamBindings with a declscope to perform lookups against
  /// and a notion of shared context.
  ParamBindings(ASTDecl &declScope);
  ParamBindings(const ParamBindings &) = default;

  /// Replace our bindings with another set.
  void operator=(ParamBindings &&other);

  /// Create a (possibly partially unbound) set of bindings for the given type.
  /// This can be used to initialize the binding set for methods. If the given
  /// type is not a parametric user defined type, this returns empty bindings.
  /// If the caller provides a known parent trait type, this will upcast the
  /// given type to it.
  static ParamBindings getForDeclaredType(ASTDecl &declScope, ASTType type,
                                          const ExprNode *expr,
                                          Type optionalParentTraitType = {});

  /// Utility function to perform substitutions of the bindings into the symbol
  /// for the given function declaration. It returns the resultant
  /// SymbolConstantAttr or produces an error message and returns null.
  TypedAttr getBoundConstAttrFor(FnOp funcOp, StringRef baseName,
                                 const ExprNode *expr) const;

  /// Return whether there are any bindings given.
  bool empty() const { return parameters.empty(); }

  // Provide access to the parameter list this represents.
  const CallOperands &getParameters() const { return parameters; }

  /// Add a bound value for pre-checked positional parameter binding. The caller
  /// is responsible for ensuring the keyword is not already present.
  void addPrechecked(const ExprNode *expr, TypedAttr precheckedBinding);

  /// Add a bound value for a positional parameter binding.
  void add(const ExprNode *expr, TypedAttr value);
  /// Add a bound value for a keyword parameter binding. The caller is
  /// responsible for ensuring the keyword is not already present.
  void add(const ExprNode *expr, PValue value, StringAttr name);

  /// The type of the function called when performing parameter inference. The
  /// hook will be provided the index of the parameter to be inferred, along
  /// with a list of existing bindings, and a parameter evaluator to be used to
  /// infer types.
  using ParameterInferenceHookTy =
      function_ref<PValue(ArrayRef<TypedAttr>, const ParameterEvaluator &)>;

  /// Describe how closely the given parameter bindings match the specified
  /// parameters and call operands.
  struct Fitness {
    /// The number of implicit conversion in the parameter bindings.
    size_t numImplicitConversions;

    /// Whether the bindings include variadic parameters.
    bool hasVariadicParams;
  };

  /// Helper class to customizing diagnostic emission for verification. The
  /// default implementation suppresses all diagnostics.
  struct DiagEmitter {
    /// Emit diagnostics for incorrect parameter count given the actual
    /// parameter count. The flag indicates if this is due to an insufficient
    /// number of positional-only parameters.
    std::function<void(size_t, bool)> emitParamCount;
    /// Emit diagnostics for incorrect type in a positional parameter.
    std::function<void(size_t, ASTExprAnd<AnyValue>, ASTType)> emitPosType;
    /// Emit diagnostics for incorrect type in a keyword parameter.
    std::function<void(StringAttr, ASTExprAnd<AnyValue>, ASTType)> emitKwType;
    /// Emit diagnostics for parameters specified by an unknown keyword.
    std::function<void(ArrayRef<StringAttr>)> emitUnknownKeywords;
    /// Emit diagnostics for a parameter specified both by position and keyword.
    std::function<void(ArrayRef<StringAttr>)> emitRedundantKeywords;
    /// Emit diagnostics for positional-only parameters specified by keyword.
    std::function<void(ArrayRef<StringAttr>)> emitPosOnlyPassedByKw;
    /// Emit diagnostics for out-of-order explicitly-specified inferred keyword.
    std::function<void(ArrayRef<StringAttr>)> emitOutOfOrderInferredKw;
    /// Emit diagnostics for failure to deduce a parameter.
    std::function<void(size_t)> emitDeductionFailure;
    /// Emit diagnostics when an unbound (i.e. `_`) is passed to a variadic.
    std::function<void(const ExprNode *)> emitUnboundInVariadic;
    /// Emit diagnostics when an unpacked unbind isn't the last parameter.
    std::function<void(const ExprNode *, bool)> emitUnpackedNotAtEnd;
    /// Emit diagnostics for failure to deduce an infer-only parameter.
    std::function<void(size_t)> emitInferOnlyFailure;
    /// Emit diagnostics for missing parameters (specified by their names).
    std::function<void(ArrayRef<StringAttr>, const Twine &)> emitMissing;
  };

  /// Verify the full parameter bindings for the given signature. If the
  /// signature doesn't match, the provided DiagEmitter will be used to emit
  /// diagnostics. A parameter inference must must be provided.
  std::pair<ParameterExprArrayAttr, Fitness>
  verifyBindings(FnTypeGeneratorType sig, const DiagEmitter &diagEmitter,
                 ParameterInferenceHookTy parameterInferenceHook) const;

  /// Attempt to bind the current set of parameters to the provided signature.
  /// This applies parameter inference and any default values to form a full
  /// binding set, which is returned along with the binding fitness. This will
  /// form a partial binding of the signature.
  ParameterExprArrayAttr verifyBindings(FnTypeGeneratorType sig) const;

  /// Attempt to bind the current set of parameters to the provided parameter
  /// types and list. This applies parameter inference and any default values to
  /// form a full binding set, which is returned along with the binding fitness.
  /// If `partial` is true, this forms a partial binding list.
  ParameterExprArrayAttr verifyBindings(ArrayRef<Type> paramTypes,
                                        PogListAttr paramList,
                                        bool partial) const;

  /// Verify the parameter bindings for the given struct. If the struct doesn't
  /// match, diagnostics will be emitted using the struct's location and the
  /// given expression location.
  ParameterExprArrayAttr verifyBindings(StructDeclOp structOp,
                                        TypeSignatureType sig, SMLoc exprLoc,
                                        bool partial) const;

  /// Verify the parameter bindings for the given signature. If the signature
  /// doesn't match, diagnostics will be emitted using the given baseName and
  /// locations.
  ParameterExprArrayAttr
  verifyBindings(FnTypeGeneratorType sig, StringRef baseName, SMLoc exprLoc,
                 std::optional<Location> opLoc = std::nullopt) const;

  /// Check that our set of parameter bindings work with the specified input
  /// parameters. If so, return a checked ParameterExprArrayAttr, along with
  /// information on how closely the bindings fit the parameters, or why
  /// they don't. The setEvaluator hook is used to install the parameter value
  /// in the evaluator used by the implementation. If the parameters do not
  /// work, this emits diagnostics using the locations and `baseName` provided.
  std::tuple<ParameterExprArrayAttr, Fitness, std::optional<InflightDiag>>
  verifyBindings(ArrayRef<Type> expectedParamTypes, PogListAttr paramListAttr,
                 const Twine &baseName, llvm::SMLoc exprLoc,
                 std::optional<Location> opLoc, bool partial) const;

  /// Method for debugging.
  LLVM_DUMP_METHOD void dump() const;

private:
  /// Check that our set of parameter bindings work with the specified input
  /// parameters. If so, return a checked ParameterExprArrayAttr, along with
  /// information on how closely the bindings fit the parameters, or why
  /// they don't. The setEvaluator hook is used to install the parameter value
  /// in the evaluator used by the implementation. This overload allows
  /// customizing diagnostics by passing a custom DiagEmitter.
  std::pair<ParameterExprArrayAttr, Fitness>
  verifyBindingsImpl(const CallOperands &operands,
                     ArrayRef<Type> expectedParamTypes,
                     PogListAttr paramListAttr,
                     ParameterInferenceHookTy parameterInferenceHook,
                     const DiagEmitter *diagEmitter, bool partial) const;

  /// This contains the values that are bound into this parameter list.
  CallOperands parameters;

  /// A list of all default parameter values declared for a type, if these are
  /// bindings for an overload set on a method.
  /// FIXME: When parameterization is rebuilt remove these two fields.
  ArrayRef<TypedAttr> defaultPosTypeParams;
  ArrayRef<TypedAttr> defaultKwTypeParams;

  /// FIXME: When parameterization is rebuilt remove this field.
  /// Store the passing kind of the original parameter so we can search the
  /// corresponding defaults array.
  SmallVector<PogMetadataAttr> ctadPogs;

  /// FIXME: When parameterization is rebuilt remove this field.
  /// The number of parameters declared for a type, if these are bindings for an
  /// overload set on a method.
  size_t numPosCtadParams = 0;
  size_t numKwOnlyCtadParams = 0;

  /// The number of pre-type-checked positional arguments.
  /// FIXME: Remove this, why is this needed?
  size_t numPreTypeChecked = 0;
};

FnTypeGeneratorType substituteTraitAliasesIntoSignature(
    DeclResolver &declResolver, ASTDecl *traitDecl, FnOp candidateFunc,
    FnTypeGeneratorType desiredSignature, PValue selfPValue);

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARAMBINDINGS_H

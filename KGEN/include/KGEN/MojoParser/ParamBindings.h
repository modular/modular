//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PARAMBINDINGS_H
#define KGEN_MOJOPARSER_PARAMBINDINGS_H

#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/TypeCheckScopeInfo.h"
#include "llvm/ADT/MapVector.h"

namespace M::KGEN {
class ParameterExprArrayAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {
using llvm::SMLoc;
class ExprNode;
class FuncOp;
class LITSignatureType;
class ParserParamEvaluator;
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
class ParamBindings : public TypeCheckScopeInfo {
public:
  /// Initialize ParamBindings with a declscope to perform lookups against
  /// and a notion of shared context.
  ParamBindings(const TypeCheckScopeInfo &scopeInfo)
      : TypeCheckScopeInfo(scopeInfo) {}
  ParamBindings(const ParamBindings &) = default;

  /// Replace our bindings with another set.
  void operator=(ParamBindings &&other);

  /// Create a (possibly partially unbound) set of bindings for the given type.
  /// This can be used to initialize the binding set for methods. If the given
  /// type is not a parametric user defined type, this returns empty bindings.
  static ParamBindings getForDeclaredType(const TypeCheckScopeInfo &scopeInfo,
                                          ASTType type, const ExprNode *expr);

  /// Utility function to perform substitutions of the bindings into the symbol
  /// for the given function declaration. It returns the resultant
  /// SymbolConstantAttr or produces an error message and returns null.
  TypedAttr getBoundConstAttrFor(LIT::FuncOp funcOp, StringRef baseName,
                                 const ExprNode *expr) const;

  /// Return whether there are any bindings given.
  bool empty() const { return posBindings.empty() && kwBindings.empty(); }

  /// Return the total number of bindings, including keyword and positional.
  size_t size() const { return posBindings.size() + kwBindings.size(); }

  ArrayRef<ASTExprAnd<AnyValue>> getPosBindings() const { return posBindings; }
  /// This contains the bound parameters given by a keyword.
  const llvm::MapVector<StringAttr, ASTExprAnd<AnyValue>,
                        SmallDenseMap<StringAttr, size_t>> &
  getKWBindings() const {
    return kwBindings;
  }

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
      function_ref<PValue(ArrayRef<TypedAttr>, const ParserParamEvaluator &)>;

  /// Describe how closely the given parameter bindings match the specified
  /// parameters and call operands.
  struct Fitness {
    /// The number of implicit conversion in the parameter bindings.
    size_t numImplicitConversions;

    /// Whether the bindings include variadic parameters.
    bool hasVariadicParams;

    /// The last expected type if there aren't enough bindings for
    /// positional-or-keyword parameters.
    Type lastExpectedType = {};
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
    /// Emit diagnostics for failure to deduce a parameter.
    std::function<void(size_t)> emitDeductionFailure;
    /// Emit diagnostics when an unbound pack (i.e. `*_`) appears in a variadic
    /// signature.
    std::function<void(ASTExprAnd<AnyValue>)> emitUnboundPackInVariadic;
    /// Emit diagnostic when unbound pack is not at the end of the param list.
    std::function<void(ASTExprAnd<AnyValue>)> emitUnboundPackNotEnd;
    /// Emit diagnostics for failure to deduce an infer-only parameter.
    std::function<void(size_t)> emitInferOnlyFailure;
    /// Emit diagnostics for missing parameters (specified by their names).
    std::function<void(ArrayRef<StringAttr>, const Twine &)> emitMissing;
    /// Emit diagnostics for too many positional parameters.
    std::function<void(size_t, size_t)> emitTooManyPositional;
  };

  /// Verify the parameter bindings for the given signature. If the signature
  /// doesn't match and a DiagEmitter was provided, it will be used to emit
  /// diagnostics. An optional hook can be provided to infer parameters. If
  /// `partial` is true, then we allow the signature to be partially bound: it
  /// can be missing parameters.
  std::pair<ParameterExprArrayAttr, Fitness>
  verifyBindings(LITSignatureType sig, const DiagEmitter *diagEmitter = nullptr,
                 ParameterInferenceHookTy parameterInferenceHook = {},
                 bool partial = true) const;

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
  verifyBindings(LITSignatureType sig, StringRef baseName, SMLoc exprLoc,
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
  verifyBindingsImpl(ArrayRef<Type> expectedParamTypes,
                     PogListAttr paramListAttr,
                     ParameterInferenceHookTy parameterInferenceHook,
                     const DiagEmitter *diagEmitter, bool partial) const;

  /// This contains a list of bound parameters given positionally.
  SmallVector<ASTExprAnd<AnyValue>> posBindings;

  /// This contains the bound parameters given by a keyword.
  llvm::MapVector<StringAttr, ASTExprAnd<AnyValue>,
                  SmallDenseMap<StringAttr, size_t>>
      kwBindings;

  /// A list of all default parameter values declared for a type, if these are
  /// bindings for an overload set on a method.
  ArrayRef<TypedAttr> defaultTypeParams;

  /// The number of parameters declared for a type, if these are bindings for an
  /// overload set on a method.
  size_t numCtadParams = 0;

  /// The number of pre-type-checked positional arguments.
  /// FIXME: Remove this, why is this needed?
  size_t numPreTypeChecked = 0;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARAMBINDINGS_H

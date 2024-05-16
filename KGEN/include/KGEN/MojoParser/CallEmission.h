//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CALLEMISSION_H
#define KGEN_MOJOPARSER_CALLEMISSION_H

#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/TypeCheckScopeInfo.h"

namespace M::KGEN {
class ParameterExprArrayAttr;
class SignatureType;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class LITSignatureType;
class PogListAttr;
class ParserParamEvaluator;
class StructDeclOp;
class TypeSignatureType;

/// Struct that carries both positional and keyword operands for a call or
/// parameter binding. This does not own any values, only references pointers
/// to their containers.
template <typename OperandType>
class OperandContainer {
public:
  /// Create call operands with positional and optional keyword arguments.
  OperandContainer(
      ArrayRef<OperandType> posOperands = {},
      const KeywordOperandContainer<OperandType> *kwOperands = nullptr)
      : posOperands(posOperands), kwOperands(kwOperands) {}

  /// Create call operands with positional arguments given a value implicitly
  /// convertible to `ArrayRef`.
  template <typename OperandsT,
            typename = std::enable_if_t<
                !std::is_same_v<OperandsT, ArrayRef<OperandType>> &&
                std::is_convertible_v<OperandsT, ArrayRef<OperandType>>>>
  OperandContainer(OperandsT &&posOperands)
      : OperandContainer(
            ArrayRef<OperandType>(std::forward<OperandsT>(posOperands))) {}

  /// Return a keyword argument value if present, or null otherwise.
  std::optional<OperandType> findKwArg(StringAttr argName) const {
    if (hasKwOperands())
      if (auto it = kwOperands->find(argName); it != kwOperands->end())
        return it->second;
    return std::nullopt;
  }

  /// Return the number of keyword operands.
  size_t getNumKwOperands() const {
    return kwOperands ? kwOperands->size() : 0;
  }

  /// Return if there are any keyword operands specified.
  bool hasKwOperands() const { return getNumKwOperands(); }

  /// The values passed as positional operands.
  ArrayRef<OperandType> posOperands;

  /// The values passed as keyword operands.
  const KeywordOperandContainer<OperandType> *kwOperands;
};

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
  struct Binding {
    /// This is the expression tree that produced the binding in the case of an
    /// Attribute, or null in the case of TypedAttr.
    const ExprNode *expr;
    /// This is the value of the binding.
    TypedAttr value;
    /// This flag is set to true if the value has been type checked.
    bool typeChecked;

    TypedAttr getValue() const {
      if (typeChecked)
        return {};
      return value;
    }

    /// Return the type of the TypedAttr or the binding.
    ASTType getType() const { return value.getType(); }
  };

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
                                          ASTType type);

  /// Utility function to perform substitutions of the bindings into the symbol
  /// for the given function declaration. It returns the resultant
  /// SymbolConstantAttr or produces an error message and returns null.
  TypedAttr getBoundConstAttrFor(LIT::FuncOp funcOp, StringRef baseName,
                                 const ExprNode *expr) const;

  /// Return whether there are any bindings given.
  bool empty() const { return posBindings.empty() && kwBindings.empty(); }

  /// Return the total number of bindings, including keyword and positional.
  size_t size() const { return posBindings.size() + kwBindings.size(); }

  /// Add a bound value for pre-checked positional parameter binding. The caller
  /// is responsible for ensuring the keyword is not already present.
  void addPrechecked(TypedAttr precheckedBinding);
  /// Add a bound value for pre-checked keyword parameter binding.
  void addPrechecked(TypedAttr precheckedBinding, StringAttr name);

  /// Add a bound value for a positional parameter binding.
  void add(const ExprNode *expr, TypedAttr value);
  /// Add a bound value for a keyword parameter binding. The caller is
  /// responsible for ensuring the keyword is not already present.
  void add(const ExprNode *expr, TypedAttr value, StringAttr name);

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

  /// This enum represents how bound a parameter list must be.
  enum class Boundness {
    /// The parameter list can be implicitly partially bound.
    /// FIXME(#32612): Require explicit unbinding.
    Partial,
    /// The parameter list can be explicitly partially bound.
    Explicit,
    /// The parameter list must be fully bound.
    Full
  };

  /// Verify the parameter bindings for the given struct. If the struct doesn't
  /// match, diagnostics will be emitted using the struct's location and the
  /// given expression location.
  ParameterExprArrayAttr verifyBindings(StructDeclOp structOp,
                                        TypeSignatureType sig,
                                        llvm::SMLoc exprLoc,
                                        Boundness boundness) const;

  /// Helper class to customizing diagnostic emission for verification. The
  /// default implementation suppresses all diagnostics.
  struct DiagEmitter {
    /// Emit diagnostics for incorrect parameter count given the actual
    /// parameter count. The flag indicates if this is due to an insufficient
    /// number of positional-only parameters.
    std::function<void(size_t, bool)> emitParamCount;
    /// Emit diagnostics for incorrect type in a positional parameter.
    std::function<void(size_t, const Binding &, ASTType)> emitPosType;
    /// Emit diagnostics for incorrect type in a keyword parameter.
    std::function<void(StringAttr, const Binding &, ASTType)> emitKwType;
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
    std::function<void(const Binding &)> emitUnboundPackInVariadic;
    /// Emit diagnostic when unbound pack is not at the end of the param list.
    std::function<void(const Binding &)> emitUnboundPackNotEnd;
    /// Emit diagnostics for failure to deduce an infer-only parameter.
    std::function<void(size_t)> emitInferOnlyFailure;
    /// Emit diagnostics for missing parameters (specified by their names).
    std::function<void(ArrayRef<StringAttr>, const Twine &)> emitMissing;
    /// Emit diagnostics for too many positional parameters.
    std::function<void(size_t, size_t)> emitTooManyPositional;
  };

  /// Verify the parameter bindings for the given signature. If the signature
  /// doesn't match  no diagnostics will be emitted.
  std::pair<ParameterExprArrayAttr, Fitness>
  verifyBindings(LITSignatureType sig) const;

  /// Verify the parameter bindings for the given signature. If the signature
  /// doesn't match, the provided DiagEmitter will be used to emit diagnostics.
  /// An optional hook can be provided to infer parameters.
  std::pair<ParameterExprArrayAttr, Fitness>
  verifyBindings(LITSignatureType sig, const DiagEmitter &diagEmitter,
                 ParameterInferenceHookTy parameterInferenceHook = {}) const;

  /// Verify the parameter bindings for the given signature. If the signature
  /// doesn't match, diagnostics will be emitted using the given baseName and
  /// locations.
  ParameterExprArrayAttr
  verifyBindings(LITSignatureType sig, StringRef baseName, llvm::SMLoc exprLoc,
                 std::optional<Location> opLoc = std::nullopt) const;

  /// Allow implicit conversion to an operand container.
  operator OperandContainer<Binding>() const {
    return {posBindings, &kwBindings};
  }

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
  verifyBindings(ArrayRef<Type> expectedParamTypes, PogListAttr paramListAttr,
                 ParameterInferenceHookTy parameterInferenceHook,
                 const DiagEmitter &diagEmitter,
                 Boundness boundness = Boundness::Explicit) const;

  /// Check that our set of parameter bindings work with the specified input
  /// parameters. If so, return a checked ParameterExprArrayAttr, along with
  /// information on how closely the bindings fit the parameters, or why
  /// they don't. The setEvaluator hook is used to install the parameter value
  /// in the evaluator used by the implementation. If the parameters do not
  /// work, this emits diagnostics using the locations and `baseName` provided.
  std::pair<ParameterExprArrayAttr, Fitness>
  verifyBindings(ArrayRef<Type> expectedParamTypes, PogListAttr paramListAttr,
                 const Twine &baseName, llvm::SMLoc exprLoc,
                 std::optional<Location> opLoc = std::nullopt,
                 Boundness boundness = Boundness::Explicit) const;

  /// This contains a list of bound parameters given positionally.
  SmallVector<Binding> posBindings;

  /// This contains the bound parameters given by a keyword.
  llvm::MapVector<StringAttr, Binding, SmallDenseMap<StringAttr, size_t>>
      kwBindings;

  /// A list of all default parameter values declared for a type, if these are
  /// bindings for an overload set on a method.
  ArrayRef<TypedAttr> defaultTypeParams;

  /// The number of parameters declared for a type, if these are bindings for an
  /// overload set on a method.
  size_t numCtadParams = 0;
};

/// When emitting a function call, this enum is used to indicate why the call
/// happened in the first place.  This allows producing better-tuned
/// diagnostics.
enum class CallSyntax : uint8_t {
  kDirectCall,         //< f()
  kIndirectCall,       //< expr()
  kMethodCall,         //< x.f()
  kTypeCall,           //< T()
  kOperator,           //< -x and x + y
  kReversedOperator,   //< y + x          (where the method was looked up on x).
  kSubscript,          // v[1, 2]
  kAttribute,          // v.x             (where x is not a static member of v).
  kImplicitConvert,    //< Conversion in an argument context
  kDestructor,         //< Destructor due to a value definition.
  kTupleGetItem,       //< Call to getitem in a tuple assignment.
  kMethodCallSynthetic //< Call to a method for synthetic checks.
};

/// Struct that carries both positional and keyword operands for a call. This
/// does not own any values, only references pointers to their containers.
class CallOperands : public OperandContainer<FuncOperand> {
public:
  using OperandContainer::OperandContainer;

  /// Inidicates if the positional operands include a self operand.
  bool hasSelfOperand = false;

  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, const CallOperands &value);

/// This class represents an unresolved overload set with partially bound
/// callees, e.g. "foo" or "a.foo" where "foo" is an overloaded declaration or
/// an incompletely bound function (e.g. one with result parameters).  This is
/// resolved when emitted to an RValue or when binding more things into it as
/// part of the expression tree.
///
/// Note that it is possible to have an overload set with methods from multiple
/// different self types that are related to each other.  For example when Mojo
/// has classes, it will be common to have super-class methods that expect
/// 'self' to be converted to a different type in order to invoke it.  For
/// nonmaterializable types like IntLiteral, we can have methods on both Int and
/// IntLiteral, etc.  Filtering the overload set will pick the appropriate
/// method.
class OverloadSet {
public:
  /// In a method reference like `x.foo`, this is the base object being invoked,
  /// e.g. `x`.
  ASTExprAnd<AnyValue> baseValue;

  /// This is the basename of the declaration set, used in diagnostics.
  StringRef baseName;

  /// The function overload set that may be called directly.
  SmallVector<ASTDecl *, 1> fnDecls;

  /// Any bound parameters.
  ParamBindings paramBindings;

  /// This is information about where this overload set was formed.
  const ExprNode *expr;
  CallSyntax syntax;

  /// When doing resolution, we should only raise new errors if previous errors
  /// haven't already been raised about functions in the overload set.  The most
  /// common issue is when one of the included declarations is erroneous.
  /// Emitting further errors about overload resolution failure can then be
  /// spurious, since we can't properly consider the erroneous declarations
  /// which otherwise might match.  This flag guards against raising those extra
  /// errors.
  bool erroneous;

  /// Form an overload set with the specified function overloads and the given
  /// parameter bindings. The parameter bindings are taken ownership of.
  OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
              ParamBindings &&paramBindings, const ExprNode *expr,
              CallSyntax syntax, bool erroneous = false);

  /// Form an OverloadSet with a lookup of a named method on the specified type,
  /// but without the candidate set filtered with operands.   If successful,
  /// this provides a non-null OverloadSet.
  ///
  /// On failure, this returns a null OverloadSet and invokes errorHandler if
  /// the problem hasn't already been diagnosed and it is non-null. This does
  /// not emit an error on failure.
  static OverloadSet lookup(const TypeCheckScopeInfo &scopeInfo, ASTType type,
                            StringRef methodName, const ExprNode *callExpr,
                            CallSyntax syntax,
                            function_ref<void()> errorHandler = {});

  /// Lookup of a named method on the specified type, filtered to match a
  /// concrete operand set. If successful, this provides a non-null PValue for a
  /// single callee. If non-null, it invokes lookupFailureErrorHandler if the
  /// lookup of the named method fails.  If that succeeds, it will complain
  /// about overload resolution when 'shouldPrintOverloadErrors' is true.
  static PValue lookup(const TypeCheckScopeInfo &scopeInfo, ASTType type,
                       StringRef methodName, const CallOperands &callOperands,
                       const ExprNode *callExpr, CallSyntax syntax,
                       function_ref<void()> lookupFailureErrorHandler,
                       bool shouldPrintOverloadErrors);

  /// Same as the above but a convenience when never emitting an error.
  static PValue lookup(const TypeCheckScopeInfo &scopeInfo, ASTType type,
                       StringRef methodName, const CallOperands &callOperands,
                       const ExprNode *callExpr, CallSyntax syntax) {
    return lookup(scopeInfo, type, methodName, callOperands, callExpr, syntax,
                  {}, false);
  }

  bool isNull() const { return fnDecls.empty(); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  /// An overload set is erroneous primarily when constructed with erroneous
  /// decls.  If an overload set is erroneous, you can't necessarily trust
  /// lookup results when processing to find further errors.
  bool isErroneous() const { return erroneous; }

  const TypeCheckScopeInfo &getScopeInfo() const { return paramBindings; }
  SharedState &getShared() const { return paramBindings.shared; }

  /// Perform substitutions of the specified bindings into the symbol, returning
  /// the resultant LITSymbolConstant attr or producing an error message and
  /// returning null. This allows producing a reference to a parameterized
  /// function without the parameters specified.  They can be bound later.
  TypedAttr getBoundConstantAttr() const;

  /// Evaluate the fnDecls candidates and see if there is an unambiguous
  /// candidate that works with the specified parameter bindings and provided
  /// arguments.  If so, return the single entry that works.  If not, generate a
  /// diagnostic (when `emitDiagnosticOnFailure` is true) and return null.
  PValue filterOverloadSet(const CallOperands &operands,
                           bool allowImplicitConversions,
                           bool emitDiagnosticOnFailure) const;
  PValue filterOverloadSet(const CallOperands &operands,
                           SmallVectorImpl<ASTDecl *> &newFnDecls,
                           bool allowImplicitConversions,
                           bool emitDiagnosticOnFailure) const;

  /// Try to resolve the overload set to a single function candidate, using the
  /// expected type if provided or using current bindings if an emitter is
  /// provided.  This emits errors if 'emitter' is non-null, but does not if it
  /// is null.
  PValue getDirectSymbol(ASTType expectedType) const;

  /// Try to emit the overload set as a PValue.
  PValue getIfPValue() const;

  /// Emit this as a CValue if it can be resolved, otherwise emit an ambiguity
  /// error and return null.
  CValue emitAsCValue(ExprEmitter &emitter, ValueDest &dest);

  /// Emit a function call to the specified callee with the specified operand
  /// values.  This emits an error and returns null on failure.
  ///
  /// `callNode` is the call like expression (e.g. a CallNode, binary operator,
  /// etc) that results in the call, or potentially a random value that is being
  /// fed into an implicit conversion.  This should only be used for location
  /// information.
  CValue emitCall(const CallOperands &callOperands, ValueDest &dest,
                  ExprEmitter &emitter);

  /// Filter down and complete this overload set based on knowledge that we need
  /// to produce a function pointer with the specified type.  This returns a
  /// PValue for the callee if resolvable or null if not.
  PValue filterOverloadSetForValueType(ASTType functionType,
                                       bool emitDiagnosticOnFailure) const;
  PValue filterOverloadSetForValueType(
      ASTType functionType,
      function_ref<InflightDiag &(llvm::SMLoc)> emitError) const;

private:
  OverloadSet(const TypeCheckScopeInfo &scopeInfo, const ExprNode *expr,
              CallSyntax syntax, bool erroneous)
      : paramBindings(scopeInfo), expr(expr), syntax(syntax),
        erroneous(erroneous){};
};

/// This provides a wrapper around OverloadSet which is reference counted,
/// allowing OverloadSetUValue to maintain it while still being copyable.
struct OverloadSetUValue::OverloadSetWrapper
    : public NonAtomicallyReferenceCounted<OverloadSetWrapper> {

  OverloadSetWrapper(OverloadSet &&overloadSet)
      : overloadSet(std::move(overloadSet)) {}
  OverloadSet overloadSet;
};

//===----------------------------------------------------------------------===//
// OverloadSetUValue implementation details
//===----------------------------------------------------------------------===//

template <typename... Args>
inline OverloadSetUValue OverloadSetUValue::create(Args &&...args) {
  return OverloadSetUValue(takeRCRef(
      new OverloadSetWrapper(OverloadSet(std::forward<Args>(args)...))));
}

inline const OverloadSet &OverloadSetUValue::operator*() const {
  return storage.getPointer()->overloadSet;
}

inline OverloadSet &OverloadSetUValue::operator*() {
  return storage.getPointer()->overloadSet;
}

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CALLEMISSION_H

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#ifndef CALLEMISSION_H
#define CALLEMISSION_H

#include "IRValues.h"

namespace M::KGEN {
class ParamDeclArrayAttr;
class ParamBindAttr;
class ParameterExprArrayAttr;
class SignatureType;
class SymbolConstantAttr;
class TypeArrayAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class FuncOp;

//===----------------------------------------------------------------------===//
// InputParamBindings
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
class InputParamBindings {
public:
  struct Binding {
    /// This is the expression tree that produced the binding in the case of an
    /// Attribute, or null in the case of ParamBindAttr.
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

  /// This contains a list of bound input parameters.
  SmallVector<Binding> bindings;

  /// Add a bound value for a pre-checked parameter bindings.  The binding must
  /// be known to be valid.
  void addPrechecked(TypedAttr precheckedBinding);

  /// Add a bound value for a parameter expression bound to a value.
  void add(const ExprNode *expr, TypedAttr value) {
    bindings.push_back({expr, value, /*typeChecked=*/false});
  }

  using ParameterInferenceHookTy =
      std::function<PValue(size_t index, Type type, ASTType expectedType,
                           ArrayRef<TypedAttr> bindings)>;

  /// Describe how closely the given parameter bindings match the specified
  /// input parameters and call operands.
  struct Fitness {
    /// The number of implicit conversion in the parameter bindings.
    size_t numImplicitConversions;

    /// Whether the bindings include variadic parameters.
    bool hasVariadicParams;
  };

  /// Check that our set of parameter bindings work with the specified input
  /// parameters and call operands (if any). If so, return a checked
  /// ParamBindArrayAttr, along with information on how closely the bindings fit
  /// the input parameters. If the parameters do not work, this emits an
  /// diagnostic (if `declOp` is non-null) and sets
  /// `incorrectBindingNo/Expectedtype` to the bad binding (or -1 if there is a
  /// count mismatch).
  ///
  /// This rejects the signature list if all the parameters are not bound.
  std::pair<ParameterExprArrayAttr, Fitness>
  verifyBindings(ArrayRef<Type> actualParamTypes,
                 ParamDeclArrayAttr actualParamDecls, StringRef baseName,
                 llvm::SMLoc loc, ssize_t &incorrectBindingNo,
                 ASTType &incorrectBindingExpectedType, ExprEmitter &emitter,
                 Operation *declOp, bool paramVarargs, bool packVarargs = false,
                 ArrayRef<ASTExprAnd<AnyValue>> callOperands = {},
                 ParameterInferenceHookTy parameterInferenceHook = {}) const;

  /// Given a candidate that may or may not be compatible with the given
  /// parameter set so far, indicate what the next parameter's expected type
  /// should be, or return null if the current parameters are incompatible with
  /// it.
  ASTType getNextExpectedBindingType(SignatureType candidateType,
                                     ExprEmitter &emitter) const;
};

/// When emitting a function call, this enum is used to indicate why the call
/// happened in the first place.  This allows producing better-tuned
/// diagnostics.
enum class CallSyntax : uint8_t {
  kDirectCall,       //< f()
  kIndirectCall,     //< expr()
  kMethodCall,       //< x.f()
  kTypeCall,         //< T()
  kOperator,         //< -x and x + y
  kReversedOperator, //< y + x          (where the method was looked up on x).
  kSubscript,        // v[1, 2]
  kAttribute,        // v.x             (where x is not a static member of v).
  kImplicitConvert,  //< Conversion in an argument context
  kDestructor,       //< Destructor due to a value definition.
  kTupleGetItem,     //< Call to getitem in a tuple assignment.
};

// Struct to that carries both positional and keyword operands for a call. This
// does not own any values, only references and pointers to their containers.
struct CallOperands {
  CallOperands() : posOperands({}){};
  CallOperands(ArrayRef<ASTExprAnd<AnyValue>> posOperands)
      : posOperands(posOperands){};
  CallOperands(ArrayRef<ASTExprAnd<AnyValue>> posOperands,
               const SmallDenseMap<StringRef, ASTExprAnd<AnyValue>> *kwOperands)
      : posOperands(posOperands), kwOperands(kwOperands) {}

  /// Return if there are any keyword operands specified.
  bool hasKwOperands() const { return kwOperands && !kwOperands->empty(); }

  /// The values passed as positional operands.
  ArrayRef<ASTExprAnd<AnyValue>> posOperands;

  /// The values passed as keyword operands.
  const SmallDenseMap<StringRef, ASTExprAnd<AnyValue>> *kwOperands = nullptr;
};

/// This class represents an unresolved overload set with partially bound
/// callees, e.g. "foo" or "a.foo" where "foo" is an overloaded declaration or
/// an incompletely bound function (e.g. one with result parameters).  This is
/// resolved when emitted to a CRValue or when binding more things into it as
/// part of the expression tree.
class OverloadSet {
public:
  /// In a method reference like `x.foo`, this is the base object being invoked,
  /// e.g. `x`.
  ASTExprAnd<AnyValue> baseValue;

  /// This is the basename of the declaration set, used in diagnostics.
  StringRef baseName;

  /// The function overload set that may be called directly.
  SmallVector<ASTDecl *, 1> fnDecls;

  /// Any bound input parameters.
  InputParamBindings inputParamBindings;

  /// This is a list of result parameters that are to be bound to the returned
  /// parameters from the call.
  std::vector<std::pair<ASTDecl *, llvm::SMLoc>> resultParams;

  /// This is information about where this overload set was formed.
  const ExprNode *expr;
  CallSyntax syntax;

  /// Form an overload set with the specified function overloads.
  OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
              ParameterExprArrayAttr bindings, const ExprNode *expr,
              CallSyntax syntax);
  OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
              ParamBindArrayAttr bindings, const ExprNode *expr,
              CallSyntax syntax);

  /// Form an OverloadSet with a lookup of a named method on the specified type.
  /// If successful, this provides a non-null OverloadSet.
  ///
  /// On failure, this returns a null OverloadSet and invokes errorHandler if
  /// the problem hasn't already been diagnosed and it is non-null. This does
  /// not emit an error on failure.
  OverloadSet(ASTType type, StringRef methodName, const ExprNode *callExpr,
              CallSyntax syntax, SharedState &shared,
              std::function<void()> errorHandler);

  /// Lookup of a named named method on the specified type, filtered to match a
  /// concrete operand set. If successful, this provides a non-null PValue for a
  /// single callee.
  static PValue lookup(ASTType type, StringRef methodName,
                       const CallOperands &callOperands,
                       const ExprNode *callExpr, CallSyntax syntax,
                       ExprEmitter &emitter,
                       std::function<void()> errorHandler);

  bool isNull() const { return fnDecls.empty(); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  /// Perform substitutions of the specified bindings into the symbol, returning
  /// the resultant LITSymbolConstant attr or producing an error message and
  /// returning null. This allows producing a reference to a parameterized
  /// function without the parameters specified.  They can be bound later.
  TypedAttr getBoundConstantAttr(ExprEmitter &emitter) const;

  /// Evaluate the fnDecls candidates and see if there is an unambiguous
  /// candidate that works with the specified parameter bindings and provided
  /// arguments.  If so, return the single entry that works.  If not, generate a
  /// diagnostic (when `emitDiagnosticOnFailure` is true) and return null.
  PValue filterOverloadSet(const CallOperands &operands,
                           bool allowImplicitConversions,
                           bool emitDiagnosticOnFailure,
                           ExprEmitter &emitter) const;

  /// Try to resolve the overload set to a single function candidate, using the
  /// expected type if provided or using current bindings if an emitter is
  /// provided.
  PValue getDirectSymbol(ExprEmitter *emitter, ASTType expectedType) const;

  /// Try to emit the overload set as a PValue.
  PValue emitAsPValue(ExprEmitter *emitter = nullptr,
                      ASTType expectedType = {}) const;

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
                                       bool emitDiagnosticOnFailure,
                                       ExprEmitter &emitter) const;

  /// Resolve the callee into either a single PValue callee (if there's only one
  /// decl provided) or a variadic that contains all the possible adaptive
  /// overloads.
  PValue getAdaptiveSet(ExprEmitter &emitter);

private:
  /// Resolve the callee into either a single PValue callee (if there's only
  /// one decl provided) or a variadic that contains all the possible adaptive
  /// overloads.
  static PValue getCallee(ArrayRef<ASTDecl *> fnDecls, StringRef baseName,
                          InputParamBindings inputParamBindings,
                          const ExprNode *expr, ExprEmitter &emitter);
};

/// This provides a wrapper around OverloadSet which is reference counted,
/// allowing ORValue to maintain it while still being copyable.
struct ORValue::OverloadSetWrapper
    : public LLCL::NonAtomicallyReferenceCounted<OverloadSetWrapper> {

  OverloadSetWrapper(OverloadSet &&overloadSet)
      : overloadSet(std::move(overloadSet)) {}
  OverloadSet overloadSet;
};

/// Returns whether the two signatures match, i.e. if they only differ in
/// argument names.
bool canZeroCostConvertSignature(SignatureType from, SignatureType to);

//===----------------------------------------------------------------------===//
// ORValue implementation details
//===----------------------------------------------------------------------===//

template <typename... Args>
inline ORValue ORValue::create(Args &&...args) {
  return ORValue(LLCL::takeRCRef(
      new OverloadSetWrapper(OverloadSet(std::forward<Args>(args)...))));
}

inline const OverloadSet &ORValue::operator*() const {
  return storage.getPointer()->overloadSet;
}

inline OverloadSet &ORValue::operator*() {
  return storage.getPointer()->overloadSet;
}

} // namespace M::KGEN::LIT

#endif // CALLEMISSION_H

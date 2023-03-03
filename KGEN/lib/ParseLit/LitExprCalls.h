//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_EXPRCALLS_H
#define LIT_EXPRCALLS_H

#include "IRValues.h"

namespace M::KGEN {
class ParamDeclArrayAttr;
class ParamBindAttr;
class SignatureType;
class SymbolConstantAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {

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
/// The type parameters (param1) will be bound as a ParamBindAttr, and the
/// param2 will be bound as the value of param2.  We cannot type check the
/// bindings until overload resolution has resolved which 'method' we are
/// talking about and when inference is complete, so we keep them as either a
/// ParamBindAttr or (Typed)Attribute for the actual value.
class InputParamBindings {
public:
  struct Binding {
    /// This is the expression tree that produced the binding in the case of an
    /// Attribute, or null in the case of ParamBindAttr.
    ExprNode *expr;
    Attribute bindingOrValue; // ParamBindAttr|TypedAttr.

    TypedAttr getValue() const { return dyn_cast<TypedAttr>(bindingOrValue); }

    /// Return the type of the TypedAttr or the binding.
    ASTType getType() const;
  };

  /// This contains a list of bound input parameters.
  SmallVector<Binding> bindings;

  /// Add a bound value for a pre-checked parameter bindings.  The binding must
  /// be known to be valid.
  void add(ParamBindAttr precheckedBinding);

  /// Add a bound value for a parameter expression bound to a value.
  void add(ExprNode *expr, TypedAttr value) {
    bindings.push_back({expr, value});
  }

  using ParameterInferenceHookTy =
      std::function<PRValue(ParamDeclAttr decl, ASTType expectedType,
                            ArrayRef<ParamBindAttr> bindings)>;

  /// Check that our set of parameter bindings work with the specified input
  /// parameters, returning a checked ParamBindArrayAttr if so.  If the
  /// parameters do not work, this emits an diagnostic (if `declOp` is non-null)
  /// and set `incorrectBindingNo/Expectedtype` to the bad binding (or -1 if
  /// there is a count mismatch).
  ///
  /// This rejects the signature list if all the parameters are not bound.
  ParamBindArrayAttr
  verifyBindings(ParamDeclArrayAttr actualParamDecls, StringRef baseName,
                 SMLoc loc, ssize_t &incorrectBindingNo,
                 ASTType &incorrectBindingExpectedType, ExprEmitter &emitter,
                 Operation *declOp, bool paramVarargs,
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
  kImplicitConvert,  //< Conversion in an argument context
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
  std::vector<std::pair<ASTDecl *, SMLoc>> resultParams;

  /// This is information about where this overload set was formed.
  const ExprNode *expr;
  CallSyntax syntax;

  /// Form an overload set with the specified function overloads.
  OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
              ParamBindArrayAttr bindings, const ExprNode *expr,
              CallSyntax syntax);

  /// Form an OverloadSet with a lookup of a named method on the specified type.
  /// If successful, this provides a non-null OverloadSet.
  ///
  /// On failure, this returns a null OverloadSet and invokes errorHandler if
  /// the problem hasn't already been diagnosed. This does not emit an error on
  /// failure.
  OverloadSet(ASTType type, StringRef methodName, const ExprNode *callExpr,
              CallSyntax syntax, LitSharedState &shared,
              std::function<void()> errorHandler);

  /// Form an OverloadSet with a lookup of a named method on the specified type,
  /// filtered to match a concrete operand set.
  /// If successful, this provides a non-null OverloadSet.
  OverloadSet(ASTType type, StringRef methodName,
              ArrayRef<ASTExprAnd<AnyValue>> operands, const ExprNode *callExpr,
              CallSyntax syntax, ExprEmitter &emitter,
              std::function<void()> errorHandler);

  bool isNull() const { return fnDecls.empty(); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  /// Perform subsitutions of the specified bindings into the symbol, returning
  /// the resultant LITSymbolConstant attr or producing an error message and
  /// returning null. This allows producing a reference to a parameterized
  /// function without the parameters specified.  They can be bound later.
  TypedAttr getBoundConstantAttr(ExprEmitter &emitter) const;

  /// Get a bound SymbolConstantAttr for a specific overload.
  TypedAttr getBoundConstAttrFor(LIT::FuncOp funcOp,
                                 ExprEmitter &emitter) const;

  /// Evaluate the fnDecls candidates and see if there is an unambiguous
  /// candidate that works with the specified parameter bindings and provided
  /// arguments.  If so, replace fnDecls with a single entry that works and
  /// return success.  If not, generate a diagnostic (when
  /// `emitDiagnosticOnFailure` is true) and return failure.
  ///
  /// On success and when `validCandidate` is non-null, `*validCandidate` is
  /// filled in with symbol for the valid callee along with its parameter
  /// bindings.
  LogicalResult filterOverloadSet(ArrayRef<ASTExprAnd<AnyValue>> operands,
                                  bool allowImplicitConversions,
                                  bool emitDiagnosticOnFailure,
                                  ExprEmitter &emitter);

  /// Emit this as a CRValue if it can be resolved, otherwise emit an ambiguity
  /// error and return null.  If `expectedType` is set, it is used to filter
  /// the overload set before emitting it.
  CRValue emitAsCRValue(ExprEmitter &emitter, ValueDest dest,
                        ASTType expectedType = {});

  /// Emit a function call to the specified callee with the specified operand
  /// values.  This emits an error and returns null on failure.
  ///
  /// `callNode` is the call like expression (e.g. a CallNode, binary operator,
  /// etc) that results in the call, or potentially a random value that is being
  /// fed into an implicit conversion.  This should only be used for location
  /// information.
  AnyValue emitCall(ArrayRef<ASTExprAnd<AnyValue>> operands, ValueDest dest,
                    ExprEmitter &emitter);

  /// Return true if 'value' may be implicitly converted to 'requiredType'
  /// by invoking (one level of) conversion operations.  This does not generate
  /// any IR.
  static bool canImplicitlyConvertToType(ASTExprAnd<AnyValue> value,
                                         ASTType requiredType,
                                         ExprEmitter &emitter);

private:
  /// Resolve the callee into either a single PRValue callee (if there's only
  /// one decl provided) or a variadic that contains all the possible adaptive
  /// overloads.
  PRValue getCallee(ExprEmitter &emitter) const;

  /// Filter down and complete this overload set based on knowledge that we need
  /// to produce a function pointer with the specified type.
  LogicalResult filterOverloadSetForValueType(ASTType functionType,
                                              ExprEmitter &emitter);
};

/// This provides a wrapper around OverloadSet which is reference counted,
/// allowing ORValue to maintain it while still being copyable.
struct ORValue::OverloadSetWrapper
    : public LLCL::NonAtomicallyReferenceCounted<OverloadSetWrapper> {

  OverloadSetWrapper(OverloadSet &&overloadSet)
      : overloadSet(std::move(overloadSet)) {}
  OverloadSet overloadSet;
};

//===----------------------------------------------------------------------===//
// ORValue implementation details
//===----------------------------------------------------------------------===//

template <typename... Args>
inline ORValue ORValue::create(Args &&...args) {
  return ORValue(LLCL::takeRCRef(
      new OverloadSetWrapper(OverloadSet(std::forward<Args>(args)...))));
}

inline OverloadSet *ORValue::operator->() {
  return &storage.getPointer()->overloadSet;
}
inline const OverloadSet *ORValue::operator->() const {
  return &storage.getPointer()->overloadSet;
}

} // namespace M::KGEN::LIT

#endif // LIT_EXPRCALLS_H

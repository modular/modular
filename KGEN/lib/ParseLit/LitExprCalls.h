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
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {
using llvm::SMLoc;
class ASTDecl;
class IREmitter;
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
/// The type parameters (param1) will be bound as a ParamBindAttr, and the
/// param2 will be bound as the value of param2.  We cannot type check the
/// bindings until overload resolution has resolved which 'method' we are
/// talking about and when inference is complete, so we keep them as either a
/// ParamBindAttr or (Typed)Attribute for the actual value.
///
/// TODO: This should grow to incorporate logic similar to KGEN::ConstraintSet.
class InputParamBindings {
public:
  struct Binding {
    /// This is the expression tree that produced the binding in the case of an
    /// Attribute, or null in the case of ParamBindAttr.
    ExprNode *expr;
    PointerUnion<ParamBindAttr, Attribute> bindingOrValue;

    TypedAttr getValue() const {
      if (auto attr = dyn_cast<Attribute>(bindingOrValue))
        return cast<TypedAttr>(attr);
      return {};
    }
    ASTType getType() const {
      if (auto attr = getValue())
        return attr.getType();
      return cast<ParamBindAttr>(bindingOrValue).getType();
    }
  };

  /// This contains a list of bound input parameters.
  SmallVector<Binding> bindings;

  /// Add a bound value for a pre-checked parameter bindings.  The binding must
  /// be known to be valid.
  void add(ParamBindAttr precheckedBinding) {
    bindings.push_back({nullptr, precheckedBinding});
  }

  /// Add a bound value for a parameter expression bound to a value.
  void add(ExprNode *expr, TypedAttr value) {
    bindings.push_back({expr, Attribute(value)});
  }

  using ParameterInferenceHookTy =
      std::function<MValue(ParamDeclAttr, ArrayRef<ParamBindAttr> bindings)>;

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
                 ASTType &incorrectBindingExpectedType, LitSharedState &shared,
                 Operation *declOp,
                 ParameterInferenceHookTy parameterInferenceHook = {}) const;
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

//===----------------------------------------------------------------------===//
// DirectCallable
//===----------------------------------------------------------------------===//

/// This struct models something that can be directly called, e.g. a global
/// symbol with any binding information.
struct DirectCallable {
  /// This is the basename of the declaration set, used in diagnostics.
  StringRef baseName;

  /// The function overload set that may be called directly.
  SmallVector<ASTDecl *, 1> fnDecls;

  /// Any bound input parameters.
  InputParamBindings inputParamBindings;

  /// This is a list of result parameters that are to be bound to the returned
  /// parameters from the call.
  std::vector<std::pair<ASTDecl *, SMLoc>> resultParams;

  /// When this is set to true, implicit conversions are not considered for
  /// argument and parameter values.
  bool disableImplicitConversions = false;

  DirectCallable(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
                 ParamBindArrayAttr bindings);

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
                                  CallSyntax syntax, const ExprNode *callExpr,
                                  bool emitDiagnosticOnFailure,
                                  LitSharedState &shared);

  /// Resolve the callee into either a single MValue callee (if there's only one
  /// decl provided) or a variadic that contains all the possible adaptive
  /// overloads. Because adaptive overloads must all have the same signature,
  /// this also returns the signature type that they all share.
  std::pair<MValue, SignatureType> getCallee();

  /// Perform subsitutions of the specified bindings into the symbol, returning
  /// the resultant LITSymbolConstant attr or producing an error message and
  /// returning null. This allows producing a reference to a parameterized
  /// function without the parmaeters specified.  They can be bound later.
  SymbolConstantAttr getBoundConstantAttr(const ExprNode *callExpr,
                                          LitSharedState &shared) const;

  /// Perform subsitutions of the specified bindings into the symbol, returning,
  /// in symConstAttrs, the resultant SymbolConstant attr for each adaptive
  /// function overload.
  /// On failure it produces an error message and returns failure.
  LogicalResult
  getBoundConstantAttrsAdaptiveSet(SmallVectorImpl<TypedAttr> &symConstAttrs,
                                   const ExprNode *callExpr,
                                   LitSharedState &shared) const;

  /// Check declarations for the result parameters and add them to
  /// resultParamDecls.  This emits and error and returns failure if an error is
  /// detected.
  LogicalResult
  getResultParamDecls(SignatureType signature,
                      SmallVectorImpl<ParamDeclAttr> &resultParamDecls,
                      IREmitter &emitter);
};

//===----------------------------------------------------------------------===//
// CallableValue
//===----------------------------------------------------------------------===//

/// This class is returned by the emitCallable hooks on AST expressions, which
/// captures aggregate callable values.  This is required to hold parametric
/// callees before their parameters are bound, e.g. in `obj.method[p1,p2](...)`
/// it may not be possible to emit `obj.method` as a RValue because it isn't
/// materializable, yet it needs to capture the dynamic value 'obj'.  Similarly
/// `obj.method` may resolve to an overload set which needs arguments to
/// disambiguate.
class CallableValue {
public:
  /// This is the expression tree this result was built from, for use in
  /// diagnostics.  This is null when the CallableValue is null.
  const ExprNode *expr;

  /// This is a dynamic value, which may either be an LValue or an RValue, that
  /// may itself be a callable, or (if targetSymbol is non-null), is the self
  /// argument to a call to the symbol.
  AnyValue baseVal;

  /// If present, this a reference to a fixed symbol or an overload set.
  std::optional<DirectCallable> direct;

  CallableValue() : expr(nullptr) {}
  CallableValue(ASTExprAnd<AnyValue> baseVal)
      : expr(baseVal.expr), baseVal(baseVal.ir) {}
  CallableValue(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
                ParamBindArrayAttr bindings, const ExprNode *expr)
      : expr(expr), direct({baseName, fnDecls, bindings}) {}

  /// Get a CallableValue for a lookup of a named method on the specified type.
  /// If successful, this provides a non-null CallableValue.
  ///
  /// On failure, this returns a null CallableValue and sets 'erroneousDecl' to
  /// indicate whether there was a problem with the callee that has already been
  /// diagnosed (allowing the client to squish downstream error messages).  This
  /// does not emit an error on failure.
  CallableValue(ASTType type, StringRef methodName, const ExprNode *callxpr,
                bool &erroneousDecl, LitSharedState &shared);

  bool isNull() const { return !baseVal && !direct; }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  /// Emit this as a flattened RValue or LValue.  This returns null on
  /// failure.
  AnyValue emitAsValue(IREmitter &emitter) const;

  /// Emit in values references of all adaptive function overloads this
  /// DirectCallable represents.
  LogicalResult emitAdaptiveSet(IREmitter &emitter,
                                SmallVectorImpl<TypedAttr> &values) const;

  /// Emit a function call to the specified callee with the specified operand
  /// values.  This emits an error and returns null on failure.
  ///
  /// `callNode` is the call like expression (e.g. a CallNode, binary operator,
  /// etc) that results in the call, or potentially a random value that is being
  /// fed into an implicit conversion.  This should only be used for location
  /// information.
  AnyValue emitFunctionCall(ArrayRef<ASTExprAnd<AnyValue>> operands,
                            CallSyntax syntax, const ExprNode *callNode,
                            IREmitter &emitter);

  /// Return true if 'value' may be implicitly converted to 'requiredType'
  /// by invoking (one level of) conversion operations.  This does not generate
  /// any IR.
  static bool canImplicitlyConvertToType(ASTExprAnd<AnyValue> value,
                                         ASTType requiredType,
                                         LitSharedState &shared);

private:
  MValue inlineFunctionCallIntoMValue(
      SMLoc callLoc, ASTDecl &callee, ParamBindArrayAttr inputParams,
      ArrayRef<ASTExprAnd<AnyValue>> argumentValues, IREmitter &emitter);
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPRCALLS_H

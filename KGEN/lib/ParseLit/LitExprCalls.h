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
class ExprEmitter;
class ExprNode;

template <typename ValueType>
struct ASTExprAnd {
  ValueType ir;

  /// This is the expression a value was produced from, carrying location and
  /// additional semantic information.
  const ExprNode *expr;

  bool isNull() const { return ir.isNull(); }
  bool operator!() const { return !ir; }
  operator bool() const { return bool(ir); }
};

//===----------------------------------------------------------------------===//
// DirectCallable
//===----------------------------------------------------------------------===//

/// This struct models something that can be directly called, e.g. a global
/// symbol with any binding information.
struct DirectCallable {
  /// This is the location of the direct-callable, e.g. in `x.method(...`, this
  /// is the location of 'method'.
  llvm::SMLoc loc;

  /// This is the basename of the declaration set, used in diagnostics.
  StringRef baseName;

  /// The function overload set that may be called directly.
  SmallVector<ASTDecl *, 1> fnDecls;

  /// Any bound parameters.  Consider something like:
  ///    SomeType[param1].method[param2](arg1)
  /// The type parameters (param1) will be bound as a ParamBindAttr, and the
  /// param2 will be bound as the value of param2.  We cannot type check the
  /// bindings until overload resolution has resolved which 'method' we are
  /// talking about, so we keep them as either a ParamBindAttr or
  /// (Typed)Attribute for the actual value.
  struct BoundParam {
    SMLoc loc;
    PointerUnion<ParamBindAttr, Attribute> bindingOrValue;

    TypedAttr getValue() const {
      if (auto attr = dyn_cast<Attribute>(bindingOrValue))
        return cast<TypedAttr>(attr);
      return {};
    }
    Type getType() const {
      if (auto attr = getValue())
        return attr.getType();
      return cast<ParamBindAttr>(bindingOrValue).getType();
    }
  };

  SmallVector<BoundParam> bindings;

  DirectCallable(SMLoc loc, StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
                 ParamBindArrayAttr bindings);

  /// Evaluate the fnDecls candidates and see if there is an unambiguous
  /// candidate that works with the specified parameter bindings and provided
  /// arguments.  If so, replace fnDecls with a single entry that works and
  /// return success.  If not, generate a diagnostic (when
  /// `emitDiagnosticOnFailure` is true) and return failure.
  LogicalResult filterOverloadSet(ArrayRef<ASTExprAnd<AnyValue>> operands,
                                  bool isMethodCall,
                                  bool emitDiagnosticOnFailure,
                                  LitSharedState &shared);

  /// Check that our set of parameter bindings work with the specified signature
  /// type, returning a checked ParamBindArrayAttr if so.  If the parameters do
  /// not work, this emits an diagnostic (if `funcLoc` is non-null) and sets
  /// `incorrectBindingNo` to the bad binding (or -1 if there is a count
  /// mismatch).
  ParamBindArrayAttr getCheckedBindings(SignatureType signature,
                                        ssize_t &incorrectBindingNo,
                                        Optional<Location> funcLoc,
                                        LitSharedState &shared) const;

  /// Perform subsitutions of the specified bindings into the symbol, returning
  /// the resultant LITSymbolConstant attr or producing an error message and
  /// returning null.
  SymbolConstantAttr getBoundConstantAttr(LitSharedState &shared) const;
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
  /// This is a dynamic value, which may either be an LValue or an RValue, that
  /// may itself be a callable, or (if targetSymbol is non-null), is the self
  /// argument to a call to the symbol.
  ASTExprAnd<AnyValue> baseVal;

  /// If present, this a reference to a fixed symbol or an overload set.
  Optional<DirectCallable> direct;

  CallableValue() {}
  CallableValue(ASTExprAnd<AnyValue> baseVal) : baseVal(baseVal) {}
  CallableValue(llvm::SMLoc loc, StringRef baseName,
                ArrayRef<ASTDecl *> fnDecls, ParamBindArrayAttr bindings)
      : direct({loc, baseName, fnDecls, bindings}) {}

  /// Get a CallableValue for a lookup of a named method on the specified type.
  /// If successful, this provides a non-null CallableValue.  On failure, it
  /// emits an error and returns a null CallableValue.
  CallableValue(ASTType type, StringRef methodName, SMLoc callLoc,
                LitSharedState &shared);

  /// Get a CallableValue for a lookup of a named method on the specified type.
  /// If successful, this provides a non-null CallableValue.
  ///
  /// On failure, this returns a null CallableValue and sets 'erroneousDecl' to
  /// indicate whether there was a problem with the callee that has already been
  /// diagnosed (allowing the client to squish downstream error messages).  This
  /// does not emit an error on failure.
  CallableValue(ASTType type, StringRef methodName, SMLoc callLoc,
                bool &erroneousDecl, LitSharedState &shared);

  bool isNull() const { return !baseVal && !direct; }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  /// Emit this as a flattened RValue or LValue.  This returns null on
  /// failure.
  AnyValue emitAsValue(ExprEmitter &emitter) const;

  /// Emit a function call to the specified callee with the specified operand
  /// values.  This emits an error and returns null on failure.
  AnyValue emitFunctionCall(ArrayRef<ASTExprAnd<AnyValue>> operands,
                            SMLoc callLoc, ExprEmitter &emitter);

private:
  void lookup(ASTType type, StringRef methodName, SMLoc callLoc,
              bool emitErrorOnFailure, bool &erroneousDecl,
              LitSharedState &shared);
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPRCALLS_H

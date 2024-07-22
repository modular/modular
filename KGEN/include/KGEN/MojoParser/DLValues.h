//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines the DLValue ("dynamic LValue") implementation details.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_DLVALUES_H
#define KGEN_MOJOPARSER_DLVALUES_H

#include "CallEmission.h"

namespace M::KGEN::LIT {

/// This DLValue implementation represents a discard pattern of _.  It discards
/// its result on store and produces an error if attempting to load it.
class DiscardDLValue : public BaseDLValue {
public:
  const ExprNode *expr;

  DiscardDLValue(ASTType elementType, const ExprNode *expr);

  void print(raw_ostream &os) const override;
  CValue emitLoad(ValueDest &dest, ExprEmitter &emitter) const override;
  void emitStore(ASTExprAnd<CValue> value, ExprEmitter &emitter) const override;
};

/// This DLValue implementation represents a stored attribute projected from
/// another DLValue, e.g. `swap(&a[i].x, ...)`.
class StoredAttributeRefDLValue : public BaseDLValue {
public:
  const ExprNode *expr;
  ASTExprAnd<DLValue> baseVal;
  Operation *fieldOp; // StructFieldOp

  StoredAttributeRefDLValue(ASTExprAnd<DLValue> baseVal, StructFieldOp fieldOp,
                            ASTType elementType, const ExprNode *expr);

  StructFieldOp getField() const;
  MBValue emitMBValueFromDefArgument(ExprEmitter &emitter) const override;

  void print(raw_ostream &os) const override;
  CValue emitLoad(ValueDest &dest, ExprEmitter &emitter) const override;
  void emitStore(ASTExprAnd<CValue> value, ExprEmitter &emitter) const override;
};

/// This DLValue implementation represents property access `a.x =`
/// and with subscript syntax `a[i,j] = `, invoking __getattr__/__setattr__ and
/// __getitem__ and __setitem__ respectively.
///
/// We allow DLValues to have getter+setter or just setter.
class SubscriptDLValue : public BaseDLValue {
public:
  /// The getter and setter to use; these may both be null.
  PValue getter;
  /// They keyword argument name for the newValue.
  StringAttr setterValueName;

  // Positional operands (including self) for the setter/getter call.
  CallOperands operands;

  const ExprNode *expr;

  /// Return true if this is a subscript, false if this is an attribute access.
  bool isSubscript() const;

  SubscriptDLValue(PValue getter, StringAttr setterValueName,
                   CallOperands &&operands, ASTType elementType,
                   const ExprNode *expr);

  void print(raw_ostream &os) const override;
  CValue emitLoad(ValueDest &dest, ExprEmitter &emitter) const override;
  void emitStore(ASTExprAnd<CValue> value, ExprEmitter &emitter) const override;
};

/// This DLValue implementation represents tuple lvalues, e.g. `(a[i], b) = x`.
class TupleDLValue : public BaseDLValue {
public:
  const ExprNode *expr;
  // These are the LValues for the sub-elements.
  std::vector<ASTExprAnd<AnyValue>> eltLValues;

  TupleDLValue(ArrayRef<ASTExprAnd<AnyValue>> eltLValues, ASTType tupleType,
               const ExprNode *expr);

  void print(raw_ostream &os) const override;
  CValue emitLoad(ValueDest &dest, ExprEmitter &emitter) const override;
  void emitStore(ASTExprAnd<CValue> value, ExprEmitter &emitter) const override;
};

/// This DLValue is used to lazily synthesize a def argument box the first time
/// the argument is mutated.  When this happens, it replaces itself in the
/// ASTDecl with a direct reference to the box.
class DefArgumentWrapperDLValue : public BaseDLValue {
public:
  DefArgumentWrapperDLValue(ASTDecl *argDecl, BValue argRef, ASTType eltType,
                            size_t argIndex);

  // This hook is called before an argument is passed inout.  The specified
  // location indicates where diagnostics should be produced if this cannot be
  // done.  This returns null on failure.
  LValue prepareForInoutAccess(llvm::SMLoc loc,
                               ExprEmitter &emitter) const override;

  void print(raw_ostream &os) const override;
  CValue emitLoad(ValueDest &dest, ExprEmitter &emitter) const override;
  void emitStore(ASTExprAnd<CValue> value, ExprEmitter &emitter) const override;

  /// If this is a def argument shadow, resolve it to the incoming immutable
  /// borrowed value without forming a local copy.  Otherwise return null.
  MBValue emitMBValueFromDefArgument(ExprEmitter &emitter) const override;
  bool isDefArgument() const override { return true; }

  /// This is the argument decl we're the representation of.
  ASTDecl *argDecl;
  // This is the MBValue or SBValue for the argument.
  BValue argRef;
  size_t argIndex;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_DLVALUES_H
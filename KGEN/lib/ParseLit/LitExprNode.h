//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ExprNode base class and support classes used for
// emission.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_EXPRNODE_H
#define LIT_EXPRNODE_H

#include "IRValues.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {
using llvm::SMLoc;
class ASTDecl;
class CallableValue;
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
// ExprNode
//===----------------------------------------------------------------------===//

/// Base class for all expression nodes.  Note that these nodes are not allowed
/// to own memory since they are bump pointer allocated and their destructors
/// are never run.
class ExprNode {
public:
  // This indicates the subclass.
  enum Kind {
    kIntLiteral,    // 42
    kFloatLiteral,  // 1.1
    kBoolLiteral,   // False
    kStringLiteral, // "Hello"
    kNoneLiteral,   // None
    kDeclRef,       // x
    kAttributeRef,  // x.y
    kCall,          // thing(a, b)
    kSubscript,     // thing[a, b:c]
    kSlice,         // :, a:, :a, ::, a:b:c, etc.  Only valid in subscripts.
    kParenExprNode, // (x+y)
    kListExprNode,  // [x, y]

    // Unary expressions.
    kNeg,     // -x
    kPos,     // +x
    kInvert,  // ~x
    kBoolNot, // not x
    kFirstUnaryOp = kNeg,
    klastUnaryOp = kBoolNot,

    // Binary expressions.
    kAdd,
    kSub,
    kMul,
    kMatMul,
    kTrueDiv,
    kFloorDiv,
    kMod,
    kBoolOr,  // x or y
    kBoolAnd, // x and y
    kCmpIn,
    kCmpNotIn,
    kCmpIs,
    kCmpIsNot,
    kCmpLT,
    kCmpLE,
    kCmpGT,
    kCmpGE,
    kCmpNE,
    kCmpEQ,
    kOr,
    kXor,
    kAnd,
    kLShift,
    kRShift,
    kPow,
    kAssign,
    // Inplace operators.
    kIAdd,
    kISub,
    kIMul,
    kIMatMul,
    kITrueDiv,
    kIFloorDiv,
    kIMod,
    kIAnd,
    kIOr,
    kIXor,
    kILShift,
    kIRShift,
    kIPow,

    // Ternary expressions.
    kIfElse,

    // Not a valid expression node.
    kInvalid,

    kFirstAssignStmt = kAssign,
    kLastAssignStmt = kIPow,

    // FIXME: This range isn't right.  It will include things like kAssign and
    // kPow which are not binary operators!
    kFirstBinOp = kAdd,
    kLastBinOp = kIFloorDiv,
  } const kind;

  ExprNode(Kind kind) : kind(kind) {}
  virtual ~ExprNode();

  /// Return the primary location for this node for error reporting purposes.
  virtual SMLoc getLoc() const = 0;
  /// Return the source range spanned by this expression.
  virtual llvm::SMRange getRange() const = 0;

  /// Emit this expression to MLIR, returning a (possibly null!) AnyValue.  The
  /// contextualType (if non-null) indicates the contextual type to use for an
  /// implicitly declared value, e.g. a/b in `def f(): (a,b) = (1,2)`.
  virtual AnyValue emitIR(ExprEmitter &emitter,
                          ASTType contextualType = {}) const = 0;

  /// Emit this expression to MLIR as a CallableValue.  On error, emit an error
  /// and return a null value.
  virtual CallableValue emitCallable(ExprEmitter &emitter,
                                     ASTType contextualType) const;
};

//===----------------------------------------------------------------------===//
// CallableValue
//===----------------------------------------------------------------------===//

/// This struct models something that can be directly called, e.g. a global
/// symbol with any binding information.
struct DirectCallable {
  llvm::SMLoc loc;

  /// The function that may be called directly.
  ASTDecl *fnDecl;

  /// Any bound parameters.
  ParamBindArrayAttr bindings;

  /// Perform subsitutions of the specified bindings into the symbol, returning
  /// the resultant LITSymbolConstant attr or producing an error message and
  /// returning null.
  SymbolConstantAttr getBoundConstantAttr(ExprEmitter &emitter) const;
};

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

  /// If present, this callable value is a reference to a fixed symbol.
  /// TODO: Extend to support overload sets.
  Optional<DirectCallable> direct;

  CallableValue() {}
  CallableValue(ASTExprAnd<AnyValue> baseVal) : baseVal(baseVal) {}
  CallableValue(llvm::SMLoc loc, ASTDecl &fnDecl, ParamBindArrayAttr bindings);

  bool isNull() const { return !baseVal && !direct; }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  /// Emit this as a flattened RValue or LValue.  This returns null on
  /// failure.
  AnyValue emitAsValue(ExprEmitter &emitter) const;
};
} // namespace M::KGEN::LIT

#endif // LIT_EXPRNODE_H

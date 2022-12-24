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

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN::LIT {
using llvm::SMLoc;
class AnyValue;
class ASTType;
class CallableValue;
class ExprEmitter;

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
    kIntLiteral,     // 42
    kFloatLiteral,   // 1.1
    kBoolLiteral,    // False
    kStringLiteral,  // "Hello"
    kNoneLiteral,    // None
    kDeclRef,        // x
    kAttributeRef,   // x.y
    kCall,           // thing(a, b)
    kSubscript,      // thing[a, b:c]
    kSubscriptArrow, // thing[x, y -> a, b]
    kSlice,          // :, a:, :a, ::, a:b:c, etc.  Only valid in subscripts.
    kParen,          // (x+y)
    kList,           // [x, y]

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
  virtual llvm::SMLoc getLoc() const = 0;
  /// Return the source range spanned by this expression.
  virtual llvm::SMRange getRange() const = 0;

  /// Emit this expression to MLIR, returning a (possibly null!) AnyValue.  The
  /// contextualType (if non-null) indicates the contextual type to use for an
  /// implicitly declared value, e.g. a/b in `def f(): (a,b) = (1,2)`.
  virtual AnyValue emitIR(ExprEmitter &emitter,
                          ASTType contextualType) const = 0;

  /// Emit this expression to MLIR as a CallableValue.  On error, emit an error
  /// and return a null value.
  virtual CallableValue emitCallable(ExprEmitter &emitter,
                                     ASTType contextualType) const;
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPRNODE_H

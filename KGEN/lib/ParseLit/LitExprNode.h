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
class LitSourceRange;
class ValueDest;

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
    kSelfLiteral,    // Self
    kStringLiteral,  // "Hello"
    kNoneLiteral,    // None
    kDeclRef,        // x
    kAttributeRef,   // x.y
    kParen,          // (x+y)
    kTuple,          // (), (x,), (x, y), etc
    kList,           // [x, y]
    kDictionary,     // {a: 1, b: 2, **dictUnpack}
    kCall,           // thing(a, b)
    kSubscript,      // thing[a, b:c]
    kSubscriptArrow, // thing[x, y -> a, b]
    kSlice,          // :, a:, :a, ::, a:b:c, etc.  Only valid in subscripts.
    kDictSubscript,  // thing{a: 1, x: 2}
    kChainedCmp,     // a < b <= c

    // Unary expressions.
    kNeg,     // -x
    kPos,     // +x
    kInvert,  // ~x
    kBoolNot, // not x
    kAwait,   // await x
    kFirstUnaryOp = kNeg,
    klastUnaryOp = kAwait,

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

    kFirstBinOp = kAdd,
    kLastBinOp = kIPow,
  } const kind;

  ExprNode(Kind kind) : kind(kind) {}
  virtual ~ExprNode();

  /// Return the primary location for this node for error reporting purposes.
  virtual llvm::SMLoc getLoc() const = 0;

  /// Return the 'loc' for this node translated to an MLIR location.
  Location getLocation(ExprEmitter &emitter) const;

  /// Return the source range spanned by this expression.
  virtual LitSourceRange getRange() const = 0;

  /// Return the start or end of the source range.
  llvm::SMLoc getRangeStart() const;
  llvm::SMLoc getRangeEnd() const;

  /// Emit this expression to MLIR, returning a (possibly null!) AnyValue.  The
  /// ValueDest indicates information about where to emit the expression result
  /// into, e.g. the a/b target in `def f(): (a,b) = (1,2)`.
  virtual AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const = 0;

  /// Emit this expression to MLIR as a CallableValue.  On error, emit an error
  /// and return a null value.
  virtual CallableValue emitCallable(ExprEmitter &emitter) const;

  /// This node is being used as the LHS target/pattern of an assignment,
  /// initialized with the specified RHS value.  On success, handle this
  /// coersion/initialization, otherwise emit an error.
  virtual MRValue emitExprResultIntoPattern(ASTExprAnd<AnyValue> value,
                                            ExprEmitter &emitter) const;
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPRNODE_H

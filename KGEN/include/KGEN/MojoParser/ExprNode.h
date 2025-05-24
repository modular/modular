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

#ifndef KGEN_MOJOPARSER_EXPRNODE_H
#define KGEN_MOJOPARSER_EXPRNODE_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace mlir {
class raw_indented_ostream;
} // namespace mlir

namespace M {
class SourceRange;
} // namespace M

namespace M::KGEN::LIT {
using llvm::SMLoc;
class AnyValue;
class ASTType;
class IREmitter;
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
    kSynthetic,      // There is no source corresponding to the IR.
    kIntLiteral,     // 42
    kFloatLiteral,   // 1.1
    kBoolLiteral,    // False
    kSelfLiteral,    // Self
    kStringLiteral,  // "Hello"
    kNoneLiteral,    // None
    kDiscardLiteral, // _
    kDeclRef,        // x
    kAttributeRef,   // x.y
    kParen,          // (x+y)
    kTuple,          // (), (x,), (x, y), etc
    kListLiteral,    // [x, y]
    kDictLiteral,    // {a: 1, b: 2, **dictUnpack}
    kSetInitLiteral, // {x, y} and {z=4, "foo"}
    kCall,           // thing(a, b)
    kSubscript,      // thing[a, b:c]
    kSubscriptArrow, // thing[x, y -> a, b]
    kSlice,          // :, a:, :a, ::, a:b:c, etc.  Only valid in subscripts.
    kChainedCmp,     // a < b <= c
    kFunctionType,   // async fn[](owned Int, &F32) capturing raises -> F64

    // Magic functions
    kGetMValueAsLitRef,        // __get_mvalue_as_litref(x)
    kGetLitRefAsMValue,        // __get_litref_as_mvalue(x)
    kGetAddressAsUninitLValue, // __get_address_as_uninit_lvalue(x)
    kGetAddressAsOwned,        // __get_address_as_owned_value(x)
    kGetNearestErrorSlot,      // __get_nearest_error_slot()
    kOriginOf,                 // __origin_of(x)
    kTypeOf,                   // __type_of(x)
    kFirstMagicFunction = kGetMValueAsLitRef,
    kLastMagicFunction = kTypeOf,

    // Prefix and Postfix unary expressions.
    kNeg,      // -x
    kPos,      // +x
    kInvert,   // ~x
    kUnpack,   // *x
    kBoolNot,  // not x
    kAwait,    // await x
    kTransfer, // x^
    kFirstUnaryOp = kNeg,
    kLastUnaryOp = kTransfer,

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
    kWalrus, // x := y aka walrus
    kAssign, // x = y aka assignment_expression

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
  Location getLocation(IREmitter &emitter) const;

  /// Return the source range spanned by this expression.
  virtual SourceRange getRange() const = 0;

  /// Print the expression node.
  virtual void print(mlir::raw_indented_ostream &os) const = 0;
  void print(raw_ostream &os) const;
  /// Dump the expression node for debugging.
  LLVM_DUMP_METHOD void dump() const;

  /// Return the start or end of the source range.
  llvm::SMLoc getRangeStart() const;
  llvm::SMLoc getRangeEnd() const;

  /// Recursively dig through noop paren nodes (if present) to find what is
  /// inside of them.
  ExprNode *getWithoutParens();
  const ExprNode *getWithoutParens() const {
    return const_cast<ExprNode *>(this)->getWithoutParens();
  }

  /// Return true if this is a TupleNode with no subexpressions.
  bool isEmptyTuple() const;

  /// Emit this expression to MLIR, returning a (possibly null!) AnyValue.  The
  /// ValueDest indicates information about where to emit the expression result
  /// into, e.g. the a/b target in `def f(): (a,b) = (1,2)`.  On success, the
  /// ValueDest /must/ be emitted into.
  virtual AnyValue emitIR(ValueDest &dest, IREmitter &emitter) const = 0;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_EXPRNODE_H

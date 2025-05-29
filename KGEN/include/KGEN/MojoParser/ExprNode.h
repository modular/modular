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

#include "KGEN/MojoParser/IRValues.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include <variant>

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
    kSynthetic,         // There is no source corresponding to the IR.
    kIntLiteral,        // 42
    kFloatLiteral,      // 1.1
    kBoolLiteral,       // False
    kSelfLiteral,       // Self
    kStringLiteral,     // "Hello"
    kNoneLiteral,       // None
    kDiscardLiteral,    // _
    kDeclRef,           // x
    kAttributeRef,      // x.y
    kParen,             // (x+y)
    kTuple,             // (), (x,), (x, y), etc
    kListLiteral,       // [x, y]
    kDictLiteral,       // {a: 1, b: 2, **dictUnpack}
    kSetInitLiteral,    // {x, y} and {z=4, "foo"}
    kListComprehension, // [x for x in range(10) if x & 1]
    kSetComprehension,  // {x for x in range(10) if x & 1}
    kDictComprehension, // {x: x * x for x in range(10) if x & 1}
    kCall,              // thing(a, b)
    kSubscript,         // thing[a, b:c]
    kSubscriptArrow,    // thing[x, y -> a, b]
    kSlice,             // :, a:, :a, ::, a:b:c, etc.  Only valid in subscripts.
    kChainedCmp,        // a < b <= c
    kFunctionType,      // async fn[](owned Int, &F32) capturing raises -> F64

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

  /// emitLValueIfImplicitlyTyped can return one of three things:
  ///   1) A CValue if it is inherently typed.
  ///   2) A (potentially changed!) ExprNode* if it is contextually typed,
  ///      unresolvable until the initializer expression is known.
  ///   3) Failure if the expression is invalid.
  ///
  /// This method is used with assignment expressions, which need to infer the
  /// LHS from the RHS when the LHS is unresolved, and the RHS from the LHS when
  /// it is known:
  ///
  ///     _, x = foo() # infer typeof _ and x from foo()
  ///     y = []       # infer type of [] from type of y.
  ///
  struct ELVIITResult {
    // Error cases can often propagate in a null AnyValue.
    ELVIITResult(AnyValue value) : storage(value) {}
    ELVIITResult(CValue value) : ELVIITResult(AnyValue(value)) {}
    ELVIITResult(const ExprNode *value) : storage(value) {
      assert(value && "Cannot init with null ExprNode*");
    }
    /*implicit*/ ELVIITResult() { return ELVIITResult(AnyValue()); }

    bool isFailure() const {
      if (auto *ptr = std::get_if<AnyValue>(&storage))
        return ptr->isNull();
      return false;
    }
    AnyValue getIfValue() const {
      if (auto *ptr = std::get_if<AnyValue>(&storage))
        return *ptr;
      return AnyValue();
    }
    const ExprNode *getIfExprNode() const {
      if (auto *ptr = std::get_if<const ExprNode *>(&storage))
        return *ptr;
      return nullptr;
    }

  private:
    /// The failure case is encoded as a null AnyValue
    std::variant<AnyValue, const ExprNode *> storage;
  };

  /// If this expression is an LValue with an inherent type, emit it and return
  /// it. Otherwise return an ExprNode* to further resolve (with emitIR) or
  /// emit an error and return failure.
  virtual ELVIITResult emitLValueIfImplicitlyTyped(IREmitter &emitter) const {
    return ELVIITResult(this);
  }
};

/// This class is a helper class used for expression nodes might be resolvable
/// to a concrete type without an explicit initializer. This allows them to
/// simplify their implementation by providing a single hook for both the
/// "speculative" and non-speculative paths.
class LValueCapableExprNode : public ExprNode {
public:
  LValueCapableExprNode(Kind kind) : ExprNode(kind) {}

  // A default implementation is provided for these that forward to emitIR.
  ELVIITResult emitLValueIfImplicitlyTyped(IREmitter &emitter) const override;
  AnyValue emitIR(ValueDest &dest, IREmitter &emitter) const override;

  virtual ELVIITResult emitLCVIR(ValueDest &dest, IREmitter &emitter,
                                 bool isSpeculative) const = 0;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_EXPRNODE_H

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides declarations for various expression nodes when in
// syntactic (not yet type checked) form.
//
// Expressions are parsed with a two-phase approach.  The first phase pulls out
// the syntactic structure of the expression, whereas the second pass does type
// checking and IR generation.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_EXPR_NODES_H
#define LIT_EXPR_NODES_H

#include "IRValues.h"
#include "LitDiags.h"
#include "LitExprNode.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN {
class SignatureType;
}

namespace M::KGEN::LIT {
class SRValue;

/// This returns an SMLoc from a StringRef that points into the source buffer.
inline SMLoc getSMLocFromStringRef(StringRef bufferRef) {
  return SMLoc::getFromPointer(bufferRef.data());
}

struct IntLiteralNode final : public ExprNode {
  IntLiteralNode(StringRef spelling)
      : ExprNode(kIntLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kIntLiteral;
  }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  LitSourceRange getRange() const override { return {getLoc(), getLoc()}; }

  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

struct FloatLiteralNode final : public ExprNode {
  FloatLiteralNode(StringRef spelling)
      : ExprNode(kFloatLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kFloatLiteral;
  }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  LitSourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

struct BoolLiteralNode final : public ExprNode {
  BoolLiteralNode(SMLoc loc, bool value)
      : ExprNode(kBoolLiteral), loc(loc), value(value) {}

  const SMLoc loc;
  const bool value;

  static bool classof(const ExprNode *node) {
    return node->kind == kBoolLiteral;
  }

  SMLoc getLoc() const override { return loc; }
  LitSourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

struct SelfLiteralNode final : public ExprNode {
  SelfLiteralNode(SMLoc loc) : ExprNode(kSelfLiteral), loc(loc) {}

  const SMLoc loc;

  static bool classof(const ExprNode *node) {
    return node->kind == kSelfLiteral;
  }

  SMLoc getLoc() const override { return loc; }
  LitSourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

struct StringLiteralNode final : public ExprNode {
  StringLiteralNode(StringRef spelling)
      : ExprNode(kStringLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kStringLiteral;
  }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  LitSourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

struct NoneLiteralNode final : public ExprNode {
  NoneLiteralNode(SMLoc loc) : ExprNode(kNoneLiteral), loc(loc) {}

  const SMLoc loc;

  static bool classof(const ExprNode *node) {
    return node->kind == kNoneLiteral;
  }
  SMLoc getLoc() const override { return loc; }
  LitSourceRange getRange() const override { return {getLoc(), getLoc()}; }

  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

struct DeclRefNode final : public ExprNode {
  DeclRefNode(StringRef spelling) : ExprNode(kDeclRef), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) { return node->kind == kDeclRef; }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  LitSourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
  CallableValue emitCallable(ExprEmitter &emitter) const override;
  AnyValue emitExprResultIntoPattern(ASTExprAnd<AnyValue> value,
                                     ExprEmitter &emitter) const override;
};

struct AttributeRefNode final : public ExprNode {
  AttributeRefNode(ExprNode *base, SMLoc dotLoc, StringRef attrSpelling)
      : ExprNode(kAttributeRef), base(base), dotLoc(dotLoc),
        attrSpelling(attrSpelling) {}

  ExprNode *const base;
  const SMLoc dotLoc;
  const StringRef attrSpelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kAttributeRef;
  }
  SMLoc getLoc() const override { return dotLoc; }
  SMLoc getAttributeNameLoc() const {
    return getSMLocFromStringRef(attrSpelling);
  }
  LitSourceRange getRange() const override {
    return {base->getRangeStart(), getAttributeNameLoc()};
  }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
  CallableValue emitCallable(ExprEmitter &emitter) const override;
};

struct CallNode final : public ExprNode {
  CallNode(ExprNode *callee, SMLoc lparenLoc, ArrayRef<ExprNode *> args,
           SMLoc rparenLoc)
      : ExprNode(kCall), callee(callee), lparenLoc(lparenLoc), args(args),
        rparenLoc(rparenLoc) {}

  ExprNode *const callee;
  const SMLoc lparenLoc;
  const ArrayRef<ExprNode *> args;
  const SMLoc rparenLoc;

  static bool classof(const ExprNode *node) { return node->kind == kCall; }
  SMLoc getLoc() const override { return lparenLoc; }
  LitSourceRange getRange() const override {
    return {callee->getRangeStart(), rparenLoc};
  }
  LitSourceRange getParenRange() const { return {lparenLoc, rparenLoc}; }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

/// This represents `A[i,j]`.  In the case of slices (e.g. `A[i, ::]`), the
/// slice will be represented with a subexpression.
struct SubscriptNode final : public ExprNode {
  SubscriptNode(ExprNode *base, SMLoc lsquareLoc, ArrayRef<ExprNode *> indices,
                SMLoc rsquareLoc)
      : ExprNode(kSubscript), base(base), lsquareLoc(lsquareLoc),
        indices(indices), rsquareLoc(rsquareLoc) {}

  ExprNode *const base;
  const SMLoc lsquareLoc;
  const ArrayRef<ExprNode *> indices;
  const SMLoc rsquareLoc;

  static bool classof(const ExprNode *node) { return node->kind == kSubscript; }
  SMLoc getLoc() const override { return lsquareLoc; }
  LitSourceRange getRange() const override {
    return {base->getRangeStart(), rsquareLoc};
  }
  /// Return a source range from '[' to ']'.
  LitSourceRange getIndexRange() const { return {lsquareLoc, rsquareLoc}; }

  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
  CallableValue emitCallable(ExprEmitter &emitter) const override;
};

/// This represents `A[i,j -> a, b]`.  In the case of slices (e.g. `A[i, ::]`),
/// the slice will be represented with a subexpression.
struct SubscriptArrowNode final : public ExprNode {
  SubscriptArrowNode(ExprNode *base, SMLoc lsquareLoc,
                     ArrayRef<ExprNode *> indices, SMLoc arrowLoc,
                     ArrayRef<ExprNode *> arrowExprs, SMLoc rsquareLoc)
      : ExprNode(kSubscriptArrow), base(base), lsquareLoc(lsquareLoc),
        indices(indices), arrowLoc(arrowLoc), arrowExprs(arrowExprs),
        rsquareLoc(rsquareLoc) {}

  ExprNode *const base;
  const SMLoc lsquareLoc;
  ArrayRef<ExprNode *> indices;
  const SMLoc arrowLoc;
  ArrayRef<ExprNode *> arrowExprs;
  const SMLoc rsquareLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kSubscriptArrow;
  }
  SMLoc getLoc() const override { return lsquareLoc; }
  LitSourceRange getRange() const override { return {lsquareLoc, rsquareLoc}; }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
  CallableValue emitCallable(ExprEmitter &emitter) const override;
};

/// This is an expression that produces a slice value in a SubscriptNode index
/// expression.  These have at least one colon in them, and one, two, or three
/// expressions, e.g. `:`, `: :`, `a:b`, `a::b` etc.
///
/// All the elements of the syntax are optional (and thus may be null!) except
/// for the first colon.
struct SliceNode final : public ExprNode {
  SliceNode(ExprNode *lower, SMLoc colon1Loc, ExprNode *upper, SMLoc colon2Loc,
            ExprNode *stride)
      : ExprNode(kSlice), lower(lower), colon1Loc(colon1Loc), upper(upper),
        colon2Loc(colon2Loc), stride(stride) {}

  ExprNode *const lower;
  SMLoc colon1Loc;
  ExprNode *const upper;
  SMLoc colon2Loc;
  ExprNode *const stride;

  static bool classof(const ExprNode *node) { return node->kind == kSlice; }
  SMLoc getLoc() const override { return colon1Loc; }

  LitSourceRange getRange() const override {
    auto startLoc = lower ? lower->getRangeStart() : colon1Loc;
    if (stride)
      return {startLoc, stride->getRangeEnd()};
    if (colon2Loc.isValid())
      return {startLoc, colon2Loc};
    if (upper)
      return {startLoc, upper->getRangeEnd()};
    return {startLoc, colon1Loc};
  }

  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

struct ParenNode final : public ExprNode {
  ParenNode(SMLoc lparenLoc, ExprNode *subExpr, SMLoc rparenLoc)
      : ExprNode(kParen), lparenLoc(lparenLoc), subExpr(subExpr),
        rparenLoc(rparenLoc) {}

  const SMLoc lparenLoc;
  ExprNode *const subExpr;
  const SMLoc rparenLoc;

  static bool classof(const ExprNode *node) { return node->kind == kParen; }
  SMLoc getLoc() const override { return lparenLoc; }
  LitSourceRange getRange() const override { return {lparenLoc, rparenLoc}; }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;

  CallableValue emitCallable(ExprEmitter &emitter) const override;
  AnyValue emitExprResultIntoPattern(ASTExprAnd<AnyValue> value,
                                     ExprEmitter &emitter) const override;
};

/// (a, b, c)
struct TupleNode final : public ExprNode {
  TupleNode(SMLoc lparenLoc, ArrayRef<ExprNode *> exprs, SMLoc rparenLoc)
      : ExprNode(kTuple), lparenLoc(lparenLoc), exprs(exprs),
        rparenLoc(rparenLoc) {}

  const SMLoc lparenLoc;
  ArrayRef<ExprNode *> exprs;
  const SMLoc rparenLoc;

  static bool classof(const ExprNode *node) { return node->kind == kTuple; }
  SMLoc getLoc() const override { return lparenLoc; }
  LitSourceRange getRange() const override { return {lparenLoc, rparenLoc}; }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

/// [a, b, c]
struct ListNode final : public ExprNode {
  ListNode(SMLoc lsquareLoc, ArrayRef<ExprNode *> exprs, SMLoc rsquareLoc)
      : ExprNode(kList), lsquareLoc(lsquareLoc), exprs(exprs),
        rsquareLoc(rsquareLoc) {}

  const SMLoc lsquareLoc;
  ArrayRef<ExprNode *> exprs;
  const SMLoc rsquareLoc;

  static bool classof(const ExprNode *node) { return node->kind == kList; }
  SMLoc getLoc() const override { return lsquareLoc; }
  LitSourceRange getRange() const override { return {lsquareLoc, rsquareLoc}; }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

/// This represents `{key1: value1, key2: value2, **dictunpack}` expressions.
/// The dictionary unpacking syntax is represented with a null key and with the
/// unpack expression as the value.
struct DictionaryNode final : public ExprNode {
  DictionaryNode(SMLoc lbraceLoc,
                 ArrayRef<std::pair<ExprNode *, ExprNode *>> values,
                 SMLoc rbraceLoc)
      : ExprNode(kDictionary), lbraceLoc(lbraceLoc), values(values),
        rbraceLoc(rbraceLoc) {}

  const SMLoc lbraceLoc;
  const ArrayRef<std::pair<ExprNode *, ExprNode *>> values;
  const SMLoc rbraceLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kDictionary;
  }
  SMLoc getLoc() const override { return lbraceLoc; }
  LitSourceRange getRange() const override { return {lbraceLoc, rbraceLoc}; }

  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

/// This represents `expr{x:y, **unpack}` using DictionaryNode as the storage.
struct DictSubscriptNode final : public ExprNode {
  DictSubscriptNode(ExprNode *base, DictionaryNode *indices)
      : ExprNode(kDictSubscript), base(base), indices(indices) {}

  ExprNode *const base;
  DictionaryNode *const indices;

  static bool classof(const ExprNode *node) {
    return node->kind == kDictSubscript;
  }
  SMLoc getLoc() const override { return indices->lbraceLoc; }
  LitSourceRange getRange() const override {
    return {base->getRangeStart(), indices->rbraceLoc};
  }

  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
  AnyValue emitTypeSubscriptIR(ASTType initType, ExprEmitter &emitter) const;
};

// trueExpr 'if' condition 'else' falseExpr
struct IfElseOpNode final : public ExprNode {
  IfElseOpNode(ExprNode *trueExpr, SMLoc ifLoc, ExprNode *condExpr,
               SMLoc elseLoc, ExprNode *falseExpr)
      : ExprNode(kIfElse), trueExpr(trueExpr), ifLoc(ifLoc), condExpr(condExpr),
        elseLoc(elseLoc), falseExpr(falseExpr) {}

  ExprNode *const trueExpr;
  const SMLoc ifLoc;
  ExprNode *const condExpr;
  const SMLoc elseLoc;
  ExprNode *const falseExpr;

  static bool classof(const ExprNode *node) { return node->kind == kIfElse; }

  SMLoc getLoc() const override { return ifLoc; }
  LitSourceRange getRange() const override {
    return {trueExpr->getRangeStart(), falseExpr->getRangeEnd()};
  }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

struct BinOpNode final : public ExprNode {
  BinOpNode(Kind kind, ExprNode *lhs, SMLoc opLoc, ExprNode *rhs)
      : ExprNode(kind), lhs(lhs), opLoc(opLoc), rhs(rhs) {}

  ExprNode *const lhs;
  const SMLoc opLoc;
  ExprNode *const rhs;

  static bool classof(const ExprNode *node) {
    return node->kind >= kFirstBinOp && node->kind <= kLastBinOp;
  }

  /// Return true if this is an "assignment stmt" node like =, +=, or *=.
  bool isAssignmentStmt() const {
    return kind >= kFirstAssignStmt && kind <= kLastAssignStmt;
  }

  SMLoc getLoc() const override { return opLoc; }
  LitSourceRange getRange() const override {
    return {lhs->getRangeStart(), rhs->getRangeEnd()};
  }

  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;

private:
  AnyValue emitAndOr(ValueDest dest, ExprEmitter &emitter) const;
  AnyValue emitAssign(ValueDest dest, ExprEmitter &emitter) const;
  AnyValue emitInplace(ValueDest dest, ExprEmitter &emitter) const;
};

struct UnaryOpNode final : public ExprNode {
  UnaryOpNode(Kind kind, SMLoc opLoc, ExprNode *subExpr)
      : ExprNode(kind), opLoc(opLoc), subExpr(subExpr) {}

  const SMLoc opLoc;
  ExprNode *const subExpr;

  static bool classof(const ExprNode *node) {
    return node->kind >= kFirstUnaryOp && node->kind <= klastUnaryOp;
  }
  SMLoc getLoc() const override { return opLoc; }
  LitSourceRange getRange() const override {
    return {opLoc, subExpr->getRangeEnd()};
  }
  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
};

/// This represents a chained comparison expression (ex. a < b <= c).
/// exprs stores all the expressions in the comparison (ex. a, b, c), while
/// ops stores the ops in between pairs of expressions (ex. <, <=).
/// Chained expressions are evaluated left to right and each expression is
/// valuated at most once: a < b <= c is equivalent to a < b and b <= c, but
/// b is evaluated only once.
struct ChainedCmpOpNode final : public ExprNode {
  ChainedCmpOpNode(ArrayRef<ExprNode *> exprs, ArrayRef<ExprNode::Kind> ops,
                   SMLoc opLoc)
      : ExprNode(ExprNode::Kind::kChainedCmp), exprs(exprs), ops(ops),
        opLoc(opLoc) {}

  const ArrayRef<ExprNode *> exprs;
  const ArrayRef<ExprNode::Kind> ops;
  const SMLoc opLoc;

  SMLoc getLoc() const override { return opLoc; }
  LitSourceRange getRange() const override {
    return {exprs.front()->getRangeStart(), exprs.back()->getRangeEnd()};
  }

  AnyValue emitIR(ExprEmitter &emitter, ValueDest dest) const override;
  AnyValue emitNextCmp(ExprEmitter &emitter, size_t opIdx, SRValue lastCmp,
                       SRValue lastExpr) const;
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPR_NODES_H

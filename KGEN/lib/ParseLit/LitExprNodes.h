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

#include "LitExprs.h"
namespace M::KGEN {
class SignatureType;
}

namespace M::KGEN::LIT {
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
  llvm::SMRange getRange() const override { return {getLoc(), getLoc()}; }

  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
};

struct FloatLiteralNode final : public ExprNode {
  FloatLiteralNode(StringRef spelling)
      : ExprNode(kFloatLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kFloatLiteral;
  }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  llvm::SMRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
};

struct StringLiteralNode final : public ExprNode {
  StringLiteralNode(StringRef spelling)
      : ExprNode(kStringLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kStringLiteral;
  }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  llvm::SMRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
};

struct NoneLiteralNode final : public ExprNode {
  NoneLiteralNode(SMLoc loc) : ExprNode(kNoneLiteral), loc(loc) {}

  const SMLoc loc;

  static bool classof(const ExprNode *node) {
    return node->kind == kNoneLiteral;
  }
  SMLoc getLoc() const override { return loc; }
  llvm::SMRange getRange() const override { return {getLoc(), getLoc()}; }

  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
};

struct DeclRefNode final : public ExprNode {
  DeclRefNode(StringRef spelling) : ExprNode(kDeclRef), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) { return node->kind == kDeclRef; }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  llvm::SMRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
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
  llvm::SMRange getRange() const override {
    return {base->getRange().Start, getSMLocFromStringRef(attrSpelling)};
  }
  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
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
  llvm::SMRange getRange() const override {
    return {callee->getRange().Start, rparenLoc};
  }
  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
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
  llvm::SMRange getRange() const override {
    return {base->getRange().Start, rsquareLoc};
  }
  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
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

  llvm::SMRange getRange() const override {
    auto startLoc = lower ? lower->getRange().Start : colon1Loc;
    if (stride)
      return {startLoc, stride->getRange().Start};
    if (colon2Loc.isValid())
      return {startLoc, colon2Loc};
    if (upper)
      return {startLoc, upper->getRange().Start};
    return {startLoc, colon1Loc};
  }

  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
};

struct ParenExprNode final : public ExprNode {
  ParenExprNode(SMLoc lparenLoc, ExprNode *subExpr, SMLoc rparenLoc)
      : ExprNode(kParenExprNode), lparenLoc(lparenLoc), subExpr(subExpr),
        rparenLoc(rparenLoc) {}

  const SMLoc lparenLoc;
  ExprNode *const subExpr;
  const SMLoc rparenLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kParenExprNode;
  }
  SMLoc getLoc() const override { return lparenLoc; }
  llvm::SMRange getRange() const override { return {lparenLoc, rparenLoc}; }
  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
};

struct ListExprNode final : public ExprNode {
  ListExprNode(SMLoc lsquareLoc, ArrayRef<ExprNode *> exprs, SMLoc rsquareLoc)
      : ExprNode(kListExprNode), lsquareLoc(lsquareLoc), exprs(exprs),
        rsquareLoc(rsquareLoc) {}

  const SMLoc lsquareLoc;
  llvm::SmallVector<ExprNode *> exprs;
  const SMLoc rsquareLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kListExprNode;
  }
  SMLoc getLoc() const override { return lsquareLoc; }
  llvm::SMRange getRange() const override { return {lsquareLoc, rsquareLoc}; }
  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
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
  llvm::SMRange getRange() const override {
    return {trueExpr->getRange().Start, falseExpr->getRange().End};
  }
  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
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
  llvm::SMRange getRange() const override {
    return {lhs->getRange().Start, rhs->getRange().End};
  }

  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
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
  llvm::SMRange getRange() const override {
    return {opLoc, subExpr->getRange().End};
  }
  AnyValue emitIR(ExprEmitter &emitter, ASTType contextualType) const override;
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPR_NODES_H

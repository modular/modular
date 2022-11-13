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

struct IntLiteralNode final : public ExprNode {
  IntLiteralNode(StringRef spelling)
      : ExprNode(kIntLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kIntLiteral;
  }
  SMLoc getLoc() const override {
    return SMLoc::getFromPointer(spelling.data());
  }
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
};

struct FloatLiteralNode final : public ExprNode {
  FloatLiteralNode(StringRef spelling)
      : ExprNode(kFloatLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kFloatLiteral;
  }
  SMLoc getLoc() const override {
    return SMLoc::getFromPointer(spelling.data());
  }
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
};

struct StringLiteralNode final : public ExprNode {
  StringLiteralNode(StringRef spelling)
      : ExprNode(kStringLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kStringLiteral;
  }
  SMLoc getLoc() const override {
    return SMLoc::getFromPointer(spelling.data());
  }
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
};

struct NoneLiteralNode final : public ExprNode {
  NoneLiteralNode(SMLoc loc) : ExprNode(kNoneLiteral), loc(loc) {}

  const SMLoc loc;

  static bool classof(const ExprNode *node) {
    return node->kind == kNoneLiteral;
  }
  SMLoc getLoc() const override { return loc; }
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
};

struct DeclRefNode final : public ExprNode {
  DeclRefNode(StringRef spelling) : ExprNode(kDeclRef), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) { return node->kind == kDeclRef; }
  SMLoc getLoc() const override {
    return SMLoc::getFromPointer(spelling.data());
  }
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
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
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
};

struct CallNode final : public ExprNode {
  CallNode(ExprNode *callee, SMLoc lparenLoc, ArrayRef<ExprNode *> args)
      : ExprNode(kCall), callee(callee), lparenLoc(lparenLoc), args(args) {}

  ExprNode *const callee;
  const SMLoc lparenLoc;
  const ArrayRef<ExprNode *> args;

  static bool classof(const ExprNode *node) { return node->kind == kCall; }
  SMLoc getLoc() const override { return lparenLoc; }
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
};

/// This represents `A[i,j]`.  In the case of slices (e.g. `A[i, ::]`), the
/// slice will be represented with a subexpression.
struct SubscriptNode final : public ExprNode {
  SubscriptNode(ExprNode *base, SMLoc lsquareLoc, ArrayRef<ExprNode *> indices)
      : ExprNode(kSubscript), base(base), lsquareLoc(lsquareLoc),
        indices(indices) {}

  ExprNode *const base;
  const SMLoc lsquareLoc;
  const ArrayRef<ExprNode *> indices;

  static bool classof(const ExprNode *node) { return node->kind == kSubscript; }
  SMLoc getLoc() const override { return lsquareLoc; }
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
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
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
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
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
};

struct TernaryOpNode final : public ExprNode {
  TernaryOpNode(Kind kind, ExprNode *condExpr, ExprNode *trueExpr,
                ExprNode *falseExpr, SMLoc opLoc)
      : ExprNode(kind), condExpr(condExpr), trueExpr(trueExpr),
        falseExpr(falseExpr), opLoc(opLoc) {}

  ExprNode *const condExpr;
  ExprNode *const trueExpr;
  ExprNode *const falseExpr;
  const SMLoc opLoc;

  SMLoc getLoc() const override { return opLoc; }
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
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
  SMLoc getLoc() const override { return opLoc; }
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
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
  ASTTypeAnd<AnyValue> emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const override;
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPR_NODES_H

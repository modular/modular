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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace M::KGEN::LIT {

/// This node is created to represent erroneous parses, but the diagnostic has
/// already been emitted.
struct ErrorNode final : public ExprNode {
  ErrorNode(SMLoc loc) : ExprNode(kError), loc(loc) {}

  const SMLoc loc;

  static bool classof(const ExprNode *node) { return node->kind == kError; }
  SMLoc getLoc() const override { return loc; }
  bool containsError() const override { return true; }
  AnyValue emitIR(ExprEmitter &state, Type contextualType) const override;
  Type emitType(ExprEmitter &state) const override;
};

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
  bool containsError() const override { return false; }
  AnyValue emitIR(ExprEmitter &state, Type contextualType) const override;
  Type emitType(ExprEmitter &state) const override;
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
  bool containsError() const override { return false; }
  AnyValue emitIR(ExprEmitter &state, Type contextualType) const override;
  Type emitType(ExprEmitter &state) const override;
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
  bool containsError() const override { return false; }
  AnyValue emitIR(ExprEmitter &state, Type contextualType) const override;
  Type emitType(ExprEmitter &state) const override;
};

struct DeclRefNode final : public ExprNode {
  DeclRefNode(StringRef spelling) : ExprNode(kDeclRef), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) { return node->kind == kDeclRef; }
  SMLoc getLoc() const override {
    return SMLoc::getFromPointer(spelling.data());
  }
  bool containsError() const override { return false; }
  AnyValue emitIR(ExprEmitter &state, Type contextualType) const override;
  Type emitType(ExprEmitter &state) const override;
};

struct CallNode final : public ExprNode {
  CallNode(ExprNode *callee, SMLoc lparenLoc, ArrayRef<ExprNode *> args)
      : ExprNode(kCall), callee(callee), lparenLoc(lparenLoc), args(args) {}

  ExprNode *const callee;
  const SMLoc lparenLoc;
  const ArrayRef<ExprNode *> args;

  static bool classof(const ExprNode *node) { return node->kind == kCall; }
  SMLoc getLoc() const override { return lparenLoc; }
  bool containsError() const override {
    return callee->containsError() || llvm::any_of(args, [&](ExprNode *exp) {
             return exp->containsError();
           });
  }
  AnyValue emitIR(ExprEmitter &state, Type contextualType) const override;
  Type emitType(ExprEmitter &state) const override;
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
  bool containsError() const override {
    return base->containsError() || llvm::any_of(indices, [&](ExprNode *exp) {
             return exp->containsError();
           });
  }
  AnyValue emitIR(ExprEmitter &state, Type contextualType) const override;
  Type emitType(ExprEmitter &state) const override;
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
  bool containsError() const override { return subExpr->containsError(); }
  AnyValue emitIR(ExprEmitter &state, Type contextualType) const override;
  Type emitType(ExprEmitter &state) const override;
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
  bool containsError() const override {
    return lhs->containsError() || rhs->containsError();
  }
  AnyValue emitIR(ExprEmitter &state, Type contextualType) const override;
  Type emitType(ExprEmitter &state) const override;
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPR_NODES_H

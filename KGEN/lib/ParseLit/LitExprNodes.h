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
// These classes are formed in the first pass, owned by a bump pointer allocator
// whose lifetime matches the ExprParser class.  These nodes are not allowed to
// own resources because their destructors are never run.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_EXPR_NODES_H
#define LIT_EXPR_NODES_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {
using llvm::SMLoc;
struct EmitterState;
class Scope;

/// When emitting an expression to MLIR as an rvalue, we get a value back that
/// is either an attribute (for parameter expressions) or an SSA value.
using MLIRValueRep = PointerUnion<Attribute, Value>;

/// Base class for all expression nodes.  Note that these nodes are not allowed
/// to own memory since they are bump pointer allocated and their destructors
/// are never run.
struct ExprNode {
  // This indicates the subclass.
  enum Kind {
    kError,         // `
    kIntLiteral,    // 42
    kFloatLiteral,  // 1.1
    kDeclRef,       // x
    kCall,          // thing(a, b)
    kParenExprNode, // (x+y)

    // Binary expressions.
    kAdd,
    kMul,
    kFirstBinOp = kAdd,
    kLastBinOp = kMul,

  } const kind;

  ExprNode(Kind kind) : kind(kind) {}
  virtual ~ExprNode();

  /// Return the primary location for this node for error reporting purposes.
  virtual SMLoc getLoc() const = 0;

  /// Return true if this expression tree contains an already-reported error.
  virtual bool containsError() const = 0;

  /// Emit this expression to MLIR, returning a (possibly null!) MLIRValueRep.
  virtual MLIRValueRep emit(EmitterState &state) const = 0;
};

/// This node is created to represent erroneous parses, but the diagnostic has
/// already been emitted.
struct ErrorNode final : public ExprNode {
  ErrorNode(SMLoc loc) : ExprNode(kError), loc(loc) {}

  const SMLoc loc;

  static bool classof(const ExprNode *node) { return node->kind == kError; }
  SMLoc getLoc() const override { return loc; }
  bool containsError() const override { return true; }
  MLIRValueRep emit(EmitterState &state) const override;
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
  MLIRValueRep emit(EmitterState &state) const override;
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
  MLIRValueRep emit(EmitterState &state) const override;
};

struct DeclRefNode final : public ExprNode {
  DeclRefNode(StringRef spelling) : ExprNode(kDeclRef), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) { return node->kind == kDeclRef; }
  SMLoc getLoc() const override {
    return SMLoc::getFromPointer(spelling.data());
  }
  bool containsError() const override { return false; }
  MLIRValueRep emit(EmitterState &state) const override;
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
  MLIRValueRep emit(EmitterState &state) const override;
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
  MLIRValueRep emit(EmitterState &state) const override;
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
  MLIRValueRep emit(EmitterState &state) const override;
};

//===----------------------------------------------------------------------===//
// EmitterState
//===----------------------------------------------------------------------===//

struct EmitterState {
  /// This is the builder to emit into.
  OpBuilder &builder;

  /// This is scope to resolve declaration references against.
  Scope *scope;

  /// This maps SMLoc's into Location's.
  std::function<Location(SMLoc)> mapLocation;

  /// This is the error handler to emit new diagnostics into.
  std::function<InFlightDiagnostic(SMLoc, const Twine &)> emitError;
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPR_NODES_H

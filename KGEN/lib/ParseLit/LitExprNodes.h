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

#include "LitSharedState.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include <memory>

namespace M::KGEN::LIT {
using llvm::SMLoc;
struct IREmitter;
struct TypeEmitter;
class Scope;

/// When emitting an expression to MLIR as an rvalue, we get a value back that
/// is either an attribute (for parameter expressions) or an SSA value.  The
/// stored attribute is always actually a TypedAttr.
class MLIRValueRep : public PointerUnion<Attribute, Value> {
public:
  using Base = PointerUnion<Attribute, Value>;
  using Base::PointerUnion;

  /// If this contains an Attribute, it is known to be a TypedAttr.  This helper
  /// performs the conversion.  This returns null if this contains a value.
  TypedAttr dyn_castTypedAttr() const;

  /// Return the type for the contained TypedAttr or Value, or null if they are
  /// both null.
  Type getType() const;
};

/// Base class for all expression nodes.  Note that these nodes are not allowed
/// to own memory since they are bump pointer allocated and their destructors
/// are never run.
class ExprNode {
public:
  // This indicates the subclass.
  enum Kind {
    kError,         // `
    kIntLiteral,    // 42
    kFloatLiteral,  // 1.1
    kStringLiteral, // "Hello"
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
  virtual MLIRValueRep emitIR(IREmitter &state) const = 0;

  /// Emit this expression tree to an MLIR type.  This returns null on error,
  /// unlike the corresponding IREmitter method.
  virtual Type emitType(IREmitter &state) const = 0;
};

/// This node is created to represent erroneous parses, but the diagnostic has
/// already been emitted.
struct ErrorNode final : public ExprNode {
  ErrorNode(SMLoc loc) : ExprNode(kError), loc(loc) {}

  const SMLoc loc;

  static bool classof(const ExprNode *node) { return node->kind == kError; }
  SMLoc getLoc() const override { return loc; }
  bool containsError() const override { return true; }
  MLIRValueRep emitIR(IREmitter &state) const override;
  Type emitType(IREmitter &state) const override;
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
  MLIRValueRep emitIR(IREmitter &state) const override;
  Type emitType(IREmitter &state) const override;
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
  MLIRValueRep emitIR(IREmitter &state) const override;
  Type emitType(IREmitter &state) const override;
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
  MLIRValueRep emitIR(IREmitter &state) const override;
  Type emitType(IREmitter &state) const override;
};

struct DeclRefNode final : public ExprNode {
  DeclRefNode(StringRef spelling) : ExprNode(kDeclRef), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) { return node->kind == kDeclRef; }
  SMLoc getLoc() const override {
    return SMLoc::getFromPointer(spelling.data());
  }
  bool containsError() const override { return false; }
  MLIRValueRep emitIR(IREmitter &state) const override;
  Type emitType(IREmitter &state) const override;
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
  MLIRValueRep emitIR(IREmitter &state) const override;
  Type emitType(IREmitter &state) const override;
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
  MLIRValueRep emitIR(IREmitter &state) const override;
  Type emitType(IREmitter &state) const override;
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
  MLIRValueRep emitIR(IREmitter &state) const override;
  Type emitType(IREmitter &state) const override;
};

//===----------------------------------------------------------------------===//
// IREmitter
//===----------------------------------------------------------------------===//

struct IREmitter {
  /// This is the shared state for the parser overall.
  LitSharedState &shared;

  /// This is scope to resolve declaration references against.
  Scope &scope;

  /// This is the current builder to emit into if we are allowed to generate a
  /// value.  This will be None when in a context that only allows parameters.
  /// It is mutable to support expressions that require internal control flow.
  Optional<OpBuilder> builder;

  IREmitter(LitSharedState &shared, Scope &scope, Optional<OpBuilder> builder)
      : shared(shared), scope(scope), builder(builder) {}

  MLIRContext *getContext() const { return shared.context; }

  /// This helper emits the specified value rep as an SSA value, materializing
  /// it as a parameter constant if it is a parameter.  This returns null if
  /// emission fails.
  Value emitAsValue(MLIRValueRep rep, SMLoc loc);

  /// This helper emits the specified value rep as an SSA value, materializing
  /// it as a parameter constant if it is a parameter.  This returns null if
  /// emission fails.
  Value emitAsValue(const ExprNode *node);

  /// This helper emits the specified expression tree as a type, e.g. turning
  /// "Int" into the type for it.  This never returns null - if the expression
  /// is erroneous, it is diagnosed and a TypeCheckErrorType is returned.
  Type emitAsType(const ExprNode *node);

  /// Emit an error through the parser's logic.
  InFlightDiagnostic emitError(SMLoc loc, const Twine &twine) const {
    shared.errorOccurred = true;
    return mlir::emitError(translateLocation(loc), twine);
  }

  /// Translate an SMLoc into an MLIR Location.
  Location translateLocation(SMLoc loc) const {
    return shared.translateLocation(loc);
  }
};

} // namespace M::KGEN::LIT

namespace llvm {

template <typename To>
struct CastInfo<To, const M::KGEN::LIT::MLIRValueRep>
    : public CastInfo<To, const M::KGEN::LIT::MLIRValueRep::Base> {};
template <typename To>
struct CastInfo<To, M::KGEN::LIT::MLIRValueRep>
    : public CastInfo<To, M::KGEN::LIT::MLIRValueRep::Base> {};

} // namespace llvm

#endif // LIT_EXPR_NODES_H

//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides machinery used when emitting expressions to MLIR, either
// as operations for runtime values or as attributes for metavalues.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_EXPRS_H
#define LIT_EXPRS_H

#include "IRValues.h"
#include "LitSharedState.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {
using llvm::SMLoc;
class ExprEmitter;
class ASTDecl;

template <typename ValueType>
struct ASTTypeAnd {
  ValueType ir; // This is the IR representation of this.
  ASTType type; // This is the AST type.

  bool isNull() const { return ir; }
  bool operator!() const { return !ir; }
  operator bool() const { return ir; }

  FullType getFullType() const { return {ir.getType(), type}; }
};

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
    kIntLiteral,    // 42
    kFloatLiteral,  // 1.1
    kStringLiteral, // "Hello"
    kNoneLiteral,   // None
    kDeclRef,       // x
    kAttributeRef,  // x.y
    kCall,          // thing(a, b)
    kSubscript,     // thing[a, b:c]
    kParenExprNode, // (x+y)
    kListExprNode,  // [x, y]

    // Unary expressions.
    kUnaryMinus,
    kUnaryPlus,
    kUnaryTilde,
    kUnaryAmp,
    kFirstUnaryOp = kUnaryMinus,
    klastUnaryOp = kUnaryAmp,

    // Binary expressions.
    kAdd,
    kSub,
    kMul,
    kMatrixMul,
    kDiv,
    kFloorDiv,
    kModulo,
    kBoolOr,
    kBoolAnd,
    kBoolNot,
    kCmpIn,
    kCmpNotIn,
    kCmpIs,
    kCmpIsNot,
    kCmpLess,
    kCmpLessEqual,
    kCmpGreater,
    kCmpGreaterEqual,
    kCmpNotEqual,
    kCmpEqual,
    kBitwiseOr,
    kBitwiseXor,
    kBitwiseAnd,
    kLeftShift,
    kRightShift,
    kExp,
    kFirstBinOp = kAdd,
    kLastBinOp = kExp,

    // Ternary expressions.
    kIfElse,
  } const kind;

  ExprNode(Kind kind) : kind(kind) {}
  virtual ~ExprNode();

  /// Return the primary location for this node for error reporting purposes.
  virtual SMLoc getLoc() const = 0;

  /// Emit this expression to MLIR, returning a (possibly null!) AnyValue.  The
  /// contextualType (if non-null) indicates the contextual type to use for an
  /// implicitly declared value, e.g. a/b in `def f(): (a,b) = (1,2)`.
  virtual ASTTypeAnd<AnyValue> emitIR(ExprEmitter &state,
                                      FullType contextualType = {}) const = 0;
};

//===----------------------------------------------------------------------===//
// ExprEmitter
//===----------------------------------------------------------------------===//

class ExprEmitter {
public:
  /// This is the shared state for the parser overall.
  LitSharedState &shared;

  /// This is scope to resolve declaration references against.
  ASTDecl &declScope;

  /// This is the current builder to emit into if we are allowed to generate a
  /// value.  This will be None when in a context that only allows parameters.
  /// It is mutable to support expressions that require internal control flow.
  Optional<OpBuilder> builder;

  /// When non-null, implicitly declared variables are added above this
  /// location.
  Operation *varDeclCursor;

  ExprEmitter(LitSharedState &shared, ASTDecl &declScope,
              Optional<OpBuilder> builder, Operation *varDeclCursor)
      : shared(shared), declScope(declScope), builder(builder),
        varDeclCursor(varDeclCursor) {}

  MLIRContext *getContext() const { return shared.context; }

  /// This helper emits the specified value rep as an RValue.
  ASTTypeAnd<RValue> emitRValue(const ExprNode *node) {
    assert(node && "cannot emit a null node");
    return emitRValue(node->emitIR(*this), node->getLoc());
  }
  ASTTypeAnd<RValue> emitRValue(ASTTypeAnd<AnyValue> rep, SMLoc loc);

  /// This helper emits the specified value rep as a DRValue which has an SSA
  /// value representation, materializing MValues and loading LValues as
  /// needed.  This returns null if emission fails.
  ASTTypeAnd<DRValue> emitDRValue(ASTTypeAnd<RValue> rep, SMLoc loc);
  ASTTypeAnd<DRValue> emitDRValue(ASTTypeAnd<AnyValue> rep, SMLoc loc) {
    return emitDRValue(emitRValue(rep, loc), loc);
  }

  /// This helper emits the specified value rep as an DRValue, materializing
  /// it as a parameter constant if it is a parameter.  This returns null if
  /// emission fails.
  ASTTypeAnd<DRValue> emitDRValue(const ExprNode *node) {
    assert(node && "cannot emit a null node");
    return emitDRValue(node->emitIR(*this), node->getLoc());
  }

  /// This helper emits the specified expression as a meta value, diagnosing the
  /// problem if the expression is only valid as a runtime value (using the
  /// specified message).  This returns null if emission fails.
  ASTTypeAnd<MValue> emitMValue(const ExprNode *node, const Twine &message);

  ASTTypeAnd<MAValue> emitMAValue(const ExprNode *node, const Twine &message) {
    auto mValue = emitMValue(node, message);
    if (!mValue.ir)
      return {};
    return {MAValue(mValue.ir.lowerToAttribute(shared, node->getLoc())),
            mValue.type};
  }

  /// Emit the specified expression as an LValue which can be loaded and stored.
  /// If contextualType is non-null, then an implicitly declared LValue will be
  /// assigned that type.
  ///
  /// This diagnoses the expression with the specified message if it isn't a
  /// valid LValue.
  ASTTypeAnd<LValue> emitLValue(const ExprNode *node, FullType contextualType,
                                const Twine &message);

  /// This helper emits the specified expression tree as a type, e.g. turning
  /// "Int" into the type for it.  This never returns null MLIR Types - if the
  /// expression is erroneous, it is diagnosed and a TypeCheckErrorType is
  /// returned, along with an erroneous AST type.
  FullType emitType(const ExprNode *node);

  /// Perform a name lookup in the current scope and return the named
  /// declaration.  This emits an error and returns null on error.
  ASTDecl *lookupDecl(StringRef name, SMLoc loc, ASTDecl &scope,
                      Twine errorMessage);

  /// Emit an error through the parser's logic.
  InFlightDiagnostic emitError(SMLoc loc, const Twine &twine) const {
    return shared.emitError(loc, twine);
  }

  /// Translate an SMLoc into an MLIR Location.
  Location translateLocation(SMLoc loc) const {
    return shared.translateLocation(loc);
  }
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPRS_H

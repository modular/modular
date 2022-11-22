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
#include "SpecialFunctions.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {
using llvm::SMLoc;
class ExprEmitter;
class ASTDecl;
class ExprNode;

template <typename ValueType>
struct ASTExprAnd {
  ValueType ir;

  bool isNull() const { return ir; }
  bool operator!() const { return !ir; }
  operator bool() const { return ir; }

  /// This is the expression a value was produced from, carrying location and
  /// additional semantic information.
  const ExprNode *expr;
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
    kSlice,         // :, a:, :a, ::, a:b:c, etc.  Only valid in subscripts.
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
    kAssign,
    kPlusAssign,
    kMinusAssign,
    kMulAssign,
    kMatMulAssign,
    kDivAssign,
    kModuloAssign,
    kBitwiseAndAssign,
    kBitwiseOrAssign,
    kBitwiseXorAssign,
    kLeftShiftAssign,
    kRightShiftAssign,
    kExpAssign,
    kFloorDivAssign,
    kFirstAssignStmt = kAssign,
    kLastAssignStmt = kFloorDivAssign,
    kFirstBinOp = kAdd,
    kLastBinOp = kFloorDivAssign,

    // Ternary expressions.
    kIfElse,
  } const kind;

  ExprNode(Kind kind) : kind(kind) {}
  virtual ~ExprNode();

  /// Return the primary location for this node for error reporting purposes.
  virtual SMLoc getLoc() const = 0;
  /// Return the source range spanned by this expression.
  virtual llvm::SMRange getRange() const = 0;

  /// Emit this expression to MLIR, returning a (possibly null!) AnyValue.  The
  /// contextualType (if non-null) indicates the contextual type to use for an
  /// implicitly declared value, e.g. a/b in `def f(): (a,b) = (1,2)`.
  virtual AnyValue emitIR(ExprEmitter &emitter,
                          ASTType contextualType = {}) const = 0;
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
  RValue emitRValue(const ExprNode *node) {
    assert(node && "cannot emit a null node");
    return emitRValue(node->emitIR(*this), node->getLoc());
  }
  RValue emitRValue(AnyValue rep, SMLoc loc);

  /// This helper emits the specified value rep as a DRValue which has an SSA
  /// value representation, materializing MValues and loading LValues as
  /// needed.  This returns null if emission fails.
  DRValue emitDRValue(RValue rep, SMLoc loc);
  DRValue emitDRValue(AnyValue rep, SMLoc loc) {
    return emitDRValue(emitRValue(rep, loc), loc);
  }

  /// This helper emits the specified value rep as an DRValue, materializing
  /// it as a parameter constant if it is a parameter.  This returns null if
  /// emission fails.
  DRValue emitDRValue(const ExprNode *node) {
    assert(node && "cannot emit a null node");
    return emitDRValue(node->emitIR(*this), node->getLoc());
  }

  /// This helper emits a method call to a special function (`kind`) on `type`
  /// with the provided `operands`. This emits an error if the special function
  /// is not implemented by the type and returns null.
  AnyValue emitSpecialMethodCall(ASTType type, SpecialFunctionKind kind,
                                 ArrayRef<ASTExprAnd<AnyValue>> operands,
                                 SMLoc callLoc);

  /// This helper emits the specified expression as a meta value, diagnosing the
  /// problem if the expression is only valid as a runtime value (using the
  /// specified message).  This returns null if emission fails.
  MValue emitMValue(const ExprNode *node, const Twine &message);

  /// Emit the specified expression as an LValue which can be loaded and stored.
  /// If contextualType is non-null, then an implicitly declared LValue will be
  /// assigned that type.
  ///
  /// This diagnoses the expression with the specified message if it isn't a
  /// valid LValue.
  LValue emitLValue(const ExprNode *node, ASTType contextualType,
                    const Twine &message);

  /// This helper emits the specified expression tree as a type, e.g. turning
  /// "Int" into the type for it.  This never returns null MLIR Types - if the
  /// expression is erroneous, it is diagnosed and a TypeCheckErrorType is
  /// returned, along with an erroneous AST type.
  ASTType emitType(const ExprNode *node);

  /// Perform a name lookup in the current scope and return the named
  /// declaration.  This emits an error and returns null on error.  When
  /// 'implicitDeclType' is non-null, a name lookup failure will insert a new
  /// variable declaration in the scope.
  ASTDecl *lookupDecl(StringRef name, SMLoc loc, ASTDecl &scope,
                      std::function<void(InFlightDiagnostic)> errorFn,
                      ASTType implicitDeclType = {});

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

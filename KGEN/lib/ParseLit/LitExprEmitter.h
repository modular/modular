//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_EXPREMITTER_H
#define LIT_EXPREMITTER_H

#include "IRValues.h"
#include "LitExprNode.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {
template <typename ValueType>
struct ASTExprAnd;
enum class SpecialFunctionKind : uint8_t;
class SpecialFunctionInfo;
enum class CallSyntax : uint8_t;
class ExprEmitter;

/// This class represents the destination context than an expression is being
/// emitted, when it may produce an RValue.  Example destinations include:
///   - an LValue:
///       This handles cases like `a.b = 42` or `var x: Int = 42`, as well as
///       a return slot with memory-primary results in `return x()`.  In this
///       case, the emitted expression must conform to type of the LValue.
///   - an untyped var decl, e.g. `var x = 42`
///       In this case, the ExprNode conforms to the initializer expression.
///   - an ExprNode:
///       This handles assignments to "targets" (Python nomenclature), e.g.:
///          1) a discard pattern, e.g. `_ = 42`
///          2) an implicitly declared var decl, e.g. `x = 42` in a def.
///          3) tuples and lists thereof, e.g. `(a, _) = foo()`
///       In this case, the ExprNode type often conforms to the expression.
///
/// Any expression may also have no proscribed result (as in the case of
/// `someExpr()`), in which case emission will create storage when needed on
/// demand.
class ValueDest {
public:
  /*implicit*/
  ValueDest(const ExprNode *target = nullptr) : representation(target) {}
  ValueDest(LValue dest) : representation(dest) {}
  ValueDest(const ValueDest &) = default;

private:
  //  This should only be accessed by ExprEmitter::emitResult.
  friend class ExprEmitter;
  // TODO: Operation* for let/vardecl.
  PointerUnion<LValue, const ExprNode *> representation;
};

/// This class is the main driver for expression emission, providing helper
/// functions used by the individual node emission hooks.
class ExprEmitter : public LitSharedStateUser {
public:
  ExprEmitter(LitSharedState &shared, ASTDecl &declScope,
              std::optional<OpBuilder> builder, Operation *varDeclCursor)
      : LitSharedStateUser(shared), builder(builder), declScope(declScope),
        varDeclCursor(varDeclCursor) {}

  //===--------------------------------------------------------------------===//
  // Emitter State.

  /// This is the current builder to emit into if we are allowed to generate a
  /// value.  This will be None when in a context that only allows parameters.
  /// It is mutable to support expressions that require internal control flow.
  std::optional<OpBuilder> builder;

  /// This is scope to resolve declaration references against.
  ASTDecl &declScope;

  /// When non-null, implicitly declared variables are added above this op.
  Operation *varDeclCursor;

  //===--------------------------------------------------------------------===//
  // Emission helpers for various value classifications.

  /// This helper emits the specified value as an RValue.
  RValue emitRValue(ASTExprAnd<AnyValue> value, ValueDest dest);
  CRValue emitCRValue(ASTExprAnd<AnyValue> value, ValueDest dest);

  /// This helper emits the specified value as a SRValue which has an SSA
  /// value representation, materializing PRValues and loading LValues as
  /// needed.  This returns null if emission fails.
  SRValue emitSRValue(ASTExprAnd<AnyValue> value);

  //===--------------------------------------------------------------------===//
  // Function Calls

  /// This helper emits a named method call with the provided `argValues`, where
  /// the first arg is the receiver of the call. This emits an error if the
  /// call is invalid and returns null.  The argValues list may not be empty.
  ///
  /// `callNode` is the call like expression (e.g. a CallNode, binary operator,
  /// etc) that results in the call, or potentially a random value that is being
  /// fed into an implicit conversion.  This should only be used for location
  /// information.
  AnyValue emitNamedMethodCall(StringRef methodName,
                               ArrayRef<ASTExprAnd<AnyValue>> argValues,
                               ValueDest dest, CallSyntax syntax,
                               const ExprNode *callNode);

  /// Emit an indirect call to a resolved value, checking for compatibility and
  /// then generating the call logic.  This emits an error and returns null on
  /// failure.
  AnyValue emitIndirectCall(CRValue callee,
                            ArrayRef<ASTExprAnd<AnyValue>> operands,
                            ValueDest dest, const ExprNode *callExpr);

  /// Emit call to a resolved and /already type checked/ callee. This does not,
  /// check for compatibility and isn't prepared to emit errors.
  AnyValue emitCallUnchecked(CRValue callee,
                             ArrayRef<ASTExprAnd<AnyValue>> operands,
                             ArrayRef<ParamDeclAttr> resultParams,
                             ValueDest dest, const ExprNode *callExpr);

  /// Return true if 'value' may be implicitly converted to 'requiredType'
  /// by invoking (one level of) conversion operations.  This does not generate
  /// any IR.
  bool canImplicitlyConvertToType(ASTExprAnd<AnyValue> value,
                                  ASTType requiredType);

  /// Convert the specified value to the expected type, invoking implicit
  /// conversions if necessary.  On error, this diagnoses it and returns null.
  AnyValue getAsExpectedType(ASTExprAnd<AnyValue> value, ASTType expectedType,
                             ValueDest dest, const Twine &errorSuffix);

  /// Convert the specified value to the expected type, invoking implicit
  /// conversions if necessary.  On error, this invokes the specified closure to
  /// diagnose the problem and returns null.
  AnyValue getAsExpectedType(ASTExprAnd<AnyValue> value, ASTType expectedType,
                             ValueDest dest,
                             std::function<void()> errorHandler);

  /// Emit the specified expression as a condition, converting it to an MLIR I1
  /// value that we can test directly, and also returning the intermediate
  /// result of calling `__bool__` (which is typically a Bool or object type,
  /// but not guaranteed).  This reports and error and returns null on error.
  RValue emitConditionValueAsI1(ASTExprAnd<AnyValue> expr,
                                AnyValue &boolResult);

  //===--------------------------------------------------------------------===//
  // Emission helpers for various value classifications.

  /// Emit the specified value into the current destination if present.  This
  /// accepts (and silently propagates) null values.
  AnyValue emitResult(AnyValue value, const ExprNode *node, ValueDest dest);

  AnyValue emitResult(TypedAttr value, const ExprNode *node, ValueDest dest) {
    return emitResult(AnyValue(value), node, dest);
  }
  AnyValue emitResult(ASTType value, const ExprNode *node, ValueDest dest) {
    return emitResult(AnyValue(value), node, dest);
  }
  AnyValue emitResult(PRValue value, const ExprNode *node, ValueDest dest) {
    return emitResult(AnyValue(value), node, dest);
  }

  /// This method is used by node implementations of emitExprResultIntoPattern
  /// to emit the result once they determine an lvalue to use.
  AnyValue emitExprResultIntoLValue(ASTExprAnd<AnyValue> value, LValue dest);

  /// This helper emits the specified value rep as an RValue.
  RValue emitExprRValue(const ExprNode *node, ValueDest dest);

  /// This helper emits the specified value rep as an RValue.
  CRValue emitExprCRValue(const ExprNode *node, ValueDest dest);

  /// This helper emits the specified value rep as an SRValue, materializing
  /// it as an operation if it is a parameter.  This returns null if emission
  /// fails.
  SRValue emitExprSRValue(const ExprNode *node);

  /// This helper emits the specified expression as a meta value, and optionally
  /// converts the result to a specified expected type.  This emits an error if
  /// the expression cannot be emitted, if it cannot be converted to the
  /// expected type, or if it isn't a valid runtime value.  This returns null if
  /// emission fails.
  PRValue emitExprPRValue(const ExprNode *node, ASTType resultType,
                          const Twine &errorSuffix);

  /// Emit the specified expression as an LValue which can be loaded and stored.
  /// If contextualType is non-null, then an implicitly declared LValue will be
  /// assigned that type.
  ///
  /// This diagnoses the expression with the specified message if it isn't a
  /// valid LValue.
  LValue emitExprLValue(SMLoc loc, const ExprNode *node, ValueDest dest,
                        const Twine &message);

  /// This helper emits the specified expression tree as a type, e.g. turning
  /// "Int" into the type for it.  This emits an error and returns null on
  /// failure.
  ASTType emitExprType(const ExprNode *node);

  /// Emit the specified expression as a condition, converting it to an MLIR I1
  /// value that we can test directly.  This reports and error and returns null
  /// on error.
  RValue emitExprConditionValueAsI1(const ExprNode *condExpr);

  /// Given a value convertable to a pop int via index conversion, emit
  /// the casting code and return the pop scalar index value
  SRValue emitBoxedIntAsPopScalar(Value numberValue, const ExprNode *source);
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPREMITTER_H

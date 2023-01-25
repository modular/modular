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

/// This class is the main driver for expression emission, providing helper
/// functions used by the individual node emission hooks.
class IREmitter : public LitSharedStateUser {
public:
  //===--------------------------------------------------------------------===//
  // General Emitter State.

  /// This is the current builder to emit into if we are allowed to generate a
  /// value.  This will be None when in a context that only allows parameters.
  /// It is mutable to support expressions that require internal control flow.
  std::optional<OpBuilder> builder;

  IREmitter(LitSharedState &shared, std::optional<OpBuilder> builder)
      : LitSharedStateUser(shared), builder(builder) {}

  //===--------------------------------------------------------------------===//
  // Emission helpers for various value classifications.

  /// This helper emits the specified value as an RValue.
  RValue emitRValue(ASTExprAnd<AnyValue> value);

  /// This helper emits the specified value as a DRValue which has an SSA
  /// value representation, materializing MValues and loading LValues as
  /// needed.  This returns null if emission fails.
  DRValue emitDRValue(ASTExprAnd<RValue> value);
  DRValue emitDRValue(ASTExprAnd<AnyValue> value);

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
                               CallSyntax syntax, const ExprNode *callNode);

  /// Convert the specified value to the expected type, invoking implicit
  /// conversions if necessary.  On error, this diagnoses it and returns null.
  AnyValue getAsExpectedType(AnyValue value, const ExprNode *expr,
                             ASTType expectedType, const Twine &errorSuffix);

  /// Convert the specified value to the expected type, invoking implicit
  /// conversions if necessary.  On error, this invokes the specified closure to
  /// diagnose the problem and returns null.
  AnyValue getAsExpectedType(AnyValue value, const ExprNode *expr,
                             ASTType expectedType,
                             std::function<void()> errorHandler);

  /// Emit the specified expression as a condition, converting it to an MLIR I1
  /// value that we can test directly, and also returning the intermediate
  /// result of calling `__bool__` (which is typically a Bool or object type,
  /// but not guaranteed).  This reports and error and returns null on error.
  DRValue emitConditionValueAsI1(ASTExprAnd<AnyValue> expr,
                                 AnyValue &boolResult);
};

/// ExprEmitter refines IREmitter, providing the additional state needed to
/// emit arbitrary nodes that require name lookup and declaration synthesis.
class ExprEmitter : public IREmitter {
public:
  ExprEmitter(LitSharedState &shared, ASTDecl &declScope,
              std::optional<OpBuilder> builder, Operation *varDeclCursor)
      : IREmitter(shared, builder), declScope(declScope),
        varDeclCursor(varDeclCursor) {}

  /// This is scope to resolve declaration references against.
  ASTDecl &declScope;

  /// When non-null, implicitly declared variables are added above this op.
  Operation *varDeclCursor;

  //===--------------------------------------------------------------------===//
  // Emission helpers for various value classifications.

  /// This helper emits the specified value rep as an RValue.
  RValue emitExprRValue(const ExprNode *node);

  /// This helper emits the specified value rep as an DRValue, materializing
  /// it as a parameter constant if it is a parameter.  This returns null if
  /// emission fails.
  DRValue emitExprDRValue(const ExprNode *node);

  /// This helper emits the specified expression as a meta value, and optionally
  /// converts the result to a specified expected type.  This emits an error if
  /// the expression cannot be emitted, if it cannot be converted to the
  /// expected type, or if it isn't a valid runtime value.  This returns null if
  /// emission fails.
  MValue emitExprMValue(const ExprNode *node, ASTType resultType,
                        const Twine &errorSuffix);

  /// Emit the specified expression as an LValue which can be loaded and stored.
  /// If contextualType is non-null, then an implicitly declared LValue will be
  /// assigned that type.
  ///
  /// This diagnoses the expression with the specified message if it isn't a
  /// valid LValue.
  LValue emitExprLValue(SMLoc loc, const ExprNode *node, ASTType contextualType,
                        const Twine &message);

  /// This helper emits the specified expression tree as a type, e.g. turning
  /// "Int" into the type for it.  This emits an error and returns null on
  /// failure.
  ASTType emitExprType(const ExprNode *node);

  /// Emit the specified expression as a condition, converting it to an MLIR I1
  /// value that we can test directly.  This reports and error and returns null
  /// on error.
  DRValue emitExprConditionValueAsI1(const ExprNode *condExpr);

  /// Given a value convertable to a pop int via index conversion, emit
  /// the casting code and return the pop scalar index value
  DRValue emitBoxedIntAsPopScalar(Value numberValue, const ExprNode *source);
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPREMITTER_H

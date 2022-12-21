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

/// This class is the main driver for expression emission, providing helper
/// functions used by the individual node emission hooks.
class ExprEmitter {
public:
  //===--------------------------------------------------------------------===//
  // General Emitter State.

  /// This is the shared state for the parser overall.
  LitSharedState &shared;

  /// This is scope to resolve declaration references against.
  ASTDecl &declScope;

  /// This is the current builder to emit into if we are allowed to generate a
  /// value.  This will be None when in a context that only allows parameters.
  /// It is mutable to support expressions that require internal control flow.
  Optional<OpBuilder> builder;

  /// When non-null, implicitly declared variables are added above this op.
  Operation *varDeclCursor;

  ExprEmitter(LitSharedState &shared, ASTDecl &declScope,
              Optional<OpBuilder> builder, Operation *varDeclCursor)
      : shared(shared), declScope(declScope), builder(builder),
        varDeclCursor(varDeclCursor) {}

  MLIRContext *getContext() const { return shared.context; }

  /// Emit an error through the parser's logic.
  InFlightDiagnostic emitError(SMLoc loc, const Twine &twine = "") const {
    return shared.emitError(loc, twine);
  }

  /// Translate an SMLoc into an MLIR Location.
  Location translateLocation(SMLoc loc) const {
    return shared.translateLocation(loc);
  }
  //===--------------------------------------------------------------------===//
  // Emission helpers for various value classifications.

  /// This helper emits the specified value rep as an RValue.
  RValue emitRValue(const ExprNode *node) {
    assert(node && "cannot emit a null node");
    return emitRValue(node->emitIR(*this, /*No Contextual Type*/ {}),
                      node->getLoc());
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
    return emitDRValue(node->emitIR(*this, /*No Contextual Type*/ {}),
                       node->getLoc());
  }

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

  //===--------------------------------------------------------------------===//
  // Function Calls

  /// This helper emits a method call to a special function (`kind`) on `type`
  /// with the provided `operands`. This emits an error if the special function
  /// is not implemented by the type and returns null.
  AnyValue emitNamedMethodCall(ASTType type, StringRef methodName,
                               ArrayRef<ASTExprAnd<AnyValue>> operands,
                               SMLoc callLoc);

  /// Convert the specified DRValue to the expected type, invoking implicit
  /// conversions if necessary.  On error, this diagnoses it and returns null.
  DRValue getAsExpectedType(DRValue value, const ExprNode *expr,
                            ASTType expectedType);

  /// Emit the specified expression as a condition, converting it to an MLIR I1
  /// value that we can test directly.  This reports and error and returns null
  /// on error.
  DRValue emitConditionValueAsI1(ExprNode *condExpr);

  /// Emit the specified expression as a condition, converting it to an MLIR I1
  /// value that we can test directly, and also returning the intermediate
  /// result of calling `__bool__` (which is typically a Bool or object type,
  /// but not guaranteed).  This reports and error and returns null on error.
  DRValue emitConditionValueAsI1(ASTExprAnd<AnyValue> expr,
                                 AnyValue &boolResult);
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPREMITTER_H
